#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Replay MegaMoE dgrad and wgrad on the exact expert splits in grouped GEMM tags.

NN/TN RTC tags are dgrad/wgrad. RTC NT forward tags generate both adjoints.
cuBLAS uses the public operand order: NT with a ragged K is accumulating wgrad.
The tags identify the input dtype only. Exercise both BF16 and FP32 gradient
buffers for accumulating wgrad. Each CSV row retains the original line number.
"""
import argparse
import csv
import ctypes
from dataclasses import dataclass
from itertools import accumulate
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import subprocess

from replay_tag import extent, parse_tag


@dataclass
class Case:
    line: int
    source: str
    layout: str
    tile: str
    origin: str
    operation: str
    splits: list[int]
    out_features: int
    in_features: int
    accumulate: bool
    dst_dtype: str


def read_cases(path):
    cases = []
    for line, text in enumerate(path.read_text().splitlines(), 1):
        if not text.strip():
            continue
        source, f = parse_tag(text)
        g, layout = int(f["groups"]), f["layout"]
        dims = []
        for key in ("m", "n", "k"):
            v = extent(f[key])
            if not f[key].startswith("["):
                v *= g
            if len(v) != g or min(v) <= 0:
                raise ValueError(f"Invalid {key} at line {line}")
            dims.append(v)
        m, n, k = dims
        acc = bool(int(f["accumulate"]))
        if f["data_type"] != "bf16":
            raise ValueError(f"Expected BF16 tag at line {line}")
        if source == "GroupedGemmRtc" and layout == "NT" and not acc:
            splits, out_features, in_features = m, n, k
            origin, ops = "from_forward", ("dgrad", "wgrad")
        elif source == "GroupedGemmRtc" and layout == "NN" and not acc:
            splits, out_features, in_features = m, k, n
            origin, ops = "captured", ("dgrad",)
        elif (source, layout) in (("GroupedGemmRtc", "TN"), ("GroupedGemmCublas", "NT")):
            splits, out_features, in_features = k, m, n
            origin, ops = "captured", ("wgrad",)
        else:
            raise ValueError(f"Unsupported tag at line {line}: {text}")
        if len(set(out_features)) != 1 or len(set(in_features)) != 1:
            raise ValueError(f"Expected shared expert weight shape at line {line}")
        for op in ops:
            for dst in ("bf16", "fp32") if acc else ("bf16",):
                cases.append(
                    Case(
                        line,
                        source,
                        layout,
                        f.get("tile", ""),
                        origin,
                        op,
                        splits,
                        out_features[0],
                        in_features[0],
                        acc,
                        dst,
                    )
                )
    return cases


class LegacyKernels:
    """Compile the specified revision's original RTC source and use its production launch geometry."""

    def __init__(self, revision, include):
        from cuda.bindings import nvrtc

        rtc_dir = "transformer_engine/common/rtc/"
        repo = Path(__file__).resolve().parents[2]

        def source(name):
            return subprocess.check_output(["git", "show", f"{revision}:{rtc_dir}{name}"], cwd=repo)

        def checked(result):
            if result[0].value != 0:
                raise RuntimeError(result)
            return result[1] if len(result) == 2 else None

        program = checked(
            nvrtc.nvrtcCreateProgram(
                source("mega_moe_backward.cu"),
                b"mega_moe_backward.cu",
                1,
                [source("mega_moe_device.cuh")],
                [b"mega_moe_device.cuh"],
            )
        )
        opts = [
            b"--std=c++17",
            b"--gpu-architecture=sm_89",
            f"-I{include}".encode(),
            b"-DNVTE_MOE_FP16=0",
        ]
        rc = nvrtc.nvrtcCompileProgram(program, len(opts), opts)
        if rc[0].value:
            log = bytearray(checked(nvrtc.nvrtcGetProgramLogSize(program)))
            nvrtc.nvrtcGetProgramLog(program, log)
            raise RuntimeError(log.decode(errors="replace"))
        binary = bytearray(checked(nvrtc.nvrtcGetCUBINSize(program)))
        checked(nvrtc.nvrtcGetCUBIN(program, binary))
        checked(nvrtc.nvrtcDestroyProgram(program))
        self.driver = ctypes.CDLL("libcuda.so.1")
        self.module = ctypes.c_void_p()
        image = ctypes.create_string_buffer(bytes(binary))
        self._call("cuModuleLoadData", ctypes.byref(self.module), image)
        self.functions = {}
        for op in ("dgrad", "wgrad"):
            fn = ctypes.c_void_p()
            self._call(
                "cuModuleGetFunction", ctypes.byref(fn), self.module, f"nvte_mega_moe_{op}".encode()
            )
            self.functions[op] = fn

    def _call(self, name, *args):
        status = getattr(self.driver, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with CUDA status {status}")

    def bind(self, case, dy, x, offsets, weights, out):
        import torch

        # The old implementation writes a fresh input-dtype wgrad. Its external
        # accumulation reads this temporary and the existing destination buffer.
        dweight = out if not case.accumulate else [torch.empty_like(t, dtype=dy.dtype) for t in out]
        pointers = weights if case.operation == "dgrad" else dweight
        table = torch.tensor([v.data_ptr() for v in pointers], dtype=torch.int64, device=dy.device)
        ptr = ctypes.c_void_p
        rows, o, i, g = len(dy), case.out_features, case.in_features, len(case.splits)
        if case.operation == "dgrad":
            raw = [dy, offsets, table, out]
            blocks = ((rows + 15) // 16 + g) * ((i + 63) // 64)
        else:
            raw = [dy, x, offsets, table]
            blocks = ((o + 15) // 16) * ((i + 63) // 64) * g
        values = [ptr(t.data_ptr()) for t in raw] + [ctypes.c_int(v) for v in [rows, o, i, g, 0]]
        args = (ptr * len(values))(*(ctypes.addressof(v) for v in values))
        stream = ptr(torch.cuda.current_stream().cuda_stream)
        kernel = self.functions[case.operation]

        def launch():
            # Retain the table and scalar storage until the last launch completes.
            _ = table, values
            self._call(
                "cuLaunchKernel", kernel, min(blocks, 65535), 1, 1, 128, 1, 1, 0, stream, args, None
            )
            if case.accumulate:
                torch._foreach_add_(out, dweight)
            return out

        return launch


def samples(case, dy, x, weights, out, initial, backend):
    """Compare 16 full reduction dot products in every expert to independent FP32 arithmetic."""
    import torch

    expected, observed = [], []
    begin = 0
    for e, length in enumerate(case.splits):
        o, i = case.out_features, case.in_features
        j = torch.arange(16, device=dy.device)
        column = (101 * j + 31 * e + 3) % i
        if case.operation == "dgrad":
            row = (997 * j + 61 * e + 1) % length
            lhs = dy[begin + row].float()
            rhs = weights[e][:, column].T.float()
            got = out[begin + row, column].float()
        else:
            row = (383 * j + 17 * e + 5) % o
            lhs = dy[begin : begin + length, row].T.float()
            rhs = x[begin : begin + length, column].T.float()
            got = out[e][row, column].float()
        target = (lhs * rhs).sum(1)
        if case.accumulate:
            target = target + initial
        expected.append(target)
        observed.append(got)
        begin += length
    ref, got = torch.cat(expected), torch.cat(observed)
    rel = float(torch.linalg.vector_norm(ref - got) / torch.linalg.vector_norm(ref))
    threshold = 0.009 if case.accumulate else 0.0045
    if case.dst_dtype == "fp32" and backend != "legacy":
        # Catch an unintended BF16 round trip before the FP32 accumulation.
        threshold = 2e-5
    if not math.isfinite(rel) or rel > threshold:
        raise AssertionError(f"line {case.line} {case.operation} sampled relative L2 {rel:g}")
    torch.testing.assert_close(got, ref, rtol=0.012, atol=0.02)
    return rel


def run(case, backend, legacy, warmup, runs, iterations):
    import torch

    device = "cuda"
    g, o, i = len(case.splits), case.out_features, case.in_features
    rows = sum(case.splits)
    init = 1.625 if case.accumulate else 0.0
    generator = torch.Generator(device=device).manual_seed(1701 + case.line)

    def tensor(shape, scale):
        t = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
        return t.mul_(scale)

    dy = tensor((rows, o), 1.0)
    offsets = torch.tensor([0, *accumulate(case.splits)], dtype=torch.int32, device=device)
    x, weights = None, None
    if case.operation == "dgrad":
        weights = [tensor((o, i), o**-0.5) for _ in range(g)]
        out = (
            torch.empty((rows, i), dtype=dy.dtype, device=device) if backend != "megamoe" else None
        )
    else:
        x = tensor((rows, i), (rows / g) ** -0.5)
        dtype = torch.bfloat16 if case.dst_dtype == "bf16" else torch.float32
        out = [torch.full((o, i), init, dtype=dtype, device=device) for _ in range(g)]
    if backend == "legacy":
        launch = legacy.bind(case, dy, x, offsets, weights, out)
    elif backend == "megamoe":
        import transformer_engine_torch as tex

        if case.operation == "dgrad":

            def launch():
                return tex.mega_moe_dgrad(dy, offsets, weights)

        else:

            def launch():
                tex.mega_moe_wgrad(dy, x, offsets, out, case.accumulate)
                return out

    else:
        from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm

        dgrad = case.operation == "dgrad"
        operands = weights if dgrad else list(x.split(case.splits))
        dys = list(dy.split(case.splits))
        destinations = [out] if dgrad else out

        def launch():
            general_grouped_gemm(
                operands,
                dys,
                destinations,
                [None] * g,
                destinations[0].dtype,
                layout="NN" if dgrad else "NT",
                m_splits=case.splits,
                grad=True,
                accumulate=case.accumulate,
                single_output=dgrad,
            )
            return out

    rel = samples(case, dy, x, weights, launch(), init, backend)
    for _ in range(warmup):
        launch()
    times = []
    for _ in range(runs):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(iterations):
            launch()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / iterations)
    names = []
    activities = None
    if backend != "legacy":
        # Profile outside timing. An unexpected fallback or cached tile cannot
        # qualify as a replay of the recorded backend.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as trace:
            launch()
            torch.cuda.synchronize()
        devices = [
            e.name for e in trace.events() if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        activities = len(devices)
        names = sorted(set(devices))
        if backend == "megamoe":
            expected = "nvte_mega_moe_tensor_" + case.operation
        elif case.source == "GroupedGemmRtc":
            shape, stages = case.tile.split("s")
            expected = f"_tb{shape}_s{stages}_"
        else:
            # PPU cuBLAS encodes the accumulator and destination in each kernel name.
            suffix = "BF16xBF16" if case.dst_dtype == "bf16" else "FP32xFP32"
            expected = "dtypeBF16xBF16xFP32x" + suffix + "_"
            if any("nvte_grouped_gemm3_" in name for name in names):
                raise RuntimeError(f"line {case.line} expected cuBLAS, observed {names}")
        if not any(expected in name for name in names):
            raise RuntimeError(f"line {case.line} expected {expected}, observed {names}")
    return rel, statistics.median(times), times, names, activities


def write_cases(args, cases, append):
    if args.backend == "tag" and cases:
        first = cases[0]
        os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1" if first.tile else "0"
        os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM_RTC"] = "1"
        os.environ["NVTE_FORCE_RTC_NO_FALLBACK"] = "1"
        if first.tile:
            os.environ["NVTE_GG_FORCE_TILE"] = first.tile.replace("s", "x").replace("x", ",")
    import torch
    import transformer_engine.pytorch

    torch.cuda.set_device(0)
    torch.empty(1, device="cuda")
    legacy = (
        LegacyKernels(args.legacy_revision, args.cuda_include) if args.backend == "legacy" else None
    )
    fields = [
        "line",
        "source",
        "layout",
        "tile",
        "origin",
        "operation",
        "splits",
        "out_features",
        "in_features",
        "accumulate",
        "dst_dtype",
        "backend",
        "relative_l2",
        "median_ms",
        "samples_ms",
        "cuda_activity",
        "num_cuda_activities",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if append else "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        if not append:
            writer.writeheader()
        for case in cases:
            rel, median, times, names, activities = run(
                case, args.backend, legacy, args.warmup, args.runs, args.iterations
            )
            writer.writerow(
                vars(case)
                | {
                    "backend": args.backend,
                    "relative_l2": rel,
                    "median_ms": median,
                    "samples_ms": times,
                    "cuda_activity": names,
                    "num_cuda_activities": activities,
                }
            )
            f.flush()
            print(f"{case.line} {case.operation} {case.dst_dtype} {median:.4f} ms", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag_file", type=Path)
    parser.add_argument("--backend", choices=("legacy", "megamoe", "tag"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-revision", default="91a4ee48")
    parser.add_argument("--cuda-include", default="/usr/local/PPU_SDK/CUDA_SDK/include")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--line", type=int, action="append")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or min(args.runs, args.iterations) < 1 or args.warmup < 0:
        parser.error("Invalid shard, runs, iterations or warmup")
    cases = read_cases(args.tag_file)
    if args.backend == "tag":
        cases = [c for c in cases if c.origin == "captured"]
    if args.line:
        cases = [c for c in cases if c.line in args.line]
    cases = cases[args.shard :: args.shards]
    if not cases:
        parser.error("The selected shard contains no matching cases")
    if args.backend != "tag":
        write_cases(args, cases, False)
        return
    # forced_tile() reads the environment once per process. Isolate arms so
    # all RTC shapes use their recorded tile and the cuBLAS arm can accumulate.
    groups = {}
    for case in cases:
        groups.setdefault((case.source, case.tile), []).append(case)
    for i, batch in enumerate(groups.values()):
        process = multiprocessing.get_context("spawn").Process(
            target=write_cases, args=(args, batch, i != 0)
        )
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"Original backend replay failed for {batch[0].source} {batch[0].tile}"
            )


if __name__ == "__main__":
    main()

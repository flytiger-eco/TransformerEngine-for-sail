#!/usr/bin/env python3
"""Replay a TE grouped GEMM from a collected '[TE] --format=...' tag line.

The tag is the exact stdout line printed under PPU_LIB_SHOW_PARAMS=1 (one per
launch, e.g. from `PPU_LIB_SHOW_PARAMS=1 python app 2>&1 | tee run_tags.txt`).
This tool turns it back into a runnable harness and re-prints the live tag so
you can diff it against the input:

  python3 replay_tag.py --tag "[TE] --format=GroupedGemmRtc,data_type:bf16,groups:8,layout:NT,m:[256,512,1024,2048,512,1024,256,4096],n:1536,k:2096,tile:256x256x64s3,accumulate:0,gpu:0"

Behavior:
  * layout NT  -> forward replay via GroupedLinear (one tag line per launch).
  * layout NN/TN -> backward replay: one GroupedLinear.backward launches
    dgrad (NN) then wgrad (TN), i.e. two tag lines per iteration; match each
    against the corresponding production line.
  * the tag's m/n/k are the GEMM axes of the captured call, so they map back
    to a layer as NT: (tokens, out, in), NN: (tokens, in, out), and
    TN: (out, in, per-group tokens) -- wgrad keeps its token split in k.
  * op GroupedGemmRtc sets NVTE_USE_CUTLASS_GROUPED_GEMM=1 automatically;
    the echoed live tag must show the same op as the input, otherwise the
    arm fell back.
  * the tag's tile (RTC arm) is pinned via NVTE_GG_FORCE_TILE by default;
    --no-force-tile lets the adaptive path re-select (same shape selects the
    same tile deterministically).
  * accumulate=1 is not reproducible through GroupedLinear (warned, replayed
    as accumulate=0).
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys


def parse_tag(tag: str):
    body = tag.split("--format=", 1)[1] if "--format=" in tag else tag
    op, rest = body.split(",", 1)
    # m:[256,512,...] embeds commas: only split on commas outside brackets.
    fields = dict(re.findall(r"(\w+):(\[[^\]]*\]|[^,]+)", rest))
    return op, fields


def extent(v: str):
    v = v.strip()
    if v.startswith("["):
        return [int(x) for x in v[1:-1].split(",") if x]
    return [int(v)]


def rtc_operand_specs(fields):
    """将后端 tag 的 GEMM 轴映射到 Python API；两者的 TN/NT 恰好相反。"""
    groups = int(fields["groups"])
    layout = fields["layout"]
    if groups <= 0 or layout not in ("NT", "NN", "TN"):
        raise ValueError("groups 必须为正数，layout 必须为 NT/NN/TN")
    dims = []
    for axis in ("m", "n", "k"):
        values = extent(fields[axis])
        # 标量表示每个 group 的尺寸，不是所有 group 的总和。
        if not fields[axis].strip().startswith("["):
            values *= groups
        if len(values) != groups or any(v <= 0 for v in values):
            raise ValueError(f"{axis} 必须包含 {groups} 个正数")
        dims.append(values)
    m, n, k = dims
    if len(set(n)) != 1 or (layout != "TN" and len(set(k)) != 1):
        raise ValueError("当前 CI 仅支持共享 n，以及 NT/NN 的共享 k")
    if layout == "NT":
        a = list(zip(n, k))
        b = list(zip(m, k))
        splits, public_layout = m, "TN"
    elif layout == "NN":
        a = list(zip(k, n))
        b = list(zip(m, k))
        splits, public_layout = m, "NN"
    else:
        a = list(zip(k, n))
        b = list(zip(k, m))
        splits, public_layout = k, "NT"
    return public_layout, a, b, list(zip(m, n)), splits, layout != "TN"


def run_ci_case(tag, output, warmup, iters, samples):
    """严格对拍并采集单个 RTC kernel 的时间；每个 case 必须独立进程运行。"""
    op, fields = parse_tag(tag)
    if op != "GroupedGemmRtc" or fields["data_type"] != "bf16":
        raise ValueError("CI 仅支持 BF16 RTC case")
    if fields.get("accumulate") != "0":
        raise ValueError("CI case 必须为 accumulate:0")
    tile = re.fullmatch(r"(\d+)x(\d+)x(\d+)s(\d+)", fields["tile"])
    if tile is None or min(map(int, tile.groups())) <= 0:
        raise ValueError("tile 必须显式指定尺寸和 stages")
    public_layout, a_shapes, b_shapes, out_shapes, splits, single = rtc_operand_specs(fields)
    if min(warmup, iters, samples) <= 0:
        raise ValueError("warmup/iters/samples 必须为正数")

    # 清除开发用开关，防止两条编译 arm 意外使用不同的实验路径。
    for key in list(os.environ):
        if key.startswith("NVTE_GG_") and key != "NVTE_GG_RTC_CACHE_DIR":
            del os.environ[key]
    os.environ.update(
        {
            "NVTE_USE_CUTLASS_GROUPED_GEMM": "1",
            "NVTE_USE_CUTLASS_GROUPED_GEMM_RTC": "1",
            "NVTE_FORCE_RTC_NO_FALLBACK": "1",
            "NVTE_GG_FORCE_TILE": ",".join(tile.groups()),
            "PPU_LIB_SHOW_PARAMS": "1",
            "PPU_LIB_PERF_INSTRUMENT": "0",
            "PPU_LIB_TFLOPS": "0",
        }
    )
    import torch
    import transformer_engine
    from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm

    gpu = int(fields["gpu"])
    if not torch.cuda.is_available():
        raise RuntimeError("CI 必须有可用的 PPU，不能跳过测试")
    torch.cuda.set_device(gpu)
    capability = torch.cuda.get_device_capability(gpu)
    if capability not in ((8, 0), (8, 9)):
        raise RuntimeError(f"RTC 不支持此设备：{capability}")
    site = Path(transformer_engine.__file__).resolve().parent.parent
    candidates = [site / "build/cmake/actlize_effective/include", site / "include/cutlass3"]
    if "NVTE_CUTLASS3_INCLUDE_DIR" not in os.environ:
        for candidate in candidates:
            if (candidate / "cutlass/cutlass.h").is_file():
                os.environ["NVTE_CUTLASS3_INCLUDE_DIR"] = str(candidate)
                break
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = f"cuda:{gpu}"
    dtype = torch.bfloat16
    a = [torch.randn(shape, device=device, dtype=dtype) for shape in a_shapes]
    b = [torch.randn(shape, device=device, dtype=dtype) for shape in b_shapes]
    if single:
        out = [
            torch.full((sum(splits), out_shapes[0][1]), float("nan"), device=device, dtype=dtype)
        ]
        views = list(torch.split(out[0], splits))
    else:
        out = [torch.full(shape, float("nan"), device=device, dtype=dtype) for shape in out_shapes]
        views = out

    def run():
        general_grouped_gemm(
            a,
            b,
            out,
            [None] * len(a),
            dtype,
            layout=public_layout,
            m_splits=splits,
            grad=fields["layout"] != "NT",
            accumulate=False,
            single_output=single,
        )

    run()
    torch.cuda.synchronize()
    max_relative_error = 0.0
    for group, (x, y, actual) in enumerate(zip(a, b, views)):
        # 逐 group 计算 FP32 参考，避免大 M 的 case 同时保留所有 FP32 输出。
        lhs = y.float().t() if public_layout == "NT" else y.float()
        rhs = x.float().t() if public_layout == "TN" else x.float()
        reference = lhs @ rhs
        error = float(
            torch.linalg.vector_norm(actual.float() - reference)
            / torch.linalg.vector_norm(reference)
        )
        if not math.isfinite(error) or error >= 4e-3:
            raise AssertionError(f"group={group} FP32 相对误差={error}")
        torch.testing.assert_close(actual.float(), reference, rtol=0.05, atol=0.05)
        max_relative_error = max(max_relative_error, error)
        del lhs, rhs, reference
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    times, names = [], set()
    expected_layout = {"NT": "LA_RM_LB_CM", "NN": "LA_RM_LB_RM", "TN": "LA_CM_LB_RM"}[
        fields["layout"]
    ]
    expected_tile = f"_tb{tile[1]}x{tile[2]}x{tile[3]}_s{tile[4]}_"
    for _ in range(samples):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(iters):
                run()
            torch.cuda.synchronize()
        kernels = [
            e
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
            and e.name.startswith("nvte_grouped_gemm3_")
        ]
        if len(kernels) != iters:
            raise AssertionError(f"期望 {iters} 个 RTC kernel，实际 {len(kernels)}")
        for event in kernels:
            if expected_tile not in event.name or expected_layout not in event.name:
                raise AssertionError(f"RTC 配置不匹配：{event.name}")
            if "bf16_bf16_bf16_acc_fp32" not in event.name:
                raise AssertionError(f"RTC dtype 不匹配：{event.name}")
            names.add(event.name)
        times.append(statistics.median(e.device_time for e in kernels))
    if not all(math.isfinite(t) and t > 0 for t in times):
        raise AssertionError(f"无效 kernel 时间：{times}")
    flops = sum(
        2 * m * n * k
        for (m, n), k in zip(
            out_shapes, extent(fields["k"]) * (len(a) if not fields["k"].startswith("[") else 1)
        )
    )
    result = {
        "tag": tag,
        "samples_us": times,
        "median_us": statistics.median(times),
        "tflops": flops / statistics.median(times) / 1e6,
        "max_relative_error": max_relative_error,
        "kernels": sorted(names),
        "device": torch.cuda.get_device_name(gpu),
        "capability": list(capability),
        "torch": torch.__version__,
        "te_path": transformer_engine.__file__,
        "headers": os.environ.get("NVTE_CUTLASS3_INCLUDE_DIR"),
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="one '[TE] --format=...' line, quoted")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument(
        "--no-force-tile", action="store_true", help="do not pin NVTE_GG_FORCE_TILE from the tag"
    )
    ap.add_argument("--ci-output", help="输出严格 RTC 对拍和逐 kernel 时间的 JSON")
    ap.add_argument("--profile-samples", type=int, default=5)
    args = ap.parse_args()
    if args.ci_output:
        if args.no_force_tile:
            ap.error("--ci-output 不允许 --no-force-tile")
        run_ci_case(args.tag, args.ci_output, args.warmup, args.iters, args.profile_samples)
        return

    op, f = parse_tag(args.tag)
    if not op.startswith("GroupedGemm"):
        sys.exit(f"unsupported op: {op}")
    g = int(f["groups"])
    layout, dtype_s = f["layout"], f["data_type"]
    if layout not in ("NT", "NN", "TN"):
        sys.exit(f"unsupported layout: {layout}")
    n = int(f["n"])
    m_raw = extent(f["m"])
    # Uniform extents serialize as a scalar (m:2048 means all groups share
    # it); bracketed lists carry one entry per group.
    m = m_raw * g if len(m_raw) == 1 else m_raw
    if len(m) != g:
        sys.exit(f"m list has {len(m)} entries but groups={g}")
    acc, tile = f.get("accumulate", "0"), f.get("tile")
    gpu = int(f.get("gpu", 0) or 0)

    # The tag records the (m, n, k) of the GEMM the backend actually ran, not
    # the layer's (in_features, out_features) pair, and the axes permute with
    # the operand role of the captured launch:
    #   NT (forward): (m, n, k) = (tokens, out_features, in_features)
    #   NN (dgrad):   (m, n, k) = (tokens, in_features, out_features)
    #   TN (wgrad):   (m, n, k) = (out_features, in_features, tokens), i.e. the
    #                 per-group token split sits in k and may be bracketed.
    # Reading them as layer dimensions swaps in/out for the backward forms.
    if layout == "TN":
        if len(set(m)) != 1:
            sys.exit(f"TN (wgrad) tag should carry one N_out per group; got m={m}")
        k_raw = extent(f["k"])
        split_list = k_raw * g if len(k_raw) == 1 else k_raw
        if len(split_list) != g:
            sys.exit(f"k list has {len(split_list)} entries but groups={g}")
        in_features, out_features = n, m[0]
    elif layout == "NN":
        split_list = m
        in_features, out_features = n, int(f["k"])
    else:
        split_list = m
        in_features, out_features = int(f["k"]), n

    # Env vars must exist before the C++ side reads them (first launch config).
    os.environ["PPU_LIB_SHOW_PARAMS"] = "1"  # echo the live tag for diffing
    # Set both routing switches explicitly per operation. Relying on the RTC
    # tag alone lets an inherited environment replay a cuBLAS capture through
    # CUTLASS (or a PPU/static capture through RTC), so the echoed tag would
    # not describe the requested backend.
    if op == "GroupedGemmRtc":
        # RTC is opt-in on both levels: without these the launch silently falls
        # back to the static CUTLASS arm or cuBLAS and the echoed tag shows a
        # different op.
        os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1"
        os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM_RTC"] = "1"
    elif op == "GroupedGemmPpu":
        # Statically compiled PPU CUTLASS arm: CUTLASS on, RTC off.
        os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1"
        os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM_RTC"] = "0"
    elif op == "GroupedGemmCublas":
        os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "0"
    if tile and not args.no_force_tile:
        tm, tn, tks = tile.split("x")
        tk, st = tks.split("s")
        os.environ["NVTE_GG_FORCE_TILE"] = f"{tm},{tn},{tk},{st}"

    import torch
    import transformer_engine.pytorch as te

    # Honour the captured device: replaying on the default GPU can select a
    # different architecture (and therefore a different backend) than the one
    # the tag was collected on.
    torch.cuda.set_device(gpu)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype_s)
    if dtype is None:
        sys.exit(
            f"unsupported data_type for this tool: {dtype_s} (fp8 needs "
            "a quantizer harness, not covered here)"
        )
    if acc == "1":
        print(
            "WARNING: accumulate=1 is not reproducible via GroupedLinear; "
            "replaying with accumulate=0."
        )

    total_m = sum(split_list)
    dev = f"cuda:{gpu}"
    torch.manual_seed(0)
    net = te.GroupedLinear(g, in_features, out_features, bias=False, params_dtype=dtype).to(dev)
    x = torch.randn(total_m, in_features, dtype=dtype, device=dev)
    splits = torch.tensor(split_list, dtype=torch.int32, device=dev)

    if layout == "NT":
        for _ in range(args.warmup + args.iters):
            y = net(x, splits)
            torch.cuda.synchronize()
    else:
        print(
            f"NOTE: layout {layout} is a backward form; running "
            "GroupedLinear.backward -> launches dgrad (NN) then wgrad (TN)."
        )
        gy = torch.randn(total_m, out_features, dtype=dtype, device=dev)
        for _ in range(args.warmup + args.iters):
            xg = x.clone().requires_grad_(True)
            y = net(xg, splits)
            y.backward(gy)
            torch.cuda.synchronize()

    pinned = tile if (tile and not args.no_force_tile) else "adaptive"
    print(
        f"\n[replay] op={op} g={g} layout={layout} tokens={total_m} "
        f"in={in_features} out={out_features} dtype={dtype_s} tile={pinned}"
    )
    print(
        "[replay] diff the [TE] lines above against the input tag: "
        "op/data_type/groups/layout/m/n/k must match; tile matches when "
        "pinned."
    )


if __name__ == "__main__":
    main()

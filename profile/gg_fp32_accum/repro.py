# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""PPU RTC grouped GEMM：in-place 累加、尾 tile 与兼容性定位。"""
import argparse
import json
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--groups", type=int, default=1)
p.add_argument("--layout", choices=("TN", "NN", "NT"), default="NT")
p.add_argument("--nonuniform", action="store_true", help="各 group 的 M/N/K 使用不同的尾块")
p.add_argument("--kernel-names", action="store_true", help="检查单次实际 RTC launch、tile 和输出 dtype")
p.add_argument("--persistent-repeat", action="store_true", help="repeat 不恢复 D，验证连续累加")
p.add_argument("--mpe", type=int, default=256)
p.add_argument("--rows", type=int, default=512)
p.add_argument("--cols", type=int, default=1024)
p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
p.add_argument("--backend", choices=("rtc", "cublas"), default="rtc",
               help="选择实测后端；默认保持 RTC")
p.add_argument("--source", choices=("randn", "zero", "one", "arange"), default="randn")
p.add_argument("--zero-a", action="store_true")
p.add_argument("--accumulate", type=int, choices=(0, 1), default=1)
p.add_argument("--repeat", type=int, default=1)
p.add_argument("--iters", type=int, default=0)
p.add_argument("--save", type=Path)
p.add_argument("--compare", type=Path)
p.add_argument("--dump", type=Path)
p.add_argument("--resources", type=Path, help="仅查询目录内 RTC cubin 的 driver 资源属性")
p.add_argument("--resource-smem", type=int, default=196608)
p.add_argument("--resource-threads", type=int, default=512)
p.add_argument("--check", action="store_true", help="按 fp32/bf16 阈值执行验收断言")
p.add_argument("--pair-c", action="store_true", help="同输入、同 dtype 配对测量 C/noC")
p.add_argument("--pair-warmup", type=int, default=8)
p.add_argument("--pair-iters", type=int, default=10)
p.add_argument("--reverse-pair", action="store_true")
p.add_argument("--smallk-arms", nargs="+", choices=("production", "matched", "contiguous", "nexpand"),
               help="fwd noC K512 的 packed ragged 四臂；tile缓存按进程初始化，每进程一臂")
p.add_argument("--smallk-profile", action="store_true",
               help="避免嵌套 torch.profiler；匹配 launch 为正确性1次+预热3次+iters次")
p.add_argument("--smallk-cubins", type=Path,
               help="按臂加载已校验的实际 RTC cubin，仅补源码调试元数据，不重编")
p.add_argument("--pair-names", action="store_true", help="只验证配对 kernel 数与 dtype")
p.add_argument("--compile-only", action="store_true", help="只用 NVRTC 编译，不初始化或运行 GPU")
p.add_argument("--compile-tile", type=int, nargs=3, default=(256, 256, 64))
p.add_argument("--compile-stages", type=int, default=3)
p.add_argument("--compile-contiguous", type=int, choices=(0, 1, 2), default=0,
               help="仅离线编译验收：0=通用，1=连续普通，2=noC NExpand4；非性能替身")
p.add_argument("--compile-arch", type=int, choices=(80, 89), default=89)
p.add_argument("--compile-ab-dtype", choices=("bf16", "fp16"), default="bf16")
p.add_argument("--compile-with-c", type=int, choices=(0, 1), default=1,
               help="离线编译的 NVTE_GG_WITH_C；用于核对实验宏在无 C cubin 中的作用域")
p.add_argument("--inspect-codegen", action="store_true", help="离线保存 ISA、资源与原子指令计数")
p.add_argument("--compile-lineinfo", action="store_true", help="离线增加源码行号，保存可复核的 PASM 交接映射")
p.add_argument("--compile-baseline-head", action="store_true", help="用 HEAD 的三个 overlay 头建立全关回归基线快照")
p.add_argument("--compile-canonical", action="store_true", help="在完整 include 快照中编译 canonical 头；不修改正在构建的 effective 树")
p.add_argument("--compare-codegen", type=Path, help="compile-only 与已有目录逐字节对比 cubin/ISA/资源并落盘")
a = p.parse_args()
if a.compare_codegen:
    assert a.compile_only and a.dump and a.inspect_codegen, "codegen 对照需要 compile-only/dump/inspect-codegen"
# ACU 的争用告警会连被测进程自己一并列出；上报自身 pid 以便采集侧只认外部占用。
print(json.dumps({"target_pid": os.getpid()}), flush=True)
for name in ("NVTE_USE_CUTLASS_GROUPED_GEMM", "NVTE_USE_CUTLASS_GROUPED_GEMM_RTC",
             "NVTE_GG_ALLOW_FP32_ACCUM", "NVTE_FORCE_RTC_NO_FALLBACK",
             "NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK"):
    os.environ[name] = "1" if a.backend == "rtc" else "0"
os.environ.setdefault("NVTE_CUTLASS_GROUPED_GEMM_RTC_VERBOSE", "1" if a.backend == "rtc" else "0")
if a.dump:
    a.dump.mkdir(parents=True, exist_ok=True)
    os.environ["NVTE_CUTLASS_RTC_DUMP_CUBIN_DIR"] = str(a.dump.resolve())

if a.compile_only:
    import ctypes

    assert a.backend == "rtc"
    root = Path(__file__).resolve().parents[2]
    include = Path(os.environ.get("NVTE_CUTLASS3_INCLUDE_DIR",
                                 str(root / "build/cmake/actlize_effective/include")))
    if a.compile_baseline_head or a.compile_canonical:
        import shutil
        import subprocess

        assert a.dump and not (a.compile_baseline_head and a.compile_canonical)
        snapshot = a.dump / ("head_include" if a.compile_baseline_head else "canonical_include")
        assert not snapshot.exists(), "不可覆盖已有 include 快照"
        shutil.copytree(include, snapshot, symlinks=False)
        relatives = ("cutlass/gemm/kernel/ppu_aiu_gemm_array_group.hpp",
                     "cutlass/gemm/collective/ppu_mma_aiu_multistage_batch_array.hpp",
                     "cutlass/epilogue/collective/ppu_epilogue_vectorized_evt.hpp")
        if a.compile_contiguous:
            assert a.compile_canonical, "专用编译验收不使用 HEAD 实验基线"
            relatives += ("cutlass/gemm/kernel/nvte_contiguous_grouped.hpp",
                          "cutlass/gemm/kernel/nvte_contiguous_grouped_params.hpp")
        for relative in relatives:
            path = f"transformer_engine/common/gemm/actlize_overlay/include/{relative}"
            contents = (subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=root)
                        if a.compile_baseline_head else (root / path).read_bytes())
            (snapshot / relative).write_bytes(contents)
        include = snapshot
    cuda = Path("/usr/local/PPU_SDK/CUDA_SDK")
    symbol = f"nvte_grouped_gemm3_compile_{a.dtype}"
    source = (f"#undef NVTE_GG_KERNEL_NAME\n#define NVTE_GG_KERNEL_NAME {symbol}\n"
              + (root / "transformer_engine/common/gemm/rtc/cutlass_grouped_gemm.cu").read_text())
    ta, tb = {"NT": (1, 0), "NN": (0, 0), "TN": (0, 1)}[a.layout]
    defines = {"NVTE_GG_TRANS_A": ta, "NVTE_GG_TRANS_B": tb, "NVTE_GG_ARCH": a.compile_arch,
               "NVTE_GG_DTYPE": int(a.compile_ab_dtype == "fp16"), "NVTE_GG_OUT_FP32": int(a.dtype == "fp32"),
               "NVTE_GG_WITH_C": a.compile_with_c,
               "NVTE_GG_CONTIGUOUS_MODE": a.compile_contiguous,
               "NVTE_GG_PRELOAD_C": int(a.dtype == "fp32" or
                                         os.environ.get("NVTE_GG_WGRAD_C_FUSION") == "1"),
               "NVTE_GG_TILE_M": a.compile_tile[0], "NVTE_GG_TILE_N": a.compile_tile[1],
               "NVTE_GG_TILE_K": a.compile_tile[2], "NVTE_GG_STAGES": a.compile_stages,
               "CUTLASS_ENABLE_TENSOR_CORE_MMA": 1, "__HGGC_FP16_H__": 1,
               "_HGGC_FP16_H_": 1, "HGGC_FP16_H": 1, "__HGGC_BF16_H__": 1,
               "__HGGC_BF16_HPP__": 1, "__BF16_DEFINED__": 1, "NDEBUG": 1, "USE_SAIL": 1}
    options = ["--no-cache", "--std=c++17", f"--gpu-architecture=sm_{a.compile_arch}",
               f"-I{cuda / 'include'}", f"-I{cuda / 'include/cccl'}", f"-I{include}",
               "-I/usr/local/PPU_SDK/targets/x86_64-linux/include"]
    options += [f"-D{k}={v}" for k, v in defines.items()]
    if a.compile_lineinfo:
        options.append("--generate-line-info")
    if os.environ.get("NVTE_GG_EXTRA_DEFINES"):
        options.extend(os.environ["NVTE_GG_EXTRA_DEFINES"].split())
    rtc = ctypes.CDLL(str(cuda / "lib64/libnvrtc.so"))
    program = ctypes.c_void_p()
    assert rtc.nvrtcCreateProgram(ctypes.byref(program), source.encode(),
                                  b"cutlass_grouped_gemm_rtc.cu", 0, None, None) == 0
    try:
        opts = (ctypes.c_char_p * len(options))(*(s.encode() for s in options))
        status = rtc.nvrtcCompileProgram(program, len(options), opts)
        size = ctypes.c_size_t()
        assert rtc.nvrtcGetProgramLogSize(program, ctypes.byref(size)) == 0
        log = ctypes.create_string_buffer(size.value)
        assert rtc.nvrtcGetProgramLog(program, log) == 0
        if a.dump:
            (a.dump / f"{symbol}.log").write_bytes(log.value)
        assert status == 0, log.value.decode(errors="replace")
        assert rtc.nvrtcGetCUBINSize(program, ctypes.byref(size)) == 0
        cubin = ctypes.create_string_buffer(size.value)
        assert rtc.nvrtcGetCUBIN(program, cubin) == 0
        result = {"compile_only": "PASS", "gpu_execution": False,
                  "cubin_bytes": size.value, "options": options}
        if a.dump:
            path = a.dump / f"{symbol}.cubin"
            path.write_bytes(cubin.raw)
            if a.inspect_codegen:
                import re
                import subprocess

                tool = "/usr/local/PPU_SDK/bin/hgobjdump"
                isa = subprocess.check_output([tool, "--dump-isa", str(path)], text=True)
                resource = subprocess.check_output(
                    [tool, "--dump-resource-usage=all", str(path)], text=True)
                (a.dump / f"{symbol}.isa").write_text(isa)
                if a.compile_lineinfo:
                    mapped = subprocess.check_output([tool, "--line-numbers", str(path)], text=True)
                    (a.dump / f"{symbol}.isa.lines").write_text(mapped)
                    # hgobjdump 的行号模式在末尾输出 DWARF 表，而非内联 file:line。
                    from bisect import bisect_right
                    files = {int(index): name for index, name in re.findall(
                        r'file_names\[\s*(\d+)\]:\s*name:\s*"([^"]+)"', mapped)}
                    rows = [(int(pc, 16), int(line), int(column), int(file))
                            for pc, line, column, file in re.findall(
                                r"^0x([0-9a-f]+)\s+(\d+)\s+(\d+)\s+(\d+)\s", mapped, re.M)]
                    pcs = [row[0] for row in rows]
                    assert pcs == sorted(pcs) and rows, "无法解析 DWARF 行号表"
                    annotations = []
                    for instruction in isa.splitlines():
                        match = re.match(r"\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+){8}(.*)", instruction)
                        if not match:
                            continue
                        pc = int(match[1], 16)
                        index = bisect_right(pcs, pc) - 1
                        if index < 0:
                            continue
                        _, line, column, file = rows[index]
                        annotations.append({"address": hex(pc), "instruction": match[2],
                                            "file": files.get(file), "line": line, "column": column})
                    result["source_line_mapping"] = any(
                        row["file"] == "ppu_epilogue_vectorized_evt.hpp" and row["line"]
                        for row in annotations)
                    assert result["source_line_mapping"], "编译未提供 overlay 源码行号"
                    (a.dump / f"{symbol}.pasm_map.json").write_text(
                        json.dumps(annotations, indent=2) + "\n")
                (a.dump / f"{symbol}.resources").write_text(resource)
                result["vreg_number"] = int(re.search(r"vreg_number:(\d+)", resource)[1])
                result["stack_size"] = int(re.search(r"STACK SIZE:(\d+)", resource)[1])
                atomic_lines = [line.strip() for line in isa.splitlines()
                                if re.search(r"\bvmem\.atm\.add\.f32\b", line)]
                result["atomic_instruction_count"] = len(atomic_lines)
                result["atomic_instruction_samples"] = atomic_lines[:4]
            if a.compare_codegen:
                import hashlib
                comparison = {"baseline": str(a.compare_codegen.resolve()), "files": {}}
                for suffix in ("cubin", "isa", "resources"):
                    current = (a.dump / f"{symbol}.{suffix}").read_bytes()
                    old = (a.compare_codegen / f"{symbol}.{suffix}").read_bytes()
                    comparison["files"][suffix] = {
                        "identical": current == old, "sha256": hashlib.sha256(current).hexdigest(),
                        "baseline_sha256": hashlib.sha256(old).hexdigest()}
                comparison["status"] = "PASS" if all(
                    v["identical"] for v in comparison["files"].values()) else "FAIL"
                (a.dump / "codegen_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
                assert comparison["status"] == "PASS", comparison
                result["codegen_comparison"] = comparison
            (a.dump / f"{symbol}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        assert rtc.nvrtcDestroyProgram(ctypes.byref(program)) == 0
    raise SystemExit(0)

import torch
from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm
from transformer_engine.common import _get_shared_object_file
print(json.dumps({"core_library": str(_get_shared_object_file("core").resolve()),
                  "torch_library": str(_get_shared_object_file("torch").resolve()),
                  "rtc_include": os.environ.get("NVTE_CUTLASS3_INCLUDE_DIR", "")}), flush=True)

if a.resources:
    import ctypes

    # CUDA lazy init 尚不保证 driver 的当前 context，先实际分配一个设备元素。
    context_probe = torch.empty(1, device="cuda")
    driver = ctypes.CDLL("libcuda.so.1")
    cubins = sorted(a.resources.glob("nvte_grouped_gemm3_*.cubin"))
    assert cubins, a.resources
    for path in cubins:
        module, function = ctypes.c_void_p(), ctypes.c_void_p()
        image = ctypes.create_string_buffer(path.read_bytes())
        status = driver.cuModuleLoadData(ctypes.byref(module), image)
        assert status == 0, (path, status)
        try:
            assert driver.cuModuleGetFunction(ctypes.byref(function), module,
                                              path.stem.encode()) == 0, path
            resources = {}
            for name, attribute in (("registers_per_thread", 4), ("local_bytes_per_thread", 3),
                                    ("static_smem_bytes", 1)):
                value = ctypes.c_int()
                assert driver.cuFuncGetAttribute(ctypes.byref(value), attribute, function) == 0
                resources[name] = value.value
            assert driver.cuFuncSetAttribute(function, 8, a.resource_smem) == 0
            active = ctypes.c_int()
            assert driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(active), function, a.resource_threads,
                ctypes.c_size_t(a.resource_smem)) == 0
            resources.update(dynamic_smem_bytes=a.resource_smem, threads=a.resource_threads,
                             active_blocks_per_cu=active.value,
                             theoretical_thread_occupancy=active.value * a.resource_threads /
                                 torch.cuda.get_device_properties(0).max_threads_per_multi_processor)
            result = {"cubin": str(path), **resources}
            (path.parent / f"{path.stem}.driver_resources.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        finally:
            assert driver.cuModuleUnload(module) == 0
    raise SystemExit(0)

if a.smallk_arms:
    import hashlib
    import math

    assert (a.layout, a.rows, a.cols, a.mpe, a.groups, a.dtype, a.accumulate) == \
        ("TN", 4096, 4096, 512, 16, "bf16", 0), "本次固定 fwd noC MNK4096x4096x512 groups16"
    assert a.backend == "rtc" and a.dump and not a.pair_c
    assert torch.cuda.get_device_capability() == (8, 9)
    assert len(a.smallk_arms) == 1, "tile/withC为进程级缓存；四臂须独立进程并校验输入摘要与输出逐位相等"
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    splits = [a.rows - (0, 1, 17, 63)[i % 4] for i in range(a.groups)]

    def packed(shapes, random):
        counts = [math.prod(s) for s in shapes]
        storage = torch.full((sum(counts) + 64,), 97, device="cuda", dtype=torch.bfloat16)
        views, offset = [], 32
        for shape, count in zip(shapes, counts):
            view = storage[offset:offset + count].view(shape)
            view.copy_(torch.randn(shape, device="cuda", dtype=torch.bfloat16)) if random else \
                view.fill_(float("nan"))
            views.append(view)
            offset += count
        return views, storage

    A, abacking = packed([(a.cols, a.mpe)] * a.groups, True)
    B, bbacking = packed([(m, a.mpe) for m in splits], True)
    D, dbacking = packed([(m, a.cols) for m in splits], False)
    # seed 与完整输入位模式摘要确保各独立 profiler 进程也使用相同输入。
    input_hashes = {name: hashlib.sha256(buf.cpu().numpy().tobytes()).hexdigest()
                    for name, buf in (("A", abacking.view(torch.int16)),
                                      ("B", bbacking.view(torch.int16)))}
    references = [y.float() @ x.float().t() for x, y in zip(A, B)]
    record = {"case": "fwd_noC", "MNK": [a.rows, a.cols, a.mpe], "splits": splits,
              "groups": a.groups, "seed": 1234, "input_sha256": input_hashes, "arms": {},
              "core_library": str(_get_shared_object_file("core").resolve()),
              "torch_library": str(_get_shared_object_file("torch").resolve()),
              "rtc_include": os.environ.get("NVTE_CUTLASS3_INCLUDE_DIR"),
              "skip_matching": 4 if a.smallk_profile and a.iters else None}
    baseline = None
    for arm in a.smallk_arms:
        dest = a.dump / arm
        dest.mkdir(exist_ok=False)
        os.environ["NVTE_CUTLASS_RTC_DUMP_CUBIN_DIR"] = str(dest.resolve())
        os.environ["NVTE_GG_RTC_CACHE_DIR"] = str((dest / "cache").resolve())
        for key in ("NVTE_GG_EXTRA_DEFINES", "NVTE_GG_N_EXPAND", "NVTE_GG_FORCE_TILE",
                    "NVTE_GG_FORCE_WITH_C", "NVTE_GG_RTC_LOAD_CUBIN_DIR", "NVTE_GG_WGRAD_C_FUSION",
                    "NVTE_GG_C_EVT_CLASSIC", "NVTE_GG_DISABLE_DIRECT_EPILOGUE"):
            os.environ.pop(key, None)
        os.environ["NVTE_GG_CONTIGUOUS"] = "1" if arm in ("contiguous", "nexpand") else "0"
        os.environ["NVTE_GG_CONTIGUOUS_N_EXPAND"] = "4" if arm == "nexpand" else "1"
        if arm != "production":
            os.environ["NVTE_GG_FORCE_TILE"] = "256,128,64,3"
        loaded_cubin = None
        if a.smallk_cubins:
            import shutil
            load_dir = (a.smallk_cubins / arm).resolve()
            cubins = list(load_dir.glob("nvte_grouped_gemm3_*.cubin"))
            assert len(cubins) == 1, load_dir
            loaded_cubin = cubins[0]
            assert loaded_cubin.with_suffix(".cu").is_file()
            # LOAD 分支不再 dump TU/cubin，保存确切加载输入；options 仍由 host 实际生成。
            for suffix in (".cubin", ".cu"):
                shutil.copyfile(loaded_cubin.with_suffix(suffix), dest / loaded_cubin.with_suffix(suffix).name)
            os.environ["NVTE_GG_RTC_LOAD_CUBIN_DIR"] = str(load_dir)

        def launch_smallk():
            general_grouped_gemm(A, B, D, [None] * a.groups, torch.bfloat16,
                                 layout="TN", m_splits=splits, grad=False,
                                 accumulate=False, single_output=False)

        for d in D:
            d.fill_(float("nan"))
        launch_smallk()
        torch.cuda.synchronize()
        rel_errors = []
        for actual, reference in zip(D, references):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual.float(), reference, rtol=0.005, atol=0.005)
            rel_errors.append(float(torch.linalg.vector_norm(actual.float() - reference.float()) /
                                    torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)))
        output = [d.cpu() for d in D]
        if baseline is None:
            baseline = output
        else:
            assert all(torch.equal(d, old) for d, old in zip(output, baseline)), arm
        if a.save:
            assert len(a.smallk_arms) == 1 and not a.save.exists()
            torch.save(output, a.save)
        if a.compare:
            old = torch.load(a.compare, weights_only=True)
            assert len(old) == len(output) and all(torch.equal(x, y) for x, y in zip(output, old))
        kernels = []
        if not a.smallk_profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                launch_smallk()
                torch.cuda.synchronize()
            kernels = [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")]
            assert len(kernels) == 1 and "bf16_bf16_bf16_acc_fp32" in kernels[0], kernels
            if arm in ("production", "matched"):
                # 生产 beta=0 的 noC 语义使用 typed C 的运行时短路，不强制实验 Cvoid。
                assert kernels[0].endswith("_Cbf16") and "_contiguous" not in kernels[0], kernels
                if arm == "matched":
                    assert "tb256x128x64_s3" in kernels[0], kernels
            else:
                assert "tb256x128x64_s3" in kernels[0], kernels
                suffix = {"contiguous": "_contiguous1", "nexpand": "_contiguous2"}[arm]
                assert kernels[0].endswith("_Cvoid" + suffix), kernels
        for storage in (abacking, bbacking, dbacking):
            assert torch.all(storage[:32] == 97) and torch.all(storage[-32:] == 97)
        mblocks = sum((m + 255) // 256 for m in splits)
        output_hash = hashlib.sha256()
        for tensor in output:
            output_hash.update(tensor.view(torch.int16).numpy().tobytes())
        entry = {"correctness": "PASS", "max_rel_error": max(rel_errors), "kernels": kernels,
                 "output_sha256": output_hash.hexdigest(),
                 "tile_override": os.environ.get("NVTE_GG_FORCE_TILE"),
                 "contiguous": os.environ["NVTE_GG_CONTIGUOUS"],
                 "n_expand": os.environ["NVTE_GG_CONTIGUOUS_N_EXPAND"],
                 "alpha": 1, "beta": 0, "accumulate": False,
                 "compiled_source_type": "void" if arm in ("contiguous", "nexpand") else "bf16",
                 "expected_contiguous_work_count": mblocks * a.cols // (512 if arm == "nexpand" else 128)
                     if arm in ("contiguous", "nexpand") else None}
        if loaded_cubin:
            entry["loaded_cubin"] = str(loaded_cubin)
            entry["loaded_cubin_sha256"] = hashlib.sha256(loaded_cubin.read_bytes()).hexdigest()
        record["arms"][arm] = entry
        print(json.dumps({"smallk_arm": arm, **entry}), flush=True)
        if a.iters:
            for _ in range(3):
                launch_smallk()
            torch.cuda.synchronize()
            for _ in range(a.iters):
                launch_smallk()
            torch.cuda.synchronize()
    (a.dump / "smallk.json").write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(0)

torch.manual_seed(1234)
torch.backends.cuda.matmul.allow_tf32 = False
dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
shapes = [(max(8, a.rows - 8 * (i % 3)), max(8, a.cols - 8 * (i % 5)),
           (8, 56, 64, 72, 128, 200, 2048)[i % 7]) if a.nonuniform
          else (a.rows, a.cols, a.mpe) for i in range(a.groups)]
A = [torch.randn((n, k) if a.layout == "TN" else (k, n),
                 device="cuda", dtype=torch.bfloat16) for m, n, k in shapes]
B = [torch.randn((k, m) if a.layout == "NT" else (m, k),
                 device="cuda", dtype=torch.bfloat16) for m, n, k in shapes]
splits = [k if a.layout == "NT" else m for m, n, k in shapes]
if a.zero_a:
    for x in A:
        x.zero_()
products = [(b.float().T if a.layout == "NT" else b.float()) @
            (x.float().T if a.layout == "TN" else x.float()) for x, b in zip(A, B)]
C = []
for m, n, k in shapes:
    if a.source == "randn":
        c = torch.randn(m, n, device="cuda", dtype=dtype)
    elif a.source == "arange":
        c = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n).to(dtype)
    else:
        c = torch.full((m, n), float(a.source == "one"), device="cuda", dtype=dtype)
    C.append(c)
guard, sentinel = 32, 97.0
storage = [torch.full((c.numel() + 2 * guard,), sentinel, device="cuda", dtype=dtype) for c in C]
D = [buf[guard:-guard].reshape(c.shape) for buf, c in zip(storage, C)]
refs = [product + (c.float() if a.accumulate else 0) for product, c in zip(products, C)]

def launch():
    general_grouped_gemm(A, B, D, [None] * a.groups, torch.bfloat16,
                         layout=a.layout, m_splits=splits,
                         grad=a.layout != "TN", accumulate=bool(a.accumulate), single_output=False)

if a.pair_c:
    # 两臂使用相同 A/B、相同输出 dtype；唯一变化是 beta 和 ptr_C。
    outputs = {False: [c.clone() for c in C], True: D}
    def paired_launch(with_c):
        for d, c in zip(outputs[with_c], C):
            d.copy_(c)
        general_grouped_gemm(
            A, B, outputs[with_c], [None] * a.groups, torch.bfloat16,
            layout=a.layout, m_splits=splits, grad=a.layout != "TN",
            accumulate=with_c, single_output=False,
        )

    # 先数值对拍再采集；每步恢复 C，避免反复 RMW 改变待测输入。
    for with_c in (False, True):
        paired_launch(with_c)
        torch.cuda.synchronize()
        errors = []
        for d, product, source in zip(outputs[with_c], products, C):
            ref = product + source.float() if with_c else product
            rel = float(torch.linalg.vector_norm(d.float() - ref)
                        / torch.linalg.vector_norm(ref).clamp_min(1e-30))
            assert torch.isfinite(d).all()
            # 深 K 的无 C 基线已约 4.4e-6；按 K 放大累加误差预算。
            # C 路径另与同一无 C 产物加 C 比较，避免混淆 mainloop 误差。
            limit = max(1e-6, 1e-6 * a.mpe / 256) if dtype == torch.float32 else 4e-3
            assert rel < limit, (a.dtype, with_c, rel, limit)
            errors.append(rel)
        print(json.dumps({"pair_check": "with_c" if with_c else "no_c",
                          "dtype": a.dtype, "groups": a.groups, "mpe": a.mpe,
                          "gemm": [a.rows, a.cols, a.mpe],
                          "max_rel_err": max(errors), "rel_errors": errors}), flush=True)
    if dtype == torch.float32:
        pair_errors, mismatches = [], []
        for no_c, with_c, source in zip(outputs[False], outputs[True], C):
            expected = no_c + source
            rel = float(torch.linalg.vector_norm(with_c - expected)
                        / torch.linalg.vector_norm(expected).clamp_min(1e-30))
            assert rel < 1e-6, ("C 路径相对同一无 C 基线异常", rel)
            pair_errors.append(rel)
            mismatches.append(int(torch.count_nonzero(with_c != expected)))
        print(json.dumps({"pair_source_check": "PASS", "max_rel_err": max(pair_errors),
                          "mismatches": mismatches}), flush=True)

    if a.pair_names:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            paired_launch(False)
            paired_launch(True)
            torch.cuda.synchronize()
        prefix = "nvte_grouped_gemm3_" if a.backend == "rtc" else "gemm_"
        kernels = [e.name for e in prof.events() if e.name.startswith(prefix)]
        per_call = 1 if a.backend == "rtc" else a.groups
        assert len(kernels) == 2 * per_call, kernels
        if a.backend == "rtc":
            assert kernels[0] == kernels[1], kernels
            assert f"bf16_bf16_{a.dtype}_acc_fp32" in kernels[0], kernels
        else:
            assert all("dtypeBF16xBF16xFP32" in name for name in kernels), kernels
        print(json.dumps({"pair_kernels": kernels, "kernels_per_call": per_call,
                          "backend": a.backend}), flush=True)
        raise SystemExit(0)

    for _ in range(a.pair_warmup):
        paired_launch(False)
        paired_launch(True)
    torch.cuda.synchronize()
    order = []
    for sample in range(a.pair_iters):
        arms = (True, False) if (sample + int(a.reverse_pair)) % 2 else (False, True)
        for with_c in arms:
            order.append("with_c" if with_c else "no_c")
            paired_launch(with_c)
        torch.cuda.synchronize()
    print(json.dumps({"pair_order": order, "skip_matching": 2 + 2 * a.pair_warmup}), flush=True)
    raise SystemExit(0)

for run in range(a.repeat):
    if not a.persistent_repeat or run == 0:
        for d, c in zip(D, C):
            d.copy_(c) if a.accumulate else d.fill_(float("nan"))
    if a.persistent_repeat and run > 0 and a.accumulate:
        refs = [product + d.float().clone() for product, d in zip(products, D)]
    launch()
    torch.cuda.synchronize()
    for buf in storage:
        assert torch.all(buf[:guard] == sentinel) and torch.all(buf[-guard:] == sentinel)
    for group, (d, ref) in enumerate(zip(D, refs)):
        err = d.float() - ref
        rel = float(torch.linalg.vector_norm(err) / torch.linalg.vector_norm(ref).clamp_min(1e-30))
        print(json.dumps({"run": run, "group": group, "dtype": a.dtype,
                          "mpe": a.mpe, "rows": a.rows, "accumulate": a.accumulate,
                          "source": a.source, "rel_err": rel,
                          "max_abs": float(err.abs().max()),
                          "lo_abs": float(err[:256].abs().max()),
                          "hi_abs": float(err[256:].abs().max()) if a.rows > 256 else None,
                          "d00": float(d[0, 0]), "ref00": float(ref[0, 0]),
                          "mismatch": int(torch.count_nonzero(err))}), flush=True)
        if a.check:
            assert torch.isfinite(d).all()
            if a.zero_a and a.accumulate:
                assert torch.equal(d, C[group]), "C 地址映射不是恒等"
            else:
                limit = max(1e-6, 1e-6 * shapes[group][2] / 256) if dtype == torch.float32 else 4e-3
                assert rel < limit, rel
                torch.testing.assert_close(d.float(), ref, rtol=0.05 if dtype == torch.bfloat16 else 1e-4,
                                           atol=0.05 if dtype == torch.bfloat16 else 5e-4 * max(1, shapes[group][2] / 256))
if a.kernel_names:
    saved_d = [d.clone() for d in D]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        launch()
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")]
    assert len(kernels) == 1, kernels
    assert f"bf16_bf16_{a.dtype}_acc_fp32" in kernels[0], kernels
    tile = os.environ.get("NVTE_GG_FORCE_TILE", "256,256,64,3").split(",")
    assert f"tb{'x'.join(tile[:3])}_s{tile[3]}" in kernels[0], kernels
    print(json.dumps({"kernels": kernels, "shapes": shapes,
                      "extra_defines": os.environ.get("NVTE_GG_EXTRA_DEFINES", "")}), flush=True)
    for d, saved in zip(D, saved_d):
        d.copy_(saved)
if a.compare:
    baseline = torch.load(a.compare, weights_only=True)
    assert len(D) == len(baseline), "基线 group 数不一致"
    bits_dtype = torch.int32 if dtype == torch.float32 else torch.int16
    for group, (d, old) in enumerate(zip(D, baseline)):
        assert d.dtype == old.dtype and d.shape == old.shape, f"group {group} 基线 dtype/shape 不一致"
        assert torch.equal(d.cpu().view(bits_dtype), old.view(bits_dtype)), f"group {group} 与基线不是逐位相同"
    print("BITWISE_BASELINE_OK", flush=True)
if a.save:
    a.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save([d.cpu() for d in D], a.save)
if a.iters:
    # 性能只用于 beta=0；重复 in-place 累加会改变输入 C 的值。
    assert not a.accumulate
    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    for _ in range(a.iters):
        launch()
    torch.cuda.synchronize()

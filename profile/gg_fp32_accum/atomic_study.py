# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""atomic epilogue 的同输入 ACU 对照；保存优化态源码、cubin、full 报告和命令。"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path(os.environ.get("NVTE_GG_STUDY_ROOT", Path(__file__).resolve().parents[2])).resolve()
METRIC = "cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
VARIANTS = {
    "fragment": "-DNVTE_GG_ATOMIC_C=0 -DNVTE_GG_PROLOGUE_C=0",
    "prologue": "-DNVTE_GG_ATOMIC_C=0 -DNVTE_GG_PROLOGUE_C=1",
    "atomic": "-DNVTE_GG_ATOMIC_C=1 -DNVTE_GG_PROLOGUE_C=0",
    "global": "-DNVTE_GG_ATOMIC_C=1 -DNVTE_GG_PROLOGUE_C=0 -DNVTE_GG_ATOMIC_GLOBAL=1",
    "interleave": "-DNVTE_GG_ATOMIC_C=1 -DNVTE_GG_PROLOGUE_C=0 -DNVTE_GG_ATOMIC_GLOBAL=1 -DNVTE_GG_ATOMIC_INTERLEAVE=1",
}
# overlap 基线明确固定 AIU C 路径，不能使用历史 fragment 标签归因。
_OVERLAP_BASE = "-DNVTE_GG_ATOMIC_C=0 -DNVTE_GG_PROLOGUE_C=0 -DNVTE_GG_C_AIU_SMEM=1 -DNVTE_GG_C_AIU_STAGES=2"
for _name, _values in {"baseline_aiu": (0, 0, 0), "next_no_c": (1, 0, 0),
                       "c_tail": (0, 1, 0), "next_with_c": (0, 0, 1),
                       "both_c": (0, 1, 1), "all_overlap": (1, 1, 1)}.items():
    VARIANTS[_name] = _OVERLAP_BASE + " " + " ".join(
        f"-D{macro}={value}" for macro, value in zip(
            ("NVTE_GG_PREFETCH_NEXT_NO_C", "NVTE_GG_PREFETCH_C_TAIL", "NVTE_GG_PREFETCH_NEXT_WITH_C"), _values))
VARIANTS["stream_no_c"] = VARIANTS["baseline_aiu"] + " -DNVTE_GG_STREAM_NEXT_NO_C=1"
for _name, _mode in (("stream_split_before", 1), ("stream_split_after", 2)):
    VARIANTS[_name] = VARIANTS["stream_no_c"] + f" -DNVTE_GG_STREAM_TAIL_MODE={_mode}"
# prologue 以 C 为 MMA 初值，累加次序与 fragment/atomic 不同，不参与逐位比较。
REORDERS_ACCUMULATION = {"prologue"}
SYMBOL = "nvte_grouped_gemm3_bf16_bf16_fp32_acc_fp32_LA_CM_LB_RM_LC_RM_sm89_tb256x256x64_s3_Cfp32"
parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=("build", "audit-build", "verify", "verify-codegen", "resources", "prepare", "run", "probe", "capture", "export", "audit-stream", "matrix", "source-reports", "tail-study", "smallk-study", "bind-verify-source"))
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["fragment", "atomic"])
parser.add_argument("--gpu", type=int, default=7)
parser.add_argument("--source-mode", choices=("noC", "withC"), default="withC")
parser.add_argument("--layout", choices=("TN", "NN", "NT"), default="NT")
parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
parser.add_argument("--save", type=Path, help="保存首次正确性 launch 的输出，供 lineinfo 等价校验")
parser.add_argument("--compare", type=Path, help="首次输出与实际 RTC 输出逐位对拍")
parser.add_argument("--offline-opt", choices=("0", "1", "2", "3"), default="3")
parser.add_argument("--actual-reference", type=Path, help="prepare 比对现有 actual_rtc 目录，不重跑 GPU")
parser.add_argument("--annotate-rtc", action="store_true", help="为 RTC 原生行号表补最小 DWARF CU，不重编机器码")
parser.add_argument("--nonuniform", action="store_true", help="复现非均匀 fwd token 分布")
parser.add_argument("--rows", type=int, default=4096)
parser.add_argument("--cols", type=int, default=1536)
parser.add_argument("--mpe", type=int, default=2048)
parser.add_argument("--groups", type=int, default=16)
parser.add_argument("--warmup", type=int, default=4)
parser.add_argument("--samples", type=int, default=1)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--full", action="store_true")
parser.add_argument("--diagnostic", action="store_true",
                    help="smallk-study 显式诊断采集：不等待完整回归，允许争用并禁止正式性能结论")
parser.add_argument("--debug-capture", action="store_true",
                    help="仅诊断：gdb定位采集侧invalid_argument，不作为性能数据")
parser.add_argument("--tag", default="", help="独立采集标签，保留失败日志与既有报告")
parser.add_argument("--rtc", action="store_true", help="不用离线替身，直接验证/采集实际 NVRTC")
parser.add_argument("--rtc-include", type=Path, help="run 使用采集绑定的完整 include 快照，避免并发重构建污染")
parser.add_argument("--test-filter", default="overlap", help="verify 阶段的 pytest -k 选择表达式")
parser.add_argument("--require-verify", type=Path, nargs="+", help="matrix 等待这些正确性回执全部 PASS 后才采集")
parser.add_argument("--verify-source-snapshot", type=Path,
                    help="bind-verify-source 保存的已验收源码，用AST验证未改测试及其公共依赖")
parser.add_argument("--resume", action="store_true", help="matrix 仅恢复已完成和未启动项；不覆盖失败或部分采集")
parser.add_argument("--families", nargs="+", choices=("step1", "step2", "step3", "combined"),
                    default=["step1", "step2", "step3", "combined"],
                    help="matrix 按实验族分卡并行；每个 job 内基线与实验臂仍同卡配对")
a = parser.parse_args()
a.out = a.out.resolve()
a.out.mkdir(parents=True, exist_ok=True)
TRANS_A, TRANS_B = {"NT": (1, 0), "NN": (0, 0), "TN": (0, 1)}[a.layout]
SYMBOL = (f"nvte_grouped_gemm3_bf16_bf16_{a.dtype}_acc_fp32_"
          f"LA_{'CM' if TRANS_A else 'RM'}_LB_{'CM' if TRANS_B else 'RM'}_LC_RM_sm89_tb256x256x64_s3_C{a.dtype}")
WITH_C = a.source_mode == "withC"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def execute(cmd, log, env=None, strict_contention=True, cwd=None):
    result = subprocess.run(cmd, cwd=cwd or ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    assert result.returncode == 0, (cmd, log, result.stdout[-3000:])
    if strict_contention:
        assert "may affect" not in result.stdout.lower(), f"争用污染：{log}"
    for error in ("==ERROR==", "Fallback to cuBLAS", "No kernels were profiled"):
        assert error not in result.stdout, f"无效结果（{error}）：{log}"
    return result.stdout


def foreign_contenders(text):
    """ACU 的争用告警会把被测进程自己也列进去；只保留外部 pid。"""
    target = {json.loads(line)["target_pid"] for line in text.splitlines()
              if line.startswith('{"target_pid"')}
    assert len(target) == 1, f"未拿到被测进程 pid：{target}"
    listed = {int(pid) for pid in re.findall(r"^\s{4,}(\d+):\S", text, re.M)}
    return sorted(listed - target), target


def device_idle():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,nounits"], text=True)
    gpu = next(r for r in csv.DictReader(io.StringIO(text)) if int(r["index"]) == a.gpu)
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,nounits"], text=True)
    # --kill=no 会留下已退出进程的僵尸条目；只有 pid 真实存活才算设备被占用。
    live = [r for r in csv.DictReader(io.StringIO(apps))
            if r["gpu_uuid"].strip() == gpu[" uuid"].strip()
            and Path(f"/proc/{r[' pid'].strip()}").exists()]
    assert not live, (live, apps)
    return gpu


def wait_idle():
    # 共享卡被他人占用时只等待，不抢占、不终止外部任务。
    while True:
        try:
            return device_idle()
        except AssertionError:
            time.sleep(60)


def annotate_native_rtc(actual_path, dest, include, symbol):
    """沿用实际 RTC 行号表，仅补 DWARF4 CU；不运行编译器、不改变原 section 内容。"""
    import struct
    assert actual_path.stem == symbol
    cubin = dest / actual_path.name
    assert not cubin.exists(), "不覆盖已注释的 cubin"
    dest.mkdir(parents=True, exist_ok=True)
    original = actual_path.read_bytes()
    image = bytearray(original)
    assert image[:6] == b"\x7fELF\x02\x01", "只支持 ELF64 little-endian"
    shoff = struct.unpack_from("<Q", image, 40)[0]
    entsize, count, strindex = struct.unpack_from("<HHH", image, 58)
    assert entsize == 64 and count < 0xff00
    headers = [list(struct.unpack_from("<IIQQQQIIQQ", image, shoff + i * 64)) for i in range(count)]
    string_header = headers[strindex]
    strings = bytes(image[string_header[4]:string_header[4] + string_header[5]])
    names = [strings[h[0]:].split(b"\0", 1)[0].decode() for h in headers]
    assert ".debug_line" in names and ".debug_info" not in names and ".debug_abbrev" not in names
    text = headers[next(i for i, name in enumerate(names) if name.startswith(".text.kernel."))]
    high_pc = text[5]
    abbrev = bytes([1, 0x11, 1, 0x03, 0x08, 0x1b, 0x08, 0x10, 0x17,
                    0x11, 0x01, 0x12, 0x07, 0x13, 0x05, 0x25, 0x08, 0x89, 0x7c, 0x08, 0, 0,
                    2, 0x2e, 0, 0x03, 0x08, 0x6e, 0x08, 0x11, 0x01, 0x12, 0x07, 0, 0, 0])
    source_mirror = dest / "build/cmake/actlize_effective/include"
    source_mirror.parent.mkdir(parents=True, exist_ok=True)
    source_mirror.symlink_to(include.resolve(), target_is_directory=True)
    assert source_mirror.resolve() == include.resolve()
    source = actual_path.with_suffix(".cu").read_bytes()
    for name in ("default", "<stdin>", f"{symbol}.cu"):
        (dest / name).write_bytes(source)
    comp_dir = str(dest.resolve())
    prefix = b"\x01<stdin>\0" + comp_dir.encode() + b"\0" + struct.pack("<I", 0)
    cu_pc_offset = 11 + len(prefix)
    body = (prefix + struct.pack("<QQH", 0, high_pc, 0x21)
            + b"hgrtc optimized RTC; minimal source-correlation CU\0[ppu:v0.1]\0"
            + b"\x02" + (symbol.encode() + b"\0") * 2 + struct.pack("<QQ", 0, high_pc) + b"\0")
    info = struct.pack("<IHIB", 7 + len(body), 4, 0, 8) + body
    line_rela = headers[names.index(".rela.debug_line")]
    _, relocation_info, addend = struct.unpack_from("<QQq", original, line_rela[4])
    rela_info = (struct.pack("<QQq", cu_pc_offset, relocation_info, addend)
                 + struct.pack("<QQq", len(info) - 17, relocation_info, addend))
    for name, contents in ((".debug_abbrev", abbrev), (".debug_info", info), (".rela.debug_info", rela_info)):
        offset = len(image)
        image.extend(contents)
        header = [len(strings), 1, 0, 0, offset, len(contents), 0, 0, 1, 0]
        if name == ".rela.debug_info":
            header[1:3] = [4, 0x40]
            header[6:10] = [line_rela[6], count + 1, 8, 24]
        headers.append(header)
        strings += name.encode() + b"\0"
    string_header[4], string_header[5] = len(image), len(strings)
    image.extend(strings)
    image.extend(b"\0" * (-len(image) % 8))
    new_shoff = len(image)
    for header in headers:
        image.extend(struct.pack("<IIQQQQIIQQ", *header))
    struct.pack_into("<Q", image, 40, new_shoff)
    struct.pack_into("<H", image, 60, len(headers))
    for header in headers[:count]:
        if header is not string_header and header[1] != 8:
            offset, size = header[4:6]
            assert image[offset:offset + size] == original[offset:offset + size]
    cubin.write_bytes(image)
    save(dest / "annotation.json", {"method": "native RTC line table + minimal DWARF4 CU",
        "nvcc_executed": False, "symbol": symbol, "original_loadable_sections_unchanged": True,
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "annotated_sha256": hashlib.sha256(image).hexdigest(),
        "compilation_directory": comp_dir, "frozen_include": str(include.resolve())})
    return cubin


def verification_source_identity(receipt, snapshot):
    import ast
    original = snapshot.read_bytes()
    current = (ROOT / "tests/pytorch/test_grouped_gemm_rtc.py").read_bytes()
    assert hashlib.sha256(original).hexdigest() == receipt["test_source_sha256"], "快照未绑定原验收"
    selected = {case["name"].split("[", 1)[0] for case in receipt["testcases"]}
    def relevant_ast(contents):
        tree = ast.parse(contents)
        available = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert selected and selected <= available
        # 保留选中的测试及全部公共fixture/helper/import/常量；仅排除未运行的测试函数。
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.FunctionDef) and node.name.startswith("test_") and node.name not in selected)]
        return ast.dump(tree, include_attributes=False)
    old_ast, new_ast = relevant_ast(original), relevant_ast(current)
    assert old_ast == new_ast, "已验收测试或公共依赖改变，必须重跑"
    return {"status": "PASS", "original_sha256": receipt["test_source_sha256"],
            "current_sha256": hashlib.sha256(current).hexdigest(), "snapshot": str(snapshot),
            "selected_tests": len(receipt["testcases"]), "common_dependency_ast_equal": True,
            "ast_sha256": hashlib.sha256(new_ast.encode()).hexdigest()}

if a.phase == "bind-verify-source":
    # 修改无关测试前保存与PASS回执逐字节绑定的源码，不改写历史回执。
    assert a.require_verify
    source = ROOT / "tests/pytorch/test_grouped_gemm_rtc.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    receipts = [json.loads(path.read_text()) for path in a.require_verify]
    assert all(r["status"] == "PASS" for r in receipts)
    if a.verify_source_snapshot:
        assert not (a.out / "result.json").exists()
        identities = [verification_source_identity(r, a.verify_source_snapshot) for r in receipts]
        save(a.out / "result.json", {"status": "PASS", "source_equivalence": identities})
        raise SystemExit(0)
    assert all(r["test_source_sha256"] == digest for r in receipts)
    snapshot = a.out / "test_source_snapshot.py"
    assert not snapshot.exists() and not (a.out / "result.json").exists()
    shutil.copy2(source, snapshot)
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest
    save(a.out / "result.json", {"status": "PASS", "test_source_sha256": digest,
                                "snapshot": str(snapshot), "verification": [str(p.resolve()) for p in a.require_verify]})
    raise SystemExit(0)

if a.phase == "smallk-study":
    # 独立于旧 streaming 实验：实际 RTC 四臂，同输入、AB/BA、3轮×3样本。
    assert (a.layout, a.dtype, a.source_mode, a.rows, a.cols, a.mpe, a.groups) == \
        ("TN", "bf16", "noC", 4096, 4096, 512, 16)
    assert a.full and a.samples == 3 and a.require_verify
    assert a.repeats == (1 if a.diagnostic else 3)
    receipt = a.out / "result.json"
    assert not receipt.exists(), "不覆盖已有采集证据"
    record = {"status": "RUNNING", "pid": os.getpid(), "reports": [],
              "case": "fwd_noC_M4096_N4096_K512_groups16_ragged", "metric": METRIC,
              "diagnostic": a.diagnostic, "performance_valid": False,
              "contention": False, "occupancy_snapshots": [], "profiler_contention": {}}
    save(receipt, record)
    def check_device(stage):
        if not a.diagnostic:
            return device_idle()
        devices = subprocess.check_output(["nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,nounits"], text=True)
        apps = subprocess.check_output(["nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,nounits"], text=True)
        gpu = next({k.strip().split(" [", 1)[0]: v.strip() for k, v in row.items()}
                   for row in csv.DictReader(io.StringIO(devices)) if int(row["index"]) == a.gpu)
        live = [{k.strip(): v.strip() for k, v in row.items()}
                for row in csv.DictReader(io.StringIO(apps))
                if row["gpu_uuid"].strip() == gpu["uuid"] and Path(f"/proc/{row[' pid'].strip()}").exists()]
        assert float(gpu["memory.total"]) - float(gpu["memory.used"]) > 8192, "剩余显存不足"
        record["contention"] |= bool(live)
        record["occupancy_snapshots"].append({"stage": stage, "device": gpu, "live": live,
            "timestamp": datetime.now(timezone.utc).isoformat(), "all_devices": devices, "all_apps": apps})
        save(receipt, record)
        return gpu
    try:
        try:
            device = check_device("start")
        except AssertionError:
            if a.diagnostic:
                raise
            record["status"] = "NO_DEVICE_AVAILABLE"
            record["occupancy"] = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv"], text=True)
            save(receipt, record)
            raise SystemExit(2)
        checks = [json.loads(path.read_text()) for path in a.require_verify]
        assert all(r["status"] == "PASS" for r in checks), checks
        library_paths = [ROOT / "libtransformer_engine.so", *ROOT.glob("transformer_engine_torch.*.so")]
        assert len(library_paths) == 2, library_paths
        library_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in library_paths}
        test_hash = hashlib.sha256((ROOT / "tests/pytorch/test_grouped_gemm_rtc.py").read_bytes()).hexdigest()
        assert all(r.get("library_sha256") == library_hashes for r in checks), "验收回执不属于当前库"
        record["verification_source_equivalence"] = []
        for check in checks:
            if check["test_source_sha256"] != test_hash:
                assert a.verify_source_snapshot, "测试文件改变，须提供绑定原验收的源码快照"
                record["verification_source_equivalence"].append(
                    verification_source_identity(check, a.verify_source_snapshot))
        cases_verified = {(c["classname"], c["name"]) for r in checks for c in r["testcases"]}
        contiguous_verified = sum("test_contiguous_" in name for _, name in cases_verified)
        full_regression = len(cases_verified) == 208 and contiguous_verified == 89
        if not a.diagnostic:
            assert full_regression, "必须覆盖89项contiguous与119项通用回归"
        record.update(device=device, verification=checks, library_sha256=library_hashes,
                      full_regression_complete=full_regression, verified_cases=len(cases_verified),
                      contiguous_verified=contiguous_verified)
        overlay = ROOT / "transformer_engine/common/gemm/actlize_overlay/include"
        include = ROOT / "build/cmake/actlize_effective/include"
        reference_record = None
        if a.actual_reference:
            assert a.diagnostic, "复用实际身份仅用于显式诊断，不替代正式验收"
            reference_record = json.loads((a.actual_reference / "result.json").read_text())
            assert reference_record["library_sha256"] == library_hashes
            for path in (ROOT / "profile/gg_fp32_accum/repro.py",
                         ROOT / "transformer_engine/common/gemm/rtc/cutlass_grouped_gemm.cu"):
                assert reference_record["source_sha256"][str(path.relative_to(ROOT))] == hashlib.sha256(path.read_bytes()).hexdigest()
            record["actual_reference"] = str(a.actual_reference.resolve())
        frozen = (a.actual_reference.resolve() / "rtc_include") if reference_record else a.out / "rtc_include"
        def digests(tree):
            return {str(p.relative_to(tree)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(tree.rglob("*")) if p.is_file()}
        before = digests(include)
        assert before and all(digests(Path(r["rtc_include"])) == before for r in checks), \
            "验收include与当前采集树不一致"
        if not reference_record:
            shutil.copytree(include, frozen)
        assert before == digests(include) == digests(frozen)
        record["rtc_include"] = str(frozen)
        save(a.out / "include_identity.json", before)
        source_paths = [ROOT / "profile/gg_fp32_accum/repro.py", Path(__file__).resolve(),
                        ROOT / "transformer_engine/common/gemm/rtc/cutlass_grouped_gemm.cu"]
        record["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in source_paths}
        for relative in ("cutlass/gemm/kernel/nvte_contiguous_grouped.hpp",
                         "cutlass/gemm/kernel/nvte_contiguous_grouped_params.hpp",
                         "cutlass/gemm/collective/ppu_mma_aiu_multistage_batch_array.hpp",
                         "cutlass/gemm/kernel/ppu_aiu_gemm_array_group.hpp",
                         "cutlass/epilogue/collective/ppu_epilogue_vectorized_evt.hpp"):
            assert (overlay / relative).read_bytes() == (frozen / relative).read_bytes() == \
                (ROOT / "include/cutlass3" / relative).read_bytes(), relative
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), LC_ALL="C", LANG="C",
                   PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   NVTE_CUTLASS3_INCLUDE_DIR=str(frozen), NVTE_GG_RTC_LINEINFO="1",
                   NVTE_CUTLASS_GROUPED_GEMM_RTC_VERBOSE="1", CUDA_LAUNCH_BLOCKING="1")
        for key in ("NVTE_GG_DISABLE_RTC_CACHE", "NVTE_GG_NO_EXTRA_NVRTC_FLAGS"):
            env.pop(key, None)
        arms = ["production", "matched", "contiguous", "nexpand"]
        base = [sys.executable, str(ROOT / "profile/gg_fp32_accum/repro.py"), "--layout", "TN",
                "--dtype", "bf16", "--rows", "4096", "--cols", "4096", "--mpe", "512",
                "--groups", "16", "--accumulate", "0", "--check"]
        actual = a.out / "actual"
        if reference_record:
            shutil.copytree(a.actual_reference / "actual", actual)
            identity = json.loads((actual / "rtc/smallk.json").read_text())
            assert identity == reference_record["actual"]
            assert all(v["correctness"] == "PASS" for v in identity["arms"].values())
            assert set(identity["arms"]) == set(arms)
            assert len({v["output_sha256"] for v in identity["arms"].values()}) == 1
            texts = [(actual / "verify.log").read_text()]
        else:
            actual.mkdir()
            (actual / "rtc").mkdir()
            identity, texts = None, []
        # FORCE_TILE/withC 在 host 首次调用后缓存，四臂独立进程；摘要与逐位对拍证明同输入。
        baseline_output = actual / "production_output.pt"
        for arm in ([] if reference_record else arms):
            check_device(f"actual_{arm}")
            job = actual / arm
            job.mkdir()
            cmd = [*base, "--dump", str(job / "rtc"), "--smallk-arms", arm,
                   "--save" if arm == "production" else "--compare", str(baseline_output)]
            save(job / "command.json", cmd)
            texts.append(execute(cmd, job / "verify.log", env=env, cwd=job, strict_contention=False))
            current = json.loads((job / "rtc/smallk.json").read_text())
            if identity is None:
                identity = current
            else:
                assert current["input_sha256"] == identity["input_sha256"]
                assert current["splits"] == identity["splits"]
                assert current["arms"][arm]["output_sha256"] == identity["arms"]["production"]["output_sha256"]
                identity["arms"].update(current["arms"])
            shutil.copytree(job / "rtc" / arm, actual / "rtc" / arm)
        text = "\n".join(texts)
        (actual / "verify.log").write_text(text)
        save(actual / "rtc/smallk.json", identity)
        record["actual"] = identity
        record["comparison_method"] = "independent processes; identical packed input hashes; bitwise output comparison"
        record["resources"] = {}
        symbols = {}
        for arm in arms:
            check_device(f"resources_{arm}")
            symbol = identity["arms"][arm]["kernels"][0]
            symbols[arm] = symbol
            dest = actual / "rtc" / arm
            cubin = dest / f"{symbol}.cubin"
            if not cubin.exists():
                # 同一生产 tile 命中进程内 cache 时，复用同 symbol 的实际字节并明确记录。
                origin = next(p for other in arms if other != arm
                              for p in (actual / "rtc" / other).glob(f"{symbol}.cubin"))
                for suffix in ("cubin", "cu"):
                    shutil.copyfile(origin.with_suffix(f".{suffix}"), dest / f"{symbol}.{suffix}")
                record.setdefault("memory_cache_reuse", {})[arm] = str(origin)
            resources = re.findall(r"kernel report: threads=(\d+).*?chosen_smem=(\d+) for " +
                                   re.escape(symbol), text)
            assert resources and len(set(resources)) == 1, (arm, resources)
            threads, smem = map(int, resources[0])
            if arm in ("contiguous", "nexpand"):
                assert (threads, smem) == (256, 147584), (arm, threads, smem)
                expected_work = identity["arms"][arm]["expected_contiguous_work_count"]
                assert re.search(re.escape(symbol) + rf" work={expected_work} grid=\d+ smem={smem}", text)
            command = [sys.executable, str(ROOT / "profile/gg_fp32_accum/repro.py"),
                       "--resources", str(dest), "--resource-smem", str(smem),
                       "--resource-threads", str(threads)]
            execute(command, dest / "driver.log", env=env)
            resource = json.loads((dest / f"{symbol}.driver_resources.json").read_text())
            assert resource["static_smem_bytes"] == 0 and resource["local_bytes_per_thread"] == 0
            assert resource["active_blocks_per_cu"] > 0
            record["resources"][arm] = resource
            execute(["hgobjdump", "--dump-isa", str(cubin)], dest / "isa.txt")
            execute(["hgobjdump", "--dump-resource-usage=all", str(cubin)], dest / "resources.txt")
            assert (dest / f"{symbol}.options.txt").is_file()
        save(receipt, record)
        raw_dir, cases = a.out / "remote-results", a.out / "cases"
        raw_dir.mkdir(); cases.mkdir()
        values = {arm: {"tensor_active_pct": [], "duration_ns": []} for arm in arms}
        sdk = subprocess.check_output(["acu", "--version"], env=env, text=True).strip()
        compiler = subprocess.check_output(["nvcc", "--version"], env=env, text=True).strip()
        sdk_version = (f"acu {re.search(r'version\s+(\S+)', sdk)[1]}, "
                       f"Data {re.search(r'Data version\s+(\S+)', sdk)[1]}, "
                       f"HGGC {re.search(r'release\s+([^,]+)', compiler)[1]}")
        sys.path.insert(0, "/usr/local/PPU_SDK/asight/extras/python")
        import acu_report
        def machine_identity(path):
            # 不同cwd会改变DWARF；逐字节验证所有非调试section，包含机器码与资源。
            import struct
            image = path.read_bytes()
            assert image[:6] == b"\x7fELF\x02\x01"
            offset = struct.unpack_from("<Q", image, 40)[0]
            size, count, string_index = struct.unpack_from("<HHH", image, 58)
            assert size == 64 and count < 0xff00
            headers = [struct.unpack_from("<IIQQQQIIQQ", image, offset + i * size) for i in range(count)]
            strings_header = headers[string_index]
            strings = image[strings_header[4]:strings_header[4] + strings_header[5]]
            result = {}
            for header in headers:
                name = strings[header[0]:].split(b"\0", 1)[0].decode()
                if not name or name == ".shstrtab" or name.startswith((".debug", ".rela.debug", ".rel.debug")):
                    continue
                contents = b"" if header[1] == 8 else image[header[4]:header[4] + header[5]]
                result[name] = {"type": header[1], "flags": header[2], "address": header[3],
                                "size": header[5], "alignment": header[8], "entry_size": header[9],
                                "sha256": hashlib.sha256(contents).hexdigest()}
            assert any(name.startswith(".text") for name in result)
            return result
        annotated = a.out / "annotated_rtc"
        for arm in arms:
            native_dir = actual / "rtc" / arm
            native = native_dir / f"{symbols[arm]}.cubin"
            dest = annotated / arm
            cubin = annotate_native_rtc(native, dest, frozen, symbols[arm])
            assert machine_identity(cubin) == machine_identity(native), "注释改变非调试section"
            shutil.copyfile(native.with_suffix(".options.txt"), cubin.with_suffix(".options.txt"))
            isa = execute(["hgobjdump", "--dump-isa", str(cubin)], dest / "isa.txt")
            resources = execute(["hgobjdump", "--dump-resource-usage=all", str(cubin)], dest / "resources.txt")
            pattern = r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2}\s+){8})"
            code = [(pc, data.split()) for pc, data in re.findall(pattern, isa, re.M)]
            old_code = [(pc, data.split()) for pc, data in re.findall(pattern, (native_dir / "isa.txt").read_text(), re.M)]
            assert code and code == old_code, "注释改变机器码"
            for pattern in (r"vreg_number:(\d+)", r"STACK SIZE:(\d+)", r"shared_memory_size:(\d+)"):
                assert re.findall(pattern, resources) == re.findall(pattern, (native_dir / "resources.txt").read_text())
            old_resource = record["resources"][arm]
            execute([sys.executable, str(ROOT / "profile/gg_fp32_accum/repro.py"),
                     "--resources", str(dest), "--resource-smem", str(old_resource["dynamic_smem_bytes"]),
                     "--resource-threads", str(old_resource["threads"])], dest / "driver.log", env=env)
            resource = json.loads((dest / f"{symbols[arm]}.driver_resources.json").read_text())
            assert {k: v for k, v in resource.items() if k != "cubin"} == \
                {k: v for k, v in old_resource.items() if k != "cubin"}, "注释改变Driver资源"
            save(dest / "lineinfo_identity.json", {"status": "PASS", "instruction_bytes_equal": True,
                "non_debug_sections_equal": True, "driver_resources_equal": True, "instructions": len(code)})
        def profiled(arm, label, full):
            check_device(f"before_{label}")
            assert library_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in library_paths}
            assert record["source_sha256"] == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in source_paths}, "采集期间源码改变"
            job = a.out / label
            job.mkdir()
            report = raw_dir / f"{label}.acurep"
            cmd = ["acu", "--set", "full", "--import-source", "yes"] if full else \
                ["acu", "--section", "LaunchStats", "--import-source", "yes"]
            debugger = []
            if a.debug_capture:
                assert a.diagnostic and not full
                debugger = ["gdb", "-batch", "-ex", "set pagination off",
                    "-ex", "set print thread-events off", "-ex", "set breakpoint pending on",
                    "-ex", "break std::__throw_invalid_argument(char const*)", "-ex", "run",
                    "-ex", "bt 30", "-ex", "info args", "--args"]
            cmd += ["-k", "regex:^" + re.escape(symbols[arm]) + "$", "-s", "4", "-c", "3",
                    "--kill=no", "--check-exit-code=yes", "-o", str(report),
                    *debugger, *base, "--smallk-arms", arm, "--smallk-profile", "--iters", "3",
                    "--smallk-cubins", str(annotated), "--dump", str(job / "rtc")]
            save(job / "command.json", {"command": cmd, "sdk_version": sdk_version,
                                       "timestamp": datetime.now(timezone.utc).isoformat()})
            log = execute(cmd, job / "profile.log", env=env, cwd=job, strict_contention=False)
            foreign, target = foreign_contenders(log)
            record["profiler_contention"][label] = {"foreign_pids": foreign,
                "target_pids": sorted(target), "warnings": log.lower().count("may affect")}
            record["contention"] |= bool(foreign)
            if not a.diagnostic:
                assert not foreign, (foreign, target)
            loaded = annotated / arm / f"{symbols[arm]}.cubin"
            assert f"loaded prebuilt cubin {loaded} " in log, "未命中已校验实际RTC加载"
            sample_identity = json.loads((job / "rtc/smallk.json").read_text())
            assert sample_identity["arms"][arm]["loaded_cubin_sha256"] == hashlib.sha256(loaded.read_bytes()).hexdigest()
            assert sample_identity["input_sha256"] == identity["input_sha256"]
            assert sample_identity["arms"][arm]["output_sha256"] == identity["arms"][arm]["output_sha256"]
            native = job / "rtc" / arm / f"{symbols[arm]}.cubin"
            reference = actual / "rtc" / arm / native.name
            native_identity = machine_identity(native)
            assert native_identity == machine_identity(reference), arm
            assert native.with_suffix(".options.txt").read_bytes() == reference.with_suffix(".options.txt").read_bytes()
            assert native.with_suffix(".cu").read_bytes() == reference.with_suffix(".cu").read_bytes()
            save(job / "machine_identity.json", native_identity)
            check_device(f"after_{label}")
            context = acu_report.load_report(str(report))
            ranges = [context.range_by_idx(i) for i in range(context.num_ranges())]
            actions = [r.action_by_idx(j) for r in ranges for j in range(r.num_actions())]
            assert len(actions) == 3
            source_manifests = []
            expected = ("nvte_contiguous_grouped.hpp" if arm in ("contiguous", "nexpand")
                        else "ppu_aiu_gemm_array_group.hpp")
            for action in actions:
                embedded = action.source_files()
                assert embedded and all(embedded.values()), "探测未实际嵌入源码，不启动full"
                for name in (expected, "ppu_mma_aiu_multistage_batch_array.hpp", "ppu_epilogue_vectorized_evt.hpp"):
                    content = next(v for p, v in embedded.items() if Path(p).name == name)
                    header = next(frozen.rglob(name))
                    assert content.splitlines() == header.read_text().splitlines(), name
                source_manifests.append({p: hashlib.sha256(v.encode()).hexdigest() for p, v in embedded.items()})
            save(job / "embedded_sources.json", source_manifests)
            if not full:
                execute(["acu", "-i", str(report), "--page", "details"], job / "details.txt", env=env)
                return
            csv_path = job / "full.csv"
            execute(["acu", "-i", str(report), "--stats-rule", "ExportMetricsByKernel",
                     "--stats-rule-output", str(csv_path)], job / "metrics_export.log", env=env)
            rows = list(csv.DictReader(io.StringIO(csv_path.read_text())))
            assert len(rows) == 3 and {r["Kernel Name"] for r in rows} == {symbols[arm]}
            sass = cases / f"{label}.sass.csv"
            execute(["acu", "-i", str(report), "--page", "source", "--csv-file", str(sass)],
                    job / "sass_export.log", env=env)
            sass_rows = list(csv.DictReader(io.StringIO(sass.read_text())))
            assert any(float(r.get("Instructions Executed", "0").replace(",", "") or 0) > 0
                       for r in sass_rows), "缺少非空动态 SASS，不能用静态 ISA 替代"
            assert len(actions) == len(rows)
            output = []
            metadata = {"Range ID", "Kernel Name", "Kernel Mangled Name", "Kernel Demangled Name",
                        "Block Size", "Grid Size", "Device"}
            for row, action in zip(rows, actions):
                def number(v):
                    try: return float(v.replace(",", ""))
                    except (ValueError, AttributeError): return v
                raw = {k: number(v) for k, v in row.items() if k not in metadata}
                assert set(raw) == set(action.metrics()), "metrics 必须全量导出"
                embedded = action.source_files()
                assert embedded and all(embedded.values()), "未实际嵌入源码，不放行采集回执"
                expected = ("nvte_contiguous_grouped.hpp" if arm in ("contiguous", "nexpand")
                            else "ppu_aiu_gemm_array_group.hpp")
                assert any(Path(path).name == expected for path in embedded), expected
                output.append({"device": "PPU-890", "kernel_name": symbols[arm], "profiler": "ACU",
                               "sdk_version": sdk_version, "process_id": next(iter(target)),
                               "device_name": row["Device"], "arm": arm, "raw_metrics": raw,
                               "kernel_mangled_name": row.get("Kernel Mangled Name"),
                               "timestamp": datetime.now(timezone.utc).isoformat(),
                               "host_name": socket.gethostname(), "diagnostic": a.diagnostic,
                               "performance_valid": not a.diagnostic,
                               "foreign_pids": foreign,
                               "source_sha256": {p: hashlib.sha256(v.encode()).hexdigest()
                                                 for p, v in embedded.items()}})
                values[arm]["tensor_active_pct"].append(raw[METRIC])
                values[arm]["duration_ns"].append(raw["ppu__time_duration.sum"])
            save(cases / f"{label}.metrics.json", output)
            record["reports"].append(str(report))
            save(receipt, record)
        record["order"], record["capture_failures"] = [], {}
        if a.diagnostic:
            # 单臂采集失败不丢失其余诊断臂；仍须先通过该臂源码探测。
            for arm in (arms[:1] if a.debug_capture else arms):
                try:
                    profiled(arm, f"probe_{arm}", False)
                    record["order"].append(arm)
                    profiled(arm, f"{arm}_full_r0", True)
                except Exception as error:
                    record["capture_failures"][arm] = str(error)
                    save(receipt, record)
            assert record["reports"], "没有完整导出成功的诊断报告，保留原始失败现场"
        else:
            for arm in arms:
                profiled(arm, f"probe_{arm}", False)
            for repeat in range(a.repeats):
                order = arms if repeat % 2 == 0 else list(reversed(arms))
                for arm in order:
                    record["order"].append(arm)
                    profiled(arm, f"{arm}_full_r{repeat}", True)
        record["samples"] = values
        record["performance_valid"] = not a.diagnostic
        record["status"] = ("PARTIAL_DIAGNOSTIC_CAPTURED" if record["capture_failures"] else
                            "DIAGNOSTIC_CAPTURED") if a.diagnostic else "PASS"
    except Exception as error:
        record.update(status="FAIL", error=str(error))
        raise
    finally:
        save(receipt, record)
    raise SystemExit(0)

if a.phase == "tail-study":
    # 四臂同输入：原生 RTC 数值/机器码身份 -> 源码探测 -> 配对 full。
    variants = ["baseline_aiu", "stream_no_c", "stream_split_before", "stream_split_after"]
    assert a.variants == variants and a.layout == "NN" and a.dtype == "bf16" and not WITH_C
    assert (a.rows, a.cols, a.mpe, a.groups) == (2048, 2048, 512, 16)
    assert a.full and a.samples >= 2 and a.repeats >= 2 and a.require_verify
    receipt = a.out / "result.json"
    dest = a.out / "evidence"
    assert not receipt.exists() and not dest.exists(), "不覆盖历史实验"
    device = device_idle()
    dest.mkdir()
    env = dict(os.environ, LC_ALL="C", LANG="C", CUDA_LAUNCH_BLOCKING="1",
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for key in ("NVTE_GG_RTC_LINEINFO", "NVTE_GG_RTC_LOAD_CUBIN_DIR", "NVTE_GG_C_EVT_CLASSIC",
                "NVTE_GG_DISABLE_RTC_CACHE", "NVTE_GG_NO_EXTRA_NVRTC_FLAGS", "NVTE_GG_FORCE_WITH_C"):
        env.pop(key, None)
    record = {"status": "RUNNING", "device": device, "variants": variants,
              "pid": os.getpid(), "collection": "--set full --import-source yes", "steps": []}
    save(receipt, record)
    common = ["--out", str(dest), "--gpu", str(a.gpu), "--layout", a.layout, "--dtype", a.dtype,
              "--source-mode", a.source_mode, "--rows", str(a.rows), "--cols", str(a.cols),
              "--mpe", str(a.mpe), "--groups", str(a.groups)]
    def run_phase(phase, selected, extra, label):
        if phase in ("run", "resources", "probe", "capture"):
            assert device_idle() == device
        phase_env = dict(env)
        if phase == "run" and "--rtc" in extra:
            phase_env["NVTE_GG_RTC_LINEINFO"] = "1"
        command = [sys.executable, str(Path(__file__).resolve()), phase, *common,
                   "--variants", *selected, *extra]
        save(dest / f"{label}.command.json", command)
        text = execute(command, dest / f"{label}.log", env=phase_env, strict_contention=False,
                       cwd=dest if phase == "run" and "--rtc" in extra else ROOT)
        record["steps"].append(label)
        save(receipt, record)
        return text
    try:
        outputs = {}
        for variant in variants:
            outputs[variant] = dest / f"{variant}.outputs.pt"
            run_phase("run", [variant], ["--rtc", "--warmup", "0", "--samples", "0",
                      "--save", str(outputs[variant])], f"actual_{variant}")
        import torch
        baseline = torch.load(outputs[variants[0]], weights_only=True)
        for variant in variants[1:]:
            current = torch.load(outputs[variant], weights_only=True)
            assert len(current) == len(baseline) and all(torch.equal(x, y) for x, y in zip(current, baseline))
        record["actual_bitwise_equal"] = True
        run_phase("resources", variants, ["--rtc"], "resources")
        run_phase("prepare", variants, ["--annotate-rtc"], "prepare")
        for variant in variants:
            identity = json.loads((dest / variant / "lineinfo_identity.json").read_text())
            assert identity["status"] == "PASS"
            text = run_phase("run", [variant], ["--warmup", "0", "--samples", "0",
                      "--compare", str(outputs[variant])], f"annotated_{variant}")
            assert "BITWISE_ACTUAL_RTC_OK" in text and "loaded prebuilt cubin" in text
        # 数值矩阵运行在另一张空闲卡；所有回执 PASS 后才允许 full 采集。
        while True:
            checks = [json.loads(path.read_text()) if path.exists() else {} for path in a.require_verify]
            assert not any(check.get("status") in ("FAIL", "NO_DEVICE_AVAILABLE") for check in checks), checks
            if all(check.get("status") == "PASS" for check in checks):
                record["verification"] = checks
                break
            time.sleep(15)
        run_phase("probe", variants, ["--warmup", str(a.warmup)], "probe")
        run_phase("capture", variants, ["--full", "--warmup", str(a.warmup), "--samples", str(a.samples),
                  "--repeats", str(a.repeats)], "capture")
        sys.path.insert(0, "/usr/local/PPU_SDK/asight/extras/python")
        import acu_report
        record["reports"] = []
        for report in sorted(dest.glob("*_full_r*.acurep")):
            metrics = json.loads(report.with_suffix(".metrics.json").read_text())
            context = acu_report.load_report(str(report))
            actions = []
            for i in range(context.num_ranges()):
                report_range = context.range_by_idx(i)
                for j in range(report_range.num_actions()):
                    action = report_range.action_by_idx(j)
                    sources = action.source_files()
                    assert set(action.metrics()) == set(metrics[len(actions)]["raw_metrics"])
                    for name in ("ppu_aiu_gemm_array_group.hpp", "ppu_mma_aiu_multistage_batch_array.hpp",
                                 "ppu_epilogue_vectorized_evt.hpp"):
                        text = next(content for path, content in sources.items() if Path(path).name == name)
                        reference = next((dest / metrics[len(actions)]["variant"] / "include").rglob(name))
                        assert text.splitlines() == reference.read_text().splitlines(), name
                    actions.append({path: {"bytes": len(content.encode()),
                        "sha256": hashlib.sha256(content.encode()).hexdigest()} for path, content in sources.items()})
            assert len(actions) == a.samples * len(variants)
            save(report.with_suffix(".sources.json"), actions)
            target = a.out / "remote-results" / report.name
            target.parent.mkdir(exist_ok=True)
            report.rename(target)
            record["reports"].append({"path": str(target), "metrics": str(report.with_suffix(".metrics.json")),
                                      "sass": str(report.with_suffix(".sass.csv")), "actions": len(actions)})
        assert len(record["reports"]) == a.repeats
        record["status"] = "PASS"
    except Exception as error:
        record.update(status="FAIL", error=str(error))
        raise
    finally:
        save(receipt, record)
    print(json.dumps(record), flush=True)
    raise SystemExit(0)


if a.phase == "source-reports":
    # 三类低 TFU 场景的单实例 full 报告；只扩展采集脚本，不修改内核。
    receipt = a.out / "result.json"
    assert not receipt.exists(), "不可覆盖已有采集"
    device = device_idle()
    sys.path.insert(0, "/usr/local/PPU_SDK/asight/extras/python")
    import acu_report
    env = dict(os.environ, LC_ALL="C", LANG="C", CUDA_LAUNCH_BLOCKING="1")
    for key in ("NVTE_GG_RTC_LINEINFO", "NVTE_GG_RTC_LOAD_CUBIN_DIR", "NVTE_GG_C_EVT_CLASSIC",
                "NVTE_GG_DISABLE_RTC_CACHE", "NVTE_GG_NO_EXTRA_NVRTC_FLAGS", "NVTE_GG_FORCE_WITH_C"):
        env.pop(key, None)
    record = {"status": "RUNNING", "device": device, "metric": METRIC,
              "tile": [256, 256, 64, 3], "groups": 16, "cases": [],
              "collection": "--set full --import-source yes", "pid": os.getpid()}
    save(receipt, record)
    script = str(Path(__file__).resolve())
    cases = [("fwd", "TN", 2048, 512, 2048, "bf16", "noC"),
             ("dgrad", "NN", 2048, 2048, 512, "bf16", "noC"),
             ("wgrad_C", "NT", 512, 2048, 2048, "fp32", "withC")]
    try:
        for name, layout, m, n, k, dtype, source in cases:
            dest = a.out / "evidence" / name
            assert not dest.exists(), "不覆盖历史产物"
            dest.mkdir(parents=True)
            common = ["--out", str(dest), "--gpu", str(a.gpu), "--variants", "baseline_aiu",
                      "--layout", layout, "--rows", str(m), "--cols", str(n), "--mpe", str(k),
                      "--groups", "16", "--dtype", dtype, "--source-mode", source]
            def run_phase(phase, extra, log):
                if phase in ("run", "capture"):
                    assert device_idle() == device, "设备改变或被占用"
                command = [sys.executable, script, phase, *common, *extra]
                save(dest / f"{log}.command.json", command)
                phase_env = dict(env, PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""))
                if phase == "run" and "--rtc" in extra:
                    phase_env["NVTE_GG_RTC_LINEINFO"] = "1"
                return execute(command, dest / f"{log}.log", env=phase_env, strict_contention=False,
                               cwd=dest if phase == "run" and "--rtc" in extra else ROOT)
            output = dest / "actual_outputs.pt"
            run_phase("run", ["--rtc", "--samples", "0", "--warmup", "0", "--save", str(output)], "actual")
            run_phase("prepare", ["--annotate-rtc"], "prepare")
            identity = json.loads((dest / "baseline_aiu/lineinfo_identity.json").read_text())
            assert identity["status"] == "PASS"
            text = run_phase("run", ["--samples", "0", "--warmup", "0", "--compare", str(output)], "lineinfo_check")
            assert "BITWISE_ACTUAL_RTC_OK" in text and "loaded prebuilt cubin" in text
            # 先用 LaunchStats 核对实际稳态目标，避免 full 抓错 launch。
            probe = dest / "probe.acurep"
            command = ["acu", "--profile=on", "--section", "LaunchStats", "--import-source", "yes",
                       "-k", "regex:^nvte_grouped_gemm3_",
                       "-s", "6", "-c", "1", "--kill=no", "--check-exit-code=yes", "-o", str(probe),
                       sys.executable, script, "run", *common, "--warmup", "4", "--samples", "1"]
            device_idle()
            save(dest / "probe.command.json", command)
            text = execute(command, dest / "probe.log", env=env, strict_contention=False)
            assert not foreign_contenders(text)[0], "探测出现外部争用"
            raw = execute(["acu", "-i", str(probe), "--page", "raw", "--csv"], dest / "probe.raw.csv", env=env)
            probe_rows = list(csv.DictReader(io.StringIO(raw)))
            assert len(probe_rows) == 1 and f"bf16_bf16_{dtype}_acc_fp32" in probe_rows[0]["Kernel Name"]
            # ctypes 子对象不持有父对象；必须保留 context，避免 source_files 悬空访问。
            probe_context = acu_report.load_report(str(probe))
            probe_range = probe_context.range_by_idx(0)
            probe_action = probe_range.action_by_idx(0)
            assert probe_action.source_files(), "轻量探测未实际嵌入源码，不能启动 full 采集"
            run_phase("capture", ["--full", "--warmup", "4", "--samples", "1", "--repeats", "1"], "capture")
            report = next(dest.glob("*_r0.acurep"))
            log = report.with_suffix(".log").read_text()
            assert "without debug line information" not in log and "cannot be collected" not in log, "源码导入失败"
            assert "loaded prebuilt cubin" in log
            metrics = json.loads(report.with_suffix(".metrics.json").read_text())
            assert len(metrics) == 1 and len(metrics[0]["raw_metrics"]) > 800
            details = report.with_suffix(".details.txt").read_text()
            assert "Source Counters" in details and "Warp State Statistics" in details
            context = acu_report.load_report(str(report))
            assert context.num_ranges() == 1 and context.range_by_idx(0).num_actions() == 1
            action = context.range_by_idx(0).action_by_idx(0)
            embedded = action.source_files()
            assert embedded and all(embedded.values()), "报告未实际嵌入源码内容"
            for expected in ("ppu_aiu_gemm_array_group.hpp", "ppu_mma_aiu_multistage_batch_array.hpp",
                             "ppu_epilogue_vectorized_evt.hpp"):
                assert any(Path(path).name == expected for path in embedded), f"缺少导入源码：{expected}"
            assert set(action.metrics()) == set(metrics[0]["raw_metrics"]), "全量指标导出不完整"
            source_manifest = {path: {"bytes": len(content.encode()),
                "sha256": hashlib.sha256(content.encode()).hexdigest()} for path, content in embedded.items()}
            save(dest / "embedded_sources.json", source_manifest)
            target = a.out / "remote-results" / f"{name}.acurep"
            target.parent.mkdir(parents=True, exist_ok=True)
            small = a.out / "cases" / name
            small.mkdir(parents=True)
            report.rename(target)
            save(small / "890.metrics.json", metrics[0])
            report.with_suffix(".sass.csv").rename(small / "890.sass.csv")
            record["cases"].append({"name": name, "layout": layout, "gemm": [m, n, k], "dtype": dtype,
                "source_mode": source, "report": str(target), "report_bytes": target.stat().st_size,
                "metrics": str(small / "890.metrics.json"), "sass": str(small / "890.sass.csv"),
                "tfu": metrics[0]["raw_metrics"][METRIC], "identity": identity,
                "source_import_warning": False, "embedded_source_files": len(embedded),
                "metric_count": len(metrics[0]["raw_metrics"]), "sections": re.findall(r"Section: (.*)", details)})
            save(receipt, record)
            print(json.dumps(record["cases"][-1]), flush=True)
        record["status"] = "PASS"
    except Exception as error:
        record.update(status="FAIL", error=str(error))
        raise
    finally:
        save(receipt, record)
    raise SystemExit(0)


if a.phase == "matrix":
    # 四组严格配对 × 八 shape + 非均匀 fwd + no-next；收益为负也不裁剪矩阵。
    assert a.rtc and a.full and a.samples == 10 and a.repeats == 3
    assert a.require_verify, "全矩阵采集必须绑定正确性验收回执"
    manifest = a.out / "matrix.json"
    shapes = [(1536, 4096), (4096, 1536), (512, 2048), (2048, 512),
              (1024, 3072), (3072, 1024), (1024, 4096), (4096, 1024)]
    cases = [(f"M{m}_N{n}", m, n, 16, "NT", False) for m, n in shapes]
    cases += [("nonuniform_fwd", 4096, 1536, 16, "TN", True),
              ("no_next", 64, 64, 1, "NT", False)]
    families = [("step1", "noC", "next_no_c"), ("step2", "withC", "c_tail"),
                ("step3", "withC", "next_with_c"), ("combined", "withC", "both_c")]
    families = [f for f in families if f[0] in a.families]
    jobs = [{"id": f"{family}_{name}", "rows": m, "cols": n, "groups": groups,
             "layout": layout, "nonuniform": mixed, "source_mode": source,
             "variants": ["baseline_aiu", arm], "status": "PENDING"}
            for name, m, n, groups, layout, mixed in cases
            for family, source, arm in families]
    record = {"status": "WAITING_FOR_CORRECTNESS", "pid": os.getpid(), "gpu": a.gpu,
              "metric": METRIC, "samples": 10, "repeats": 3, "K": 2048,
              "require_verify": [str(p.resolve()) for p in a.require_verify], "jobs": jobs}
    if a.resume:
        assert manifest.exists(), "没有可恢复的矩阵回执"
    if manifest.exists():
        assert a.resume, "请使用新目录或显式 --resume"
        previous = json.loads(manifest.read_text())
        assert previous["gpu"] == a.gpu and previous["require_verify"] == record["require_verify"]
        assert [{k: v for k, v in j.items() if k not in ("status", "failed_attempts")} for j in previous["jobs"]] == [
            {k: v for k, v in j.items() if k != "status"} for j in jobs]
        assert all(j["status"] in ("PASS", "PENDING") for j in previous["jobs"]), \
            "有失败/部分采集；保留历史证据并使用新目录"
        record["jobs"] = previous["jobs"]
    source_paths = [ROOT / "profile/gg_fp32_accum/atomic_study.py", ROOT / "profile/gg_fp32_accum/repro.py",
                    ROOT / "transformer_engine/common/gemm/rtc/cutlass_grouped_gemm.cu",
                    ROOT / "tests/pytorch/test_grouped_gemm_rtc.py",
                    *[ROOT / "transformer_engine/common/gemm/actlize_overlay/include" / relative
                      for relative in ("cutlass/gemm/kernel/ppu_aiu_gemm_array_group.hpp",
                                       "cutlass/gemm/collective/ppu_mma_aiu_multistage_batch_array.hpp",
                                       "cutlass/epilogue/collective/ppu_epilogue_vectorized_evt.hpp")]]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    if a.resume:
        assert source_hashes == previous["source_hashes"], "源码改变，不能恢复旧矩阵"
    record["source_hashes"] = source_hashes
    save(manifest, record)
    try:
        while True:
            receipts = [json.loads(p.read_text()) for p in a.require_verify]
            assert all(r["status"] in ("running", "PASS") for r in receipts), receipts
            pending = [r for r in receipts if r["status"] == "running"]
            assert all(Path(f"/proc/{r['pid']}").exists() for r in pending), \
                "验收进程已经退出但没有最终回执，不能开始采集"
            if not pending:
                break
            time.sleep(15)
        assert all(r.get("counts", {}).get("tests", 0) > 0 and not any(
            r["counts"][k] for k in ("failures", "errors", "skipped")) for r in receipts)
        record.update(status="RUNNING", correctness=receipts)
        save(manifest, record)
        for job in record["jobs"]:
            dest = a.out / job["id"]
            if job["status"] == "PASS":
                assert json.loads((dest / "result.json").read_text())["status"] == "PASS"
                assert len(list(dest.glob("*.summary.json"))) == 1
                continue
            assert {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in source_paths} == source_hashes, "源码改变，停止矩阵并保留已采历史"
            assert not dest.exists(), "不覆盖已启动或失败采集"
            cmd = [sys.executable, str(Path(__file__).resolve()), "capture", "--out", str(dest),
                   "--rtc", "--full", "--gpu", str(a.gpu), "--samples", "10", "--repeats", "3",
                   "--variants", *job["variants"], "--source-mode", job["source_mode"],
                   "--layout", job["layout"], "--groups", str(job["groups"]), "--mpe", "2048",
                   "--rows", str(job["rows"]), "--cols", str(job["cols"])]
            if job["nonuniform"]:
                cmd.append("--nonuniform")
            job["status"] = "RUNNING"
            save(manifest, record)
            for attempt in range(4):
                wait_idle()
                try:
                    execute(cmd, a.out / f"{job['id']}.capture.log", strict_contention=False)
                    break
                except AssertionError:
                    # 争用或中断的采集整体改名保留，换新目录重采，不覆盖证据。
                    job.setdefault("failed_attempts", []).append(f"{dest.name}.failed{attempt}")
                    save(manifest, record)
                    if dest.exists():
                        dest.rename(dest.with_name(f"{dest.name}.failed{attempt}"))
                    log = a.out / f"{job['id']}.capture.log"
                    if log.exists():
                        log.rename(log.with_name(f"{log.name}.failed{attempt}"))
                    if attempt == 3:
                        raise
            resource_cmd = [sys.executable, str(Path(__file__).resolve()), "resources",
                            "--out", str(dest), "--rtc", "--gpu", str(a.gpu),
                            "--layout", job["layout"], "--variants", *job["variants"]]
            for attempt in range(4):
                wait_idle()
                try:
                    execute(resource_cmd, a.out / f"{job['id']}.resources.log", strict_contention=False)
                    break
                except AssertionError:
                    receipt = dest / "result.json"
                    if receipt.exists():
                        receipt.rename(receipt.with_name(f"result.json.failed{attempt}"))
                    if attempt == 3:
                        raise
            summary = json.loads(next(dest.glob("*.summary.json")).read_text())
            assert all(len(v[k]["samples"]) == 30 for v in summary.values()
                       for k in ("tfu", "duration_ns"))
            assert len(list(dest.glob("*.acurep"))) == len(list(dest.glob("*.metrics.json"))) == \
                len(list(dest.glob("*.sass.csv"))) == 3
            for runtime_path in dest.glob("*.runtime.json"):
                runtime = json.loads(runtime_path.read_text())
                assert runtime["multi_work_per_cta"] == (job["groups"] == 16)
            job["status"] = "PASS"
            save(manifest, record)
            print(json.dumps({"completed": job["id"], "total": len(jobs)}), flush=True)
        record["status"] = "PASS"
    except Exception as error:
        record.update(status="FAIL", error=str(error))
        raise
    finally:
        save(manifest, record)
    raise SystemExit(0)


if a.phase in ("build", "audit-build", "verify", "verify-codegen", "resources"):
    # 将完成回执落盘，终端状态通道超时也不会误判或重复启动任务。
    result_path = a.out / "result.json"
    assert not result_path.exists(), "请使用新目录，不能覆盖已有构建/验收证据"
    record = {"phase": a.phase, "status": "running", "pid": os.getpid()}
    save(result_path, record)
    try:
        if a.phase in ("build", "audit-build"):
            if a.phase == "build":
                commands = [["cmake", "-S", "transformer_engine/common", "-B", "build/cmake"],
                            ["cmake", "--build", "build/cmake", "-j", "8"],
                            ["cmake", "--install", "build/cmake"]]
                save(a.out / "commands.json", commands)
                for label, cmd in zip(("configure", "build", "install"), commands):
                    execute(cmd, a.out / f"{label}.log", strict_contention=False)
            overlay = ROOT / "transformer_engine/common/gemm/actlize_overlay/include"
            paths = ("cutlass/gemm/kernel/ppu_aiu_gemm_array_group.hpp",
                     "cutlass/gemm/collective/ppu_mma_aiu_multistage_batch_array.hpp",
                     "cutlass/epilogue/collective/ppu_epilogue_vectorized_evt.hpp")
            contiguous_headers = ("cutlass/gemm/kernel/nvte_contiguous_grouped.hpp",
                                  "cutlass/gemm/kernel/nvte_contiguous_grouped_params.hpp")
            # 专用头缺失必须失败，不能跳过后仍给出构建一致性 PASS。
            paths += contiguous_headers
            record["headers"] = []
            for relative in paths:
                source = (overlay / relative).read_bytes()
                effective = (ROOT / "build/cmake/actlize_effective/include" / relative).read_bytes()
                installed = (ROOT / "include/cutlass3" / relative).read_bytes()
                assert source == effective, f"effective include 未刷新：{relative}"
                assert source == installed, f"安装 include 未刷新：{relative}"
                record["headers"].append({"path": relative, "sha256": hashlib.sha256(source).hexdigest(),
                                          "installed_sha256": hashlib.sha256(installed).hexdigest()})
            tu = (ROOT / "transformer_engine/common/gemm/rtc/cutlass_grouped_gemm.cu").read_bytes()
            baked = (ROOT / "build/cmake/string_headers/string_code_gemm_rtc_cutlass_grouped_gemm_cu.h").read_bytes()
            baked_tu = baked.split(b'R"__STRING_DELIM__(', 1)[1].split(b')__STRING_DELIM__"', 1)[0]
            assert tu == baked_tu, "RTC TU 尚未重新 configure 烘焙"
            record["rtc_tu_sha256"] = hashlib.sha256(tu).hexdigest()
            if a.phase == "audit-build":
                import transformer_engine.pytorch
                from transformer_engine.common import _get_shared_object_file
                record["libraries"] = {}
                for kind in ("core", "torch"):
                    loaded = _get_shared_object_file(kind).resolve()
                    assert loaded.parent == ROOT, (kind, loaded)
                    if kind == "core":
                        built = ROOT / "build/cmake/libtransformer_engine.so"
                    else:
                        # 按当前 ABI 找唯一 extension，不把 Python 版本写死。
                        candidates = list((ROOT / "build").glob(f"lib.*/{loaded.name}"))
                        assert len(candidates) == 1, (kind, candidates)
                        built = candidates[0]
                    digest = hashlib.sha256(loaded.read_bytes()).hexdigest()
                    assert digest == hashlib.sha256(built.read_bytes()).hexdigest(), (kind, built, loaded)
                    record["libraries"][kind] = {"path": str(loaded), "built": str(built), "sha256": digest}
        elif a.phase == "resources":
            assert a.rtc, "资源审计仅使用已落盘的实际 RTC cubin"
            record["device"] = device_idle()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
            record["resources"] = {}
            for variant in a.variants:
                dest = a.out / variant / "actual_rtc"
                cmd = [sys.executable, str(ROOT / "profile/gg_fp32_accum/repro.py"), "--resources", str(dest)]
                save(dest / "resource_command.json", cmd)
                execute(cmd, dest / "driver_resources.log", env=env)
                record["resources"][variant] = json.loads((dest / f"{SYMBOL}.driver_resources.json").read_text())
                resource = record["resources"][variant]
                assert resource["dynamic_smem_bytes"] == 196608 and resource["threads"] == 512
                # 静态 smem 会平移动态 smem 基址、破坏 mainloop swizzle 对齐。
                assert resource["static_smem_bytes"] == 0 and resource["local_bytes_per_thread"] == 0, resource
                assert resource["active_blocks_per_cu"] > 0, resource
        else:
            env = os.environ.copy()
            if a.phase == "verify-codegen":
                assert a.test_filter == "stream_codegen", "无设备阶段只允许编译验收"
                env["CUDA_VISIBLE_DEVICES"] = ""
                record["gpu_execution"] = False
            else:
                record["device"] = device_idle()
                env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
                source = ROOT / "build/cmake/actlize_effective/include"
                frozen = a.out / "rtc_include"
                def header_manifest(tree):
                    return {str(path.relative_to(tree)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(tree.rglob("*")) if path.is_file()}
                before = header_manifest(source)
                assert before, "effective include 树为空"
                shutil.copytree(source, frozen)
                # 复制前后逐文件对拍；构建中的中间树不能成为验收快照。
                assert before == header_manifest(frozen) == header_manifest(source), \
                    "effective include 在冻结期间改变，请等待构建完成后使用新目录"
                save(a.out / "include_identity.json", before)
                env["NVTE_GG_TEST_INCLUDE_DIR"] = str(frozen)
                record["rtc_include"] = str(frozen)
                record["include_files"] = len(before)
                library_paths = [ROOT / "libtransformer_engine.so", *ROOT.glob("transformer_engine_torch.*.so")]
                assert len(library_paths) == 2, library_paths
                record["library_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in library_paths}
                test_source = ROOT / "tests/pytorch/test_grouped_gemm_rtc.py"
                record["test_source_sha256"] = hashlib.sha256(test_source.read_bytes()).hexdigest()
                save(result_path, record)
            cmd = [sys.executable, "-m", "pytest", "tests/pytorch/test_grouped_gemm_rtc.py",
                   "-k", a.test_filter, "--basetemp", str(a.out / "tests"),
                   "--junitxml", str(a.out / "junit.xml"), "-x", "-q"]
            save(a.out / "command.json", cmd)
            execute(cmd, a.out / "pytest.log", env=env, strict_contention=False)
            import xml.etree.ElementTree as ET
            suites = ET.parse(a.out / "junit.xml").getroot().findall("testsuite")
            counts = {key: sum(int(s.get(key, 0)) for s in suites)
                      for key in ("tests", "failures", "errors", "skipped")}
            assert counts["tests"] > 0 and not any(counts[k] for k in ("failures", "errors", "skipped")), counts
            if a.phase == "verify":
                assert record["library_sha256"] == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                     for p in library_paths}, "验收期间库改变"
                assert record["test_source_sha256"] == hashlib.sha256(test_source.read_bytes()).hexdigest()
                record["testcases"] = [{"classname": case.get("classname"), "name": case.get("name")}
                                       for suite in suites for case in suite.findall("testcase")]
                assert len(record["testcases"]) == counts["tests"]
            record["counts"] = counts
            record["test_filter"] = a.test_filter
        record["status"] = "PASS"
    except Exception as error:
        record.update(status="FAIL", error=str(error))
        raise
    finally:
        save(result_path, record)
    print(json.dumps(record), flush=True)
    raise SystemExit(0)

if a.phase == "prepare":
    for variant in a.variants:
        dest = a.out / variant
        dest.mkdir(exist_ok=True)
        include = dest / "include"
        assert not (dest / "build.json").exists(), "不可覆盖已经编译成功的源码快照"
        if not include.exists():
            include_source = (a.actual_reference.resolve().parent / "include"
                              if a.annotate_rtc and a.actual_reference
                              else ROOT / "build/cmake/actlize_effective/include")
            shutil.copytree(include_source, include, symlinks=False)
        source = dest / f"{SYMBOL}.cu"
        source.write_text(f"#undef NVTE_GG_KERNEL_NAME\n#define NVTE_GG_KERNEL_NAME {SYMBOL}\n" +
                          (ROOT / "transformer_engine/common/gemm/rtc/cutlass_grouped_gemm.cu").read_text())
        flags = ["-std=c++17", f"-O{a.offline_opt}", "-lineinfo", "-cubin", "--expt-relaxed-constexpr",
                 "--generate-code=arch=compute_89,code=sm_89",
                 "-I/usr/local/PPU_SDK/CUDA_SDK/include", "-I/usr/local/PPU_SDK/CUDA_SDK/include/cccl",
                 "-I/usr/local/PPU_SDK/CUDA_SDK/include/cccl/nv",
                 f"-I{include}", "-I/usr/local/PPU_SDK/targets/x86_64-linux/include"]
        defines = {"NVTE_GG_TRANS_A": TRANS_A, "NVTE_GG_TRANS_B": TRANS_B, "NVTE_GG_ARCH": 89,
                   "NVTE_GG_DTYPE": 0, "NVTE_GG_OUT_FP32": int(a.dtype == "fp32"), "NVTE_GG_WITH_C": 1,
                   "NVTE_GG_PRELOAD_C": int(a.dtype == "fp32"), "NVTE_GG_TILE_M": 256, "NVTE_GG_TILE_N": 256,
                   "NVTE_GG_TILE_K": 64, "NVTE_GG_STAGES": 3, "CUTLASS_ENABLE_TENSOR_CORE_MMA": 1,
                   "__HGGC_FP16_H__": 1, "_HGGC_FP16_H_": 1, "HGGC_FP16_H": 1,
                   "__HGGC_BF16_H__": 1, "__HGGC_BF16_HPP__": 1, "__BF16_DEFINED__": 1,
                   "NDEBUG": 1, "USE_SAIL": 1, "NVTE_GG_PROFILE_RTC": 1}
        flags += [f"-D{k}={v}" for k, v in defines.items()] + VARIANTS[variant].split()
        cubin = dest / f"{SYMBOL}.cubin"
        cmd = [shutil.which("nvcc"), *flags, str(source), "-o", str(cubin)]
        if not a.annotate_rtc:
            save(dest / "command.json", cmd)
        if a.annotate_rtc:
            save(dest / "command.json", {"method": "annotate-rtc", "argv": sys.argv,
                                       "nvcc_executed": False})
            # 仅给原生 .debug_line 加 CU/函数索引；所有原有可加载 section 保持逐字节不变。
            import struct
            actual_path = (a.actual_reference.resolve() if a.actual_reference else dest / "actual_rtc") / f"{SYMBOL}.cubin"
            original = actual_path.read_bytes()
            image = bytearray(original)
            assert image[:6] == b"\x7fELF\x02\x01", "只支持 ELF64 little-endian"
            shoff = struct.unpack_from("<Q", image, 40)[0]
            entsize, count, strindex = struct.unpack_from("<HHH", image, 58)
            assert entsize == 64 and count < 0xff00
            headers = [list(struct.unpack_from("<IIQQQQIIQQ", image, shoff + i * 64)) for i in range(count)]
            string_header = headers[strindex]
            strings = bytes(image[string_header[4]:string_header[4] + string_header[5]])
            names = [strings[h[0]:].split(b"\0", 1)[0].decode() for h in headers]
            assert ".debug_line" in names and ".debug_info" not in names and ".debug_abbrev" not in names
            text = headers[next(i for i, name in enumerate(names) if name.startswith(".text.kernel."))]
            high_pc = text[5]
            # DWARF4 compile_unit + subprogram，行号完全沿用 hgrtc 的原生表。
            abbrev = bytes([1, 0x11, 1, 0x03, 0x08, 0x1b, 0x08, 0x10, 0x17,
                            0x11, 0x01, 0x12, 0x07, 0x13, 0x05, 0x25, 0x08, 0x89, 0x7c, 0x08, 0, 0,
                            2, 0x2e, 0, 0x03, 0x08, 0x6e, 0x08, 0x11, 0x01, 0x12, 0x07, 0, 0, 0])
            # 原生行表使用 build/cmake/... 相对路径；编译目录绑定该臂冻结 include。
            source_mirror = dest / "build/cmake/actlize_effective/include"
            source_mirror.parent.mkdir(parents=True, exist_ok=True)
            if not source_mirror.exists():
                source_mirror.symlink_to(include, target_is_directory=True)
            assert source_mirror.resolve() == include.resolve()
            # HGRTC 对翻译单元使用 default 名称，保存原 TU 供源码导入。
            (dest / "default").write_bytes(actual_path.with_suffix(".cu").read_bytes())
            comp_dir = str(dest)
            prefix = b"\x01<stdin>\0" + comp_dir.encode() + b"\0" + struct.pack("<I", 0)
            cu_pc_offset = 11 + len(prefix)
            body = (prefix + struct.pack("<QQH", 0, high_pc, 0x21)
                    + b"hgrtc optimized RTC; minimal source-correlation CU\0[ppu:v0.1]\0"
                    + b"\x02" + (SYMBOL.encode() + b"\0") * 2 + struct.pack("<QQ", 0, high_pc) + b"\0")
            info = struct.pack("<IHIB", 7 + len(body), 4, 0, 8) + body
            line_rela = headers[names.index(".rela.debug_line")]
            _, relocation_info, addend = struct.unpack_from("<QQq", original, line_rela[4])
            rela_info = (struct.pack("<QQq", cu_pc_offset, relocation_info, addend)
                         + struct.pack("<QQq", len(info) - 17, relocation_info, addend))
            for name, contents in ((".debug_abbrev", abbrev), (".debug_info", info), (".rela.debug_info", rela_info)):
                offset = len(image)
                image.extend(contents)
                header = [len(strings), 1, 0, 0, offset, len(contents), 0, 0, 1, 0]
                if name == ".rela.debug_info":
                    header[1:3] = [4, 0x40]
                    header[6:10] = [line_rela[6], count + 1, 8, 24]
                headers.append(header)
                strings += name.encode() + b"\0"
            string_header[4], string_header[5] = len(image), len(strings)
            image.extend(strings)
            image.extend(b"\0" * (-len(image) % 8))
            new_shoff = len(image)
            for header in headers:
                image.extend(struct.pack("<IIQQQQIIQQ", *header))
            struct.pack_into("<Q", image, 40, new_shoff)
            struct.pack_into("<H", image, 60, len(headers))
            for header in headers[:count]:
                if header is not string_header and header[1] != 8:
                    offset, size = header[4:6]
                    assert image[offset:offset + size] == original[offset:offset + size]
            cubin.write_bytes(image)
            save(dest / "annotation.json", {"method": "native RTC line table + minimal DWARF4 CU",
                "original_loadable_sections_unchanged": True, "original_sha256": hashlib.sha256(original).hexdigest(),
                "annotated_sha256": hashlib.sha256(image).hexdigest(),
                "compilation_directory": comp_dir, "frozen_include": str(include)})
        else:
            execute(cmd, dest / "compile.log")
        tool = "/usr/local/PPU_SDK/bin/hgobjdump"
        isa = execute([tool, "--dump-isa", str(cubin)], dest / "isa.txt")
        resources = execute([tool, "--dump-resource-usage=all", str(cubin)], dest / "resources.txt")
        debug = execute(["readelf", "-S", str(cubin)], dest / "sections.txt")
        assert ".debug_info" in debug and ".debug_line" in debug
        actual = (a.actual_reference.resolve() if a.actual_reference else dest / "actual_rtc") / f"{SYMBOL}.cubin"
        if actual.exists():
            assert source.read_bytes() == actual.with_suffix(".cu").read_bytes(), "实际 RTC TU 不一致"
            actual_isa = (actual.parent / "isa.txt").read_text()
            pattern = r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2}\s+){8})"
            current_code = [(pc, data.split()) for pc, data in re.findall(pattern, isa, re.M)]
            actual_code = [(pc, data.split()) for pc, data in re.findall(pattern, actual_isa, re.M)]
            assert current_code and current_code == actual_code, "lineinfo 与实际 RTC 机器码不同"
            old_resources = (actual.parent / "resources.txt").read_text()
            for pattern in (r"vreg_number:(\d+)", r"STACK SIZE:(\d+)", r"shared_memory_size:(\d+)"):
                assert re.findall(pattern, resources) == re.findall(pattern, old_resources), "资源不同"
            save(dest / "lineinfo_identity.json", {"status": "PASS", "instruction_bytes_equal": True,
                "instructions": len(current_code), "actual_cubin_sha256": hashlib.sha256(actual.read_bytes()).hexdigest(),
                "lineinfo_cubin_sha256": hashlib.sha256(cubin.read_bytes()).hexdigest()})
        record = {"variant": variant, "symbol": SYMBOL, "cubin_bytes": cubin.stat().st_size,
                  "sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
                  "vreg": int(re.search(r"vreg_number:(\d+)", resources)[1]),
                  "stack": int(re.search(r"STACK SIZE:(\d+)", resources)[1]),
                  "atomic_instructions": len(re.findall(r"\bvmem\.atm\.add\.f32\b", isa))}
        save(dest / "build.json", record)
        print(json.dumps(record), flush=True)
    raise SystemExit(0)

if a.phase == "probe":
    # 先验证每个实验臂的 launch 身份和实际嵌入源码，再启动 full。
    assert not a.rtc, "探测使用已校验机器码身份的 annotated RTC cubin"
    probe = a.out / f"probe{('_' + a.tag) if a.tag else ''}.acurep"
    receipt = probe.with_suffix(".sources.json")
    assert not probe.exists() and not receipt.exists(), "不覆盖历史探测"
    device = device_idle()
    command = ["acu", "--profile=on", "--section", "LaunchStats", "--import-source", "yes",
               "-k", "regex:^nvte_grouped_gemm3_", "-s", str((2 + a.warmup) * len(a.variants)),
               "-c", str(len(a.variants)), "--kill=no", "--check-exit-code=yes", "-o", str(probe),
               sys.executable, str(Path(__file__).resolve()), "run", "--out", str(a.out),
               "--gpu", str(a.gpu), "--variants", *a.variants, "--layout", a.layout,
               "--dtype", a.dtype, "--source-mode", a.source_mode, "--rows", str(a.rows),
               "--cols", str(a.cols), "--mpe", str(a.mpe), "--groups", str(a.groups),
               "--warmup", str(a.warmup), "--samples", "1"]
    if a.nonuniform:
        command.append("--nonuniform")
    save(probe.with_suffix(".command.json"), command)
    env = dict(os.environ, LC_ALL="C", LANG="C", CUDA_LAUNCH_BLOCKING="1")
    text = execute(command, probe.with_suffix(".log"), env=env, strict_contention=False)
    assert not foreign_contenders(text)[0], "探测出现外部争用"
    assert "loaded prebuilt cubin" in text
    order = next(json.loads(line)["order"] for line in text.splitlines() if line.startswith('{"order"'))
    assert order == a.variants, order
    launches = re.findall(r"grid=\((\d+),(\d+),(\d+)\) block=\((\d+),(\d+),(\d+)\) smem=(\d+)B", text)
    assert launches and len(set(launches)) == 1, "实验臂 launch 身份不一致"
    assert tuple(map(int, launches[0][3:6])) == (512, 1, 1) and int(launches[0][6]) == 196608
    raw = execute(["acu", "-i", str(probe), "--page", "raw", "--csv"],
                  probe.with_suffix(".raw.csv"), env=env)
    probe_rows = list(csv.DictReader(io.StringIO(raw)))
    assert len(probe_rows) == len(order) and {r["Kernel Name"] for r in probe_rows} == {SYMBOL}
    assert {r["Device"] for r in probe_rows} == {str(a.gpu)}, "探测设备不匹配"
    assert len({r["Grid Size"] for r in probe_rows}) == len({r["Block Size"] for r in probe_rows}) == 1
    sys.path.insert(0, "/usr/local/PPU_SDK/asight/extras/python")
    import acu_report
    context = acu_report.load_report(str(probe))
    actions = []
    for i in range(context.num_ranges()):
        report_range = context.range_by_idx(i)
        for j in range(report_range.num_actions()):
            action = report_range.action_by_idx(j)
            sources = action.source_files()
            assert sources and all(sources.values()), "探测未实际嵌入源码"
            for expected in ("ppu_aiu_gemm_array_group.hpp", "ppu_mma_aiu_multistage_batch_array.hpp",
                             "ppu_epilogue_vectorized_evt.hpp"):
                assert any(Path(path).name == expected for path in sources), expected
            actions.append({"sources": {path: {"bytes": len(content.encode()),
                "sha256": hashlib.sha256(content.encode()).hexdigest()} for path, content in sources.items()}})
    assert len(actions) == len(order), (len(actions), order)
    save(receipt, {"status": "PASS", "device": device, "order": order, "launch": launches[0], "actions": actions})
    device_idle()
    print(json.dumps({"status": "PASS", "probe": str(probe), "order": order}), flush=True)
    raise SystemExit(0)

if a.phase == "run":
    print(json.dumps({"target_pid": os.getpid()}), flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    for name in ("NVTE_USE_CUTLASS_GROUPED_GEMM", "NVTE_USE_CUTLASS_GROUPED_GEMM_RTC",
                 "NVTE_FORCE_RTC_NO_FALLBACK", "NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK",
                 "NVTE_GG_ALLOW_FP32_ACCUM"):
        os.environ[name] = "1"
    os.environ["NVTE_GG_FORCE_TILE"] = "256,256,64,3"
    os.environ["NVTE_GG_WGRAD_C_FUSION"] = "0"
    os.environ["NVTE_GG_RTC_CACHE_DIR"] = str(a.out / ("rtc_cache" if a.rtc else "offline_cache"))
    os.environ["NVTE_CUTLASS3_INCLUDE_DIR"] = str(
        a.rtc_include.resolve() if a.rtc_include else ROOT / "build/cmake/actlize_effective/include")
    os.environ["NVTE_CUTLASS_GROUPED_GEMM_RTC_VERBOSE"] = "1"
    os.environ.pop("NVTE_GG_DISABLE_RTC_CACHE", None)
    os.environ.pop("NVTE_GG_WITH_C", None)
    import torch
    from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm

    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokens = [a.rows - (0, 1, 9, 24, 5, 64, 17, 40)[i % 8] if a.nonuniform else a.rows
              for i in range(a.groups)]
    A = [torch.randn((a.cols, a.mpe) if a.layout == "TN" else (a.mpe, a.cols),
                     device="cuda", dtype=torch.bfloat16) for _ in tokens]
    B = [torch.randn((a.mpe, m) if a.layout == "NT" else (m, a.mpe),
                     device="cuda", dtype=torch.bfloat16) for m in tokens]
    output_dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    C = [torch.randn(m, a.cols, device="cuda", dtype=output_dtype) for m in tokens]
    D = [c.clone() for c in C]
    ref = [(b.float().T if a.layout == "NT" else b.float()) @
           (x.float().T if a.layout == "TN" else x.float()) + (c if WITH_C else 0)
           for x, b, c in zip(A, B, C)]

    def launch(variant, accumulate=WITH_C):
        os.environ["NVTE_GG_EXTRA_DEFINES"] = VARIANTS[variant]
        if a.rtc:
            os.environ.pop("NVTE_GG_RTC_LOAD_CUBIN_DIR", None)
            dump = a.out / variant / "actual_rtc"
            # host 的 ofstream 不创建目录；必须在编译前建立完整路径。
            dump.mkdir(parents=True, exist_ok=True)
            os.environ["NVTE_CUTLASS_RTC_DUMP_CUBIN_DIR"] = str(dump)
        else:
            os.environ["NVTE_GG_RTC_LOAD_CUBIN_DIR"] = str(a.out / variant)
        for d, c in zip(D, C):
            d.copy_(c)
        general_grouped_gemm(A, B, D, [None] * a.groups, torch.bfloat16,
                             layout=a.layout, m_splits=[a.mpe] * a.groups if a.layout == "NT" else tokens,
                             grad=a.layout != "TN", accumulate=accumulate, single_output=False)

    baseline = None
    for variant in a.variants:
        launch(variant)
        torch.cuda.synchronize()
        errors = [float(torch.linalg.vector_norm(d.float() - r) / torch.linalg.vector_norm(r)) for d, r in zip(D, ref)]
        limit = max(1e-6, 1e-6 * a.mpe / 256) if a.dtype == "fp32" else 4e-3
        assert max(errors) < limit, errors
        if a.save or a.compare:
            assert len(a.variants) == 1, "输出身份校验仅支持单变体"
            if a.compare:
                saved = torch.load(a.compare, weights_only=True)
                assert len(saved) == len(D) and all(torch.equal(d.cpu(), old) for d, old in zip(D, saved)), \
                    "lineinfo 输出与实际 RTC 不一致"
                print("BITWISE_ACTUAL_RTC_OK", flush=True)
            if a.save:
                a.save.parent.mkdir(parents=True, exist_ok=True)
                torch.save([d.cpu() for d in D], a.save)
        # baseline 只由逐位同语义的变体建立，使闸门与 launch 顺序无关。
        exact = variant not in REORDERS_ACCUMULATION
        if exact:
            if baseline is None:
                baseline = [d.clone() for d in D]
            else:
                bits_dtype = torch.int32 if a.dtype == "fp32" else torch.int16
                assert all(torch.equal(d.view(bits_dtype), r.view(bits_dtype))
                           for d, r in zip(D, baseline)), variant
        print(json.dumps({"check": variant, "rel_error": max(errors), "bitwise": exact}), flush=True)
        if a.rtc:
            from transformer_engine.common import _get_shared_object_file
            dest = a.out / variant / "actual_rtc"
            cubin = dest / f"{SYMBOL}.cubin"
            assert cubin.is_file(), (variant, SYMBOL, list(dest.glob("*.cubin")))
            tool = "/usr/local/PPU_SDK/bin/hgobjdump"
            resources = execute([tool, "--dump-resource-usage=all", str(cubin)], dest / "resources.txt")
            isa = execute([tool, "--dump-isa", str(cubin)], dest / "isa.txt")
            from collections import Counter
            opcodes = Counter(re.findall(r"^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+){8}(\S+)", isa, re.M))
            assert opcodes, "无法解析 PASM opcode"
            save(dest / "opcode_counts.json", dict(sorted(opcodes.items())))
            identity = {"symbol": SYMBOL, "variant": variant, "layout": a.layout,
                        "source_mode": a.source_mode, "tokens": tokens,
                        "defines": VARIANTS[variant], "sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
                        "core_library": str(_get_shared_object_file("core").resolve()),
                        "torch_library": str(_get_shared_object_file("torch").resolve()),
                        "include": os.environ["NVTE_CUTLASS3_INCLUDE_DIR"],
                        "vreg": int(re.search(r"vreg_number:(\d+)", resources)[1]),
                        "stack": int(re.search(r"STACK SIZE:(\d+)", resources)[1])}
            save(dest / "identity.json", identity)
            print(json.dumps({"identity": identity}), flush=True)
    # noC 额外预热各变体；NaN 旧 D 的闸门由参数化单测执行。
    for variant in a.variants:
        launch(variant, False)
        torch.cuda.synchronize()
        assert all(torch.isfinite(d).all() for d in D)
    for _ in range(a.warmup):
        for variant in a.variants:
            launch(variant)
    torch.cuda.synchronize()
    order = []
    for sample in range(a.samples):
        variants = a.variants if sample % 2 == 0 else list(reversed(a.variants))
        for variant in variants:
            order.append(variant)
            launch(variant)
        torch.cuda.synchronize()
    print(json.dumps({"order": order, "skip_matching": (2 + a.warmup) * len(a.variants)}), flush=True)
    raise SystemExit(0)

stem = a.out / f"M{a.rows}_N{a.cols}_K{a.mpe}_{a.layout}_{a.source_mode}_{'nonuniform' if a.nonuniform else 'uniform'}_{'_'.join(a.variants)}_{'rtc' if a.rtc else 'offline'}_{'full' if a.full else 'metrics'}{('_' + a.tag) if a.tag else ''}"
if a.layout == "NT" and WITH_C and not a.nonuniform and not any(v in VARIANTS and v not in
        ("fragment", "prologue", "atomic", "global", "interleave") for v in a.variants):
    # 保持旧报告的 export 路径兼容。
    stem = a.out / f"M{a.rows}_N{a.cols}_K{a.mpe}_{'_'.join(a.variants)}_{'rtc' if a.rtc else 'offline'}_{'full' if a.full else 'metrics'}{('_' + a.tag) if a.tag else ''}"
if a.phase == "capture":
    # 一份采集目录绑定源码与 include；后续源码改变必须换新目录。
    snapshot = a.out / "source_snapshot"
    paths = ["transformer_engine/common/gemm/rtc/cutlass_grouped_gemm.cu",
             "transformer_engine/common/gemm/actlize_overlay/include/cutlass/gemm/kernel/ppu_aiu_gemm_array_group.hpp",
             "transformer_engine/common/gemm/actlize_overlay/include/cutlass/gemm/collective/ppu_mma_aiu_multistage_batch_array.hpp",
             "transformer_engine/common/gemm/actlize_overlay/include/cutlass/epilogue/collective/ppu_epilogue_vectorized_evt.hpp",
             "profile/gg_fp32_accum/atomic_study.py", "profile/gg_fp32_accum/repro.py",
             "tests/pytorch/test_grouped_gemm_rtc.py"]
    for path in paths:
        if "/actlize_overlay/include/" in path:
            relative = path.split("/actlize_overlay/include/", 1)[1]
            assert (ROOT / path).read_bytes() == (ROOT / "build/cmake/actlize_effective/include" / relative).read_bytes(), \
                f"canonical/effective 不一致，必须先重构建：{relative}"
        dest = snapshot / path
        if dest.exists():
            assert dest.read_bytes() == (ROOT / path).read_bytes(), f"源码改变，不能复用历史目录：{dest}"
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / path, dest)
    include_snapshot = snapshot / "effective_include"
    if not include_snapshot.exists():
        shutil.copytree(ROOT / "build/cmake/actlize_effective/include", include_snapshot, symlinks=False)
    for repeat in range(a.repeats):
        gpu = device_idle()
        report_stem = Path(str(stem) + f"_r{repeat}")
        assert not report_stem.with_suffix(".acurep").exists(), report_stem
        variants = a.variants if repeat % 2 == 0 else list(reversed(a.variants))
        cmd = ["acu", "--profile=on", "--metrics", METRIC + ",ppu__time_duration.sum",
               "-k", "regex:^nvte_grouped_gemm3_", "-s", str((2 + a.warmup) * len(variants)),
               "-c", str(a.samples * len(variants)), "--kill=no", "--check-exit-code=yes", "-o", str(report_stem)]
        if a.full:
            cmd += ["--set", "full", "--import-source", "yes"]
        cmd += [sys.executable, str(snapshot / "profile/gg_fp32_accum/atomic_study.py"), "run", "--out", str(a.out),
                "--variants", *variants, "--gpu", str(a.gpu), "--rows", str(a.rows), "--cols", str(a.cols),
                "--mpe", str(a.mpe), "--groups", str(a.groups), "--warmup", str(a.warmup), "--samples", str(a.samples),
                "--layout", a.layout, "--source-mode", a.source_mode, "--dtype", a.dtype]
        if a.nonuniform:
            cmd += ["--nonuniform"]
        if a.rtc:
            cmd += ["--rtc", "--rtc-include", str(include_snapshot)]
        save(report_stem.with_suffix(".command.json"), {
            "gpu": gpu, "command": cmd, "timestamp": datetime.now(timezone.utc).isoformat(),
            "host_name": socket.gethostname()})
        capture_env = dict(os.environ, NVTE_GG_STUDY_ROOT=str(ROOT))
        text = execute(cmd, report_stem.with_suffix(".log"), env=capture_env, strict_contention=False)
        foreign, target = foreign_contenders(text)
        assert not foreign, f"争用污染（外部进程 {foreign}，被测 {target}）：{report_stem}.log"
        order = next(json.loads(line)["order"] for line in text.splitlines() if line.startswith('{"order"'))
        assert len(order) == a.samples * len(a.variants)
        launches = re.findall(r"grid=\((\d+),(\d+),(\d+)\) block=\((\d+),(\d+),(\d+)\) smem=(\d+)B", text)
        assert launches and len(set(launches)) == 1, "跨实验臂的 tile/grid/资源不一致"
        grid = tuple(map(int, launches[0][:3]))
        assert tuple(map(int, launches[0][3:6])) == (512, 1, 1) and int(launches[0][6]) == 196608
        runtime = {"grid": grid, "block": [512, 1, 1], "dynamic_smem_bytes": 196608,
                   "source_mode": a.source_mode, "layout": a.layout, "groups": a.groups,
                   "output_tiles": sum(((m + 255) // 256) * ((a.cols + 255) // 256)
                                       for m in (a.rows - (0, 1, 9, 24, 5, 64, 17, 40)[i % 8]
                                                 if a.nonuniform else a.rows for i in range(a.groups)))}
        runtime["multi_work_per_cta"] = runtime["output_tiles"] > grid[0] * grid[1] * grid[2]
        if repeat:
            previous = json.loads(Path(str(stem) + "_r0.runtime.json").read_text())
            assert json.loads(json.dumps(runtime)) == previous, "跨重复采集的工作集/grid 不一致"
        save(report_stem.with_suffix(".runtime.json"), runtime)
        assert report_stem.with_suffix(".acurep").is_file(), "未生成 ACU 报告"
        save(report_stem.with_suffix(".order.json"), order)
        print(f"完成：{report_stem}.acurep", flush=True)
    device_idle()

if a.phase in ("capture", "export"):
    values = {v: {"tfu": [], "duration_ns": []} for v in a.variants}
    for repeat in range(a.repeats):
        report_stem = Path(str(stem) + f"_r{repeat}")
        report = report_stem.with_suffix(".acurep")
        order = json.loads(report_stem.with_suffix(".order.json").read_text())
        raw = execute(["acu", "-i", str(report), "--page", "raw", "--csv"], report_stem.with_suffix(".raw.csv"))
        rows = list(csv.DictReader(io.StringIO(raw)))
        assert len(rows) == len(order) and {r["Kernel Name"] for r in rows} == {SYMBOL}
        assert {r["Device"] for r in rows} == {str(a.gpu)}
        assert len({r["Grid Size"] for r in rows}) == len({r["Block Size"] for r in rows}) == 1
        for variant, row in zip(order, rows):
            values[variant]["tfu"].append(float(row[METRIC].replace(",", "")))
            values[variant]["duration_ns"].append(float(row["ppu__time_duration.sum"].replace(",", "")))
        if a.full:
            execute(["acu", "-i", str(report), "--page", "source"], report_stem.with_suffix(".source.txt"))
            execute(["acu", "-i", str(report), "--page", "details"], report_stem.with_suffix(".details.txt"))
            sass = report_stem.with_suffix(".sass.csv")
            execute(["acu", "-i", str(report), "--page", "source", "--csv-file", str(sass)],
                    report_stem.with_suffix(".sass_export.log"))
            source_rows = list(csv.DictReader(io.StringIO(sass.read_text())))
            assert source_rows and any(float(r.get("Instructions Executed", "0").replace(",", "") or 0) > 0
                                       for r in source_rows), "缺少动态 SASS 执行证据"
            full_csv = report_stem.with_suffix(".full.csv")
            env = dict(os.environ, LC_ALL="C", LANG="C")
            execute(["acu", "-i", str(report), "--stats-rule", "ExportMetricsByKernel",
                     "--stats-rule-output", str(full_csv)], report_stem.with_suffix(".full_export.log"), env=env)
            full_rows = list(csv.DictReader(io.StringIO(full_csv.read_text())))
            assert len(full_rows) == len(order), (len(full_rows), len(order))
            metadata = {"Range ID", "Kernel Name", "Kernel Mangled Name", "Kernel Demangled Name",
                        "Block Size", "Grid Size", "Device"}
            def metric_value(value):
                try:
                    return float(value.replace(",", ""))
                except (ValueError, AttributeError):
                    return value
            acu_banner = subprocess.check_output(["acu", "--version"], text=True).strip()
            compiler_banner = subprocess.check_output(["nvcc", "--version"], text=True).strip()
            acu_version = re.search(r"version\s+(\S+)", acu_banner)[1]
            compiler_version = re.search(r"release\s+([^,]+)", compiler_banner)[1]
            data_version = re.search(r"Data version\s+(\S+)", acu_banner)
            sdk = f"acu {acu_version}, Data {data_version[1] if data_version else 'unknown'}, HGGC {compiler_version}"
            context = json.loads(report_stem.with_suffix(".command.json").read_text())
            log = report_stem.with_suffix(".log").read_text()
            target_pid = next(json.loads(line)["target_pid"] for line in log.splitlines()
                              if line.startswith('{"target_pid"'))
            all_metrics = [{"device": "PPU-890", "kernel_name": row["Kernel Name"],
                            "kernel_mangled_name": row.get("Kernel Mangled Name"),
                            "timestamp": context.get("timestamp"), "host_name": context.get("host_name"),
                            "process_id": target_pid, "compiler_version_banner": compiler_banner,
                            "profiler": "ACU", "sdk_version": sdk, "variant": variant,
                            "source_mode": a.source_mode, "device_name": row.get("Device"),
                            "grid": row.get("Grid Size"), "block": row.get("Block Size"),
                            "raw_metrics": {key: metric_value(value) for key, value in row.items()
                                            if key not in metadata}}
                           for variant, row in zip(order, full_rows)]
            save(report_stem.with_suffix(".metrics.json"), all_metrics)
            # 导出成功不等于源码实际导入；逐 action 验收 SDK 的全量指标和源码内容。
            sys.path.insert(0, "/usr/local/PPU_SDK/asight/extras/python")
            import acu_report
            report_context = acu_report.load_report(str(report))
            embedded_manifest = []
            for i in range(report_context.num_ranges()):
                report_range = report_context.range_by_idx(i)
                for j in range(report_range.num_actions()):
                    action = report_range.action_by_idx(j)
                    embedded = action.source_files()
                    assert embedded and all(embedded.values()), "full 报告未实际嵌入源码"
                    for expected in ("ppu_aiu_gemm_array_group.hpp", "ppu_mma_aiu_multistage_batch_array.hpp",
                                     "ppu_epilogue_vectorized_evt.hpp"):
                        assert any(Path(path).name == expected for path in embedded), expected
                    index = len(embedded_manifest)
                    assert set(action.metrics()) == set(all_metrics[index]["raw_metrics"]), "全量指标导出不完整"
                    embedded_manifest.append({"variant": order[index], "sources": {
                        path: {"bytes": len(content.encode()), "sha256": hashlib.sha256(content.encode()).hexdigest()}
                        for path, content in embedded.items()}})
            assert len(embedded_manifest) == len(order), "full 报告 action 数不匹配"
            save(report_stem.with_suffix(".embedded_sources.json"), embedded_manifest)
    summary = {v: {k: {"samples": samples, "median": statistics.median(samples)} for k, samples in val.items()}
               for v, val in values.items()}
    save(stem.with_suffix(".summary.json"), summary)
    print(json.dumps(summary), flush=True)

if a.phase == "audit-stream":
    # 同名多臂按 order 逐 action 对齐，各自使用对应 cubin 的相对 PC。
    from collections import Counter, defaultdict
    sys.path.insert(0, "/usr/local/PPU_SDK/asight/extras/python")
    import acu_report
    receipt = a.out / "stream_audit.json"
    assert not receipt.exists(), "不覆盖动态审计证据"
    assert a.layout == "NN" and a.dtype == "bf16" and not WITH_C and a.full
    result = {"status": "RUNNING", "metric": METRIC, "actions": [], "summary": {}}
    headers = ("cutlass/gemm/kernel/ppu_aiu_gemm_array_group.hpp",
               "cutlass/gemm/collective/ppu_mma_aiu_multistage_batch_array.hpp",
               "cutlass/epilogue/collective/ppu_epilogue_vectorized_evt.hpp")
    sites = (0x2798, 0x27d8, 0x2998, 0x2b50, 0x2d70, 0x2e08,
             0x3dc8, 0x3e10, 0x3f90, 0x4010, 0x40e8, 0x41d0, 0x42d0, 0x4490,
             0x44b0, 0x55c0, 0x5620, 0x57c0, 0x5840, 0x5918, 0x5a00, 0x5b10,
             0x5c60, 0x5c98, 0x5e08, 0x6048, 0x6118, 0x6300, 0x6308)
    for repeat in range(a.repeats):
        report_stem = Path(str(stem) + f"_r{repeat}")
        order = json.loads(report_stem.with_suffix(".order.json").read_text())
        metrics = json.loads(report_stem.with_suffix(".metrics.json").read_text())
        context = acu_report.load_report(str(report_stem.with_suffix(".acurep")))
        index = 0
        for i in range(context.num_ranges()):
            report_range = context.range_by_idx(i)
            for j in range(report_range.num_actions()):
                action = report_range.action_by_idx(j)
                variant = order[index]
                raw = metrics[index]["raw_metrics"]
                assert metrics[index]["variant"] == variant
                sources = action.source_files()
                for relative in headers:
                    frozen = a.out / variant / "include" / relative
                    content = frozen.read_bytes()
                    for tree in ("transformer_engine/common/gemm/actlize_overlay/include",
                                 "build/cmake/actlize_effective/include", "include/cutlass3"):
                        assert content == (ROOT / tree / relative).read_bytes(), (variant, relative, tree)
                    embedded = [v for k, v in sources.items() if Path(k).name == Path(relative).name]
                    assert len(embedded) == 1 and embedded[0].encode() == content
                native = {int(pc, 16): code.split()[0] for pc, code in re.findall(
                    r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+){8}(\S+[^\n]*)",
                    (a.out / variant / "isa.txt").read_text(), re.M)}
                sass = action.sass()
                base = min(sass)
                rows = []
                counts, stalls = Counter(), Counter()
                lines = defaultdict(Counter)
                for pc, instruction in sass.items():
                    offset = pc - base
                    opcode = instruction.sass_code.split()[0]
                    assert offset in native and opcode == native[offset], (variant, hex(offset), opcode)
                    executed = instruction.warp_executed_count
                    stall = instruction.stall
                    counts[opcode] += executed
                    stalls[opcode] += stall.total
                    line = f"{Path(instruction.source_file_name).name}:{instruction.source_file_line_number}"
                    lines[line]["executed"] += executed
                    for reason in stall.__slots__:
                        lines[line][reason] += getattr(stall, reason)
                    rows.append({"pc": hex(offset), "instruction": instruction.sass_code,
                                 "executed": executed, "stall_total": stall.total,
                                 "stall_sync": stall.sync, "stall_memory_dependency": stall.memory_dependency,
                                 "stall_pipe_busy": stall.pipe_busy, "source": line})
                pc_rows = {int(r["pc"], 16): r for r in rows}
                mma = sum(n for op, n in counts.items() if op.startswith("v.mma"))
                record = {"variant": variant, "repeat": repeat, "action": index,
                          "report": str(report_stem.with_suffix(".acurep")),
                          "tensor_active_pct": raw[METRIC], "duration_ns": raw["ppu__time_duration.sum"],
                          "metric_count": len(raw), "embedded_sources": len(sources),
                          "dynamic_mma_warp_instructions": mma,
                          "executed_by_opcode": dict(counts), "stall_samples_by_opcode": dict(stalls),
                          "hot_source_lines": [{"source": k, **v} for k, v in sorted(
                              lines.items(), key=lambda pair: pair[1]["total"], reverse=True)[:15]],
                          "hot_instructions": sorted(rows, key=lambda r: r["stall_total"], reverse=True)[:15],
                          "stall_per_issue": {k: v for k, v in raw.items() if
                              k.startswith("pu__we_average_warps_issue_stalled_") and k.endswith("per_issue_active.ratio")}}
                if variant == "stream_no_c":
                    record["stream_sites"] = [pc_rows[pc] for pc in sites if pc in pc_rows]
                    record["warm_prologue_copies"] = [r for r in rows if
                        0x27e8 <= int(r["pc"], 16) < 0x2d70 and "vmem.aiu" in r["instruction"]]
                result["actions"].append(record)
                index += 1
        assert index == len(order) == len(metrics)
    assert len({r["dynamic_mma_warp_instructions"] for r in result["actions"]}) == 1, "MMA 工作量不一致"
    for variant in a.variants:
        samples = [r["tensor_active_pct"] for r in result["actions"] if r["variant"] == variant]
        result["summary"][variant] = {"samples": samples, "median": statistics.median(samples),
                                      "min": min(samples), "max": max(samples)}
    baseline = result["summary"]["baseline_aiu"]["median"]
    for stats in result["summary"].values():
        stats["delta_percentage_points"] = stats["median"] - baseline
        stats["relative_pct"] = (stats["median"] / baseline - 1) * 100
    result["status"] = "PASS"
    save(receipt, result)
    print(json.dumps({"status": "PASS", "summary": result["summary"], "audit": str(receipt)}, ensure_ascii=False), flush=True)

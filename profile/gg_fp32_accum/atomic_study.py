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
def nexpand_case_manifest(splits=None):
    """Keep the historic uniform manifest unchanged unless explicit ragged splits are supplied."""
    if splits is not None:
        assert splits and all(type(s) is int and s >= 0 and s % 256 == 0 for s in splits)
    shapes = [(1536, 4096), (4096, 1536), (512, 2048), (2048, 512),
              (1024, 3072), (3072, 1024), (1024, 4096), (4096, 1024)]
    cases = []
    for index, (k, n) in enumerate(shapes):
        for direction, layout, mnk in (("fwd", "TN", [2048, n, k]),
                                        ("dgrad", "NN", [2048, k, n]),
                                        ("wgrad", "NT", [n, k, 2048])):
            cases.append({"id": f"{index + 1:02d}_{direction}_K{k}_N{n}",
                          "index": len(cases), "direction": direction, "layout": layout,
                          "model_KN": [k, n], "MNK": mnk, "groups": 16,
                          "splits": [2048] * 16, "dtype": "fp32" if direction == "wgrad" else "bf16",
                          "beta": int(direction == "wgrad"), "control": False, "seed": 1234})
    for direction, layout, model in (("fwd", "TN", [2048, 4096]),
                                      ("dgrad", "NN", [4096, 2048])):
        cases.append({"id": f"control_K2048_{direction}", "index": len(cases),
                      "direction": direction, "layout": layout, "model_KN": model,
                      "MNK": [2048, 4096, 2048], "groups": 16, "splits": [2048] * 16,
                      "dtype": "bf16", "beta": 0, "control": True, "seed": 1234})
    if splits is not None:
        for case in cases:
            case.update(splits=list(splits), groups=len(splits),
                        split_axis="K" if case["layout"] == "NT" else "M")
    return cases


def nexpand_validate_admission(case, config):
    """Recompute strict256 admission and exact work from the actual per-group workload."""
    m, n, k = case["MNK"]
    splits = case["splits"]
    assert len(splits) == case["groups"] and splits
    assert all(type(s) is int and s >= 0 for s in splits)
    shapes = [(m, n, s) if case["layout"] == "NT" else (s, n, k) for s in splits]
    assert all(not (gm and gn) or all(d % 256 == 0 for d in (gm, gn, gk))
               for gm, gn, gk in shapes), "NVTE_GG_CONTIGUOUS requires each output-bearing MNK256"
    assert config["requested_ne"] in (1, 4), "NE8 runtime admission is not supported"
    bn, ne, kmax = config["tile_n"], config["requested_ne"], config["kmax"]
    assert bn in (128, 256) and kmax in (512, 1024, 2048)
    eligible = (sum(gm > 0 and 0 < gk <= kmax and gn % (bn * ne) == 0
                    for gm, gn, gk in shapes) if not case["beta"] and ne > 1 else 0)
    effective = ne if eligible else 1
    work = sum(gm // 256 * (gn // bn // effective) for gm, gn, _ in shapes)
    expected = {"eligible_groups": eligible, "effective_ne": effective,
                "effective_kmax": kmax if effective > 1 else 512,
                "mode": 2 if effective > 1 else 1, "expected_work": work,
                "bypass_reason": ("withC_NE1" if case["beta"] else "requested_NE1" if ne == 1 else
                                  "no_work" if work == 0 else "eligible" if eligible else
                                  "N_not_divisible" if n % (bn * ne) else "no_eligible_K")}
    assert all(config[key] == value for key, value in expected.items()), (config, expected)
    if "splits" in config:
        assert config["splits"] == splits
        assert [list(s) for s in config["per_group_MNK"]] == [list(s) for s in shapes]
    return expected


def nexpand_statistics(rounds):
    assert len(rounds) == 3 and all(len(values) == 3 for values in rounds)
    values = [value for row in rounds for value in row]
    assert all(isinstance(value, (int, float)) and 0 <= value <= 100 for value in values)
    medians = [statistics.median(row) for row in rounds]
    return {"samples": rounds, "round_medians": medians,
            "median": statistics.median(medians), "min": min(values), "max": max(values),
            "round_min": min(medians), "round_max": max(medians)}


def rtc_machine_identity(path):
    """忽略debug布局，但逐字节绑定所有机器码、资源和非debug section。"""
    import struct
    image = path.read_bytes()
    assert image[:6] == b"\x7fELF\x02\x01"
    offset = struct.unpack_from("<Q", image, 40)[0]
    size, count, string_index = struct.unpack_from("<HHH", image, 58)
    assert size == 64 and count < 0xff00
    headers = [struct.unpack_from("<IIQQQQIIQQ", image, offset + i * size) for i in range(count)]
    string_header = headers[string_index]
    strings = image[string_header[4]:string_header[4] + string_header[5]]
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


def nexpand_validate_codegen(gate):
    """完整配置清单与默认不变性必须同时满足，不能仅检查PASS字符串。"""
    assert gate["status"] == "PASS" and gate["gpu_execution"] is False
    defaults = {(mode, bn, c, layout, dtype, 512)
                for mode, bn, c in ((0, 256, 0), (0, 256, 1), (1, 128, 0), (1, 128, 1),
                                   (1, 256, 0), (2, 128, 0), (2, 256, 0))
                for layout in ("TN", "NN", "NT") for dtype in ("bf16", "fp32")}
    added = {(2, bn, 0, layout, dtype, kmax) for kmax in (1024, 2048)
             for bn in (128, 256) for layout in ("TN", "NN", "NT") for dtype in ("bf16", "fp32")}
    cases = gate["cases"]
    keys = [(c["mode"], c["bn"], int(c["with_C"]), c["layout"], c["dtype"], c["kmax"]) for c in cases]
    assert len(keys) == len(set(keys)) == 66 and set(keys) == defaults | added
    for key, case in zip(keys, cases):
        mode, bn, with_c, layout, dtype, kmax = key
        compiled = case["compile_result"]
        assert compiled["compile_only"] == "PASS" and compiled["gpu_execution"] is False
        assert compiled["stack_size"] == compiled["atomic_instruction_count"] == 0
        ta, tb = {"TN": (0, 1), "NN": (0, 0), "NT": (1, 0)}[layout]
        macros = {"NVTE_GG_CONTIGUOUS_MODE": mode, "NVTE_GG_WITH_C": with_c,
                  "NVTE_GG_TRANS_A": ta, "NVTE_GG_TRANS_B": tb,
                  "NVTE_GG_OUT_FP32": int(dtype == "fp32"), "NVTE_GG_DTYPE": 0,
                  "NVTE_GG_ARCH": 89, "NVTE_GG_TILE_M": 256, "NVTE_GG_TILE_N": bn,
                  "NVTE_GG_TILE_K": 64, "NVTE_GG_STAGES": 3}
        for name, value in macros.items():
            assert [o for o in compiled["options"] if o.startswith(f"-D{name}=")] == [f"-D{name}={value}"]
        assert [o for o in compiled["options"] if o.startswith("-DNVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX=")] == \
            ([f"-DNVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX={kmax}"] if kmax != 512 else [])
        comparison = case["default_comparison"]
        if key in defaults:
            assert comparison["machine_identity_identical"] is True and comparison["resources_identical"] is True
        else:
            assert comparison is None


def nexpand_codegen_artifacts(gate):
    """重验实际cubin/资源/编译回执并绑定哈希，不信任可篡改的比较布尔值。"""
    nexpand_validate_codegen(gate)
    baseline = json.loads((Path(gate["baseline"]) / "result.json").read_text())
    assert baseline["status"] == "PASS" and len(baseline["cases"]) == 42
    old = {(c["mode"], c["bn"], int(c["with_C"]), c["layout"], c["dtype"]): c for c in baseline["cases"]}
    assert len(old) == 42
    hashes = {}
    for case in gate["cases"]:
        dest, symbol = Path(case["path"]), f"nvte_grouped_gemm3_compile_{case['dtype']}"
        assert json.loads((dest / f"{symbol}.json").read_text()) == case["compile_result"]
        assert re.findall(r"STACK SIZE:(\d+)", (dest / f"{symbol}.resources").read_text()) == ["0"]
        assert not re.search(r"\bvmem\.atm\.add\.f32\b", (dest / f"{symbol}.isa").read_text())
        if case["kmax"] == 512:
            key = (case["mode"], case["bn"], int(case["with_C"]), case["layout"], case["dtype"])
            previous = Path(old[key]["path"])
            assert rtc_machine_identity(dest / f"{symbol}.cubin") == rtc_machine_identity(previous / f"{symbol}.cubin")
            assert (dest / f"{symbol}.resources").read_bytes() == (previous / f"{symbol}.resources").read_bytes()
        for suffix in ("json", "cubin", "isa", "resources"):
            path = dest / f"{symbol}.{suffix}"
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def nexpand_validate_capture(record, digest, stage=None, kmax=None):
    """完整26例、每臂三轮三样本、无争用才允许作为后档门禁。"""
    assert record.get("scope") != "runtime-comparison", "Runtime comparison is never a formal nexpand capture"
    assert record["status"] == "PASS" and record["performance_valid"] is True and not record["contention"]
    assert record["case_digest"] == digest and record["manifest"] == nexpand_case_manifest(record.get("packed_study_splits"))
    assert record["core_24_complete"] and record["samples_per_round"] == record["rounds"] == 3
    if stage is not None:
        assert record["stage"] == stage and record.get("kmax", 512) == kmax
    assert set(record["cases"]) == {c["id"] for c in record["manifest"]}
    for case in record["manifest"]:
        entry = record["cases"][case["id"]]
        assert entry["status"] == "PASS" and entry["case"] == case and len(entry["reports"]) == 6
        assert entry["order"] == [["production", "contiguous"] if (case["index"] + r) % 2 == 0
                                  else ["contiguous", "production"] for r in range(3)]
        for arm in ("production", "contiguous"):
            assert sum(report["arm"] == arm for report in entry["reports"]) == 3
            assert nexpand_statistics(entry["summary"][arm]["samples"]) == entry["summary"][arm]


def nexpand_validate_current_anchor(stage, uuid, baseline, anchor, digest, binding):
    """首次换卡补采无需已有补采；新阈值换卡仍必须有同卡完整锚点。"""
    if baseline is None or stage == "baseline" or uuid == baseline["device"]["uuid"]:
        return
    assert anchor is not None, "换卡须先补采同卡NE4/Kmax512锚点，禁止跨卡绝对相减"
    nexpand_validate_capture(anchor, digest, "baseline", 512)
    assert anchor["device"]["uuid"] == uuid and anchor["baseline_audit"] == binding
    assert all(c["production_anchor_identity"] == c["current_anchor_identity"] == "PASS"
               for c in anchor["cases"].values())


def nexpand_duration_statistics(rounds):
    """Duration uses the same paired three-round/three-action estimator, without a TFU cap."""
    assert len(rounds) == 3 and all(len(row) == 3 for row in rounds)
    values = [value for row in rounds for value in row]
    assert all(isinstance(value, (int, float)) and 0 < value < float("inf") for value in values)
    medians = [statistics.median(row) for row in rounds]
    return {"samples": rounds, "round_medians": medians, "median": statistics.median(medians)}


def fulltile_source_identity(action, arm, include, rtc_source, annotated):
    embedded = action.source_files()
    assert embedded and all(embedded.values())
    include = include.resolve()
    mirrors = {rtc_source.resolve(), *(annotated / name for name in
               ("default", "<stdin>", rtc_source.name))}
    mirrors = {p.resolve() for p in mirrors}
    for path, content in embedded.items():
        source = Path(path).resolve()
        assert source.is_relative_to(include) or source in mirrors, ("Unbound embedded source", path)
        expected = source if source.is_relative_to(include) else rtc_source
        assert content.splitlines() == expected.read_text().splitlines(), ("Embedded source differs", path)
    required = ("nvte_contiguous_grouped.hpp" if arm == "contiguous" else "ppu_aiu_gemm_array_group.hpp",
                "ppu_mma_aiu_multistage_batch_array.hpp", "ppu_epilogue_vectorized_evt.hpp")
    for name in required:
        assert len([p for p in embedded if Path(p).name == name]) == 1, ("Missing or ambiguous embedded source", name)
    return {p: hashlib.sha256(v.encode()).hexdigest() for p, v in embedded.items()}


def fulltile_dynamic_work(rows, actions, symbol, isa_path):
    def normalize(code):
        return " ".join(re.sub(r"\s*<[^>]+>", "", code).split())

    isa = {int(match[1], 16): normalize(match[2]) for line in isa_path.read_text().splitlines()
           if (match := re.match(r"^\s*([0-9a-fA-F]+):\s+(?:[0-9a-fA-F]{2}\s+){8}(.+)$", line))}
    assert isa, "Missing bound cubin ISA"
    sequences = [str(action.capture_sequence_id()) for action in actions]
    assert len(actions) == len(set(sequences)) == 3
    assert set(sequences) == {row["Range Id"] for row in rows}, "Dynamic SASS/action identities differ"
    dynamic = {}
    for sequence, action in zip(sequences, actions):
        sass, functions = action.sass(), action.functions()
        assert sass and set(functions.values()) == {symbol}
        selected = [row for row in rows if row["Range Id"] == sequence]
        addresses = [int(row["Address"], 16) for row in selected]
        assert len(addresses) == len(set(addresses)) == len(sass) and set(addresses) == set(sass)
        base = min(sass) - min(isa)
        counts = {"instructions": 0, "sass_mma": 0, "store_bytes": 0}
        for row, pc in zip(selected, addresses):
            instruction = sass[pc]
            executed = int(row["Instructions Executed"].replace(",", ""))
            threads = int(row["Thread Instructions Executed"].replace(",", ""))
            code = normalize(row["Source"])
            assert row["Function Name"] == functions[instruction.function_id] == symbol
            assert instruction.address == pc and code == normalize(instruction.sass_code) == isa[pc - base]
            assert executed == instruction.warp_executed_count and threads == instruction.thread_executed_count
            counts["instructions"] += executed
            if re.search(r"\bv\.mma\.f32\.bf16\.", code):
                counts["sass_mma"] += executed
            if store := re.search(r"\bvmem\.st\.b(\d+)(?:x(\d+))?\b", code):
                counts["store_bytes"] += threads * int(store[1]) // 8 * int(store[2] or 1)
        assert counts["instructions"] > 0
        dynamic[sequence] = counts
    return dynamic


def fulltile_verify_gate(paths, library_hashes, python_hashes, include, include_identity):
    """Revalidate finalized CLI GPU records and bind their actual files; not a regression gate."""
    bindings, coverage, rejected, verified = {}, set(), set(), []
    checked_includes = set()

    def bind(path, expected=None):
        path = Path(path).resolve()
        assert path.is_file(), ("Missing CLI evidence", path)
        if str(path) in bindings:
            assert expected is None or bindings[str(path)] == expected
            return path  # All bindings are rehashed by check_identity before any GPU launch.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert expected is None or digest == expected, ("Changed CLI evidence", path)
        assert bindings.get(str(path), digest) == digest
        bindings[str(path)] = digest
        return path

    def include_hashes(tree):
        tree = Path(tree)
        assert tree.is_dir()
        return {str(p.relative_to(tree)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(tree.rglob("*")) if p.is_file()}

    assert include_identity and include_hashes(include) == include_identity
    assert len(paths) == len({str(p.resolve()) for p in paths})
    for path in paths:
        assert path.name == "contiguous_contract.json", "Require real contiguous-contract CLI summaries"
        check = json.loads(bind(path).read_text())
        assert check["status"] == "PASS" and check["gpu_execution"] is True
        ids = [row["id"] for row in check["cases"]]
        assert ids and len(ids) == len(set(ids))
        assert check["selected_cases"] == ids, "Incomplete or duplicate finalized CLI case selection"
        assert "--contiguous-contract" in check["command"]
        for row in check["cases"]:
            assert row["status"] == "PASS" and not row.get("error")
            dest = Path(row["path"]).resolve()
            assert dest == path.resolve().parent / row["id"]
            evidence = json.loads(bind(dest / "evidence.json").read_text())
            case = json.loads(bind(dest / "case.json").read_text())
            assert evidence["case"] == case and case["id"] == row["id"]
            assert evidence["status"] == "PASS" and type(evidence["returncode"]) is int and evidence["returncode"] == 0
            command = json.loads(bind(dest / "command.json").read_text())
            assert "--contract-worker" in command["argv"] and case["id"] in command["argv"]
            assert evidence["python_sources"] == python_hashes, "CLI production Python differs from repository"
            for kind, filename in (("core", "libtransformer_engine.so"),
                                   ("torch", next(n for n in library_hashes if n != "libtransformer_engine.so"))):
                lib = evidence["libraries"][kind]
                assert lib["sha256"] == library_hashes[filename], "CLI libraries differ from frozen runtime"
                bind(lib["path"], lib["sha256"])
            tree = Path(evidence["rtc_include"]).resolve()
            if tree not in checked_includes:
                assert include_hashes(tree) == include_identity, "CLI include differs from supplied frozen include"
                for name, digest in include_identity.items():
                    bind(tree / name, digest)
                checked_includes.add(tree)
            assert evidence["full_backing_guards"] is True, "Require live full-backing GPU guards"
            log_path = bind(evidence["stdout"])
            assert log_path == dest / "stdout.log"
            log = log_path.read_text()
            assert not any(error in log for error in ("==ERROR==", "Fallback to cuBLAS", "Traceback (most recent call last):\n"))
            # The worker prints before the parent finalizes returncode/host receipts.
            printed = [json.loads(line)["public_runtime_observation"] for line in log.splitlines()
                       if line.startswith('{"public_runtime_observation"')]
            assert len(printed) == 1 and all(evidence.get(k) == v for k, v in printed[0].items())
            launches = [line for line in log.splitlines() if "NVTE_GG_CONTIGUOUS LAUNCHED" in line]
            setup = evidence.get("setup_host_launch_receipts", [])
            assert launches == setup + evidence["host_launch_receipts"]
            assert evidence["host_no_work_receipts"] == [line for line in log.splitlines()
                                                        if "NVTE_GG_CONTIGUOUS NO_WORK" in line]
            fields = [dict(re.findall(r"\b(\w+)=([^\s]+)", line)) for line in evidence["host_launch_receipts"]]
            assert fields == evidence["host_receipt_fields"]
            profiles = [evidence["profile"], *[evidence[k] for k in ("forward_setup", "ordinary_baseline") if k in evidence]]
            measured_symbols = []
            for profile in profiles:
                trace_path = bind(profile["trace"])
                assert trace_path.parent == dest
                events = json.loads(trace_path.read_text())["traceEvents"]
                rtc = [e["name"] for e in events if e.get("cat") == "kernel"
                       and e.get("name", "").startswith("nvte_grouped_gemm3_")]
                assert rtc == profile["rtc_kernels"], "Trace/worker RTC launch mismatch"
                assert rtc == [e["name"] for e in profile["events"]
                               if e["device_type"].endswith("CUDA") and e["name"].startswith("nvte_grouped_gemm3_")]
                assert profile["launch_host_events"] == [e for e in profile["events"] if "launch" in e["name"].lower()]
                if case.get("bad") and profile is evidence["profile"]:
                    # Graph entry may launch torch FillFunctor bookkeeping; forbid GEMM/driver launches,
                    # not those observed auxiliary kernels, and retain their trace in the binding.
                    assert not rtc and not any(e["name"].lower().startswith("culaunch") for e in profile["launch_host_events"])
                    assert not any(e.get("name", "").lower().startswith("culaunch") for e in events)
                    assert not any(re.search(r"gemm|cublas", name, re.I) for name in profile["cuda_kernels"])
                measured_symbols.extend(rtc)
            artifacts = evidence["runtime_artifacts"]
            artifact_paths = [str(bind(item["path"], item["sha256"])) for item in artifacts]
            assert len(artifact_paths) == len(set(artifact_paths))
            actual_artifacts = {str(p.resolve()) for pattern in ("*.cubin", "*.cu", "*.options.txt") for p in dest.glob(pattern)}
            assert set(artifact_paths) == actual_artifacts, "Incomplete runtime artifact binding"
            options = {p.name: p.read_text() for p in dest.glob("*.options.txt")}
            assert options == evidence["runtime_compile_options"]
            for symbol in measured_symbols:
                for suffix in (".cubin", ".cu", ".options.txt"):
                    assert str(dest / (symbol + suffix)) in actual_artifacts
            if case.get("bad"):
                assert evidence["observed_error"] and "NVTE_GG_CONTIGUOUS" in evidence["observed_error"]["message"]
                assert evidence["full_D_backing_unchanged"] is True
                assert not evidence["profile"]["rtc_kernels"] and not fields
                if "workspace_unchanged_before_rejection" in evidence:
                    assert evidence["workspace_unchanged_before_rejection"] is True
                rejected.add((case["api"], case["bad"]))
            else:
                assert evidence["observed_error"] is None and evidence["numeric_comparisons"]
                if case.get("no_work"):
                    assert not fields and not evidence["profile"]["rtc_kernels"]
                    assert evidence["host_no_work_receipts"] and evidence["full_D_backing_unchanged"] is True
                if not case.get("ordinary"):
                    assert evidence["host_admission_work_matches"] is True
                    expected = evidence["expected_launches"]
                    assert len(fields) == len(expected) == len(evidence["profile"]["rtc_kernels"])
                    for got, want, symbol in zip(fields, expected, evidence["profile"]["rtc_kernels"]):
                        assert got["symbol"] == symbol and "_fulltile" in symbol and got["full_tile"] == "1"
                        assert got["params_bytes"] == "80"
                        assert all(got[key] == str(want[key]) for key in ("work", "effective_ne", "eligible_groups", "effective_kmax"))
                        assert all(fragment in symbol for fragment in want["symbol_fragments"])
                # Coverage comes from successful, guarded, nonempty numeric case data, not selectors/IDs.
                shapes = evidence.get("per_group_MNK", [])
                if case["api"] in ("native", "gemm", "direct") and not case.get("ordinary") and any(all(d > 0 for d in shape) for shape in shapes):
                    assert evidence["profile"]["rtc_kernels"] and evidence.get("operands", {}).get("D")
                    assert all(evidence["operands"].get(operand) and all(view["storage_offset"] >= 32 for view in evidence["operands"][operand])
                               for operand in ("A", "B", "D")), "Numeric coverage needs actual guarded packed operands"
                    if case["api"] == "native":
                        assert evidence["native_metadata"] and all(m["no_copy"] is True and m["metadata_device"].startswith("cuda")
                            and m["first_dims_dtype"] == "torch.int64" for m in evidence["native_metadata"])
                    assert all("oracle" in comparison for comparison in evidence["numeric_comparisons"])
                    assert case["layout"] in ("TN", "NN", "NT") and case["dtype"] in ("bf16", "fp32") and case["beta"] in (0, 1)
                    expected_shapes = [[case.get("m", 256), case.get("n", 256), s] if case["layout"] == "NT"
                                       else [s, case.get("n", 256), case.get("k", 256)] for s in case["splits"]]
                    assert shapes == expected_shapes and evidence["splits"] == case["splits"]
                    layout_fragment = {"TN": "_LA_RM_LB_CM_LC_RM_", "NN": "_LA_RM_LB_RM_LC_RM_", "NT": "_LA_CM_LB_RM_LC_RM_"}[case["layout"]]
                    bn = 128 if case["beta"] else 256
                    requested_ne = int(evidence["environment"]["NVTE_GG_CONTIGUOUS_N_EXPAND"])
                    assert requested_ne in (1, 4)
                    kmax = case.get("kmax", 512)
                    eligible = sum(m > 0 and 0 < k <= kmax and n % (bn * requested_ne) == 0 for m, n, k in shapes) if not case["beta"] and requested_ne > 1 else 0
                    ne = requested_ne if eligible else 1
                    work = sum(m // 256 * (n // bn // ne) for m, n, _ in shapes)
                    for got in fields:
                        symbol = got["symbol"]
                        assert layout_fragment in symbol and f"_bf16_bf16_{case['dtype']}_acc_fp32_" in symbol
                        assert f"_C{case['dtype'] if case['beta'] else 'void'}_contiguous{2 if ne > 1 else 1}" in symbol
                        assert got["work"] == str(work) and got["effective_ne"] == str(ne) and got["eligible_groups"] == str(eligible)
                        assert got["effective_kmax"] == str(kmax if ne > 1 else 512)
                        compile_options = options[symbol + ".options.txt"].splitlines()
                        assert f"-DNVTE_GG_WITH_C={case['beta']}" in compile_options
                        assert f"-DNVTE_GG_TILE_N={bn}" in compile_options and "-DNVTE_GG_TILE_M=256" in compile_options
                    coverage.add((case["api"], case["layout"], case["dtype"], case["beta"]))
            verified.append({"summary": str(path.resolve()), "id": case["id"], "api": case["api"],
                             "guarded": True, "rejected": bool(case.get("bad"))})
    required = {(api, layout, dtype, beta) for api in ("native", "gemm")
                for layout in ("TN", "NN", "NT") for dtype in ("bf16", "fp32") for beta in (0, 1)}
    assert required <= coverage, ("Missing actual guarded numeric coverage", sorted(required - coverage))
    assert {api for api, _ in rejected} >= {"native", "direct", "gemm"}, "Missing native/direct/list rejection boundaries"
    assert ("direct", "lists") in rejected, "Missing direct tensor-list boundary"
    assert any(key[0] == "direct" for key in coverage), "Missing successful guarded direct boundary"
    return bindings, {"cases": verified, "numeric_coverage": [list(key) for key in sorted(coverage)],
                      "rtc_includes": [str(p) for p in sorted(checked_includes)],
                      "rejection_boundaries": [list(key) for key in sorted(rejected)]}


def nexpand_csv_row(case, summary, uuid, status, reports):
    """完整账本一例一行，保留九样本、负收益、旁路及资源信息。"""
    summary = summary or {}
    cfg, p, x = summary.get("config", {}), summary.get("production", {}), summary.get("contiguous", {})
    return {"id": case["id"], "direction": case["direction"], "model_K": case["model_KN"][0],
            "model_N": case["model_KN"][1], "M": case["MNK"][0], "N": case["MNK"][1], "K": case["MNK"][2],
            "dtype": case["dtype"], "beta": case["beta"], "status": status, "gpu_uuid": uuid,
            "scope": summary.get("scope", case.get("scope", "")),
            "formal_acceptance": summary.get("formal_acceptance", case.get("formal_acceptance", "")),
            "performance_valid": summary.get("performance_valid", case.get("performance_valid", "")),
            "selected": case.get("selected", ""),
            "production_tfu": p.get("median"), "current_tfu": x.get("median"),
            "delta_pp": summary.get("delta_pp"), "relative_pct": summary.get("relative_pct"),
            **{key: cfg.get(key) for key in ("requested_ne", "effective_ne", "kmax", "effective_kmax", "eligible_groups", "bypass_reason")},
            "production_samples": json.dumps(p.get("samples", [])), "current_samples": json.dumps(x.get("samples", [])),
            "paired_round_delta_pp": json.dumps([b - a for a, b in zip(p.get("round_medians", []), x.get("round_medians", []))]),
            **{f"{label}_{key}": stats.get(key) for label, stats in (("production", p), ("current", x))
               for key in ("min", "max", "round_min", "round_max")},
            "production_duration_ns": summary.get("duration_ns", {}).get("production", {}).get("median", ""),
            "current_duration_ns": summary.get("duration_ns", {}).get("contiguous", {}).get("median", ""),
            "production_duration_round_medians_ns": json.dumps(summary["duration_ns"]["production"]["round_medians"]) if "duration_ns" in summary else "",
            "current_duration_round_medians_ns": json.dumps(summary["duration_ns"]["contiguous"]["round_medians"]) if "duration_ns" in summary else "",
            "speedup": summary.get("speedup", ""),
            "launch_geometry": json.dumps(summary["launch_geometry"]) if "launch_geometry" in summary else "",
            "resources": json.dumps(summary.get("resources", {})), "reports": json.dumps(reports)}


def nexpand_validate_runtime(log, symbol, config, case=None):
    """准入必须由真实host回执及实际symbol证明，不使用requested元数据代替。"""
    if case is not None:
        nexpand_validate_admission(case, config)
    lines = [line for line in log.splitlines() if "NVTE_GG_CONTIGUOUS LAUNCHED" in line and
             f"symbol={symbol} " in line]
    assert lines, "缺少contiguous实际launch回执"
    expected = {"requested_ne": config["requested_ne"], "effective_ne": config["effective_ne"],
                "requested_kmax": config["kmax"], "effective_kmax": config["effective_kmax"],
                "eligible_groups": config["eligible_groups"], "bypass_reason": config["bypass_reason"],
                "work": config["expected_work"]}
    for line in lines:
        fields = dict(re.findall(r"\b(\w+)=([^\s]+)", line))
        assert all(fields.get(key) == str(value) for key, value in expected.items()), (fields, expected)


parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=("build", "audit-build", "verify", "verify-codegen", "resources", "prepare", "run", "probe", "capture", "export", "audit-stream", "matrix", "source-reports", "tail-study", "smallk-study", "nexpand-matrix", "fulltile-compare", "audit-nexpand", "audit-nexpand-codegen", "audit-epilogue", "bind-verify-source", "verify-representative"))
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
                    help="smallk-study 显式诊断采集：不等待完整回归，仍要求空闲设备并禁止正式性能结论")
parser.add_argument("--nexpand-stage", choices=("baseline", "kmax", "ne8"), default="baseline")
parser.add_argument("--nexpand-kmax", type=int, choices=(512, 1024, 2048), default=512)
parser.add_argument("--packed-study-splits", help="Explicit comma-separated MNK256 ragged M/TN/NN or K/NT; uniform manifest stays unchanged")
parser.add_argument("--nexpand-baseline", type=Path, help="阶段0完整独立审计目录，包含production锚点")
parser.add_argument("--nexpand-prior", type=Path, help="前一档完整采集result.json；2048必须绑定1024")
parser.add_argument("--nexpand-codegen", type=Path, help="当前阶段默认不变性/新档位编译门禁result.json")
parser.add_argument("--nexpand-current-anchor", type=Path, help="换卡时已完成的同卡NE4/Kmax512补采result.json")
parser.add_argument("--nexpand-capture", type=Path, help="audit-nexpand独立审计的完整采集目录，不运行GPU")
parser.add_argument("--smallk-bn256", action="store_true",
                    help="smallk-study 额外采集 noC BN256 普通/NE4，保留默认四臂")
parser.add_argument("--debug-capture", action="store_true",
                    help="仅诊断：gdb定位采集侧invalid_argument，不作为性能数据")
parser.add_argument("--tag", default="", help="独立采集标签，保留失败日志与既有报告")
parser.add_argument("--rtc", action="store_true", help="不用离线替身，直接验证/采集实际 NVRTC")
parser.add_argument("--rtc-include", type=Path, help="run 使用采集绑定的完整 include 快照，避免并发重构建污染")
parser.add_argument("--rtc-runtime", type=Path, help="fulltile-compare only: frozen runtime containing both shared libraries and production Python")
parser.add_argument("--comparison-cases", nargs="+", help="fulltile-compare only: exact original26 manifest IDs; omitted means all26")
parser.add_argument("--test-filter", default="overlap", help="verify 阶段的 pytest -k 选择表达式")
parser.add_argument("--require-verify", type=Path, nargs="+", help="matrix 等待这些正确性回执全部 PASS 后才采集；fulltile-compare requires finalized contiguous_contract.json GPU receipts")
parser.add_argument("--verify-scope", choices=("full", "representative"), default="full",
                    help="验证范围；representative 只支撑开发采集，不声明全量正确性，也不构成24+2正式验收")
parser.add_argument("--nodes", type=Path, help="verify-representative 的显式 node ID 清单，逐项核对真实收集")
parser.add_argument("--verify-source-snapshot", type=Path,
                    help="bind-verify-source 保存的已验收源码，用AST验证未改测试及其公共依赖")
parser.add_argument("--resume", action="store_true", help="matrix 仅恢复已完成和未启动项；不覆盖失败或部分采集")
parser.add_argument("--families", nargs="+", choices=("step1", "step2", "step3", "combined"),
                    default=["step1", "step2", "step3", "combined"],
                    help="matrix 按实验族分卡并行；每个 job 内基线与实验臂仍同卡配对")
a = parser.parse_args()
if a.phase in ("build", "prepare", "run", "verify", "verify-codegen", "resources",
               "probe", "matrix", "tail-study", "smallk-study", "source-reports"):
    parser.error(
        f"phase {a.phase!r} uses retired experiment macros and cannot run against cleaned "
        "production source (its arms would be identical). Use the frozen historical drivers "
        "under build/ for those experiments; report/capture audits, nexpand-matrix and "
        "fulltile-compare remain available."
    )
assert a.phase == "fulltile-compare" or (a.rtc_runtime is None and a.comparison_cases is None), \
    "--rtc-runtime and --comparison-cases are only for fulltile-compare"
if a.packed_study_splits:
    assert a.phase == "nexpand-matrix", "Ragged workload selection is only for nexpand-matrix"
    a.packed_study_splits = [int(s.strip()) for s in a.packed_study_splits.split(",")]
    assert a.packed_study_splits and all(s >= 0 and s % 256 == 0 for s in a.packed_study_splits)
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

if a.phase == "verify-representative":
    # 独立、显式的开发验证范围：只跑经收集核对的显式 node 子集，回执带 scope 标记，
    # 不冒充完整收集集合回执，也不改变 verify 阶段的全量门禁。
    import xml.etree.ElementTree as ET

    receipt = a.out / "result.json"
    record = {"status": "FAIL", "scope": "representative-development",
              "phase": "verify-representative", "formal_acceptance": False,
              "full_regression_complete": False,
              "note": "代表性子集开发验证：不声明全量正确性，不构成 24+2 正式验收",
              "gpu_execution": True}
    try:
        assert a.verify_scope == "representative", "该入口只接受 --verify-scope representative"
        assert a.nodes and a.nodes.is_file(), "必须显式提供 --nodes 清单，不接受 -k 猜测式筛选"
        selected = [line.strip() for line in a.nodes.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
        assert selected and all(node.startswith("tests/pytorch/test_grouped_gemm_rtc.py::")
                                for node in selected), selected
        names = [node.split("::", 1)[1] for node in selected]
        assert len(set(names)) == len(names), "选中清单含重复 node"
        record.update(gpu_index=a.gpu, selected_nodes=selected, selected_count=len(selected), device=device_idle())
        env = dict(os.environ)
        for key in [k for k in env if k.startswith("NVTE_GG_")]:
            env.pop(key, None)
        env.pop("NVTE_CUTLASS3_INCLUDE_DIR", None)
        env.pop("PYTHONPATH", None)
        env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
        env["NVTE_CUTLASS_GROUPED_GEMM_RTC_VERBOSE"] = "1"
        source = ROOT / "build/cmake/actlize_effective/include"
        frozen = a.out / "rtc_include"
        def header_manifest(tree):
            return {str(path.relative_to(tree)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(tree.rglob("*")) if path.is_file()}
        before = header_manifest(source)
        assert before, "effective include 树为空"
        if frozen.exists():
            shutil.rmtree(frozen)
        shutil.copytree(source, frozen)
        # 复制前后逐文件对拍；构建中的中间树不能成为验收快照。
        assert before == header_manifest(frozen) == header_manifest(source), \
            "effective include 在冻结期间改变，请等待构建完成后使用新目录"
        save(a.out / "include_identity.json", before)
        env["NVTE_GG_TEST_INCLUDE_DIR"] = str(frozen)
        env["NVTE_GG_RTC_CACHE_DIR"] = str(a.out / "rtc_cache")
        library_paths = [ROOT / "libtransformer_engine.so", *ROOT.glob("transformer_engine_torch.*.so")]
        assert len(library_paths) == 2, library_paths
        test_source = ROOT / "tests/pytorch/test_grouped_gemm_rtc.py"
        collection_cmd = [sys.executable, "-m", "pytest", str(test_source), "--collect-only", "-q"]
        save(a.out / "collection_command.json", collection_cmd)
        collected = execute(collection_cmd, a.out / "collection.log", env=env, strict_contention=False)
        collected_names = [line.split("::", 1)[1] for line in collected.splitlines()
                           if line.startswith("tests/pytorch/test_grouped_gemm_rtc.py::")]
        assert collected_names, "未取得测试收集集合"
        missing = sorted(set(names) - set(collected_names))
        assert not missing, ("选中 node 不在当前收集集合", missing)
        record.update(collected_total=len(collected_names),
                      selection_is_strict_subset=set(names) != set(collected_names))
        cmd = [sys.executable, "-m", "pytest", *selected, "--basetemp", str(a.out / "tests"),
               "--junitxml", str(a.out / "junit.xml"), "-v", "-rA"]
        save(a.out / "command.json", cmd)
        execute(cmd, a.out / "pytest.log", env=env, strict_contention=False)
        suites = ET.parse(a.out / "junit.xml").getroot().findall("testsuite")
        counts = {key: sum(int(s.get(key, 0)) for s in suites)
                  for key in ("tests", "failures", "errors", "skipped")}
        testcases = [{"classname": case.get("classname"), "name": case.get("name")}
                     for suite in suites for case in suite.findall("testcase")]
        assert counts["tests"] == len(selected), (counts, len(selected))
        assert not any(counts[k] for k in ("failures", "errors", "skipped")), counts
        assert {case["name"] for case in testcases} == set(names), "实际运行 node 与选中清单不一致"
        record.update(counts=counts, testcases=testcases, test_filter=None,
                      library_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in library_paths},
                      test_source_sha256=hashlib.sha256(test_source.read_bytes()).hexdigest(),
                      rtc_include=str(frozen), include_files=len(before))
        record["status"] = "PASS"
    except Exception as error:
        record.update(status="FAIL", error=str(error))
        raise
    finally:
        save(receipt, record)
    print(json.dumps(record), flush=True)
    raise SystemExit(0)

if a.phase == "audit-epilogue":
    # 只审计已有 full 导出；不启动 GPU、不重写采集回执或历史报告。
    from collections import Counter, defaultdict
    capture = json.loads((a.out / "result.json").read_text())
    assert capture["status"] in ("DIAGNOSTIC_CAPTURED", "PASS")
    assert not capture["contention"] and not capture["capture_failures"]
    assert capture["case"] == "fwd_noC_M4096_N4096_K512_groups16_ragged"
    output = a.out / "epilogue_audit.json"
    assert not output.exists(), "不覆盖已有动态审计"
    frozen = Path(capture["rtc_include"])
    manifest = json.loads((a.out / "include_identity.json").read_text())
    assert manifest == {str(p.relative_to(frozen)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(frozen.rglob("*")) if p.is_file()}
    for check in capture["verification"]:
        assert check["status"] == "PASS" and check["counts"]["tests"] == len(check["testcases"])
        assert not any(check["counts"][k] for k in ("failures", "errors", "skipped"))
    expected_bytes = sum(capture["actual"]["splits"]) * 4096 * 2
    metrics_kept = (METRIC, "ppu__time_duration.sum", "kvd__bytes_pipe_lsu_mem_global_op_st.sum",
                    "kvd__requests_store_pipe_lsu.sum", "kvd__bytes_store_pipe_l2.sum",
                    "kvd__transactions_store_pipe_l2.sum")
    result = {"status": "RUNNING", "capture": str(a.out / "result.json"),
              "diagnostic": capture["diagnostic"], "performance_valid": capture["performance_valid"],
              "expected_D_bytes": expected_bytes, "actions": [], "summary": {},
              "gpu_regression_counts": [r["counts"] for r in capture["verification"]],
              "note": "b16占比按实际thread执行次数和store位宽换算输出元素；KVD字节取原metric，不以向量宽度换算事务。"}
    for report_path in capture["reports"]:
        report = Path(report_path)
        assert report.parent == a.out / "remote-results" and report.is_file()
        label = report.stem
        metrics = json.loads((a.out / "cases" / f"{label}.metrics.json").read_text())
        arm = metrics[0]["arm"]
        groups = defaultdict(list)
        with (a.out / "cases" / f"{label}.sass.csv").open() as stream:
            for row in csv.DictReader(stream):
                groups[(row["Range Id"], row["Function Id"])].append(row)
        assert len(groups) == len(metrics) == 3
        embedded = json.loads((a.out / label / "embedded_sources.json").read_text())
        assert len(embedded) == len(metrics)
        symbol = capture["actual"]["arms"][arm]["kernels"][0]
        native_isa = (a.out / "actual" / "rtc" / arm / "isa.txt").read_text()
        native = {int(pc, 16): op for pc, op in re.findall(
            r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+){8}(\S+)", native_isa, re.M)}
        assert native
        for index, (rows, metric) in enumerate(zip(groups.values(), metrics)):
            assert metric["arm"] == arm and metric["kernel_name"] == symbol
            assert not metric["foreign_pids"]
            assert len(metric["raw_metrics"]) > 0
            assert metric["source_sha256"] == embedded[index]
            for path, digest in embedded[index].items():
                source = Path(path)
                assert source.is_relative_to(frozen) and hashlib.sha256(source.read_bytes()).hexdigest() == digest
            warp, thread, store_bytes = Counter(), Counter(), Counter()
            base = min(int(r["Address"], 16) for r in rows)
            for row in rows:
                assert row["Function Name"] == symbol
                opcode = row["Source"].split()[0]
                offset = int(row["Address"], 16) - base
                assert native.get(offset) == opcode, (arm, offset, opcode)
                executed = float(row["Instructions Executed"].replace(",", "") or 0)
                lanes = float(row["Thread Instructions Executed"].replace(",", "") or 0)
                assert executed.is_integer() and lanes.is_integer() and 0 <= lanes <= executed * 32
                warp[opcode] += int(executed)
                thread[opcode] += int(lanes)
                if opcode.startswith("vmem.st."):
                    width = re.fullmatch(r"vmem\.st\.b(16|32|64|128)(?:x(\d+))?(?:\.[a-z]+)*", opcode)
                    assert width, opcode
                    store_bytes[opcode] += int(lanes) * (int(width[1]) // 8) * int(width[2] or 1)
            assert sum(store_bytes.values()) == expected_bytes, (arm, store_bytes, expected_bytes)
            raw = metric["raw_metrics"]
            assert raw["kvd__bytes_store_pipe_l2.sum"] == expected_bytes
            action = {"arm": arm, "sample": index, "report": str(report),
                      "metric_count": len(raw), "embedded_sources": len(embedded[index]),
                      "store_thread_instructions": {op: thread[op] for op in store_bytes},
                      "store_warp_instructions": {op: warp[op] for op in store_bytes},
                      "store_effective_bytes": dict(store_bytes),
                      "scalar_output_pct": 100 * sum(value for op, value in store_bytes.items()
                          if re.fullmatch(r"vmem\.st\.b16(?:\.[a-z]+)*", op)) / expected_bytes,
                      "dynamic_warp_instructions": sum(warp.values()),
                      "metrics": {key: raw[key] for key in metrics_kept}}
            result["actions"].append(action)
    assert {r["arm"] for r in result["actions"]} == set(capture["actual"]["arms"])
    for arm in capture["actual"]["arms"]:
        actions = [r for r in result["actions"] if r["arm"] == arm]
        fields = {"scalar_output_pct": [r["scalar_output_pct"] for r in actions]}
        fields.update({key: [r["metrics"][key] for r in actions] for key in metrics_kept})
        result["summary"][arm] = {key: {"median": statistics.median(values),
                                      "min": min(values), "max": max(values)}
                                  for key, values in fields.items()}
        result["summary"][arm]["resources"] = capture["resources"][arm]
    result["status"] = "PASS"
    save(output, result)
    print(json.dumps({"status": "PASS", "audit": str(output), "summary": result["summary"]},
                     ensure_ascii=False), flush=True)
    raise SystemExit(0)

if a.phase == "audit-nexpand-codegen":
    output = a.out / "result.json"
    assert not output.exists(), "不覆盖已有codegen独立审计"
    result = {"status": "RUNNING", "gpu_execution": False}
    save(output, result)
    try:
        assert a.nexpand_codegen, "必须显式绑定66组codegen门禁"
        gate = json.loads(a.nexpand_codegen.read_text())
        hashes = nexpand_codegen_artifacts(gate)
        frozen = a.nexpand_codegen.parent / "rtc_include"
        effective = ROOT / "build/cmake/actlize_effective/include"
        def identity(tree):
            return {str(p.relative_to(tree)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(tree.rglob("*")) if p.is_file()}
        assert gate["include_identity"] == identity(frozen) == identity(effective)
        tu = ROOT / "transformer_engine/common/gemm/rtc/cutlass_grouped_gemm.cu"
        assert gate["rtc_tu_sha256"] == hashlib.sha256(tu.read_bytes()).hexdigest()
        result.update(status="PASS", cases=66, default_comparisons=42, new_kmax_cases=24,
                      stack_bytes=0, atomic_instructions=0, artifact_sha256=hashes,
                      codegen=str(a.nexpand_codegen.resolve()),
                      codegen_sha256=hashlib.sha256(a.nexpand_codegen.read_bytes()).hexdigest(),
                      rtc_tu_sha256=gate["rtc_tu_sha256"])
    except Exception as error:
        result.update(status="FAIL", error=str(error))
        raise
    finally:
        save(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifact_sha256"}), flush=True)
    raise SystemExit(0)


if a.phase == "audit-nexpand":
    output = a.out / "result.json"
    assert not output.exists(), "不覆盖已有独立审计"
    result = {"status": "RUNNING", "gpu_execution": False, "performance_valid": False,
              "full_reports": 0, "full_actions": 0, "probe_reports": 0, "audited_cases": []}
    save(output, result)

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def tree_sha(path):
        return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}

    def report_actions(path):
        context = acu_report.load_report(str(path))
        ranges = [context.range_by_idx(i) for i in range(context.num_ranges())]
        return context, ranges, [group.action_by_idx(j) for group in ranges for j in range(group.num_actions())]

    try:
        assert a.nexpand_capture, "必须显式绑定完整采集目录"
        capture = a.nexpand_capture.resolve()
        record = json.loads((capture / "result.json").read_text())
        digest = hashlib.sha256(json.dumps(nexpand_case_manifest(record.get("packed_study_splits")), sort_keys=True).encode()).hexdigest()
        nexpand_validate_capture(record, digest)
        assert record["stage"] in ("baseline", "kmax") and record["kmax"] in (512, 1024, 2048)
        result.update(capture=str(capture), capture_sha256=sha(capture / "result.json"),
                      stage=record["stage"], kmax=record["kmax"], gpu_uuid=record["device"]["uuid"], case_digest=digest)
        frozen = capture / "rtc_include"
        assert record["include_identity"] == tree_sha(frozen)
        for relative, expected in record["source_sha256"].items():
            assert sha(capture / "source_snapshot" / relative) == expected
        for name, expected in record["library_sha256"].items():
            assert sha(capture / "libraries" / name) == expected
        assert record["gate_artifact_sha256"] and all(sha(Path(p)) == h for p, h in record["gate_artifact_sha256"].items())
        result["freeze_identity"] = "PASS"
        gate = json.loads(Path(record["codegen_gate"]["path"]).read_text())
        assert gate["include_identity"] == record["include_identity"]
        assert gate["rtc_tu_sha256"] == record["source_sha256"]["transformer_engine/common/gemm/rtc/cutlass_grouped_gemm.cu"]
        assert all(record["gate_artifact_sha256"].get(p) == h for p, h in nexpand_codegen_artifacts(gate).items())
        result["codegen"] = {"status": "PASS", "cases": 66, "default_comparisons": 42}
        anchors = json.loads((Path(record["baseline_audit"]["path"]) / "production_anchors.json").read_text())
        assert all(not snap["live"] for snap in record["occupancy_snapshots"])
        assert record["profiler_contention"] and all(not v["foreign_pids"] and v["target_identity_valid"]
                                                      for v in record["profiler_contention"].values())
        sys.path.insert(0, "/usr/local/PPU_SDK/asight/extras/python")
        import acu_report
        tables = {False: [], True: []}
        for case in record["manifest"]:
            entry = record["cases"][case["id"]]
            assert all(sha(Path(p)) == h for p, h in entry["artifacts"].items())
            attempt = Path(entry["attempts"][-1])
            actual, resources = entry["actual"], entry["resources"]
            assert actual["production"]["input_sha256"] == actual["contiguous"]["input_sha256"]
            assert actual["production"]["output_sha256"] == actual["contiguous"]["output_sha256"]
            assert entry["production_anchor_identity"] == "PASS"
            samples = {"production": [], "contiguous": []}
            for arm in samples:
                identity, resource = actual[arm], resources[arm]
                assert identity["correctness"] == "PASS" and identity["MNK"] == case["MNK"]
                assert identity["beta"] == case["beta"] and identity["splits"] == case["splits"]
                assert ("initial_C" in identity["input_sha256"]) == bool(case["beta"])
                symbol = identity["kernels"][0]
                rtc = attempt / "actual" / arm / "rtc"
                assert resource["static_smem_bytes"] == resource["local_bytes_per_thread"] == 0
                assert resource["active_blocks_per_cu"] > 0
                assert json.loads((rtc / f"{symbol}.driver_resources.json").read_text()) == resource
                assert re.findall(r"STACK SIZE:(\d+)", (rtc / "resources.txt").read_text()) == ["0"]
                assert not re.search(r"\bvmem\.atm\.add\.f32\b", (rtc / "isa.txt").read_text())
                machine = rtc_machine_identity(rtc / f"{symbol}.cubin")
                assert machine == rtc_machine_identity(attempt / "annotated_rtc" / arm / f"{symbol}.cubin")
                options = (rtc / f"{symbol}.options.txt").read_text().splitlines()
                if arm == "production":
                    anchor = anchors[case["id"]]
                    assert symbol == anchor["symbol"] and machine == anchor["machine_identity"]
                    assert identity["input_sha256"] == anchor["input_sha256"] and identity["output_sha256"] == anchor["output_sha256"]
                    assert {k: v for k, v in resource.items() if k != "cubin"} == {k: v for k, v in anchor["resources"].items() if k != "cubin"}
                    assert "-DNVTE_GG_CONTIGUOUS_MODE=0" in options
                else:
                    cfg = identity["config"]
                    assert cfg["kmax"] == record["kmax"] and cfg["requested_ne"] == (1 if case["beta"] else 4)
                    ending = f"_C{'fp32' if case['beta'] else 'void'}_contiguous{cfg['mode']}"
                    if cfg["effective_ne"] > 1 and cfg["kmax"] != 512:
                        ending += f"_kmax{cfg['kmax']}"
                    ending += "_fulltile"
                    if cfg.get("mma_init", False) and cfg["effective_ne"] == 4:
                        ending += "_mma_init"
                    assert symbol.endswith(ending)
                    assert f"-DNVTE_GG_CONTIGUOUS_MODE={cfg['mode']}" in options
                    nexpand_validate_runtime((rtc.parent / "verify.log").read_text(), symbol, cfg, case)
                expected_kmax = identity["config"]["kmax"] if arm == "contiguous" and identity["config"]["effective_ne"] > 1 else 512
                assert [o for o in options if o.startswith("-DNVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX=")] == \
                    ([f"-DNVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX={expected_kmax}"] if expected_kmax != 512 else [])
                context, report_ranges, actions = report_actions(attempt / "remote-results" / f"probe_{arm}.acurep")
                assert len(actions) == 3
                result["probe_reports"] += 1
            for report in entry["reports"]:
                arm = report["arm"]
                context, report_ranges, actions = report_actions(Path(report["report"]))
                raw = json.loads(Path(report["metrics"]).read_text())
                assert len(raw) == len(actions) == 3
                for metric, action in zip(raw, actions):
                    assert set(metric["raw_metrics"]) == set(action.metrics())
                    assert metric["performance_valid"] and not metric["foreign_pids"] and metric["gpu_uuid"] == record["device"]["uuid"]
                    assert metric["case_digest"] == digest and metric["kernel_name"] == actual[arm]["kernels"][0]
                    assert abs(metric["raw_metrics"][METRIC] - action.metric_by_name(METRIC).as_double()) < 1e-9
                    sources = action.source_files()
                    assert sources and all(sources.values())
                    assert metric["source_sha256"] == {p: hashlib.sha256(v.encode()).hexdigest() for p, v in sources.items()}
                    for name in ("nvte_contiguous_grouped.hpp" if arm == "contiguous" else "ppu_aiu_gemm_array_group.hpp",
                                 "ppu_mma_aiu_multistage_batch_array.hpp", "ppu_epilogue_vectorized_evt.hpp"):
                        embedded = next(v for p, v in sources.items() if Path(p).name == name)
                        assert embedded.splitlines() == next(frozen.rglob(name)).read_text().splitlines()
                assert any(float(r.get("Instructions Executed", "0").replace(",", "") or 0) > 0
                           for r in csv.DictReader(io.StringIO(Path(report["sass"]).read_text())))
                if arm == "production":
                    assert all(all(launch[k] == anchors[case["id"]]["launches"][0][k] for k in ("Block Size", "Grid Size")) for launch in report["launch"])
                samples[arm].append([m["raw_metrics"][METRIC] for m in raw])
                result["full_reports"] += 1
                result["full_actions"] += 3
            assert all(nexpand_statistics(values) == entry["summary"][arm] for arm, values in samples.items())
            p, x = entry["summary"]["production"], entry["summary"]["contiguous"]
            assert entry["summary"]["delta_pp"] == x["median"] - p["median"]
            assert entry["summary"]["relative_pct"] == 100 * (x["median"] / p["median"] - 1)
            tables[case["control"]].append({**case, **entry["summary"], "status": "PASS", "gpu_uuid": record["device"]["uuid"],
                "paired_round_delta_pp": [b - a for a, b in zip(p["round_medians"], x["round_medians"])], "actual": actual})
            result["audited_cases"].append(case["id"])
            save(output, result)
        main, controls = tables[False], tables[True]
        assert (len(main), len(controls), result["full_reports"], result["full_actions"], result["probe_reports"]) == (24, 2, 156, 468, 52)
        assert sum(row["config"]["effective_ne"] == 4 for row in main) == {512: 2, 1024: 6, 2048: 8}[record["kmax"]]
        assert sum(row["config"]["effective_ne"] == 4 for row in controls) == (2 if record["kmax"] == 2048 else 0)
        for name, table in (("current_vs_production", main), ("K2048_controls", controls)):
            save(a.out / f"{name}.json", table)
            with (a.out / f"{name}.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(nexpand_csv_row(table[0], table[0], record["device"]["uuid"], "PASS", [])))
                writer.writeheader()
                for row in table:
                    writer.writerow(nexpand_csv_row(row, row, row["gpu_uuid"], row["status"], row["reports"]))
        result.update(status="PASS", performance_valid=True, main_pass=24, control_pass=2)
    except Exception as error:
        result.update(status="FAIL", error=str(error))
        raise
    finally:
        save(output, result)
    raise SystemExit(0)


if a.phase in ("nexpand-matrix", "fulltile-compare"):
    import xml.etree.ElementTree as ET

    comparison = a.phase == "fulltile-compare"
    scope = "runtime-comparison" if comparison else a.verify_scope
    formal_acceptance = not comparison and a.verify_scope == "full"
    # The comparison shares actual collection, never the historical pytest/full-regression gate.
    receipt = a.out / "result.json"
    manifest = nexpand_case_manifest(a.packed_study_splits)
    manifest_digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    selected_ids = a.comparison_cases if a.comparison_cases is not None else [c["id"] for c in manifest]
    selected_manifest = [c for c in manifest if c["id"] in selected_ids]
    anchors, baseline_record, codegen_gate = None, None, None
    gate_bindings = {}
    try:
        assert len(selected_ids) == len(set(selected_ids)) == len(selected_manifest), "Use unique exact original26 manifest IDs"
        if comparison:
            assert a.nexpand_stage == "baseline" and a.nexpand_kmax == 512
            assert not any((a.nexpand_baseline, a.nexpand_prior, a.nexpand_codegen, a.nexpand_current_anchor))
            assert a.rtc_runtime and a.rtc_include, "Comparison requires an explicit frozen runtime and include"
            a.rtc_runtime, a.rtc_include = a.rtc_runtime.resolve(), a.rtc_include.resolve()
            assert a.rtc_runtime.is_dir() and a.rtc_include.is_dir()
            libraries = [a.rtc_runtime / "libtransformer_engine.so", *a.rtc_runtime.glob("transformer_engine_torch.*.so")]
            assert len(libraries) == 2 and all(p.is_file() for p in libraries)
            runtime_python = sorted((a.rtc_runtime / "transformer_engine").rglob("*.py"))
            assert runtime_python, "Frozen runtime must include production Python"
            for p in runtime_python:
                relative = p.relative_to(a.rtc_runtime)
                assert p.read_bytes() == (ROOT / relative).read_bytes(), ("Frozen production Python differs from repository", relative)
            python_hashes = {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() for relative in (
                "profile/gg_fp32_accum/repro.py", "transformer_engine/pytorch/cpp_extensions/gemm.py",
                "transformer_engine/pytorch/module/grouped_linear.py", "transformer_engine/pytorch/ops/basic/grouped_linear.py")}
            assert all((a.rtc_runtime / relative).is_file() for relative in python_hashes if relative.startswith("transformer_engine/"))
            assert a.require_verify and a.full and a.samples == a.repeats == 3 and not a.diagnostic and not a.debug_capture
            supplied_identity = {str(p.relative_to(a.rtc_include)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(a.rtc_include.rglob("*")) if p.is_file()}
            runtime_library_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries}
            gate_bindings, cli_verification = fulltile_verify_gate(a.require_verify,
                runtime_library_hashes, python_hashes, a.rtc_include, supplied_identity)
            for p in libraries:
                gate_bindings[str(p.resolve())] = runtime_library_hashes[p.name]
            for relative, digest in python_hashes.items():
                gate_bindings[str((ROOT / relative).resolve())] = digest
                if relative.startswith("transformer_engine/"):
                    gate_bindings[str((a.rtc_runtime / relative).resolve())] = digest
        assert a.nexpand_stage in ("baseline", "kmax"), "NE8须在两档NE4矩阵完成后接入"
        assert (a.nexpand_stage == "baseline" and a.nexpand_kmax == 512) or \
            (a.nexpand_stage == "kmax" and a.nexpand_kmax in (1024, 2048))
        assert a.full and a.samples == 3 and a.repeats == 3 and a.require_verify
        assert not a.diagnostic and not a.actual_reference
        if a.nexpand_stage != "baseline" or a.nexpand_baseline:
            assert a.nexpand_baseline and a.nexpand_codegen, "新阶段须绑定完整阶段0独立审计与当前codegen门禁"
            audit_path = a.nexpand_baseline / "result.json"
            audit = json.loads(audit_path.read_text())
            assert audit["status"] == "PASS" and audit["performance_valid"]
            assert (audit["main_pass"], audit["control_pass"], audit["full_reports"], audit["full_actions"]) == (24, 2, 156, 468)
            baseline_path = Path(audit["capture"]) / "result.json"
            baseline_record = json.loads(baseline_path.read_text())
            nexpand_validate_capture(baseline_record, manifest_digest, "baseline", 512)
            anchor_path = a.nexpand_baseline / "production_anchors.json"
            anchors = json.loads(anchor_path.read_text())
            assert set(anchors) == {c["id"] for c in manifest}
            for case in manifest:
                old, anchor = baseline_record["cases"][case["id"]], anchors[case["id"]]
                assert anchor["input_sha256"] == old["actual"]["production"]["input_sha256"]
                assert anchor["output_sha256"] == old["actual"]["production"]["output_sha256"]
                assert anchor["machine_identity"] == rtc_machine_identity(Path(anchor["resources"]["cubin"]))
            codegen_gate = json.loads(a.nexpand_codegen.read_text())
            gate_bindings = nexpand_codegen_artifacts(codegen_gate)
            for path in (audit_path, baseline_path, anchor_path, a.nexpand_codegen):
                gate_bindings[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
            if a.nexpand_stage == "kmax" and a.nexpand_kmax == 2048:
                assert a.nexpand_prior, "先交付Kmax1024的完整24+2账本"
                prior = json.loads(a.nexpand_prior.read_text())
                nexpand_validate_capture(prior, manifest_digest, "kmax", 1024)
                assert prior["baseline_audit"]["anchors_sha256"] == gate_bindings[str(anchor_path.resolve())]
                assert prior["codegen_gate"]["sha256"] == gate_bindings[str(a.nexpand_codegen.resolve())]
                gate_bindings[str(a.nexpand_prior.resolve())] = hashlib.sha256(a.nexpand_prior.read_bytes()).hexdigest()
        if a.nexpand_current_anchor:
            gate_bindings[str(a.nexpand_current_anchor.resolve())] = hashlib.sha256(a.nexpand_current_anchor.read_bytes()).hexdigest()
    except Exception as error:
        if not receipt.exists():
            save(receipt, {"status": "FAIL", "failure_phase": "prerequisites", "gpu_execution": False,
                           "stage": a.nexpand_stage, "kmax": a.nexpand_kmax, "error": str(error),
                           "case_digest": manifest_digest, "manifest": manifest, "selected_case_ids": selected_ids,
                           "scope": scope, "formal_acceptance": False, "measurement_valid": False,
                           "performance_valid": False})
        raise
    if receipt.exists():
        assert a.resume, "不覆盖既有批次；续采须显式--resume"
        record = json.loads(receipt.read_text())
        assert record["case_digest"] == manifest_digest and record["stage"] == a.nexpand_stage
        assert record.get("kmax", 512) == a.nexpand_kmax
        assert int(record["gpu_index"]) == a.gpu, "同批次续采不得换物理卡"
        assert record.get("verification_scope", "full") == scope, "同批次续采不得更换验证范围"
        if comparison:
            assert record["scope"] == scope and record["formal_acceptance"] is False and record["performance_valid"] is False
            assert record["manifest"] == manifest and record["selected_case_ids"] == selected_ids
            assert record["rtc_runtime"] == str(a.rtc_runtime) and record["supplied_rtc_include"] == str(a.rtc_include)
    else:
        record = {"status": "RUNNING", "stage": a.nexpand_stage, "pid": os.getpid(),
                  "case_digest": manifest_digest, "manifest": manifest, "cases": {},
                  "gpu_index": a.gpu, "contention": False, "occupancy_snapshots": [],
                  "profiler_contention": {}, "performance_valid": False,
                  "metric": METRIC, "samples_per_round": 3, "rounds": 3,
                  "core_24_complete": False,
                  "verification_scope": scope,
                  "formal_acceptance": formal_acceptance}
    assert record.get("gate_artifact_sha256", gate_bindings) == gate_bindings, "续采门禁身份改变"
    record["gate_artifact_sha256"] = gate_bindings
    record.update(pid=os.getpid(), status="RUNNING", performance_valid=False, requested_ne=4,
                  kmax=a.nexpand_kmax, verification_scope=scope,
                  formal_acceptance=formal_acceptance)
    if comparison:
        record.update(scope=scope, measurement_valid=False, selected_case_ids=selected_ids,
                      rtc_runtime=str(a.rtc_runtime), supplied_rtc_include=str(a.rtc_include),
                      gate_bindings=gate_bindings, cli_verification=cli_verification,
                      verification=[str(p.resolve()) for p in a.require_verify],
                      verified_cases=len(cli_verification["cases"]))
    if a.packed_study_splits is not None:
        record["packed_study_splits"] = a.packed_study_splits
    if anchors:
        record["baseline_audit"] = {"path": str(a.nexpand_baseline.resolve()),
            "result_sha256": hashlib.sha256((a.nexpand_baseline / "result.json").read_bytes()).hexdigest(),
            "anchors_sha256": hashlib.sha256((a.nexpand_baseline / "production_anchors.json").read_bytes()).hexdigest()}
        record["codegen_gate"] = {"path": str(a.nexpand_codegen.resolve()),
            "sha256": hashlib.sha256(a.nexpand_codegen.read_bytes()).hexdigest()}
    save(receipt, record)
    runtime_root = a.rtc_runtime if comparison else ROOT
    library_paths = [runtime_root / "libtransformer_engine.so", *runtime_root.glob("transformer_engine_torch.*.so")]
    assert len(library_paths) == 2
    source_paths = [Path(__file__).resolve(), ROOT / "profile/gg_fp32_accum/repro.py",
                    ROOT / "tests/pytorch/test_grouped_gemm_rtc.py",
                    ROOT / "tests/pytorch/test_rtc_grouped_gemm_ci.py",
                    ROOT / "transformer_engine/common/gemm/cutlass_grouped_gemm_rtc.cu",
                    ROOT / "transformer_engine/common/gemm/rtc/cutlass_grouped_gemm.cu",
                    ROOT / "transformer_engine/pytorch/cpp_extensions/gemm.py",
                    ROOT / "transformer_engine/pytorch/module/grouped_linear.py",
                    ROOT / "transformer_engine/pytorch/ops/basic/grouped_linear.py"]
    if comparison:
        source_paths = sorted(set(source_paths) | {ROOT / p.relative_to(runtime_root) for p in runtime_python})
    effective = a.rtc_include if comparison else ROOT / "build/cmake/actlize_effective/include"
    frozen = a.out / "rtc_include"
    script = str(Path(__file__).resolve())
    repro = str(ROOT / "profile/gg_fp32_accum/repro.py")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), LC_ALL="C", LANG="C",
               PYTHONPATH=str(runtime_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               NVTE_CUTLASS3_INCLUDE_DIR=str(frozen), NVTE_GG_RTC_LINEINFO="1",
               NVTE_CUTLASS_GROUPED_GEMM_RTC_VERBOSE="1", CUDA_LAUNCH_BLOCKING="1")

    def digests(tree):
        return {str(p.relative_to(tree)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(tree.rglob("*")) if p.is_file()}

    def source_identity():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}

    def library_identity():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in library_paths}

    def check_device(stage):
        devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                                           "--format=csv,nounits"], text=True)
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                                       "--format=csv,nounits"], text=True)
        gpu = next({k.strip().split(" [", 1)[0]: v.strip() for k, v in row.items()}
                   for row in csv.DictReader(io.StringIO(devices)) if int(row["index"]) == a.gpu)
        live = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(io.StringIO(apps))
                if row["gpu_uuid"].strip() == gpu["uuid"] and Path(f"/proc/{row[' pid'].strip()}").exists()]
        record["occupancy_snapshots"].append({"stage": stage, "device": gpu, "live": live,
            "all_devices": devices, "all_apps": apps, "timestamp": datetime.now(timezone.utc).isoformat()})
        record["contention"] |= bool(live)
        save(receipt, record)
        assert not live, ("NO_DEVICE_AVAILABLE", stage, live)
        assert float(gpu["memory.total"]) - float(gpu["memory.used"]) > 8192
        if "device" in record:
            assert gpu["uuid"] == record["device"]["uuid"]
        return gpu

    def check_identity():
        assert record["library_sha256"] == library_identity(), "采集期间库改变"
        assert record["source_sha256"] == source_identity(), "采集期间源码改变"
        assert record["include_identity"] == digests(frozen) == digests(effective), "采集期间头文件改变"
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                   for path, digest in record["gate_artifact_sha256"].items()), "采集期间前档或codegen门禁改变"
        if comparison:
            assert all(digests(Path(tree)) == record["include_identity"] for tree in record["cli_verification"]["rtc_includes"]), \
                "CLI include tree changed during comparison"
            current_python = sorted((runtime_root / "transformer_engine").rglob("*.py"))
            assert current_python == runtime_python and all(
                p.read_bytes() == (ROOT / p.relative_to(runtime_root)).read_bytes() for p in current_python), \
                "Frozen production Python changed during comparison"

    def command_case(case, arm, dest):
        m, n, k = case["MNK"]
        command = [sys.executable, repro, "--packed-study-arm", arm, "--packed-study-ne",
                   "1" if case["beta"] or arm == "production" else "4",
                   "--packed-study-kmax", str(a.nexpand_kmax), "--layout", case["layout"], "--dtype", case["dtype"],
                   "--rows", str(m), "--cols", str(n), "--mpe", str(k), "--groups", str(case["groups"]),
                   "--accumulate", str(case["beta"]), "--check", "--dump", str(dest)]
        if a.packed_study_splits is not None:
            command += ["--packed-study-splits", ",".join(map(str, case["splits"]))]
        return command

    def launch_identity(log, arm, symbol, identity):
        tile = re.search(r"_tb(\d+)x(\d+)x(\d+)_s(\d+)_", symbol)
        assert tile and len(identity["kernels"]) <= 1
        tm, tn, tk, stages = map(int, tile.groups())
        if arm == "contiguous":
            nexpand_validate_runtime(log, symbol, identity["config"], case)
            fields = [dict(re.findall(r"\b(\w+)=([^\s]+)", line)) for line in log.splitlines()
                      if "NVTE_GG_CONTIGUOUS LAUNCHED" in line and f"symbol={symbol} " in line]
            assert fields and all(f["full_tile"] == "1" and f["params_bytes"] == "80" for f in fields)
            launches = [{"grid": [int(f["grid"]), 1, 1], "block": [int(f["threads"]), 1, 1],
                         "smem": int(f["smem"]), "work": int(f["work"])} for f in fields]
        else:
            assert "_contiguous" not in symbol and identity["config"]["mode"] == 0
            work = sum(((m + tm - 1) // tm) * ((n + tn - 1) // tn) for m, n, _ in identity["per_group_MNK"])
            fields = re.findall(r"kernel_func=\S+ grid=\((\d+),(\d+),(\d+)\) block=\((\d+),(\d+),(\d+)\)"
                                r" smem=(\d+)B\s+num_gemms=(\d+) tile=\((\d+),(\d+),(\d+)\) stages=(\d+)", log)
            assert fields, "Missing actual ordinary RTC launch geometry"
            launches = []
            for values in fields:
                gx, gy, gz, bx, by, bz, smem, groups, lm, ln, lk, ls = map(int, values)
                assert (groups, lm, ln, lk, ls) == (case["groups"], tm, tn, tk, stages)
                launches.append({"grid": [gx, gy, gz], "block": [bx, by, bz], "smem": smem, "work": work})
        assert all(item == launches[0] for item in launches), "Within-arm host work/grid/block drift"
        launch = launches[0]
        assert all(d > 0 for d in launch["grid"] + launch["block"]) and launch["work"] > 0
        return launch

    def write_ledger():
        for control, filename in ((False, "current_vs_production"), (True, "K2048_controls")):
            rows = []
            for case in manifest:
                if case["control"] != control:
                    continue
                entry = record["cases"].get(case["id"], {})
                row = {**case, "status": entry.get("status", "NOT_SELECTED" if comparison and case["id"] not in selected_ids else "PENDING"),
                       "gpu_uuid": record.get("device", {}).get("uuid"),
                       "metric": METRIC, "result": entry.get("summary"),
                       "actual": entry.get("actual"), "reports": entry.get("reports"),
                       "error": entry.get("error")}
                if comparison:
                    row.update(scope=scope, formal_acceptance=False, performance_valid=False,
                               selected=case["id"] in selected_ids)
                rows.append(row)
            ledger = {"case_digest": manifest_digest, "rows": rows,
                      "scope": scope, "formal_acceptance": formal_acceptance,
                      "complete": all(row["status"] == "PASS" for row in rows)}
            if comparison:
                ledger.update(manifest=manifest, selected_case_ids=selected_ids, performance_valid=False,
                              measurement_valid=record["measurement_valid"],
                              core_24_complete=record["core_24_complete"])
            save(a.out / f"{filename}.json", ledger)
            # CSV每case一行，不以嵌套结果或赢家子集替代全量账本。
            with (a.out / f"{filename}.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(nexpand_csv_row(manifest[0], {}, None, "PENDING", [])))
                writer.writeheader()
                for row in rows:
                    writer.writerow(nexpand_csv_row(row, row["result"], row["gpu_uuid"], row["status"], row["reports"]))

    try:
        if comparison and not frozen.exists():
            assert digests(effective) == supplied_identity
            shutil.copytree(effective, frozen)
            assert digests(frozen) == digests(effective) == supplied_identity
            record.update(include_identity=supplied_identity, source_sha256=source_identity(),
                          library_sha256=library_identity())
            for p in source_paths:
                dest = a.out / "source_snapshot" / p.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dest)
            for p in library_paths:
                dest = a.out / "libraries" / p.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dest)
            save(a.out / "case_manifest.json", manifest)
            save(receipt, record)
        if comparison:
            check_identity()
            assert json.loads((a.out / "case_manifest.json").read_text()) == manifest
            write_ledger()
        record["device"] = check_device("start")
        current_anchor = json.loads(a.nexpand_current_anchor.read_text()) if a.nexpand_current_anchor else None
        nexpand_validate_current_anchor(a.nexpand_stage, record["device"]["uuid"], baseline_record,
                                       current_anchor, manifest_digest, record.get("baseline_audit"))
        if a.nexpand_stage != "baseline" and baseline_record and record["device"]["uuid"] != baseline_record["device"]["uuid"]:
            record["same_device_current_anchor"] = str(a.nexpand_current_anchor.resolve())
            record["gate_artifact_sha256"][str(a.nexpand_current_anchor.resolve())] = hashlib.sha256(a.nexpand_current_anchor.read_bytes()).hexdigest()
        if not frozen.exists():
            checks = [json.loads(path.read_text()) for path in a.require_verify]
            current_test_hash = hashlib.sha256((ROOT / "tests/pytorch/test_grouped_gemm_rtc.py").read_bytes()).hexdigest()
            before = digests(effective)
            if codegen_gate:
                assert codegen_gate["include_identity"] == before
                assert codegen_gate["rtc_tu_sha256"] == hashlib.sha256((ROOT / "transformer_engine/common/gemm/rtc/cutlass_grouped_gemm.cu").read_bytes()).hexdigest()
            development = a.verify_scope == "representative"
            if development:
                # 代表性/开发范围：只接受显式声明子集的回执，且回执自带 scope 标记；
                # 采集仅用于开发筛查，不声明全量正确性，也不能充当 24+2 正式验收。
                assert all(r.get("scope") == "representative-development" for r in checks), \
                    "representative 范围只接受 verify-representative 回执"
                assert all(r["status"] == "PASS" and r.get("formal_acceptance") is False
                           and r.get("full_regression_complete") is False for r in checks), checks
                assert all(r["collected_total"] > r["selected_count"] == r["counts"]["tests"] for r in checks), \
                    "代表性范围必须是当前收集集合的真子集"
                assert all(not any(r["counts"][k] for k in ("failures", "errors", "skipped")) for r in checks)
                selected = {n.split("::", 1)[1] for r in checks for n in r["selected_nodes"]}
                verified = {c["name"] for r in checks for c in r["testcases"]}
                assert verified == selected, "回执实际用例与显式清单不一致"
                expected = None
            else:
                assert all(r.get("scope") in (None, "full") for r in checks), \
                    "full 范围不接受代表性/开发回执"
                expected = set(checks[0]["collected_test_names"])
                assert expected and all(set(r["collected_test_names"]) == expected for r in checks)
                verified = {c["name"] for r in checks for c in r["testcases"]}
                assert verified == expected and all(r["status"] == "PASS" for r in checks)
            for path, check in zip(a.require_verify, checks):
                assert check["library_sha256"] == library_identity() and check["test_source_sha256"] == current_test_hash
                assert digests(Path(check["rtc_include"])) == before
                suites = ET.parse(path.parent / "junit.xml").getroot().findall("testsuite")
                counts = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
                junit_cases = [{"classname": c.get("classname"), "name": c.get("name")} for s in suites for c in s.findall("testcase")]
                assert counts == check["counts"] and junit_cases == check["testcases"]
                assert not any(counts[key] for key in ("failures", "errors", "skipped"))
                assert re.search(rf"\b{counts['tests']} passed\b", (path.parent / "pytest.log").read_text())
            record["verification"] = [str(p) for p in a.require_verify]
            record["verified_cases"] = len(verified)
            record["verification_scope"] = "representative" if development else "full"
            record["representative_verification"] = development
            cpu_dir = a.out / "cpu_gate"
            cpu_dir.mkdir()
            cpu_python = ROOT / ".smoke_venv18/bin/python"
            # 新环境未建历史 venv 时回落到当前解释器，CPU 门禁仍须真实跑满，不许跳过。
            if not cpu_python.is_file():
                cpu_python = Path(sys.executable)
            assert cpu_python.is_file()
            cpu_env = dict(env, PATH=str(cpu_python.parent) + os.pathsep + env["PATH"])
            command = [str(cpu_python), "-m", "pytest", "tests/pytorch/test_rtc_grouped_gemm_ci.py", "-q",
                       "--junitxml", str(cpu_dir / "junit.xml")]
            save(cpu_dir / "command.json", command)
            execute(command, cpu_dir / "pytest.log", env=cpu_env)
            suites = ET.parse(cpu_dir / "junit.xml").getroot().findall("testsuite")
            counts = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
            assert counts["tests"] >= 160 and not any(counts[k] for k in ("failures", "errors", "skipped"))
            record["cpu_gate"] = counts
            execute([sys.executable, script, "audit-build", "--out", str(a.out / "build_identity")],
                    a.out / "audit_build.log", env=env)
            assert json.loads((a.out / "build_identity/result.json").read_text())["status"] == "PASS"
            shutil.copytree(effective, frozen)
            assert before == digests(frozen) == digests(effective)
            record.update(include_identity=before, source_sha256=source_identity(), library_sha256=library_identity())
            snapshots = a.out / "source_snapshot"
            for p in source_paths:
                dest = snapshots / p.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dest)
            for p in library_paths:
                dest = a.out / "libraries" / p.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dest)
            save(a.out / "case_manifest.json", manifest)
            save(receipt, record)
        check_identity()
        sdk = subprocess.check_output(["acu", "--version"], text=True)
        compiler = subprocess.check_output(["nvcc", "--version"], text=True)
        sdk_version = (f"acu {re.search(r'version\s+(\S+)', sdk)[1]}, "
                       f"Data {re.search(r'Data version\s+(\S+)', sdk)[1]}, "
                       f"HGGC {re.search(r'release\s+([^,]+)', compiler)[1]}")
        sys.path.insert(0, "/usr/local/PPU_SDK/asight/extras/python")
        import acu_report
        for case in selected_manifest:
            case_id = case["id"]
            previous = record["cases"].get(case_id, {})
            if previous.get("status") == "PASS":
                assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in previous["artifacts"].items())
                continue
            root = a.out / a.nexpand_stage / f"ne4_k{a.nexpand_kmax}" / case_id
            root.mkdir(parents=True, exist_ok=True)
            attempt = root / f"attempt_{len(previous.get('attempts', [])):03d}"
            attempt.mkdir()
            entry = {"status": "RUNNING", "attempts": [*previous.get("attempts", []), str(attempt)],
                     "case": case, "reports": [], "samples": {"production": [], "contiguous": []}}
            if comparison:
                entry.update(scope=scope, formal_acceptance=False, performance_valid=False)
            record["cases"][case_id] = entry
            save(receipt, record)
            write_ledger()
            actual, resources, symbols = {}, {}, {}
            baseline_output = attempt / "production_output.pt"
            for arm in ("production", "contiguous"):
                check_device(f"{case_id}:actual:{arm}")
                check_identity()
                job = attempt / "actual" / arm
                job.mkdir(parents=True)
                command = command_case(case, arm, job / "rtc")
                command += ["--save" if arm == "production" else "--compare", str(baseline_output)]
                save(job / "command.json", command)
                log = execute(command, job / "verify.log", env=env, cwd=job, strict_contention=False)
                identity = json.loads((job / "rtc/packed_study.json").read_text())
                assert identity["correctness"] == "PASS" and identity["MNK"] == case["MNK"] and identity["splits"] == case["splits"]
                assert (identity["layout"], identity["dtype"], identity["beta"], identity["groups"]) == \
                    (case["layout"], case["dtype"], case["beta"], case["groups"])
                assert Path(identity["rtc_include"]).resolve() == frozen.resolve()
                for key, filename in (("core_library", "libtransformer_engine.so"),
                                      ("torch_library", library_paths[1].name)):
                    assert hashlib.sha256(Path(identity[key]).read_bytes()).hexdigest() == record["library_sha256"][filename]
                if arm == "contiguous":
                    assert identity["input_sha256"] == actual["production"]["input_sha256"]
                    assert identity["output_sha256"] == actual["production"]["output_sha256"]
                actual[arm] = identity
                symbol = identity["kernels"][0]
                symbols[arm] = symbol
                runtime = re.findall(r"kernel report: threads=(\d+).*?chosen_smem=(\d+) for " + re.escape(symbol), log)
                assert runtime and len(set(runtime)) == 1
                threads, smem = map(int, runtime[0])
                launch = launch_identity(log, arm, symbol, identity)
                assert launch["block"] == [threads, 1, 1] and launch["smem"] == smem
                entry.setdefault("launch_geometry", {})[arm] = launch
                if arm == "contiguous":
                    cfg = identity["config"]
                    assert re.search(re.escape(symbol) + rf" work={cfg['expected_work']} grid=\d+ smem={smem}", log)
                    nexpand_validate_runtime(log, symbol, cfg, case)
                dest = job / "rtc"
                cubin = dest / f"{symbol}.cubin"
                options = (dest / f"{symbol}.options.txt").read_text().splitlines()
                tm, tn, tk, stages = map(int, re.search(r"_tb(\d+)x(\d+)x(\d+)_s(\d+)_", symbol).groups())
                ta, tb = {"TN": (0, 1), "NN": (0, 0), "NT": (1, 0)}[case["layout"]]
                macros = {"NVTE_GG_CONTIGUOUS_MODE": identity["config"]["mode"], "NVTE_GG_TRANS_A": ta,
                          "NVTE_GG_TRANS_B": tb, "NVTE_GG_OUT_FP32": int(case["dtype"] == "fp32"),
                          "NVTE_GG_WITH_C": int(arm == "production" or bool(case["beta"])),
                          "NVTE_GG_TILE_M": tm, "NVTE_GG_TILE_N": tn, "NVTE_GG_TILE_K": tk, "NVTE_GG_STAGES": stages}
                for name, value in macros.items():
                    assert [o for o in options if o.startswith(f"-D{name}=")] == [f"-D{name}={value}"]
                expected_kmax = a.nexpand_kmax if arm == "contiguous" and identity["config"]["effective_ne"] > 1 else 512
                assert [o for o in options if o.startswith("-DNVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX=")] == \
                    ([f"-DNVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX={expected_kmax}"] if expected_kmax != 512 else [])
                execute([sys.executable, repro, "--resources", str(dest), "--resource-smem", str(smem),
                         "--resource-threads", str(threads)], dest / "driver.log", env=env)
                resource = json.loads((dest / f"{symbol}.driver_resources.json").read_text())
                assert resource["static_smem_bytes"] == resource["local_bytes_per_thread"] == 0
                assert resource["active_blocks_per_cu"] > 0
                resources[arm] = resource
                if anchors and arm == "production":
                    anchor = anchors[case_id]
                    assert symbol == anchor["symbol"] and identity["input_sha256"] == anchor["input_sha256"]
                    assert identity["output_sha256"] == anchor["output_sha256"]
                    assert rtc_machine_identity(cubin) == anchor["machine_identity"], "production机器码漂移"
                    assert {k: v for k, v in resource.items() if k != "cubin"} == \
                        {k: v for k, v in anchor["resources"].items() if k != "cubin"}, "production资源漂移"
                    options = (dest / f"{symbol}.options.txt").read_text().splitlines()
                    assert "-DNVTE_GG_CONTIGUOUS_MODE=0" in options
                    assert not any("CONTIGUOUS_N_EXPAND_K_MAX" in o for o in options)
                    entry["production_anchor_identity"] = "PASS"
                if anchors and arm == "contiguous" and a.nexpand_stage == "baseline":
                    old = baseline_record["cases"][case_id]
                    old_symbol = old["actual"]["contiguous"]["kernels"][0]
                    old_cubin = Path(old["resources"]["contiguous"]["cubin"])
                    assert symbol == old_symbol and rtc_machine_identity(cubin) == rtc_machine_identity(old_cubin)
                    assert identity["input_sha256"] == old["actual"]["contiguous"]["input_sha256"]
                    assert identity["output_sha256"] == old["actual"]["contiguous"]["output_sha256"]
                    assert {k: v for k, v in resource.items() if k != "cubin"} == \
                        {k: v for k, v in old["resources"]["contiguous"].items() if k != "cubin"}
                    entry["current_anchor_identity"] = "PASS"
                execute(["hgobjdump", "--dump-isa", str(cubin)], dest / "isa.txt")
                text = execute(["hgobjdump", "--dump-resource-usage=all", str(cubin)], dest / "resources.txt")
                assert re.findall(r"STACK SIZE:(\d+)", text) == ["0"]
                annotated = attempt / "annotated_rtc" / arm
                image = annotate_native_rtc(cubin, annotated, frozen, symbol)
                assert rtc_machine_identity(image) == rtc_machine_identity(cubin)
                execute([sys.executable, repro, "--resources", str(annotated), "--resource-smem", str(smem),
                         "--resource-threads", str(threads)], annotated / "driver.log", env=env)
                annotated_resource = json.loads((annotated / f"{symbol}.driver_resources.json").read_text())
                assert {k: v for k, v in resource.items() if k != "cubin"} == {k: v for k, v in annotated_resource.items() if k != "cubin"}
                save(annotated / "machine_identity.json", rtc_machine_identity(image))
            entry.update(actual=actual, resources=resources)
            save(receipt, record)

            def profiled(arm, label, full):
                check_device(f"{case_id}:before:{label}")
                check_identity()
                job = attempt / label
                job.mkdir()
                raw_dir = attempt / "remote-results"
                raw_dir.mkdir(exist_ok=True)
                report = raw_dir / f"{label}.acurep"
                annotated = attempt / "annotated_rtc" / arm
                symbol = symbols[arm]
                command = ["acu", "--set", "full"] if full else ["acu", "--section", "LaunchStats"]
                command += ["--import-source", "yes", "-k", "regex:^" + re.escape(symbol) + "$", "-s", "4", "-c", "3",
                            "--kill=no", "--check-exit-code=yes", "-o", str(report),
                            *command_case(case, arm, job / "rtc"), "--packed-study-profile", "--iters", "3",
                            "--packed-study-cubins", str(annotated), "--compare", str(baseline_output)]
                save(job / "command.json", {"command": command, "sdk_version": sdk_version,
                    "timestamp": datetime.now(timezone.utc).isoformat()})
                try:
                    log = execute(command, job / "profile.log", env=env, cwd=job, strict_contention=False)
                finally:
                    log = (job / "profile.log").read_text() if (job / "profile.log").is_file() else ""
                    try:
                        foreign, target = foreign_contenders(log)
                    except AssertionError:
                        foreign, target = [], set()
                    record["profiler_contention"][f"{case_id}:{attempt.name}:{label}"] = {
                        "foreign_pids": foreign, "target_pids": sorted(target), "target_identity_valid": bool(target)}
                    record["contention"] |= bool(foreign)
                    save(receipt, record)
                assert target and not foreign, (target, foreign)
                identity = json.loads((job / "rtc/packed_study.json").read_text())
                loaded = annotated / f"{symbol}.cubin"
                assert f"loaded prebuilt cubin {loaded} " in log
                assert identity["loaded_cubin_sha256"] == hashlib.sha256(loaded.read_bytes()).hexdigest()
                assert identity["input_sha256"] == actual[arm]["input_sha256"]
                assert identity["output_sha256"] == actual[arm]["output_sha256"]
                assert identity["correctness"] == "PASS" and identity["config"] == actual[arm]["config"]
                assert identity["per_group_MNK"] == actual[arm]["per_group_MNK"]
                assert identity["MNK"] == case["MNK"] and identity["splits"] == case["splits"]
                assert Path(identity["rtc_include"]).resolve() == frozen.resolve()
                assert launch_identity(log, arm, symbol, identity) == entry["launch_geometry"][arm]
                for key, filename in (("core_library", "libtransformer_engine.so"), ("torch_library", library_paths[1].name)):
                    assert hashlib.sha256(Path(identity[key]).read_bytes()).hexdigest() == record["library_sha256"][filename]
                assert rtc_machine_identity(job / "rtc" / loaded.name) == rtc_machine_identity(loaded)
                check_device(f"{case_id}:after:{label}")
                context = acu_report.load_report(str(report))
                report_ranges = [context.range_by_idx(i) for i in range(context.num_ranges())]
                actions = [group.action_by_idx(j) for group in report_ranges for j in range(group.num_actions())]
                assert len(actions) == 3 and all(action.name() == symbol for action in actions), "Actual ACU action name mismatch"
                sources = []
                for action in actions:
                    if comparison:
                        sources.append(fulltile_source_identity(action, arm, frozen,
                            attempt / "actual" / arm / "rtc" / f"{symbol}.cu", annotated))
                    else:
                        embedded = action.source_files()
                        assert embedded and all(embedded.values())
                        required = ["nvte_contiguous_grouped.hpp" if arm == "contiguous" else "ppu_aiu_gemm_array_group.hpp",
                                    "ppu_mma_aiu_multistage_batch_array.hpp", "ppu_epilogue_vectorized_evt.hpp"]
                        for name in required:
                            content = next(v for p, v in embedded.items() if Path(p).name == name)
                            assert content.splitlines() == next(frozen.rglob(name)).read_text().splitlines()
                        sources.append({p: hashlib.sha256(v.encode()).hexdigest() for p, v in embedded.items()})
                save(job / "embedded_sources.json", sources)
                if not full:
                    return
                execute(["acu", "-i", str(report), "--stats-rule", "ExportMetricsByKernel", "--stats-rule-output", str(job / "full.csv")],
                        job / "metrics_export.log", env=env)
                rows = list(csv.DictReader(io.StringIO((job / "full.csv").read_text())))
                assert len(rows) == 3 and {r["Kernel Name"] for r in rows} == {symbol}
                if anchors and arm == "production":
                    expected_launch = anchors[case_id]["launches"][0]
                    assert all(all(row[key] == expected_launch[key] for key in ("Block Size", "Grid Size")) for row in rows), "production launch几何漂移"
                cases_dir = attempt / "cases"
                cases_dir.mkdir(exist_ok=True)
                sass = cases_dir / f"{label}.sass.csv"
                execute(["acu", "-i", str(report), "--page", "source", "--csv-file", str(sass)], job / "sass_export.log", env=env)
                sass_rows = list(csv.DictReader(io.StringIO(sass.read_text())))
                assert any(float(r.get("Instructions Executed", "0").replace(",", "") or 0) > 0 for r in sass_rows)
                if comparison:
                    dynamic = fulltile_dynamic_work(sass_rows, actions, symbol,
                        attempt / "actual" / arm / "rtc" / "isa.txt")
                    assert len({row["Range ID"] for row in rows}) == 3
                    assert [row["Range ID"] for row in rows] == [str(action.capture_sequence_id()) for action in actions]
                    expected_mma = sum(m * n * k for m, n, k in actual[arm]["per_group_MNK"]) // 4096
                    expected_store = sum(m * n for m, n, _ in actual[arm]["per_group_MNK"]) * (4 if case["dtype"] == "fp32" else 2)
                metadata = {"Range ID", "Kernel Name", "Kernel Mangled Name", "Kernel Demangled Name", "Block Size", "Grid Size", "Device"}
                output = []
                for row, action, source_hashes in zip(rows, actions, sources):
                    def number(value):
                        try:
                            return float(value.replace(",", ""))
                        except (ValueError, AttributeError):
                            return value
                    raw = {k: number(v) for k, v in row.items() if k not in metadata}
                    assert set(raw) == set(action.metrics())
                    assert raw[METRIC] > 0 and raw["ppu__time_duration.sum"] > 0
                    assert abs(raw[METRIC] - action.metric_by_name(METRIC).as_double()) < 1e-9
                    assert abs(raw["ppu__time_duration.sum"] - action.metric_by_name("ppu__time_duration.sum").as_double()) < 1e-9
                    launch = entry["launch_geometry"][arm]
                    for key, dimensions in (("Grid Size", launch["grid"]), ("Block Size", launch["block"])):
                        assert [int(v) for v in re.findall(r"\d+", row[key])] == dimensions, "ACU/host launch geometry mismatch"
                    for kind in ("grid", "block"):
                        assert [raw[f"launch__{kind}_dim_{axis}"] for axis in ("x", "y", "z")] == launch[kind]
                    output.append({"device": "PPU-890", "kernel_name": symbol,
                        "kernel_mangled_name": row.get("Kernel Mangled Name"), "profiler": "ACU",
                        "sdk_version": sdk_version, "timestamp": datetime.now(timezone.utc).isoformat(),
                        "process_id": next(iter(target)), "host_name": socket.gethostname(),
                        "device_name": row["Device"], "gpu_uuid": record["device"]["uuid"],
                        "case_id": case_id, "case_digest": manifest_digest, "arm": arm,
                        "raw_metrics": raw, "source_sha256": source_hashes, "foreign_pids": foreign,
                        "diagnostic": False, "performance_valid": not comparison})
                    if comparison:
                        counts = dynamic[row["Range ID"]]
                        assert raw["cu__inst_executed_pipe_tensor_bf16.sum"] == action.metric_by_name("cu__inst_executed_pipe_tensor_bf16.sum").as_double() == expected_mma, "Dynamic tensor work differs from report/MNK"
                        assert counts["store_bytes"] == expected_store, "Dynamic output-store bytes differ from MNK/dtype"
                        output[-1].update(scope=scope, formal_acceptance=False, launch_geometry=launch,
                            capture_sequence_id=action.capture_sequence_id(),
                            dynamic_work={**counts, "hardware_mma": raw["cu__inst_executed_pipe_tensor_bf16.sum"],
                                "expected_mma": expected_mma, "expected_store_bytes": expected_store,
                                "sass_mma_matches_hardware": counts["sass_mma"] == expected_mma})
                metrics = cases_dir / f"{label}.metrics.json"
                save(metrics, output)
                entry["samples"][arm].append([r["raw_metrics"][METRIC] for r in output])
                entry["reports"].append({"arm": arm, "report": str(report), "metrics": str(metrics), "sass": str(sass),
                                         "duration_ns": [r["raw_metrics"]["ppu__time_duration.sum"] for r in output],
                                         "launch": [{key: row[key] for key in ("Block Size", "Grid Size", "Device")} for row in rows]})
                if comparison:
                    entry["reports"][-1].update(scope=scope, formal_acceptance=False, performance_valid=False,
                                                round=int(label.rsplit("_r", 1)[1]), action_names=[action.name() for action in actions],
                                                launch_geometry=entry["launch_geometry"][arm], actions=3)
                save(receipt, record)
            for arm in ("production", "contiguous"):
                profiled(arm, f"probe_{arm}", False)
            for repeat in range(3):
                order = ("production", "contiguous") if (case["index"] + repeat) % 2 == 0 else ("contiguous", "production")
                entry.setdefault("order", []).append(list(order))
                for arm in order:
                    profiled(arm, f"{arm}_full_r{repeat}", True)
            summary = {arm: nexpand_statistics(values) for arm, values in entry["samples"].items()}
            p, x = summary["production"]["median"], summary["contiguous"]["median"]
            durations = {arm: nexpand_duration_statistics([r["duration_ns"] for r in entry["reports"] if r["arm"] == arm])
                         for arm in ("production", "contiguous")}
            summary.update(delta_pp=x - p, relative_pct=100 * (x / p - 1), duration_ns=durations,
                           speedup=durations["production"]["median"] / durations["contiguous"]["median"],
                           launch_geometry=entry["launch_geometry"],
                           work_basis={"production": "actual per-group shapes and RTC tile", "contiguous": "host launch receipt"},
                           config=actual["contiguous"]["config"], resources=resources,
                           reports=entry["reports"], input_sha256=actual["production"]["input_sha256"])
            if comparison:
                summary.update(scope=scope, formal_acceptance=False, performance_valid=False)
            entry.update(status="PASS", summary=summary, artifacts={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(attempt.rglob("*")) if p.is_file() and (comparison or p.suffix in (".json", ".csv", ".acurep", ".cubin", ".cu", ".txt"))})
            record["core_24_complete"] = all(record["cases"].get(c["id"], {}).get("status") == "PASS" for c in manifest if not c["control"])
            save(receipt, record)
            write_ledger()
            print(json.dumps({"case_complete": case_id, "delta_pp": x - p, "relative_pct": summary["relative_pct"],
                              "core_24_complete": record["core_24_complete"]}), flush=True)
        if comparison:
            check_identity()
            assert not record["contention"] and set(record["cases"]) == set(selected_ids)
            for case in selected_manifest:
                entry = record["cases"][case["id"]]
                assert entry["status"] == "PASS" and entry["case"] == case and len(entry["reports"]) == 6
                assert entry["scope"] == scope and entry["formal_acceptance"] is False and entry["performance_valid"] is False
                assert entry["order"] == [["production", "contiguous"] if (case["index"] + r) % 2 == 0
                                          else ["contiguous", "production"] for r in range(3)]
                assert [(r["round"], r["arm"]) for r in entry["reports"]] == [(i, arm) for i, order in enumerate(entry["order"]) for arm in order]
                assert entry["summary"]["scope"] == scope and entry["summary"]["formal_acceptance"] is False and entry["summary"]["performance_valid"] is False
                assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in entry["artifacts"].items())
                attempt = Path(entry["attempts"][-1])
                for arm in ("production", "contiguous"):
                    actual = entry["actual"][arm]
                    assert json.loads((attempt / "actual" / arm / "rtc/packed_study.json").read_text()) == actual
                    assert actual["correctness"] == "PASS"
                    assert actual["input_sha256"] == entry["actual"]["production"]["input_sha256"]
                    assert actual["output_sha256"] == entry["actual"]["production"]["output_sha256"]
                    reports = [r for r in entry["reports"] if r["arm"] == arm]
                    assert len(reports) == 3 and [r["round"] for r in reports] == [0, 1, 2]
                    samples, durations = [], []
                    symbol = actual["kernels"][0]
                    for r in reports:
                        assert r["actions"] == 3 and r["action_names"] == [symbol] * 3
                        assert r["scope"] == scope and r["performance_valid"] is False and r["formal_acceptance"] is False
                        context = acu_report.load_report(r["report"])
                        actions = [context.range_by_idx(i).action_by_idx(j) for i in range(context.num_ranges())
                                   for j in range(context.range_by_idx(i).num_actions())]
                        assert len(actions) == 3 and all(action.name() == symbol for action in actions)
                        metrics = json.loads(Path(r["metrics"]).read_text())
                        assert len(metrics) == 3
                        dynamic = fulltile_dynamic_work(list(csv.DictReader(io.StringIO(Path(r["sass"]).read_text()))),
                            actions, symbol, attempt / "actual" / arm / "rtc" / "isa.txt")
                        assert [metric["capture_sequence_id"] for metric in metrics] == [action.capture_sequence_id() for action in actions]
                        for metric, action in zip(metrics, actions):
                            assert metric["scope"] == scope and metric["formal_acceptance"] is False and metric["performance_valid"] is False
                            assert metric["gpu_uuid"] == record["device"]["uuid"] and not metric["foreign_pids"]
                            assert metric["case_id"] == case["id"] and metric["case_digest"] == manifest_digest and metric["arm"] == arm
                            assert metric["kernel_name"] == symbol and set(metric["raw_metrics"]) == set(action.metrics())
                            work = metric["dynamic_work"]
                            assert work["hardware_mma"] == work["expected_mma"] == sum(m * n * k for m, n, k in actual["per_group_MNK"]) // 4096
                            assert work["hardware_mma"] == metric["raw_metrics"]["cu__inst_executed_pipe_tensor_bf16.sum"]
                            assert work["store_bytes"] == work["expected_store_bytes"] == sum(m * n for m, n, _ in actual["per_group_MNK"]) * (4 if case["dtype"] == "fp32" else 2)
                            assert all(work[key] == value for key, value in dynamic[str(action.capture_sequence_id())].items())
                            assert work["sass_mma_matches_hardware"] == (work["sass_mma"] == work["hardware_mma"])
                            assert metric["source_sha256"] == fulltile_source_identity(action, arm, frozen,
                                attempt / "actual" / arm / "rtc" / f"{symbol}.cu", attempt / "annotated_rtc" / arm)
                            for key in (METRIC, "ppu__time_duration.sum", "cu__inst_executed_pipe_tensor_bf16.sum"):
                                assert abs(metric["raw_metrics"][key] - action.metric_by_name(key).as_double()) < 1e-9
                        assert r["launch_geometry"] == entry["launch_geometry"][arm]
                        assert all([int(v) for v in re.findall(r"\d+", launch[key])] == entry["launch_geometry"][arm][kind]
                                   for launch in r["launch"] for key, kind in (("Grid Size", "grid"), ("Block Size", "block")))
                        samples.append([m["raw_metrics"][METRIC] for m in metrics])
                        durations.append([m["raw_metrics"]["ppu__time_duration.sum"] for m in metrics])
                        assert durations[-1] == r["duration_ns"]
                    assert nexpand_statistics(samples) == entry["summary"][arm]
                    assert nexpand_duration_statistics(durations) == entry["summary"]["duration_ns"][arm]
                summary = entry["summary"]
                assert summary["speedup"] == summary["duration_ns"]["production"]["median"] / summary["duration_ns"]["contiguous"]["median"]
                assert summary["delta_pp"] == summary["contiguous"]["median"] - summary["production"]["median"]
                assert summary["relative_pct"] == 100 * (summary["contiguous"]["median"] / summary["production"]["median"] - 1)
            record["core_24_complete"] = all(record["cases"].get(c["id"], {}).get("status") == "PASS" for c in manifest if not c["control"])
            record.update(status="COMPARISON-PASS", measurement_valid=True, performance_valid=False,
                          manifest_complete=all(record["cases"].get(c["id"], {}).get("status") == "PASS" for c in manifest),
                          selected_pairs=len(selected_ids), full_reports=6 * len(selected_ids), full_actions=18 * len(selected_ids),
                          scope=scope, formal_acceptance=False)
        else:
            assert len(record["cases"]) == len(manifest) and all(e["status"] == "PASS" for e in record["cases"].values())
            expected_hits = {512: 2, 1024: 6, 2048: 8}[a.nexpand_kmax]
            assert sum(record["cases"][c["id"]]["summary"]["config"]["effective_ne"] == 4
                       for c in manifest if not c["control"]) == expected_hits
            assert sum(record["cases"][c["id"]]["summary"]["config"]["effective_ne"] == 4
                       for c in manifest if c["control"]) == (2 if a.nexpand_kmax == 2048 else 0)
        if not comparison and record.get("verification_scope") == "representative":
            # 代表性/开发批次只标开发通过：正式门禁入口(nexpand_validate_capture/audit-nexpand)会拒绝，
            # 保留同卡配对数值供开发筛查，但不声明正式收益或全量正确性。
            record.update(status="DEV-PASS", performance_valid=False, formal_acceptance=False,
                          dev_measurement_valid=True)
        elif not comparison:
            record.update(status="PASS", performance_valid=True, formal_acceptance=True)
    except Exception as error:
        record.update(status="NO_DEVICE_AVAILABLE" if "NO_DEVICE_AVAILABLE" in str(error) else "FAIL", error=str(error))
        if "entry" in locals() and entry["status"] == "RUNNING":
            entry.update(status="FAIL", error=str(error))
        raise
    finally:
        save(receipt, record)
        write_ledger()
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
        # 拒绝占用卡之前先保存现场，失败回执也必须准确记录争用。
        record["contention"] |= bool(live)
        record["occupancy_snapshots"].append({"stage": stage, "device": gpu, "live": live,
            "timestamp": datetime.now(timezone.utc).isoformat(), "all_devices": devices, "all_apps": apps})
        save(receipt, record)
        # 诊断仅放宽正式回归门禁，不允许在有外部 live 进程的设备上采集。
        assert not live, (stage, live, apps)
        assert float(gpu["memory.total"]) - float(gpu["memory.used"]) > 8192, "剩余显存不足"
        return gpu
    try:
        try:
            device = check_device("start")
        except AssertionError:
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
        # 按当前源码的完整收集集合验收，新增测试不能被历史固定数量漏掉。
        collections = [set(r.get("collected_test_names", [])) for r in checks]
        expected_cases = collections[0]
        full_regression = (bool(expected_cases) and all(c == expected_cases for c in collections)
                           and {name for _, name in cases_verified} == expected_cases
                           and len(cases_verified) == len(expected_cases))
        if not a.diagnostic:
            assert full_regression, "必须逐项覆盖当前完整收集集合，不能沿用历史数量"
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
        if a.smallk_bn256:
            wide_tests = {name for name in expected_cases if "test_contiguous_bn256_" in name}
            assert wide_tests and wide_tests <= {name for _, name in cases_verified}, \
                "BN256采集必须先完整通过当前宽BN正确性集合"
            arms += ["contiguous256", "nexpand256"]
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
            if identity["arms"][arm]["contiguous"] == "1":
                wide = arm in ("contiguous256", "nexpand256")
                assert (threads, smem) == ((512, 196736) if wide else (256, 147584)), (arm, threads, smem)
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
            try:
                log = execute(cmd, job / "profile.log", env=env, cwd=job, strict_contention=False)
            finally:
                # 非零退出也保留 profiler 的外部 pid，不能被 execute 的断言跳过。
                log = (job / "profile.log").read_text() if (job / "profile.log").is_file() else ""
                try:
                    foreign, target = foreign_contenders(log)
                except AssertionError:
                    foreign, target = [], set()
                record["profiler_contention"][label] = {"foreign_pids": foreign,
                    "target_pids": sorted(target), "target_identity_valid": bool(target),
                    "warnings": log.lower().count("may affect")}
                record["contention"] |= bool(foreign)
                save(receipt, record)
            assert target, "未取得被测进程身份，不接受采集结果"
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
            expected = ("nvte_contiguous_grouped.hpp" if arm in ("contiguous", "nexpand", "contiguous256", "nexpand256")
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
                expected = ("nvte_contiguous_grouped.hpp" if arm in ("contiguous", "nexpand", "contiguous256", "nexpand256")
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
                        # 按当前 ABI 找唯一 extension，不把 Python 版本写死；
                        # 兼容 extension 直接放 lib.*/ 与放 lib.*/transformer_engine/ 两种布局。
                        candidates = list((ROOT / "build").glob(f"lib.*/**/{loaded.name}"))
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
                collection_cmd = [sys.executable, "-m", "pytest", str(test_source), "--collect-only", "-q"]
                save(a.out / "collection_command.json", collection_cmd)
                collected = execute(collection_cmd, a.out / "collection.log", env=env,
                                    strict_contention=False)
                record["collected_test_names"] = [line.split("::", 1)[1] for line in collected.splitlines()
                                                  if line.startswith("tests/pytorch/test_grouped_gemm_rtc.py::")]
                assert record["collected_test_names"], "未取得完整测试收集集合"
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

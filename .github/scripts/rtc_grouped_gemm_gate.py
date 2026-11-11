#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.
"""RTC GroupGemm 的逐 case 测量、配对性能判定和报告；控制面不依赖 torch。"""

import argparse
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "benchmarks/gemm/grouped_gemm_rtc_ci_cases.txt"
DEFAULT_REPLAY = ROOT / "benchmarks/gemm/replay_tag.py"
# 与 workflow 的 pull_request.paths 保持一致；不泛选所有 util/csrc/3rdparty。
SCOPE_PATHS = (
    "transformer_engine/common/gemm/cutlass_grouped_gemm*",
    "transformer_engine/common/gemm/rtc/**",
    "transformer_engine/common/gemm/actlize_overlay/**",
    "transformer_engine/common/gemm/ppu_adaptive_tile.hpp",
    "transformer_engine/common/gemm/cublaslt_grouped_gemm.cu",
    "transformer_engine/common/gemm/config.cpp",
    "transformer_engine/common/gemm/config.h",
    "transformer_engine/common/gemm/te_profiling_interface.h",
    "transformer_engine/common/util/rtc.cpp",
    "transformer_engine/common/util/rtc.h",
    "transformer_engine/common/include/transformer_engine/gemm.h",
    "transformer_engine/common/CMakeLists.txt",
    "transformer_engine/pytorch/cpp_extensions/gemm.py",
    "transformer_engine/pytorch/csrc/extensions/gemm.cpp",
    "transformer_engine/pytorch/module/grouped_linear.py",
    "3rdparty/actlize",
    "3rdparty/actlize/**",
    "3rdparty/cudafy-for-sail",
    "3rdparty/cudafy-for-sail/**",
    "include/cutlass3/**",
    "include/cutlass3_overlay/**",
    "build_tools/build_ext.py",
    "build_tools/pytorch.py",
    "build_tools/utils.py",
    "setup.py",
    "pyproject.toml",
    "MANIFEST.in",
    ".gitmodules",
    "benchmarks/gemm/replay_tag.py",
    "benchmarks/gemm/grouped_gemm_rtc_ci_cases.txt",
    "tests/pytorch/test_grouped*",
    "tests/pytorch/test_rtc_grouped_gemm_ci.py",
    "qa/L0_pytorch_grouped_gemm_rtc/**",
    ".github/scripts/rtc_grouped_gemm_gate.py",
    ".github/workflows/ppu-890-rtc-grouped-gemm-gate.yml",
)
SCOPE = re.compile("^(?:" + "|".join(fnmatch.translate(path) for path in SCOPE_PATHS) + ")")


def policy_for(subjects, labels, manual=False):
    """与参考 CI 一致：手动/perf/refactor 阻断退化，其他变更只报告性能。"""
    if manual:
        return "enforced"
    for policy in ("perf", "refactor"):
        pattern = rf"(?im)(\[{policy}(?::[^\]]*)?\]|^{policy}(\([^)]*\))?!?:)"
        if re.search(pattern, subjects) or any(label.lower() == policy for label in labels):
            return policy
    return "advisory"


def load_cases(path):
    cases = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("case 列表不能为空或包含重复项")
    return cases


def measure(args):
    cases = load_cases(args.cases)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results, errors = [], []
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    # base/head 的缓存完全隔离；同一 arm 的 sweep 复用该 arm 自己的 cubin。
    env["NVTE_GG_RTC_CACHE_DIR"] = str(out.parent / "rtc-cache")
    for index, tag in enumerate(cases, 1):
        if args.only_case and index != args.only_case:
            continue
        log = out / f"case-{index:02d}.log"
        result_file = out / f"case-{index:02d}.json"
        print(f"[RTC CI] case {index}/{len(cases)}: {tag}", flush=True)
        command = [
            args.python,
            str(Path(args.replay).resolve()),
            "--tag",
            tag,
            "--ci-output",
            str(result_file),
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--profile-samples",
            str(args.samples),
        ]
        try:
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout
                )
            if completed.returncode:
                raise RuntimeError(f"worker 退出码 {completed.returncode}；详见 {log.name}")
            # tag 含真实尺寸、backend、tile、stages，不能仅验证 override 配置。
            live = re.findall(r"\[TE\] --format=[^\r\n]+", log.read_text(encoding="utf-8"))
            if not live or any(line != tag for line in live):
                raise RuntimeError(f"实际 launch tag 与 case 不同或缺失；详见 {log.name}")
            result = json.loads(result_file.read_text(encoding="utf-8"))
            if result["tag"] != tag:
                raise RuntimeError("结果 tag 不匹配")
            expected_prefix = os.environ.get("RTC_CI_EXPECTED_PREFIX")
            if expected_prefix and not Path(result["te_path"]).is_relative_to(expected_prefix):
                raise RuntimeError(f"导入了错误的 TE：{result['te_path']}")
            result["id"] = index
            results.append(result)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            subprocess.TimeoutExpired,
        ) as error:
            errors.append({"id": index, "tag": tag, "error": str(error)})
            print(f"[RTC CI] ERROR: {error}", flush=True)
        # 每个 case 发布检查点，pod 超时也保留已完成的测试和原始日志。
        payload = {
            "results": results,
            "errors": errors,
            "case_digest": hashlib.sha256("\n".join(cases).encode()).hexdigest(),
            "settings": {"iters": args.iters, "samples": args.samples, "warmup": args.warmup},
        }
        (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 1 if errors else 0


def compare(cases, measurements, tolerance):
    """每个 case 取配对 sweep 的速度比中位数，禁止汇总增益掩盖单项退化。"""
    if not 0 <= tolerance < 1 or not measurements:
        raise ValueError("需要配对测量，且性能容差必须在 [0, 1) 内")
    rows = []
    digest = hashlib.sha256("\n".join(cases).encode()).hexdigest()
    for base, head in measurements:
        for arm in (base, head):
            if arm.get("errors") or arm.get("case_digest") != digest:
                raise ValueError(f"测量失败或 case 列表不一致：{arm.get('errors')}")
            if [r.get("id") for r in arm["results"]] != list(range(1, len(cases) + 1)):
                raise ValueError("结果缺失、重复或顺序错误")
        if base.get("settings") != head.get("settings"):
            raise ValueError("base/head 的测量设置不一致")
    for index, tag in enumerate(cases):
        base_times, head_times, ratios = [], [], []
        for base, head in measurements:
            b, h = base["results"][index], head["results"][index]
            if b["tag"] != tag or h["tag"] != tag:
                raise ValueError("case tag 不一致")
            for key in ("device", "capability", "torch"):
                if b.get(key) != h.get(key):
                    raise ValueError(f"base/head 运行环境不一致：{key}")
            bt, ht = b["median_us"], h["median_us"]
            if not all(math.isfinite(t) and t > 0 for t in (bt, ht)):
                raise ValueError("无效的时间测量")
            base_times.append(bt)
            head_times.append(ht)
            ratios.append(ht / bt)
        ratio = statistics.median(ratios)
        rows.append(
            {
                "id": index + 1,
                "tag": tag,
                "base_us": statistics.median(base_times),
                "head_us": statistics.median(head_times),
                "latency_ratio": ratio,
                "speedup": 1 / ratio,
                "regression": ratio > 1 + tolerance,
            }
        )
    return rows


def plot_report(gate, output):
    """无显示环境绘制延迟及配对变化；不把独立中位数之比冒充配对比率。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = gate["rows"]
    figure, axes = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True)
    try:
        figure.suptitle(
            f"RTC GroupGemm (890P) | {gate['verdict']} | "
            f"{gate['base_sha'][:8]} -> {gate['head_sha'][:8]} | {gate['sweeps']} AB/BA sweeps"
        )
        if gate["error"]:
            # 缺测时不得绘制空柱并暗示性能通过。
            for axis in axes:
                axis.set_axis_off()
            axes[0].text(0.5, 0.5, "ERROR: no valid paired measurements", ha="center", fontsize=18)
            axes[1].text(0.5, 0.5, "See the report and case/build logs.", ha="center", fontsize=14)
        else:
            ids = [row["id"] for row in rows]
            axes[0].bar([i - 0.2 for i in ids], [r["base_us"] for r in rows], 0.4, label="base")
            axes[0].bar([i + 0.2 for i in ids], [r["head_us"] for r in rows], 0.4, label="head")
            axes[0].set_ylabel("RTC kernel latency (μs)")
            axes[0].legend()
            changes = [(row["latency_ratio"] - 1) * 100 for row in rows]
            colors = [
                "#d62728" if r["regression"] else "#2ca02c" if c < 0 else "#7f7f7f"
                for r, c in zip(rows, changes)
            ]
            axes[1].bar(ids, changes, color=colors)
            axes[1].axhline(0, color="black", linewidth=0.8)
            axes[1].axhline(
                gate["tolerance"] * 100,
                color="#d62728",
                linestyle="--",
                label=f"+{gate['tolerance']:.0%} tolerance",
            )
            axes[1].set_ylabel("Paired head/base latency change (%)")
            axes[1].legend()
            for axis in axes:
                axis.set_xticks(ids, [f"{i:02d}" for i in ids])
                axis.set_xlabel("case (lower latency is better)")
                axis.grid(axis="y", alpha=0.25)
                axis.set_axisbelow(True)
        figure.savefig(output, dpi=140)
    finally:
        plt.close(figure)


def report(args):
    out = Path(args.output)
    # NAS 由 pod 写入；runner 只读测量，将报告写到自己的可写目录。
    source = Path(getattr(args, "input", None) or args.output)
    out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.cases)
    rows, error = [], None
    try:
        measurements = []
        for sweep in range(args.sweeps):
            pair = []
            for arm in ("base", "head"):
                path = source / arm / f"sweep-{sweep}" / "results.json"
                pair.append(json.loads(path.read_text(encoding="utf-8")))
            measurements.append(pair)
        rows = compare(cases, measurements, args.tolerance)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        error = str(exc)
    regressions = [r for r in rows if r["regression"]]
    verdict = "ERROR" if error else "REGRESSION" if regressions else "PASS"
    gate = {
        "verdict": verdict,
        "policy": args.policy,
        "tolerance": args.tolerance,
        "base_sha": args.base_sha,
        "head_sha": args.head_sha,
        "sweeps": args.sweeps,
        "rows": rows,
        "error": error,
    }
    (out / "gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    lines = [
        "### TE RTC GroupGemm 性能门禁（890P）",
        f"结论：**{verdict}**；策略：`{args.policy}`；逐 case 延迟容差：{args.tolerance:.0%}。",
        f"base `{args.base_sha}` → head `{args.head_sha}`；{args.sweeps} 轮配对 sweep。",
        "计时只统计 RTC GEMM kernel；JIT、FP32 对拍、CPU dispatch 不计入。",
        "数值、backend、实际 tag/tile/stages 或测量缺失始终使检查失败。",
        "",
    ]
    if error:
        lines += [f"测量未完成或校验失败：`{error}`", "详见 build 日志和逐 case 日志。"]
    else:
        geometric_speedup = math.exp(statistics.mean(math.log(r["speedup"]) for r in rows))
        lines += [
            f"几何平均加速比：{geometric_speedup:.3f}x；退化：{len(regressions)}/{len(rows)}。",
            "",
            "| case | base μs | head μs | 加速比 | 结果 |",
            "|---|---:|---:|---:|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['id']:02d} | {row['base_us']:.2f} | {row['head_us']:.2f} | "
                f"{row['speedup']:.3f}x | {'退化' if row['regression'] else '通过'} |"
            )
        lines += [
            "",
            "<details><summary>完整 case 列表</summary>",
            "",
            "```text",
            *cases,
            "```",
            "</details>",
        ]
    (out / "gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if getattr(args, "plot", False):
        try:
            plot_report(gate, out / "gate.png")
        except Exception as exc:
            # 绘图仅用于展示；文字报告和原有门禁退出码仍以实际测量为准。
            print(f"::warning::RTC GroupGemm 图表生成失败：{exc}", file=sys.stderr)
    print(
        f"::notice::RTC GroupGemm: {verdict}, policy={args.policy}, regressions={len(regressions)}"
    )
    if error:
        return 3
    if regressions and args.policy != "advisory":
        return 2
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("measure")
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--replay", default=str(DEFAULT_REPLAY))
    run.add_argument("--output", required=True)
    run.add_argument("--cases", default=str(DEFAULT_CASES))
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--iters", type=int, default=20)
    run.add_argument("--samples", type=int, default=5)
    run.add_argument("--timeout", type=int, default=600)
    run.add_argument("--only-case", type=int, help="本地诊断时仅运行指定 case（从 1 开始）")
    summary = commands.add_parser("report")
    summary.add_argument("--output", required=True, help="gate.json/gate.md 的输出目录")
    summary.add_argument("--input", help="测量结果输入目录；省略时与 --output 相同")
    summary.add_argument("--plot", action="store_true", help="生成 gate.png；需要 matplotlib")
    summary.add_argument("--cases", default=str(DEFAULT_CASES))
    summary.add_argument("--base-sha", required=True)
    summary.add_argument("--head-sha", required=True)
    summary.add_argument("--sweeps", type=int, default=4)
    summary.add_argument("--tolerance", type=float, default=0.05)
    summary.add_argument(
        "--policy", choices=("perf", "refactor", "advisory", "enforced"), default="enforced"
    )
    args = parser.parse_args()
    if args.command == "measure":
        if min(args.warmup, args.iters, args.samples, args.timeout) <= 0:
            parser.error("测量参数必须为正数")
        if args.only_case and not 1 <= args.only_case <= len(load_cases(args.cases)):
            parser.error("--only-case 超出 case 范围")
        return measure(args)
    return report(args)


if __name__ == "__main__":
    sys.exit(main())

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
    "transformer_engine/common/common.h",
    "transformer_engine/pytorch/cpp_extensions/gemm.py",
    "transformer_engine/pytorch/csrc/extensions/gemm.cpp",
    "transformer_engine/pytorch/csrc/type_converters.cpp",
    "transformer_engine/pytorch/module/grouped_linear.py",
    "transformer_engine/pytorch/ops/basic/grouped_linear.py",
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


def validate_result(result, api, tag, samples, *, head=False):
    if result.get("tag") != tag or result.get("api") != api:
        raise ValueError(f"case tag/API 与 {api} 不一致")
    names = result.get("kernels")
    if not names or not all(
        isinstance(name, str) and name.startswith("nvte_grouped_gemm3_")
        and ("_grouped_tensor" in name) == (api == "grouped_tensor") for name in names
    ):
        raise ValueError(f"{api} RTC kernel 缺失或选路错误")
    if head and "n_expand:4" in tag.split(",") and any("_ne4" not in name for name in names):
        raise ValueError(f"{api} 未命中该 case 要求的 N_expand=4")
    if not 0 <= result["max_relative_error"] < 4e-3:
        raise ValueError(f"{api} 数值对拍失败")
    for values, median in (("samples_us", "median_us"), ("e2e_samples_us", "e2e_us")):
        times = result[values]
        if not isinstance(times, list) or len(times) != samples or not all(
            type(t) in (int, float) and math.isfinite(t) and t > 0 for t in times
        ) or result[median] != statistics.median(times):
            raise ValueError(f"{api} {median} 测量无效")


def measure(args):
    cases = load_cases(args.cases)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results, errors = [], []
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NVTE_GG_RTC_CACHE_DIR=str(out.parent / "rtc-cache"))
    apis = ("ordinary", "grouped_tensor")
    if args.sweep % 2:
        apis = apis[::-1]
    for index, tag in enumerate(cases, 1):
        if args.only_case and index != args.only_case:
            continue
        row = {"id": index}
        for api in apis:
            log = out / f"case-{index:02d}.{api}.log"
            result_file = log.with_suffix(".json")
            print(f"[RTC CI] case {index}/{len(cases)} {api}: {tag}", flush=True)
            try:
                result_file.unlink(missing_ok=True)
                with log.open("w", encoding="utf-8") as stream:
                    subprocess.run([
                        args.python, str(Path(args.replay).resolve()), "--tag", tag, "--api", api,
                        "--ci-output", str(result_file), "--warmup", str(args.warmup),
                        "--iters", str(args.iters), "--profile-samples", str(args.samples),
                    ], env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout,
                       check=True)
                result = json.loads(result_file.read_text())
                validate_result(result, api, tag, args.samples, head=args.arm == "head")
                prefix = env.get("RTC_CI_EXPECTED_PREFIX")
                if prefix and not Path(result["te_path"]).resolve().is_relative_to(Path(prefix).resolve()):
                    raise ValueError(f"导入了错误的 TE：{result['te_path']}")
                row[api] = result
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                message = f"{exc} ({log.name})"
                errors.append({"id": index, "tag": tag, "api": api, "error": message})
                print(f"[RTC CI] ERROR: {message}", flush=True)
        results.append(row)
        payload = {
            "results": results, "errors": errors, "apis": list(apis),
            "case_digest": hashlib.sha256("\n".join(cases).encode()).hexdigest(),
            "settings": {"iters": args.iters, "samples": args.samples, "warmup": args.warmup},
        }
        (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 1 if errors else 0


def compare(cases, measurements, tolerance):
    """Compare adjacent arms within each sweep before aggregating across sweeps."""
    if not measurements or not cases or not 0 <= tolerance < 1:
        raise ValueError("需要配对测量，且性能容差必须在 [0, 1) 内")
    digest = hashlib.sha256("\n".join(cases).encode()).hexdigest()
    settings = measurements[0][0]["settings"]
    if not all(type(settings[key]) is int and settings[key] > 0 for key in ("samples", "iters", "warmup")):
        raise ValueError("测量设置无效")
    for pair in measurements:
        for arm in pair:
            if arm["errors"] != [] or arm["case_digest"] != digest or arm["settings"] != settings:
                raise ValueError(f"测量失败或设置不一致：{arm['errors']}")
            if [r["id"] for r in arm["results"]] != list(range(1, len(cases) + 1)):
                raise ValueError("结果缺失、重复或顺序错误")
    rows = []
    for index, tag in enumerate(cases):
        sweeps = []
        environment = None
        base_grouped = []
        for base, head in measurements:
            b, h = base["results"][index], head["results"][index]
            trio = b["ordinary"], h["ordinary"], h["grouped_tensor"]
            for arm, result in zip(("base", "head", "grouped_tensor"), trio):
                api = "grouped_tensor" if arm == "grouped_tensor" else "ordinary"
                validate_result(result, api, tag, settings["samples"], head=arm != "base")
                current = tuple(result[key] for key in ("device", "device_uuid", "capability", "torch"))
                if not all(current) or (environment is not None and environment != current):
                    raise ValueError("base/head/API/sweep 运行环境不一致")
                environment = current
            if "grouped_tensor" in b or "grouped_tensor" in base.get("apis", ()):
                result = b["grouped_tensor"]
                validate_result(result, "grouped_tensor", tag, settings["samples"])
                if tuple(result[key] for key in ("device", "device_uuid", "capability", "torch")) != environment:
                    raise ValueError("基线 GroupedTensor 运行环境不一致")
                base_grouped.append(result)
            sweeps.append(trio)
        if base_grouped and len(base_grouped) != len(sweeps):
            raise ValueError("部分 sweep 缺少基线 GroupedTensor")

        row = {"id": index + 1, "tag": tag}
        for metric, suffix in (("median_us", ""), ("e2e_us", "_e2e")):
            row[f"base_grouped{suffix}_us"] = (
                statistics.median(result[metric] for result in base_grouped) if base_grouped else None
            )
            for arm, times in zip(("base", "head", "grouped"), zip(*sweeps)):
                row[f"{arm}{suffix}_us"] = statistics.median(t[metric] for t in times)
            row[f"ordinary{suffix}_ratio"] = statistics.median(h[metric] / b[metric] for b, h, g in sweeps)
            row[f"grouped{suffix}_ratio"] = statistics.median(g[metric] / h[metric] for b, h, g in sweeps)
        row["regression"] = max(row["ordinary_ratio"], row["ordinary_e2e_ratio"],
                                row["grouped_e2e_ratio"]) > 1 + tolerance
        rows.append(row)
    return rows


def plot_report(gate, output):
    """绘制两种 API 在 base/head 上的设备延迟和端到端延迟。"""
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
            for axis, suffix, label in zip(axes, ("", "_e2e"), ("Device", "End-to-end")):
                series = [("base", "Base ordinary"), ("head", "PR ordinary"),
                          ("base_grouped", "Base GroupedTensor"), ("grouped", "PR GroupedTensor")]
                for i, (arm, name) in enumerate(series):
                    values = [r[f"{arm}{suffix}_us"] for r in rows]
                    points = [(case + (i - 1.5) * 0.2, value) for case, value in zip(ids, values)
                              if value is not None]
                    if points:
                        x, y = zip(*points)
                        axis.bar(x, y, 0.2, label=name)
                axis.set_ylabel(f"{label} latency (μs)")
                axis.legend()
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
        "",
        f"**{verdict}**。本次采用 `{args.policy}` 策略，每个 case 的延迟容差为 {args.tolerance:.0%}。",
        f"比较基线 `{args.base_sha}` 与 PR `{args.head_sha}`，配对轮数为 {args.sweeps}。",
        "",
        "设备时间包含 RTC GEMM 和实际启动的元数据 kernel。端到端时间还包含 Python 提交和批次末尾同步。",
        "各组使用相同输入和 tile。加速比来自逐轮配对的延迟比，大于 1× 表示加速。",
        "",
    ]
    if error:
        lines += ["测量或校验失败。", "", f"`{error}`", "", "详见构建日志和各 case 的日志。"]
    else:
        case_ids = ", ".join(f"{row['id']:02d}" for row in regressions)
        lines += [
            f"{len(regressions)}/{len(rows)} 个 case 超过容差" + (f"（{case_ids}）。" if case_ids else "。"),
            "普通接口比较设备和端到端延迟。GroupedTensor 比较其相对 PR 普通接口的端到端延迟。",
        ]
        for suffix, title in (("", "设备"), ("_e2e", "端到端")):
            lines += ["", f"**{title}**", "",
                      "| case | 基线普通 μs | PR 普通 μs | 基线 GroupedTensor μs | PR GroupedTensor μs | 普通加速比 | 接口额外加速比 |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for row in rows:
                baseline = row[f"base_grouped{suffix}_us"]
                baseline_text = "未采集" if baseline is None else f"{baseline:.2f}"
                lines.append(
                    f"| {row['id']:02d} | {row[f'base{suffix}_us']:.2f} | {row[f'head{suffix}_us']:.2f} | "
                    f"{baseline_text} | {row[f'grouped{suffix}_us']:.2f} | {1/row[f'ordinary{suffix}_ratio']:.3f}× | "
                    f"{1/row[f'grouped{suffix}_ratio']:.3f}× |"
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
    run.add_argument("--sweep", type=int, default=0, help="奇数轮反转接口测量顺序")
    run.add_argument("--arm", choices=("base", "head"), default="head")
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

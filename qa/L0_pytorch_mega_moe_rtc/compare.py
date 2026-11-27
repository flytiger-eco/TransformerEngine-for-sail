# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Compare paired base/head sweeps of the same route capture."""

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("out", type=Path)
parser.add_argument(
    "--head-vs-deepep", action="store_true", help="compare two backends from the same wheel"
)
args = parser.parse_args()
out = args.out
arms = ["head", "deepep"] if args.head_vs_deepep else ["base", "head", "deepep"]
failed = False
charts = []
columns = ["Case", "EP", "Stage"]
titles = {"base": "MegaMoE base", "head": "MegaMoE PR", "deepep": "DeepEP + TE RTC GroupGEMM"}
columns.extend(titles[arm] + " (ms)" for arm in arms)
columns.append("DeepEP / PR")
summary = ["| " + " | ".join(columns) + " |", "|---|---:|---|" + "---:|" * (len(arms) + 1)]
for line in (out / "cases.txt").read_text().splitlines():
    name, world_text, tag = line.split()
    world = int(world_text)
    ratios = []
    runs = []
    for sweep in (0, 1):
        sweep_runs = {
            arm: json.loads((out / arm / f"{name}-{sweep}.json").read_text()) for arm in arms
        }
        head, deepep = sweep_runs["head"], sweep_runs["deepep"]
        for run in sweep_runs.values():
            if run["tag"] != tag or run["world"] != world or run["backward"] != head["backward"]:
                raise ValueError("incompatible performance reports")
            if not run["relative_l2"] < 0.012 or not all(
                math.isfinite(v) and v > 0
                for k, v in run["median_ms"].items()
                if k != "backward" or run["backward"]
            ):
                raise ValueError(f"invalid EP{world} replay result")
        if deepep["backend"] != "deepep" or head["backend"] != "megamoe":
            raise ValueError("incompatible replay backends")
        if (
            args.head_vs_deepep
            and head["rows_per_global_expert"] != deepep["rows_per_global_expert"]
        ):
            raise ValueError("backends replayed different expert loads")
        runs.append(sweep_runs)
        if "base" in sweep_runs:
            ratios.append(head["median_ms"]["total"] / sweep_runs["base"]["median_ms"]["total"])
    if ratios:
        ratio = statistics.median(ratios)
        print(f"{name} (EP{world}) head/base step latency: {ratio:.3f}x (sweeps: {ratios})")
        failed |= ratio > 1.05
    for stage in ("forward", "backward", "total"):
        if stage == "backward" and not head["backward"]:
            continue
        times = {arm: statistics.median(s[arm]["median_ms"][stage] for s in runs) for arm in arms}
        values = " | ".join(f"{v:.3f}" for v in times.values())
        summary.append(
            f"| {name} | {world} | {stage} | {values} | {times['deepep']/times['head']:.3f}× |"
        )
    labels = ("Forward L1", "Forward L2", "Dgrad L2", "Dgrad L1", "Wgrad L2", "Wgrad L1")
    keys = ("fwd_l1", "fwd_l2", "dgrad_l2", "dgrad_l1", "wgrad_l2", "wgrad_l1")
    profiles = [runs[0][arm]["kernel_profile"] for arm in ("head", "deepep")]
    if any((p["layers_per_step"], p["steps"]) != (4, 3) for p in profiles):
        raise ValueError("incompatible kernel profiles")
    deepep = runs[0]["deepep"]
    expert_rows = deepep.get("rows_per_global_expert")
    if expert_rows is None or any(expert_rows):
        for key in keys if deepep["backward"] else keys[:2]:
            names = profiles[1]["kernels"][key]
            if any(n.startswith("nvte_mega_moe_") for n in names) or not any(
                n.startswith("nvte_grouped_gemm3_") for n in names
            ):
                raise ValueError(f"{name} DeepEP {key}: expected TE RTC GroupGEMM, got {names}")
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    charts.extend(
        [
            "",
            f"#### {name} · EP{world}",
            "",
            "| Kernel stage | MegaMoE PR (ms) | DeepEP + TE RTC GroupGEMM (ms) |",
            "|---|---:|---:|",
        ]
    )
    columns = []
    for index, (profile, legend) in enumerate(
        zip(profiles, ("MegaMoE PR", "DeepEP + TE RTC GroupGEMM"))
    ):
        column = []
        for row, key in enumerate(keys):
            names = profile["kernels"][key]
            has_gemm = any(
                n.startswith(("nvte_mega_moe_", "nvte_grouped_gemm")) and "bulk_" not in n
                for n in names
            )
            has_comm = (
                index == 0
                or row >= 4
                or any(("dispatch" if row in (0, 2) else "combine") in n for n in names)
            )
            value = profile["mean_ms"][key]
            y = row + (index - 0.5) * 0.36
            if has_gemm and has_comm and math.isfinite(value) and value > 0:
                bar = ax.barh(
                    y,
                    value,
                    height=0.34,
                    color=("steelblue", "darkorange")[index],
                    label=legend if row == 0 else None,
                )
                ax.bar_label(bar, fmt="%.2f", padding=3, fontsize=8)
                column.append(f"{value:.3f}")
            else:
                if args.head_vs_deepep:
                    raise ValueError(
                        f"{name} {legend} {key}: missing measurable kernel stage: {names}"
                    )
                ax.text(0, y, "N/A", va="center", fontsize=8)
                column.append("N/A")
        columns.append(column)
    for label, h, d in zip(labels, *columns):
        charts.append(f"| {label} | {h} | {d} |")
    deep_ops = (
        "dispatch + FC1",
        "FC2 + combine",
        "dispatch + FC2 dgrad",
        "FC1 dgrad + combine",
        "FC2 wgrad",
        "FC1 wgrad",
    )
    ax.set_yticks(range(len(labels)), [f"{label}\n{op}" for label, op in zip(labels, deep_ops)])
    ax.invert_yaxis()
    ax.margins(x=0.17)
    ax.set_xlabel("Mean kernel time per step (ms, four replays of the same layer)")
    ax.set_title(f"{name}  ·  EP{world}")
    ax.legend()
    fig.savefig(out / f"stages-{name}.png", dpi=150)
    plt.close(fig)
summary.extend(
    [
        "",
        (
            "Stage timings sum selected device kernels over four replays of the same "
            "captured layer, take the maximum rank per stage, then average three profiled steps. "
            "DeepEP uses TE RTC GroupGEMM for forward (TN), dgrad (NN) and wgrad (NT). "
            "Its times include dispatch with forward L1 / dgrad L2 and combine with forward L2 / "
            "dgrad L1. MegaMoE uses the executed GEMMs (pull/scatter when available) and any "
            "explicit bulk payload kernels. Separate layout, activation, recompute, collectives "
            "outside these kernels and local reductions are excluded. N/A indicates a missing "
            "kernel match. Whole-step timings above measure one unprofiled layer."
        ),
    ]
)
report = "\n".join(summary + charts) + "\n"
print(report)
(out / "comparison.md").write_text(report)
sys.exit(int(failed and os.getenv("MEGA_MOE_CI_POLICY", "enforced") != "advisory"))

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
from matplotlib.patches import Patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("out", type=Path)
comparison = parser.add_mutually_exclusive_group()
comparison.add_argument(
    "--baseline-commit", help="merge-base SHA used for the profiled baseline wheel"
)
comparison.add_argument(
    "--head-vs-deepep", action="store_true", help="compare two backends from the same wheel"
)
args = parser.parse_args()
out = args.out
arms = ["head", "deepep"] if args.head_vs_deepep else ["base", "head", "deepep"]
failed = False
charts = []
columns = ["Case", "EP", "Stage"]
titles = {"base": "MegaMoE baseline", "head": "MegaMoE PR", "deepep": "DeepEP + TE RTC GroupGEMM"}
columns.extend(titles[arm] + " (ms)" for arm in arms)
if "base" in arms:
    columns.append("Baseline / PR")
columns.append("DeepEP / PR")
summary = ["| " + " | ".join(columns) + " |", "|---|---:|---|" + "---:|" * (len(columns) - 3)]
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
        for arm, run in sweep_runs.items():
            if run["backend"] != ("deepep" if arm == "deepep" else "megamoe"):
                raise ValueError(f"{name} {arm}: incompatible replay backend")
            if run.get("rows_per_global_expert") != head.get("rows_per_global_expert"):
                raise ValueError(f"{name} {arm}: different expert loads")
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
        if "base" in times:
            values += f" | {times['base']/times['head']:.3f}×"
        summary.append(
            f"| {name} | {world} | {stage} | {values} | {times['deepep']/times['head']:.3f}× |"
        )
    labels = ("Forward L1", "Forward L2", "Dgrad L2", "Dgrad L1", "Wgrad L2", "Wgrad L1")
    keys = ("fwd_l1", "fwd_l2", "dgrad_l2", "dgrad_l1", "wgrad_l2", "wgrad_l1")
    profiles = {
        arm: runs[0][arm]["kernel_profile"] for arm in arms if "kernel_profile" in runs[0][arm]
    }
    required = arms if args.baseline_commit else ("head", "deepep")
    if any(arm not in profiles for arm in required):
        raise ValueError(f"{name}: missing stage profile for a required arm")
    if any((p["layers_per_step"], p["steps"]) != (4, 3) for p in profiles.values()):
        raise ValueError("incompatible kernel profiles")
    deepep = runs[0]["deepep"]
    expert_rows = deepep.get("rows_per_global_expert")
    if expert_rows is None or any(expert_rows):
        for key in keys if deepep["backward"] else keys[:2]:
            names = profiles["deepep"]["kernels"][key]
            if any(n.startswith("nvte_mega_moe_") for n in names) or not any(
                n.startswith("nvte_grouped_gemm3_") for n in names
            ):
                raise ValueError(f"{name} DeepEP {key}: expected TE RTC GroupGEMM, got {names}")
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    charts.extend(
        [
            "",
            f"#### {name} · EP{world}",
            "",
            "| Kernel stage | " + " | ".join(titles[arm] + " (ms)" for arm in profiles) + " |",
            "|---|" + "---:|" * len(profiles),
        ]
    )
    stage_columns = []
    colors = {"base": "slategray", "head": "steelblue", "deepep": "darkorange"}
    spacing = 0.82 / len(profiles)
    for index, (arm, profile) in enumerate(profiles.items()):
        column = []
        for row, key in enumerate(keys):
            names = profile["kernels"][key]
            has_gemm = any(
                n.startswith(("nvte_mega_moe_", "nvte_grouped_gemm")) and "bulk_" not in n
                for n in names
            )
            has_comm = (
                arm != "deepep"
                or row >= 4
                or any(("dispatch" if row in (0, 2) else "combine") in n for n in names)
            )
            value = profile["mean_ms"][key]
            y = row + (index - (len(profiles) - 1) / 2) * spacing
            if has_gemm and has_comm and math.isfinite(value) and value > 0:
                bar = ax.barh(
                    y,
                    value,
                    height=spacing * 0.92,
                    color=colors[arm],
                )
                ax.bar_label(bar, fmt="%.2f", padding=3, fontsize=8)
                column.append(f"{value:.3f}")
            else:
                if world == 16 and arm != "base":
                    raise ValueError(
                        f"{name} {arm} {key}: missing measurable kernel stage: {names}"
                    )
                ax.text(0, y, "N/A", va="center", fontsize=8)
                column.append("N/A")
        stage_columns.append(column)
    for values in zip(labels, *stage_columns):
        charts.append("| " + " | ".join(values) + " |")
    axis_label = "Mean kernel time per step (ms, four replays of the same layer)"
    if world == 16 and args.baseline_commit:
        paths = runs[0]["base"]["rank_paths"]
        if sorted(p["rank"] for p in paths) != list(range(world)):
            raise ValueError(f"{name}: incomplete EP16 baseline path evidence")
        counts = [
            sum(bool(p[phase]["collectives"]) for p in paths) for phase in ("forward", "backward")
        ]
        charts += [
            "",
            f"Baseline all-to-all observed on {counts[0]}/{world} forward ranks and "
            f"{counts[1]}/{world} backward ranks. Total latency includes these collectives.",
        ]
        if any(counts):
            axis_label += "\nBaseline all-to-all is measured in whole-layer latency and excluded from these bars."
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
    ax.set_xlabel(axis_label)
    fig.suptitle(f"{name}  ·  EP{world}")
    ax.legend(
        handles=[Patch(color=colors[arm], label=titles[arm]) for arm in profiles],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncols=len(profiles),
        frameon=False,
    )
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
if args.baseline_commit:
    summary.extend(
        [
            "",
            f"MegaMoE baseline uses merge-base `{args.baseline_commit}`. "
            "Ratios divide the named reference latency by the PR latency.",
        ]
    )
report = "\n".join(summary + charts) + "\n"
print(report)
(out / "comparison.md").write_text(report)
sys.exit(int(failed and os.getenv("MEGA_MOE_CI_POLICY", "enforced") != "advisory"))

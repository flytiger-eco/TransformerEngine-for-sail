# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Compare paired base/head sweeps of the same route capture."""

import json
import math
import os
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
failed = False
totals = []
summary = [
    "| Case | EP | Stage | MegaMoE base (ms) | MegaMoE PR (ms) | DeepEP + GroupGEMM (ms) | DeepEP / PR |",
    "|---|---:|---|---:|---:|---:|---:|",
]
for line in (out / "cases.txt").read_text().splitlines():
    name, world_text, tag = line.split()
    world = int(world_text)
    ratios = []
    runs = []
    for sweep in (0, 1):
        base, head, deepep = [
            json.loads((out / arm / f"{name}-{sweep}.json").read_text())
            for arm in ("base", "head", "deepep")
        ]
        if any(run["tag"] != tag or run["world"] != world for run in (base, head, deepep)):
            raise ValueError("incompatible performance reports")
        for run in (base, head, deepep):
            total = run["median_ms"]["total"]
            if not math.isfinite(total) or total <= 0 or not run["relative_l2"] < 0.012:
                raise ValueError(f"invalid EP{world} replay result")
        if deepep["backend"] != "deepep" or len({r["backward"] for r in (base, head, deepep)}) != 1:
            raise ValueError("incompatible DeepEP baseline")
        runs.append((base, head, deepep))
        ratios.append(head["median_ms"]["total"] / base["median_ms"]["total"])
    ratio = statistics.median(ratios)
    print(f"{name} (EP{world}) head/base step latency: {ratio:.3f}x (sweeps: {ratios})")
    for stage in ("forward", "backward", "total"):
        if stage == "backward" and not head["backward"]:
            continue
        b, h, d = [statistics.median(r["median_ms"][stage] for r in arm) for arm in zip(*runs)]
        if stage == "total":
            totals.append((name, b, h, d))
        summary.append(f"| {name} | {world} | {stage} | {b:.3f} | {h:.3f} | {d:.3f} | {d/h:.3f}× |")
    failed |= ratio > 1.05
report = "\n".join(summary) + "\n"
print(report)
(out / "comparison.md").write_text(report)
fig, ax = plt.subplots(figsize=(9, max(3, 0.95 * len(totals) + 1.6)), layout="constrained")
for index, label in enumerate(("MegaMoE base", "MegaMoE PR", "DeepEP + GroupGEMM")):
    bars = ax.barh(
        [row + (index - 1) * 0.25 for row in range(len(totals))],
        [row[index + 1] for row in totals],
        height=0.24,
        label=label,
    )
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
ax.set_yticks(range(len(totals)), [row[0] for row in totals])
ax.invert_yaxis()
ax.margins(x=0.13)
ax.set_xlabel("Total step latency (ms), lower is faster")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=3)
fig.savefig(out / "comparison.png", dpi=150)
plt.close(fig)
sys.exit(int(failed and os.getenv("MEGA_MOE_CI_POLICY", "enforced") != "advisory"))

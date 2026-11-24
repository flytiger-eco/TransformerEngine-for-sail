# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Compare paired base/head sweeps of the same route capture."""

import json
import math
import os
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
failed = False
for line in (out / "cases.txt").read_text().splitlines():
    name, world_text, tag = line.split()
    world = int(world_text)
    ratios = []
    for sweep in (0, 1):
        base, head = [
            json.loads((out / arm / f"{name}-{sweep}.json").read_text()) for arm in ("base", "head")
        ]
        if any(run["tag"] != tag or run["world"] != world for run in (base, head)):
            raise ValueError("incompatible performance reports")
        for run in (base, head):
            total = run["median_ms"]["total"]
            if not math.isfinite(total) or total <= 0 or not run["relative_l2"] < 0.012:
                raise ValueError(f"invalid EP{world} replay result")
        ratios.append(head["median_ms"]["total"] / base["median_ms"]["total"])
    ratio = statistics.median(ratios)
    print(f"{name} (EP{world}) head/base step latency: {ratio:.3f}x (sweeps: {ratios})")
    failed |= ratio > 1.05
sys.exit(int(failed and os.getenv("MEGA_MOE_CI_POLICY", "enforced") != "advisory"))

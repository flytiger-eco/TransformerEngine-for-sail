# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Run the head / DeepEP / DeepEP / head arms of one case in one torchrun."""

from copy import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/gemm"))
from replay_mega_moe import parse_args, run_replays


def main():
    out, name = Path(sys.argv[1]), sys.argv[2]
    common = parse_args(sys.argv[3:])
    cases = []
    for sweep, arms in enumerate((("head", "deepep"), ("deepep", "head"))):
        for arm in arms:
            args = copy(common)
            args.backend = "megamoe" if arm == "head" else "deepep"
            args.expect_ep8_training = common.expect_ep8_training and arm == "head"
            args.profile_stages = sweep == 0
            args.output = out / arm / f"{name}-{sweep}.json"
            cases.append(args)
    run_replays(cases)


if __name__ == "__main__":
    main()

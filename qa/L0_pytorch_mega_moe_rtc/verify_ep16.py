# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Check the two-node Qwen training capture against the kernel evidence from every rank."""

import json
import sys
from collections import defaultdict
from pathlib import Path


def verify(report):
    paths = report["rank_paths"]
    if (
        report["world"] != 16
        or report["backend"] != "megamoe"
        or not report["backward"]
        or report["autograd_node"] != "_TrainBackward"
        or sorted(p["rank"] for p in paths) != list(range(16))
        or len(report["rows_per_global_expert"]) != 256
        or sum(report["rows_per_global_expert"]) != 952664
    ):
        raise ValueError("expected the complete Qwen EP16 training capture")
    nodes = defaultdict(list)
    for path in paths:
        rank = path["rank"]
        nodes[path["node"]].append(path["local_rank"])
        fwd, bwd = path["forward"], path["backward"]
        if (
            fwd["collectives"]
            or bwd["collectives"]
            or fwd["kernels"].get("nvte_mega_moe_fwd_pull") != 1
            or fwd["kernels"].get("nvte_mega_moe_fwd_scatter") != 1
            or bwd["kernels"].get("nvte_mega_moe_dgrad_pull") != 1
            or bwd["kernels"].get("nvte_mega_moe_dgrad_scatter") != 1
            or bwd["kernels"].get("nvte_mega_moe_tensor_wgrad") != 2
        ):
            raise ValueError(f"rank {rank}: unexpected forward/backward path: {path}")
    if len(nodes) != 2 or any(sorted(ranks) != list(range(8)) for ranks in nodes.values()):
        raise ValueError(
            f"expected two physical nodes with local devices 0..7 on each: {dict(nodes)}"
        )


if __name__ == "__main__":
    verify(json.loads(Path(sys.argv[1]).read_text()))
    print("Verified fused Qwen EP16 forward and backward on 2 x 8 PPUs.")

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Create a deterministic distributed CI case through the public capture entry point."""

from contextlib import redirect_stdout
import io
import json
import os
import sys
from unittest.mock import patch
from pathlib import Path

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/gemm"))
from mega_moe_tags import PREFIX, decode_rank, unpack


def main():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    world, rank = dist.get_world_size(), dist.get_rank()
    k = world
    t = 0 if rank == 0 else 2048 + rank
    torch.manual_seed(713 + rank)
    x = torch.randn(t, 512, dtype=torch.bfloat16, device=device, requires_grad=True)
    ids = torch.randint(world * 2, (k, t), dtype=torch.int32, device=device).T
    probs = torch.rand(k, t, device=device).T
    ids[::7, 1] = ids[::7, 0]
    ids[::11, 0] = -1
    probs[::13, 1] = 0
    w1, w2 = [
        [torch.randn(shape, device=device, dtype=x.dtype).requires_grad_() for _ in range(2)]
        for shape in ((512, 512), (512, 256))
    ]
    with patch.object(torch.cuda.nvtx, "range", wraps=torch.cuda.nvtx.range) as nvtx:
        te.bf16_mega_moe(x, ids, probs.requires_grad_(), w1, w2, ep_group=dist.group.WORLD)
    tag = nvtx.call_args.args[0]
    fields = dict(item.split(":", 1) for item in tag.removeprefix(PREFIX).split(","))
    shard = decode_rank(fields, json.loads(unpack(fields["meta"])))
    assert torch.equal(shard["ids"], ids.cpu())
    assert torch.equal(shard["probs"] == 0, probs.cpu() == 0)
    if "exact" in fields:
        assert torch.equal(shard["probs"], probs.cpu())
    # One bad source must reach the same collective validation as its peers.
    if rank == 1:
        ids[0, 0] = world * len(w1)
    with redirect_stdout(io.StringIO()) as captured:
        try:
            te.bf16_mega_moe(x, ids, probs, w1, w2, ep_group=dist.group.WORLD)
        except ValueError as exc:
            assert "expert IDs in range or -1" in str(exc), str(exc)
        else:
            raise AssertionError(f"rank {rank} accepted an invalid distributed route")
    if rank == 1:
        assert PREFIX not in captured.getvalue()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

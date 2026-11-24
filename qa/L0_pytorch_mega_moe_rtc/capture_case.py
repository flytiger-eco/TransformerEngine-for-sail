# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Create a deterministic distributed CI case through the public capture entry point."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te


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
    te.bf16_mega_moe(x, ids, probs.requires_grad_(), w1, w2, ep_group=dist.group.WORLD)
    (folder,) = Path(os.environ["NVTE_MOE_CAPTURE_DIR"]).glob("*/manifest.json")
    manifest = json.loads(folder.read_text())
    assert manifest["global_ranks"] == list(range(world))
    shard = torch.load(folder.parent / f"rank-{rank}.pt", weights_only=True)
    assert torch.equal(shard["ids"], ids.cpu())
    assert torch.equal(shard["probs"], probs.cpu())
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

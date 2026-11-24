# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Opt-in eager capture of one routed MegaMoE invocation per EP group."""

import json
import os
from pathlib import Path
from uuid import uuid4

import torch
import torch.distributed as dist


def _layout(t):
    return {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "dtype": str(t.dtype).removeprefix("torch."),
        "offset_bytes": t.data_ptr() % 512,
        "requires_grad": t.requires_grad,
    }


def capture(directory, x, ids, probs, fc1, fc2, group):
    rank = group.rank() if group is not None else 0
    world = group.size() if group is not None else 1
    ranks = (
        dist.get_process_group_ranks(group)
        if group is not None
        else [dist.get_rank() if dist.is_initialized() else 0]
    )
    key = [uuid4().hex if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(key, src=ranks[0], group=group, device=x.device)
    folder = Path(directory) / key[0]
    error = None
    try:
        folder.mkdir(parents=True, exist_ok=True)
        tensors = (x, ids, probs, *fc1, *fc2)
        shard = {
            "id": key[0],
            "rank": rank,
            "global_rank": ranks[rank],
            "grad_enabled": torch.is_grad_enabled(),
            "experts": len(fc1),
            "settings": {
                k: v
                for k, v in os.environ.items()
                if k.startswith("NVTE_MOE_") and k != "NVTE_MOE_CAPTURE_DIR"
            },
            "layouts": [_layout(t) for t in tensors],
            "ids": ids.detach().to(device="cpu", copy=True).contiguous(),
            "probs": probs.detach().to(device="cpu", copy=True).contiguous(),
        }
        torch.save(shard, folder / f"rank-{rank}.pt")
    except Exception as exc:
        error = f"rank {rank}: {exc}"
    errors = [error] * world
    if world > 1:
        dist.all_gather_object(errors, error, group=group)
    if any(errors):
        raise RuntimeError(f"MegaMoE capture failed: {errors}")
    tag = f"te_mega_moe:v1:{key[0]}"
    if rank == 0:
        try:
            manifest = {"version": 1, "tag": tag, "global_ranks": ranks, "local": group is None}
            tmp = folder / "manifest.tmp"
            tmp.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            tmp.replace(folder / "manifest.json")
        except OSError as exc:
            error = str(exc)
    status = [error]
    if world > 1:
        dist.broadcast_object_list(status, src=ranks[0], group=group, device=x.device)
    if status[0]:
        raise RuntimeError(f"MegaMoE manifest publication failed: {status[0]}")
    return tag

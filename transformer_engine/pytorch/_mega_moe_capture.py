# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""One self-contained NVTX/log string per source rank; no capture collectives."""

import base64
from collections import defaultdict
import itertools
import json
import os
from urllib.parse import quote
import zlib

import numpy as np
import torch
import torch.distributed as dist


_sequences = defaultdict(itertools.count)


def _layout(t):
    return {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "dtype": str(t.dtype).removeprefix("torch."),
        "offset_bytes": t.data_ptr() % 512,
        "requires_grad": t.requires_grad,
    }


def _pack(data):
    return base64.urlsafe_b64encode(zlib.compress(data, level=1)).decode("ascii")


def capture(mode, x, ids, probs, fc1, fc2, group):
    if mode not in ("routes", "exact"):
        raise ValueError("NVTE_MOE_CAPTURE_TAG must be routes or exact")
    ranks = (
        dist.get_process_group_ranks(group)
        if group is not None
        else [dist.get_rank() if dist.is_initialized() else 0]
    )
    rank = group.rank() if group is not None else 0
    name = group.group_name if group is not None else "local"
    run = os.getenv("NVTE_MOE_CAPTURE_RUN") or os.getenv("TORCHELASTIC_RUN_ID", "local")
    group_id = ".".join(map(str, ranks))
    seq = next(_sequences[(run, name, group_id)])
    call = f"{quote(run, safe='')}/{os.getenv('TORCHELASTIC_RESTART_COUNT', '0')}/{quote(name, safe='')}/{group_id}/{seq}"
    experts = len(fc1) * len(ranks)
    width = 1 if experts <= 255 else 2 if experts <= 65535 else 4
    raw_ids = ids.detach().cpu().numpy()
    if np.any(raw_ids < -1) or np.any(raw_ids >= experts):
        raise ValueError("invalid expert ID for MegaMoE capture")
    counts = np.bincount(raw_ids[raw_ids >= 0], minlength=experts)
    m_src = ".".join(map(str, counts))
    routes = _pack((raw_ids.astype(np.int64) + 1).astype(f"<u{width}").tobytes())
    if mode == "exact":
        weights = probs.detach().cpu().numpy().astype("<f4", copy=False).tobytes()
    else:
        weights = np.packbits((probs.detach() == 0).cpu().numpy(), bitorder="little").tobytes()
    meta = {
        "layouts": [_layout(t) for t in (x, ids, probs, *fc1, *fc2)],
        "grad_enabled": torch.is_grad_enabled(),
        "local": group is None,
        "global_ranks": ranks,
        "settings": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("NVTE_MOE_") and not k.startswith("NVTE_MOE_CAPTURE_")
        },
    }
    h, i = x.size(1), fc2[0].size(1)
    tag = (
        f"[TE] --format=MegaMoE,v:2,call:{call},rank:{rank},ep:{len(ranks)},"
        f"dtype:{x.dtype},t:{x.size(0)},topk:{ids.size(1)},groups:{len(fc1)},"
        f"fc1:{h}x{2*i},fc2:{i}x{h},m_src:{m_src},ids{width}:{routes},{mode}:{_pack(weights)},"
        f"meta:{_pack(json.dumps(meta, separators=(',', ':')).encode())}"
    )
    print(tag, flush=True)
    return tag

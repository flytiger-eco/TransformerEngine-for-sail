# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Offline reader for per-rank TE MegaMoE range names (also works with a base wheel)."""

import base64
import json
import zlib
from pathlib import Path

import numpy as np
import torch


PREFIX = "[TE] --format=MegaMoE,"


def unpack(text):
    return zlib.decompress(base64.b64decode(text, altchars=b"-_", validate=True))


def read_tags(path, call, world, rank):
    groups = {}
    with Path(path).open() as stream:
        for line in stream:
            if PREFIX not in line:
                continue
            fields = dict(
                item.split(":", 1) for item in line.split(PREFIX, 1)[1].strip().split(",")
            )
            if fields["v"] != "2" or (call and fields["call"] != call):
                continue
            if groups and fields["call"] not in groups:
                raise ValueError("multiple MegaMoE invocations; select one with --call")
            peers = groups.setdefault(fields["call"], {})
            r = int(fields["rank"])
            if r in peers:
                raise ValueError(f"duplicate rank {r} in {fields['call']}")
            peers[r] = fields
    if len(groups) != 1:
        raise ValueError(f"select exactly one invocation with --call; found {list(groups)}")
    key, peers = next(iter(groups.items()))
    if set(peers) != set(range(world)):
        raise ValueError(f"expected {world} EP ranks, found {sorted(peers)}")
    f = peers[rank]
    meta = json.loads(unpack(f["meta"]))
    global_m = np.zeros(world * int(f["groups"]), dtype=np.int64)
    for r, other in peers.items():
        if any(other[k] != f[k] for k in ("ep", "dtype", "groups", "fc1", "fc2")):
            raise ValueError(f"inconsistent EP/GEMM geometry at rank {r}")
        counts = [int(v) for v in other["m_src"].split(".")]
        if len(counts) != global_m.size or min(counts) < 0:
            raise ValueError(f"invalid per-expert counts at rank {r}")
        global_m += counts
        if int(other["ep"]) != world:
            raise ValueError("torchrun world size differs from tag")
        if json.loads(unpack(other["meta"]))["global_ranks"] != meta["global_ranks"]:
            raise ValueError("inconsistent group membership")
    shard = decode_rank(f, meta)
    shard["expert_rows"] = global_m.tolist()
    return key, meta["local"], shard


def decode_rank(f, meta):
    world = int(f["ep"])
    shape = int(f["t"]), int(f["topk"])
    width = next(w for w in (1, 2, 4) if f"ids{w}" in f)
    arr = np.frombuffer(unpack(f[f"ids{width}"]), dtype=f"<u{width}").astype(np.int64) - 1
    if np.any(arr >= world * int(f["groups"])):
        raise ValueError("expert ID outside EP group")
    counts = np.bincount(arr[arr >= 0], minlength=world * int(f["groups"]))
    if ".".join(map(str, counts)) != f["m_src"]:
        raise ValueError("per-expert counts disagree with source slots")
    layouts = meta["layouts"]
    t, h = layouts[0]["shape"]
    i = layouts[-1]["shape"][1]
    if (
        shape != tuple(layouts[1]["shape"])
        or t != shape[0]
        or f["dtype"] != "torch." + layouts[0]["dtype"]
        or f["fc1"] != f"{h}x{2*i}"
        or f["fc2"] != f"{i}x{h}"
    ):
        raise ValueError("tag geometry disagrees with tensor layouts")
    ids = torch.from_numpy(arr.astype(np.int32).reshape(shape))
    if "exact" in f:
        scores = np.frombuffer(unpack(f["exact"]), dtype="<f4").copy().reshape(shape)
    else:
        bits = unpack(f["routes"])
        if len(bits) != (ids.numel() + 7) // 8:
            raise ValueError("invalid zero-weight mask")
        zeros = np.unpackbits(
            np.frombuffer(bits, dtype=np.uint8), bitorder="little", count=ids.numel()
        )
        scores = ((1 - zeros.astype(np.float32)) / shape[1]).reshape(shape)
    return {**meta, "ids": ids, "probs": torch.from_numpy(scores), "experts": int(f["groups"])}

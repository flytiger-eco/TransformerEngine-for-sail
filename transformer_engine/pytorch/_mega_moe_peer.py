# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Intra-node MegaMoE dispatch and combine through symmetric PPU memory."""

import socket
import threading
from importlib.util import find_spec
from weakref import WeakKeyDictionary, finalize

import torch
import torch.distributed as dist
import transformer_engine_torch as tex

from ._mega_moe_routing import make_route_plan

try:
    import torch.distributed._symmetric_memory as symm
except ImportError:
    symm = None


class _Workspace:
    def __init__(self):
        self.lock = threading.RLock()
        self.supported = None
        self.bulk_supported = False
        self.device = None
        self.storage = None
        self.handle = None
        self.done = None

    def wait_for_completion(self):
        if self.done is not None:
            self.done.synchronize()

    def buffers(self, group, shape, x):
        nbytes = shape[0] * shape[1] * x.element_size()
        if self.storage is None or self.storage.numel() < nbytes:
            # Local and remote kernels may still use the old mapping. Each rank
            # finishes its local stream before the collective new rendezvous.
            # Keep both old storage and handle alive until all peers arrive.
            self.wait_for_completion()
            storage = symm.empty(nbytes, device=x.device, dtype=torch.uint8)
            handle = symm.rendezvous(storage, group)
            self.storage, self.handle = storage, handle
            self.done = torch.cuda.Event()
        else:
            torch.cuda.current_stream(x.device).wait_event(self.done)
        return [self.handle.get_buffer(rank, shape, x.dtype) for rank in range(group.size())]


_workspaces = WeakKeyDictionary()
_workspace_lock = threading.Lock()


def _workspace(group):
    with _workspace_lock:
        if group not in _workspaces:
            state = _Workspace()
            _workspaces[group] = state
            # A short-lived subgroup can disappear while its last operation is
            # still queued. Retain its mappings until the local combine finishes.
            finalize(group, state.wait_for_completion).atexit = False
        return _workspaces[group]


def _peer_supported(group, device):
    """Agree on CUDA symmetric-memory access for every ordered pair in this group."""
    if dist.get_backend(group) != "nccl":
        return False
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    state = _workspace(group)
    with state.lock:
        if state.supported is None:
            supported = (
                device.type == "cuda"
                and torch.cuda.get_device_capability(device) == (8, 9)
                and hasattr(torch.cuda, "can_device_access_peer")
                and symm is not None
                and all(hasattr(symm, name) for name in ("empty", "rendezvous", "get_backend"))
                and all(
                    hasattr(getattr(symm, "_SymmetricMemory", None), name)
                    for name in ("barrier", "get_buffer", "get_backend")
                )
                and symm.get_backend(device) == "CUDA"
            )
            properties = torch.cuda.get_device_properties(device) if supported else None
            uuid = getattr(properties, "uuid", None)
            uuid = str(uuid) if uuid is not None else None
            reachable = set()
            if uuid is not None:
                for ordinal in range(torch.cuda.device_count()):
                    if ordinal == device.index or torch.cuda.can_device_access_peer(
                        device, ordinal
                    ):
                        peer_uuid = getattr(torch.cuda.get_device_properties(ordinal), "uuid", None)
                        if peer_uuid is not None:
                            reachable.add(str(peer_uuid))
            # UUIDs also handle per-rank CUDA_VISIBLE_DEVICES permutations. Each
            # direction must be addressable from the participating device.
            peers = [None] * group.size()
            dist.all_gather_object(
                peers, (socket.gethostname(), uuid, reachable, find_spec("triton") is not None), group
            )
            devices = {item[1] for item in peers}
            state.supported = (
                len({item[0] for item in peers}) == 1
                and None not in devices
                and len(devices) == len(peers)
                and all(devices <= item[2] for item in peers)
            )
            state.bulk_supported = group.size() == 2 and all(item[3] for item in peers)
            state.device = device
        return state.supported


def peer_forward(x, ids, probs, fc1, fc2, ep_group_name):
    """Read remote input rows and write remote output slots on a P2P-capable group."""
    if not ep_group_name:
        return None
    group = dist.distributed_c10d._resolve_process_group(ep_group_name)
    if group.size() == 1 or not x.is_cuda:
        return None
    with torch.cuda.device(x.device):
        if not _peer_supported(group, x.device):
            return None
        state = _workspace(group)
        with state.lock:
            if state.bulk_supported:
                from .triton.mega_moe_bulk import bulk_forward

                output = bulk_forward(x, ids, probs, fc1, fc2, group, state)
                if output is not None:
                    return output
            tokens, topk = ids.shape
            hidden = x.size(1)
            fields = (tokens, ids.numel(), x.device == state.device)
            plan, experts, metadata = make_route_plan(x, ids, len(fc1), group, fields)
            if not all(m[2] for m in metadata):
                return None
            max_tokens = max(m[0] for m in metadata)
            max_slots = max(m[1] for m in metadata)
            if not max_slots or not hidden:
                return torch.zeros_like(x, memory_format=torch.contiguous_format)

            # Exchange only route metadata. Each source slot has a unique
            # writer even when a token selects the same expert twice.
            slots = plan.slots
            routes = torch.stack(
                (experts.long(), torch.full_like(slots, group.rank()), slots // topk, slots), dim=1
            )
            received = plan.to_experts(routes)
            received_probs = plan.to_experts(probs.reshape(-1).index_select(0, slots))
            sorted_ids, order = received[:, 0].sort(stable=True)
            bounds = torch.arange(len(fc1) + 1, device=x.device, dtype=sorted_ids.dtype)
            offsets = torch.searchsorted(sorted_ids, bounds, out_int32=True)
            route_info = received[:, 1:].index_select(0, order)
            sorted_probs = received_probs.index_select(0, order)

            buffers = state.buffers(group, (max_tokens + max_slots, hidden), x)
            local = buffers[group.rank()]
            local[:tokens].copy_(x)
            output_slots = local[max_tokens : max_tokens + ids.numel()]
            output_slots.zero_()
            state.handle.barrier(channel=0)
            tex.mega_moe_peer_forward(
                buffers, route_info, sorted_probs, offsets, fc1, fc2, max_tokens
            )
            state.handle.barrier(channel=1)
            output = tex.mega_moe_combine(output_slots.view(tokens, topk, hidden))
            state.done.record(torch.cuda.current_stream(x.device))
            return output

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Reusable IPC mappings for the SM89 three-kernel BF16 forward (EP <= 8)."""

import torch
import torch.distributed._symmetric_memory as symm
import transformer_engine_torch as tex

from ._mega_moe_peer import _collective_allocation


def _mapping(group, tensor):
    handle = symm.rendezvous(tensor, group)
    buffers = [handle.get_buffer(r, tensor.shape, tensor.dtype) for r in range(group.size())]
    return tensor, handle, buffers


class FusedWorkspace:
    def __init__(self, group, device):
        # Initialize before rendezvous so a remote rank can safely signal its
        # first epoch as soon as it acquires the mapped addresses.
        self.allocation_status = torch.empty(1, dtype=torch.int32, device=device)
        with _collective_allocation(group, self.allocation_status):
            control = symm.empty(1024, dtype=torch.int32, device=device)
        control.zero_()
        torch.cuda.current_stream(device).synchronize()
        self.control = _mapping(group, control)
        self.status = tex.MegaMoEFusedStatus(control)
        self.payload = None
        self.local = None
        self.shape = None
        self.capacity = 0
        self.epoch = 0
        self.done = torch.cuda.Event()

    def _resize(self, group, shape, capacity, device):
        # Keep the old mapping alive until all ranks allocated replacements.
        # A peer that cannot allocate must signal before entering rendezvous.
        with _collective_allocation(group, self.allocation_status):
            peer_bytes, local_bytes, flags, flag_bytes = tex.mega_moe_fused_workspace_sizes(
                capacity, *shape
            )
            local = torch.empty(local_bytes, dtype=torch.uint8, device=device)
            local.narrow(0, flags, flag_bytes).zero_()
            payload = symm.empty(peer_bytes, dtype=torch.uint8, device=device)
        self.payload = _mapping(group, payload)
        self.capacity, self.local, self.shape = capacity, local, shape

    def forward(self, x, ids, probs, fc1, fc2, group):
        stream = torch.cuda.current_stream(self.control[0].device)
        stream.wait_event(self.done)
        shape = (x.size(1), fc2[0].size(1), len(fc1), group.size(), ids.size(1))
        while True:
            self.epoch += 1
            configured = self.shape == shape
            out, status, max_tokens = tex.mega_moe_fused_forward(
                x,
                ids,
                probs,
                fc1,
                fc2,
                self.control[2],
                self.payload[2] if configured else [],
                self.local if configured else None,
                self.status,
                group.rank(),
                self.capacity if configured else 0,
                self.epoch,
            )
            self.done.record(stream)
            if status == 0:
                return out
            if status == 1:
                return None
            if status == 2:
                raise ValueError("Expected expert IDs in range or -1")
            if status == 4:
                raise torch.OutOfMemoryError("A rank could not allocate its MegaMoE output")
            if status != 3:
                raise RuntimeError(f"Unexpected MegaMoE pre-dispatch status {status}")
            # All ranks finished the previous epoch and requested the same
            # capacity. The second pre-dispatch uses the new mappings.
            capacity = (max_tokens + 255) // 256 * 256
            self._resize(group, shape, capacity, x.device)

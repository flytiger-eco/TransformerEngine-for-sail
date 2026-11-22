# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Route MegaMoE token copies to the ranks that own their experts."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class _RoutePlan:
    group: Optional[dist.ProcessGroup]
    send_counts: list[int]
    recv_counts: list[int]
    slots: torch.Tensor
    tokens: int
    topk: int

    def to_experts(self, values):
        return _exchange(values, self.group, self.send_counts, self.recv_counts)


def _exchange(values, group, send_counts, recv_counts):
    if group is None:
        return values
    output = values.new_empty((sum(recv_counts), *values.shape[1:]))
    dist.all_to_all_single(output, values.contiguous(), recv_counts, send_counts, group)
    return output


def make_route_plan(x, ids, num_local_experts, group, peer_fields=()):
    """Partition valid slots by destination, exchanging counts and optional peer metadata."""
    world = dist.get_world_size(group) if group is not None else 1
    tokens, topk = ids.shape
    flat_ids = ids.reshape(-1)
    valid = (flat_ids >= 0) & (flat_ids < world * num_local_experts)
    bad_id = ((flat_ids != -1) & ~valid).any()
    slots = valid.nonzero(as_tuple=True)[0]
    experts = flat_ids.index_select(0, slots)

    if world == 1:
        if num_local_experts <= 0 or bool(bad_id):
            raise ValueError("Expected at least one local expert and expert IDs in range or -1")
        plan = _RoutePlan(None, [], [], slots, tokens, topk)
        peer_metadata = []
    else:
        owners = experts.long() // max(num_local_experts, 1)
        order = owners.argsort(stable=True)
        slots = slots.index_select(0, order)
        experts = experts.index_select(0, order)
        send = torch.bincount(owners, minlength=world)

        # Exchange counts and validation together. Every rank sees an invalid ID or
        # inconsistent expert partition before anyone enters the payload collectives.
        expert_counts = send.new_full((world,), num_local_experts)
        metadata = torch.stack((send, bad_id.expand(world), expert_counts), dim=1)
        payload = [x.size(1), int(x.dtype == torch.float16)]
        fields = send.new_tensor([*payload, *peer_fields]).expand(world, -1)
        metadata = torch.cat((metadata, fields), dim=1)
        incoming = torch.empty_like(metadata)
        dist.all_to_all_single(incoming, metadata, group=group)
        counts = torch.cat((send[:, None], incoming), dim=1).tolist()
        if any(c[2] or c[3] != num_local_experts or c[3] <= 0 for c in counts):
            raise ValueError(
                "Expected equal positive local expert counts and expert IDs in range or -1"
            )
        if any(c[4:6] != payload for c in counts):
            raise ValueError("Expert-parallel input width and dtype must match on every rank")
        plan = _RoutePlan(
            group, [c[0] for c in counts], [c[1] for c in counts], slots, tokens, topk
        )
        experts = experts % num_local_experts
        peer_metadata = [row[6:] for row in counts] if peer_fields else []

    return plan, experts, peer_metadata


def dispatch(x, ids, probs, num_local_experts, ep_group_name):
    """Return expert-local x [R,H], probabilities [R], local IDs [R], and the inverse map."""
    group = dist.distributed_c10d._resolve_process_group(ep_group_name) if ep_group_name else None
    plan, experts, _ = make_route_plan(x, ids, num_local_experts, group)
    return (
        plan.to_experts(x.index_select(0, plan.slots // plan.topk)),
        plan.to_experts(probs.reshape(-1).index_select(0, plan.slots)),
        plan.to_experts(experts),
        plan,
    )


def return_routes(local_values, plan):
    """Restore [R,...] to source [T,K,...], filling unused slots with zero."""
    returned = _exchange(local_values, plan.group, plan.recv_counts, plan.send_counts)
    output = returned.new_zeros((plan.tokens * plan.topk, *returned.shape[1:]))
    output.index_copy_(0, plan.slots, returned)
    return output.view(plan.tokens, plan.topk, *returned.shape[1:])


def redispatch(source_values, plan):
    """Apply the original dispatch order to source [T,K,...] values."""
    # Backward can pass dY expanded over K, with stride zero on the top-k axis.
    selected = source_values[plan.slots // plan.topk, plan.slots % plan.topk]
    return plan.to_experts(selected)

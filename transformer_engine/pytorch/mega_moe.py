# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Routed BF16 and FP16 SwiGLU training on the PPU RTC backend."""

from typing import Optional, Sequence

import torch
from torch import Tensor
import transformer_engine_torch as tex

from ._mega_moe_routing import dispatch, redispatch, return_routes


def _pack(x, ids, probs, experts, group):
    if not x.is_cuda:
        raise NotImplementedError("MegaMoE RTC requires a CUDA/PPU device")
    x, probs, local_ids, plan = dispatch(x, ids, probs, experts, group)
    # Keep variable-size expert segments entirely on the device, including
    # repeated boundaries for empty experts.
    sorted_ids, order = local_ids.sort(stable=True)
    boundaries = torch.arange(experts + 1, device=x.device, dtype=torch.int32)
    offsets = torch.searchsorted(sorted_ids, boundaries, out_int32=True)
    return x.index_select(0, order), probs.index_select(0, order), offsets, order, plan


def _restore(values, order, plan):
    received = torch.empty_like(values).index_copy_(0, order, values)
    return return_routes(received, plan)


def _rtc_forward(
    x: Tensor, ids: Tensor, probs: Tensor, fc1: list[Tensor], fc2: list[Tensor], ep_group: str
) -> Tensor:
    if ep_group and x.is_cuda:
        from ._mega_moe_peer import peer_forward

        output = peer_forward(x, ids, probs, fc1, fc2, ep_group)
        if output is not None:
            return output
    x, probs, offsets, order, plan = _pack(x, ids, probs, len(fc1), ep_group)
    routed = tex.mega_moe_local_forward(x, probs, offsets, fc1, fc2)
    return tex.mega_moe_combine(_restore(routed, order, plan))


def _rtc_backward(
    dy: Tensor,
    x: Tensor,
    ids: Tensor,
    probs: Tensor,
    fc1: list[Tensor],
    fc2: list[Tensor],
    ep_group: str,
) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor]]:
    x, probs, offsets, order, plan = _pack(x, ids, probs, len(fc1), ep_group)
    routed_dy = redispatch(dy.unsqueeze(1).expand(-1, ids.size(1), -1), plan)
    dx, dp, dw1, dw2 = tex.mega_moe_local_backward(
        routed_dy.index_select(0, order), x, probs, offsets, fc1, fc2
    )
    dx = tex.mega_moe_combine(_restore(dx, order, plan))
    return dx, _restore(dp, order, plan), dw1, dw2


def _forward_shape(x, ids, probs, fc1, fc2, ep_group):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _backward_shapes(dy, x, ids, probs, fc1, fc2, ep_group):
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty_like(probs, memory_format=torch.contiguous_format),
        [torch.empty_like(w, memory_format=torch.contiguous_format) for w in fc1],
        [torch.empty_like(w, memory_format=torch.contiguous_format) for w in fc2],
    )


def _save_inputs(ctx, inputs, output):
    x, ids, probs, fc1, fc2, ctx.ep_group = inputs
    ctx.num_experts = len(fc1)
    ctx.save_for_backward(x, ids, probs, *fc1, *fc2)


def _register_ops(dtype_name):
    prefix = f"te_moe::{dtype_name}_mega_moe"
    forward = torch.library.custom_op(prefix + "_fwd", _rtc_forward, mutates_args=())
    backward = torch.library.custom_op(prefix + "_bwd", _rtc_backward, mutates_args=())
    forward.register_fake(_forward_shape)
    backward.register_fake(_backward_shapes)

    def autograd_backward(ctx, dy):
        x, ids, probs, *weights = ctx.saved_tensors
        dx, dp, dw1, dw2 = backward(
            dy, x, ids, probs, weights[: ctx.num_experts], weights[ctx.num_experts :], ctx.ep_group
        )
        return dx, None, dp, dw1, dw2, None

    forward.register_autograd(autograd_backward, setup_context=_save_inputs)
    return forward


_bf16_forward = _register_ops("bf16")
_fp16_forward = _register_ops("fp16")


def _mega_moe(x, expert_ids, routing_weights, fc1_weights, fc2_weights, ep_group, dtype, forward):
    if x.ndim != 2 or x.dtype != dtype:
        raise ValueError(f"x must have dtype {dtype} and shape [T,H]")
    if expert_ids.dtype != torch.int32 or routing_weights.dtype != torch.float32:
        raise ValueError("expert_ids must be int32 and routing_weights must be float32")
    if (
        expert_ids.ndim != 2
        or expert_ids.shape != routing_weights.shape
        or expert_ids.size(0) != x.size(0)
        or expert_ids.size(1) == 0
    ):
        raise ValueError("expert_ids and routing_weights must have the same [T,K] shape with K > 0")
    if x.device != expert_ids.device or x.device != routing_weights.device:
        raise ValueError("routes and x must be on the same device")
    if not fc1_weights or len(fc1_weights) != len(fc2_weights):
        raise ValueError("expected one FC1/FC2 weight pair per local expert")
    hidden = x.size(1)
    if fc2_weights[0].ndim != 2:
        raise ValueError("FC2 weight must be [H,I]")
    intermediate = fc2_weights[0].size(1)
    for w1, w2 in zip(fc1_weights, fc2_weights):
        if w1.shape != (2 * intermediate, hidden) or w2.shape != (hidden, intermediate):
            raise ValueError("FC1 and FC2 weights must have shapes [2I,H] and [H,I]")
        if any(w.dtype != dtype or w.device != x.device for w in (w1, w2)):
            raise ValueError("expert weights must have the same dtype and device as x")
    return forward(
        x,
        expert_ids,
        routing_weights,
        list(fc1_weights),
        list(fc2_weights),
        "" if ep_group is None else ep_group.group_name,
    )


def bf16_mega_moe(
    x: Tensor,
    expert_ids: Tensor,
    routing_weights: Tensor,
    fc1_weights: Sequence[Tensor],
    fc2_weights: Sequence[Tensor],
    *,
    ep_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tensor:
    """Apply routed BF16 SwiGLU experts and return BF16 [tokens, hidden].

    The input is [T,H]. Global expert IDs (int32) and routing weights (float32)
    are [T,K]. ID -1 skips a slot. Each rank supplies an equal number of experts
    in contiguous group-rank order, with FC1 [2I,H] (gate then up) and FC2 [H,I].
    Multiply SwiGLU by the routing weight before FC2. Accumulate the returned
    routes in FP32. Shared experts are caller-owned. ep_group=None is local.

    The PPU runtime compiles the expert kernels for the input precision. On
    peer-accessible EP groups, tokens and results travel through symmetric buffers.
    Large two-rank BF16 batches push tokens directly into expert-major rows for
    grouped GEMM and pull the results into an FP32 top-k reduction. Other peer
    batches read input rows in FC1 and return output slots in FC2. Groups without
    peer access use variable-split all-to-all. First-order backward routes gradients to the
    expert owners, recomputes FC1, and returns gradients for x, routing weights
    and both local expert parameter lists.
    """
    return _mega_moe(
        x,
        expert_ids,
        routing_weights,
        fc1_weights,
        fc2_weights,
        ep_group,
        torch.bfloat16,
        _bf16_forward,
    )


def fp16_mega_moe(
    x: Tensor,
    expert_ids: Tensor,
    routing_weights: Tensor,
    fc1_weights: Sequence[Tensor],
    fc2_weights: Sequence[Tensor],
    *,
    ep_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tensor:
    """FP16 MegaMoE with the routing, weight layouts and gradients of bf16_mega_moe."""
    return _mega_moe(
        x,
        expert_ids,
        routing_weights,
        fc1_weights,
        fc2_weights,
        ep_group,
        torch.float16,
        _fp16_forward,
    )

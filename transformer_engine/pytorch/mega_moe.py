# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Routed BF16 and FP16 SwiGLU training on the PPU RTC backend."""

import os
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch._subclasses.fake_tensor import is_fake
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
    if ep_group and x.is_cuda:
        from ._mega_moe_peer import peer_backward

        gradients = peer_backward(dy, x, ids, probs, fc1, fc2, ep_group)
        if gradients is not None:
            return gradients
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
    if (
        not torch.compiler.is_compiling()
        and (mode := os.getenv("NVTE_MOE_CAPTURE_TAG"))
        and x.is_cuda
        and not is_fake(x)
    ):
        from ._mega_moe_capture import capture

        tag = capture(mode, x, expert_ids, routing_weights, fc1_weights, fc2_weights, ep_group)
        with torch.cuda.nvtx.range(tag):
            return _run_mega_moe(
                x, expert_ids, routing_weights, fc1_weights, fc2_weights, ep_group, dtype, forward
            )
    return _run_mega_moe(
        x, expert_ids, routing_weights, fc1_weights, fc2_weights, ep_group, dtype, forward
    )


def _run_mega_moe(
    x, expert_ids, routing_weights, fc1_weights, fc2_weights, ep_group, dtype, forward
):
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
    if (
        not torch.compiler.is_compiling()
        and not is_fake(x)
        and ep_group is not None
        and ep_group.size() == 8
        and x.is_cuda
    ):
        from ._mega_moe_peer import _peer_supported, _workspace

        with torch.cuda.device(x.device):
            if _peer_supported(ep_group, x.device) and _workspace(ep_group).training_supported:
                # All eager EP8 peers must agree before entering different transports.
                # Sum two binary flags to distinguish mixed precision and mixed grad mode.
                mode = x.new_tensor(
                    [dtype == torch.bfloat16, torch.is_grad_enabled()], dtype=torch.int32
                )
                torch.distributed.all_reduce(mode, group=ep_group)
                bf16, grad = mode.tolist()
                if bf16 not in (0, 8):
                    raise ValueError("Expert-parallel input dtype must match on every rank")
                if grad not in (0, 8):
                    raise ValueError("Expert-parallel grad mode must match on every rank")
                if bf16 and grad:
                    from ._mega_moe_train import training_forward

                    output = training_forward(
                        x, expert_ids, routing_weights, fc1_weights, fc2_weights, ep_group
                    )
                    if output is not None:
                        return output
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
    Inference batches with K <= 8 on two or eight SM89 PPUs use three kernels after workspace
    setup and collective admission when H and I are multiples of 256, the BF16 weights are
    contiguous and aligned, each rank has at most 16 experts, and at least one rank has 2048 or
    more input tokens. Pre-dispatch builds expert route lists. FC1
    overlaps fetching expert input tiles with the gate/up projection and weighted
    SwiGLU. FC2 makes output columns available for cross-rank FP32 reduction.
    Other peer shapes use the bulk or generic peer implementation. Groups without
    peer access use variable-split all-to-all.

    Eager BF16 training on eight peer-accessible SM89 PPUs with Triton, K=8,
    the same tile and batch requirements, and I <= 16384 saves expert-local
    inputs, gate/up and SwiGLU outputs for backward. FC2 dgrad gathers source
    output gradients, FC1 dgrad scatters per-route input gradients, and both
    expert weight gradients are local grouped GEMMs. Backward returns gradients
    for x, routing weights, and both expert weight lists.

    The fallback first-order backward recomputes FC1 and returns gradients for x, routing
    weights and both local expert weight lists. Two peer-accessible SM89 PPUs
    with Triton use symmetric backward dispatch when H >= 256 and I >= 128
    are multiples of eight, each rank has at most 64 experts and K <= 8, and
    at least one rank has 4096 or more T*K slots. The sources send x, output
    gradients and routing weights to expert-major rows on their owners. On the
    grouped GEMM path, a separate stream returns the per-route input and
    routing-weight gradients once FC1 dgrad finishes, while the owner computes
    its two weight gradients on the caller's stream. Other EP shapes use
    variable-split all-to-all for backward dispatch and return.
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

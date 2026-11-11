# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Regression tests for the host descriptor cache of `general_grouped_gemm`.

`te_general_grouped_gemm` keeps one descriptor per expert in thread-local
storage and refreshes it in place when it is handed the same tensor again. The
identity it compared was `(PyObject*, data_ptr, dtype)`, which a torch view can
satisfy while carrying a different shape: `GroupedLinear.backward` re-creates
its split views in every layer and the allocator hands the same address back, so
an fc1 dgrad could leave a 1024-wide B behind for fc2's dgrad and trip the
dimension check in `getGemmOutputShape`.
"""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm

RTOL = 0.02
ATOL = 0.02

# fc1 and fc2 of a MoE block, as (N, K) of an expert weight.
FC1 = (1024, 2048)
FC2 = (2048, 512)
M = 173
G = 4
SPLITS = [M, 0, 0, 0]

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and te.is_bf16_available()),
    reason="grouped GEMM requires a CUDA device with bf16 support",
)


def _dgrad(weights, grad_out_parts, m_splits, k):
    """D = grad_out @ W. NN convention: A=(N,K), B=(m,N), out=(m,K)."""
    out = torch.empty(sum(m_splits), k, device="cuda", dtype=torch.bfloat16)
    general_grouped_gemm(
        weights,
        grad_out_parts,
        [out],
        [None] * len(weights),
        torch.bfloat16,
        layout="NN",
        m_splits=m_splits,
        grad=True,
        single_output=True,
    )
    return out


def test_descriptor_shape_change():
    """Same PyObject and data_ptr, new shape: the descriptor must be refreshed.

    Deterministic repro of the stale-shape assertion: `resize_` keeps the buffer
    in place, so the identity the cache compares is unchanged while the width
    goes 1024 -> 2048.
    """
    weight1 = torch.randn(*FC1, device="cuda", dtype=torch.bfloat16)
    weight2 = torch.randn(*FC2, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(M, FC2[0], device="cuda", dtype=torch.bfloat16)
    ptr, obj = grad_out.data_ptr(), id(grad_out)

    grad_out.resize_(M, FC1[0])
    out1 = _dgrad([weight1], [grad_out], [M], FC1[1])
    torch.testing.assert_close(
        out1.float(), grad_out.float() @ weight1.float(), rtol=RTOL, atol=ATOL
    )

    grad_out.resize_(M, FC2[0])
    assert id(grad_out) == obj and grad_out.data_ptr() == ptr
    out2 = _dgrad([weight2], [grad_out], [M], FC2[1])
    torch.testing.assert_close(
        out2.float(), grad_out.float() @ weight2.float(), rtol=RTOL, atol=ATOL
    )


def test_descriptor_slot_handoff_with_empty_groups():
    """Replay the backward call pattern: zero-token experts hand their slot on."""
    weights1 = [torch.randn(*FC1, device="cuda", dtype=torch.bfloat16) for _ in range(G)]
    weights2 = [torch.randn(*FC2, device="cuda", dtype=torch.bfloat16) for _ in range(G)]
    for _ in range(2):
        for weights, (n, k) in ((weights1, FC1), (weights2, FC2)):
            grad_out = torch.randn(sum(SPLITS), n, device="cuda", dtype=torch.bfloat16)
            parts = list(torch.split(grad_out, SPLITS))
            out = _dgrad(weights, parts, SPLITS, k)
            torch.testing.assert_close(
                out.float(), grad_out.float() @ weights[0].float(), rtol=RTOL, atol=ATOL
            )

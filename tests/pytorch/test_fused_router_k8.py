import pytest
import torch
import transformer_engine_torch as tex


@pytest.mark.parametrize("num_tokens", [1, 13])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("function,pre", [("softmax", False), ("softmax", True), ("sigmoid", False), ("sqrtsoftplus", False)])
def test_k8_final_selection_matches_indices(num_tokens, grouped, function, pre):
    experts = 192
    groups, group_topk = (8, 4) if grouped else (None, None)
    expert = torch.arange(experts, device="cuda")[None, :]
    token = torch.arange(num_tokens, device="cuda")[:, None]
    logits = (((expert + token) % 11) - 5).float() * 0.25
    logits = torch.where((token % 4 == 0) & (expert % 2 == 0), -0.0, logits)
    logits = torch.where((token % 4 == 0) & (expert % 2 == 1), 0.0, logits)
    logits = torch.where((token % 4 == 1) & (expert % 17 == 0), -torch.inf, logits)
    if function == "softmax" and not pre:
        logits = torch.where((token % 4 == 2) & (expert % 13 == 0), torch.nan, logits)
    args = (logits, 8, pre, groups, group_topk, 1.2, function, None, int(tex.NVTERoutingMapFormat.BYTEMAP))
    probs, mask, saved = tex.fused_topk_with_score_function_fwd(*args, None)
    indices = torch.empty((num_tokens, 8), device="cuda", dtype=torch.int32)
    expected_probs, _, expected_saved = tex.fused_topk_with_score_function_fwd(*args, indices)
    expected_mask = torch.zeros_like(mask).scatter_(1, indices.long(), True)
    assert torch.equal(mask, expected_mask)
    assert torch.equal(probs.view(torch.int32), expected_probs.view(torch.int32))
    assert torch.equal(saved.view(torch.int32), expected_saved.view(torch.int32))


@pytest.mark.parametrize("num_tokens", [1, 5, 13])
@pytest.mark.parametrize("scale", [0.0, 1.2, -0.75])
def test_k8_masked_dot_rounding(num_tokens, scale):
    experts = 192
    lanes = torch.tensor([0, 32, 64, 1, 33, 95, 127, 191], device="cuda")
    token = torch.arange(num_tokens, device="cuda")[:, None]
    idx = (lanes[None, :] + token * 7) % experts
    mask = torch.zeros((num_tokens, experts), device="cuda", dtype=torch.bool).scatter_(1, idx, True)
    bit = torch.arange(8, device="cuda")
    packed = (mask.view(num_tokens, -1, 8).long() << bit).sum(-1).to(torch.uint8)
    generator = torch.Generator().manual_seed(331)
    weights = torch.rand(num_tokens, experts, generator=generator).cuda().masked_fill(~mask, 0)
    saved = (weights / weights.sum(1, keepdim=True)).masked_fill(~mask, -torch.inf)
    grad = torch.randn(num_tokens, experts, generator=generator).cuda()
    grad[::2, :] += 0.9375
    grad.masked_fill_(~mask, torch.nan)
    dense = torch.full_like(grad, torch.nan)
    control = torch.full_like(grad, torch.nan)
    for routing, result, fmt in ((mask, dense, tex.NVTERoutingMapFormat.BYTEMAP),
                                 (packed, control, tex.NVTERoutingMapFormat.BITMAP_U8)):
        tex.fused_topk_with_score_function_bwd(routing, saved, grad, result, 8, False, scale,
                                              "softmax", False, int(fmt))
    assert torch.equal(dense.view(torch.int32), control.view(torch.int32))
    assert not torch.count_nonzero(dense[~mask].view(torch.int32))
    assert torch.isfinite(dense[mask]).all()

import pytest
import torch
import transformer_engine_torch as tex


def check_k16_forward(logits):
    args = (logits, 16, False, 8, 4, 1.2, "softmax", None, int(tex.NVTERoutingMapFormat.BYTEMAP))
    probs, mask, saved = tex.fused_topk_with_score_function_fwd(*args, None)
    indices = torch.empty((logits.size(0), 16), device=logits.device, dtype=torch.int32)
    reference, _, reference_saved = tex.fused_topk_with_score_function_fwd(*args, indices)
    expected_mask = torch.zeros_like(mask).scatter_(1, indices.long(), True)
    assert torch.equal(mask, expected_mask)
    assert torch.all(mask.sum(1) == 16)
    assert torch.equal(probs.view(torch.int32), reference.view(torch.int32))
    assert torch.equal(saved.view(torch.int32), reference_saved.view(torch.int32))
    return mask


@pytest.mark.parametrize("tokens", [1, 5, 13])
@pytest.mark.parametrize("distribution", ["random", "ties", "signed_zeros", "nonfinite"])
def test_grouped_k16_radix_matches_indices(tokens, distribution):
    logits = torch.randn(tokens, 192, generator=torch.Generator().manual_seed(43)).cuda()
    expert = torch.arange(192, device=logits.device)[None, :]
    token = torch.arange(tokens, device=logits.device)[:, None]
    if distribution == "ties":
        logits = ((expert + 7 * token) % 5).float()
    elif distribution == "signed_zeros":
        logits = torch.where((expert + token) % 3 == 0, -0.0, torch.zeros_like(logits))
    elif distribution == "nonfinite":
        logits[:, ::19] = -torch.inf
        logits[:, 1::23] = torch.inf
        logits.view(torch.int32)[:, 2::29] = 0x7FC00017
        logits.view(torch.int32)[:, 3::31] = -0x003FFFED
    check_k16_forward(logits)


def test_grouped_k16_uses_radix_gather_order_for_fp32_group_sum():
    logits = torch.full((1, 8, 24), -float(2**26), device="cuda")
    logits[0, 0, :4] = torch.tensor([1.0, 1.0, 2**24, -(2**24)], device="cuda")
    logits[0, 1:4, :4] = 100.0
    logits[0, 4, :4] = 0.25
    mask = check_k16_forward(logits.flatten(1)).view(8, 24)
    assert mask[0, 2] and not mask[4].any()


def test_k16_cutoff_ties_across_expert_tiles():
    logits = torch.full((1, 8, 24), -4.0, device="cuda")
    groups = [0, 2, 4, 7]
    logits[0, groups, :6] = 2.0
    logits[0, groups, -1] = 3.0
    mask = check_k16_forward(logits.flatten(1)).view(8, 24)
    expected = torch.zeros_like(mask)
    expected[groups, -1] = True
    expected[[0, 2], :6] = True
    assert torch.equal(mask, expected)


def test_k16_all_mask_sentinels_keep_original_indices():
    logits = torch.full((5, 192), -torch.inf, device="cuda")
    mask = check_k16_forward(logits)
    assert mask[:, :16].all()
    assert not mask[:, 16:].any()


@pytest.mark.parametrize("tokens", [1, 5, 13])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("scale", [0.0, 1.2, -0.75, 0.07])
def test_k16_backward_matches_bitmap_with_real_forward(tokens, grouped, scale):
    generator = torch.Generator().manual_seed(129)
    logits = torch.randn(tokens, 192, generator=generator).cuda()
    logits[::3] = logits[::3].round()
    groups, group_topk = (8, 4) if grouped else (None, None)
    dense_fmt = int(tex.NVTERoutingMapFormat.BYTEMAP)
    packed_fmt = int(tex.NVTERoutingMapFormat.BITMAP_U8)
    _, mask, saved = tex.fused_topk_with_score_function_fwd(
        logits, 16, False, groups, group_topk, scale, "softmax", None, dense_fmt, None
    )
    bit = torch.arange(8, device=logits.device)
    packed = (mask.view(tokens, -1, 8).long() << bit).sum(-1).to(torch.uint8)
    grad = torch.randn(tokens, 192, generator=generator).cuda()
    grad.masked_fill_(~mask, torch.nan)
    if tokens > 1:
        grad[1].masked_fill_(mask[1], torch.inf)
    actual = torch.full_like(grad, torch.nan)
    expected = torch.full_like(grad, torch.nan)
    for routing, output, fmt in ((mask, actual, dense_fmt), (packed, expected, packed_fmt)):
        tex.fused_topk_with_score_function_bwd(
            routing, saved, grad, output, 16, False, scale, "softmax", False, fmt
        )
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert not torch.count_nonzero(actual[~mask].view(torch.int32))
    assert torch.isfinite(actual[0, mask[0]]).all()

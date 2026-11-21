import pytest
import torch
import transformer_engine_torch as tex


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk", [1, 2, 7, 8, 9, 12, 15, 16, 17, 24])
@pytest.mark.parametrize("experts,groups,keep", [(192, 8, 1), (193, None, None)])
@pytest.mark.parametrize("function", ["softmax", "sigmoid", "sqrtsoftplus"])
def test_lowp_routing_matches_bitmap(dtype, topk, experts, groups, keep, function):
    generator = torch.Generator().manual_seed(1681 + experts + topk)
    logits = torch.randn(5, experts, generator=generator).round().to(device="cuda", dtype=dtype)
    grad = torch.randn(5, experts, generator=generator).to(device="cuda", dtype=dtype)
    bias = None if function == "softmax" else torch.linspace(-0.25, 0.25, experts, device="cuda")
    dense, bits = int(tex.NVTERoutingMapFormat.BYTEMAP), int(tex.NVTERoutingMapFormat.BITMAP_U8)

    def forward(fmt):
        return tex.fused_topk_with_score_function_fwd(
            logits, topk, False, groups, keep, -0.75, function, bias, fmt, None
        )

    probs, mask, saved = forward(dense)
    ref_probs, packed, ref_saved = forward(bits)
    positions = torch.arange(experts, device="cuda")
    reference_mask = ((packed[:, positions // 8].long() >> (positions % 8)) & 1).bool()
    assert torch.equal(mask, reference_mask)
    assert torch.equal(probs.view(torch.int16), ref_probs.view(torch.int16))
    assert torch.equal(saved.view(torch.int32), ref_saved.view(torch.int32))
    grad.masked_fill_(~mask, torch.nan)
    actual, expected = torch.empty_like(grad), torch.empty_like(grad)
    for routing, act, output, fmt in ((mask, saved, actual, dense), (packed, ref_saved, expected, bits)):
        tex.fused_topk_with_score_function_bwd(
            routing, act, grad, output, topk, False, -0.75, function, False, fmt
        )
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk", [1, 2, 7, 8, 9, 12, 15, 16, 17, 24])
def test_lowp_postsoftmax_matches_fp64(dtype, topk):
    experts = 192
    columns = torch.arange(experts)
    logits = ((columns * 53 + 11) % experts).double().reshape(1, -1) / 128
    upstream = ((columns % 31) - 15).double().reshape(1, -1) / 16
    dense = int(tex.NVTERoutingMapFormat.BYTEMAP)
    probs, mask, saved = tex.fused_topk_with_score_function_fwd(
        logits.to(device="cuda", dtype=dtype), topk, False, None, None, 1.2, "softmax", None, dense, None
    )
    grad = upstream.to(device="cuda", dtype=dtype)
    actual = torch.empty_like(grad)
    tex.fused_topk_with_score_function_bwd(mask, saved, grad, actual, topk, False, 1.2, "softmax", False, dense)
    values, indices = logits.topk(topk)
    weights = values.softmax(-1)
    scaled = upstream.gather(1, indices) * 1.2
    expected_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, indices, True)
    assert torch.equal(mask.cpu(), expected_mask)
    for output, reference in ((probs, weights * 1.2), (actual, weights * (scaled - (scaled * weights).sum(-1, keepdim=True)))):
        expected = torch.zeros_like(logits).scatter_(1, indices, reference)
        torch.testing.assert_close(output.cpu().double(), expected, rtol=torch.finfo(dtype).eps * 1.1, atol=3e-7)

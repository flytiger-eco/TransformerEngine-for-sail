import pytest
import torch
import transformer_engine_torch as tex


@pytest.mark.parametrize("experts", [31, 33, 64, 127, 129, 192, 193, 255, 256, 257])
@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("scale", [1.2, -0.75])
def test_postsoftmax_vjp_matches_bitmap(experts, topk, scale):
    generator = torch.Generator().manual_seed(317 + experts + topk)
    logits = torch.randn(5, experts, generator=generator).cuda()
    grad = torch.randn(5, experts, generator=generator).cuda()
    dense_format = int(tex.NVTERoutingMapFormat.BYTEMAP)
    bitmap_format = int(tex.NVTERoutingMapFormat.BITMAP_U8)
    _, mask, saved = tex.fused_topk_with_score_function_fwd(
        logits, topk, False, None, None, scale, "softmax", None, dense_format, None
    )
    grad.masked_fill_(~mask, torch.nan)
    padded = torch.nn.functional.pad(mask, (0, -experts % 8))
    shifts = torch.arange(8, device="cuda")
    bitmap = (padded.reshape(5, -1, 8).long() << shifts).sum(-1).to(torch.uint8)
    actual, reference = torch.empty_like(grad), torch.empty_like(grad)
    for routing, output, fmt in ((bitmap, reference, bitmap_format), (mask, actual, dense_format)):
        tex.fused_topk_with_score_function_bwd(
            routing, saved, grad, output, topk, False, scale, "softmax", False, fmt
        )
    assert torch.equal(actual.view(torch.int32), reference.view(torch.int32))
    assert not torch.count_nonzero(actual[~mask].view(torch.int32))

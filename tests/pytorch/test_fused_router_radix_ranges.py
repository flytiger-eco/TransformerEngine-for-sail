import pytest
import torch
import transformer_engine_torch as tex


@pytest.mark.parametrize("experts", [31, 33, 128, 191, 192, 193, 255, 256, 257])
@pytest.mark.parametrize("requested", [16, 17, 23, 24, 32, 48, None])
def test_radix_ordered_key_gather(experts, requested):
    topk = experts if requested is None else min(experts, requested)
    logits = ((torch.arange(experts) * 37) % 17 - 8).float()
    logits[::9], logits[1::11], logits[3::13] = -0.0, 0.0, -torch.inf
    logits[5], logits[7] = torch.nan, torch.inf
    bits = [value & 0xFFFFFFFF for value in logits.view(torch.int32).tolist()]
    keys = [value ^ (0xFFFFFFFF if value >> 31 else 0x80000000) for value in bits]
    cutoff = sorted(keys, reverse=True)[topk - 1]
    expected = [pos for pos, key in enumerate(keys) if key > cutoff]
    expected.extend([pos for pos, key in enumerate(keys) if key == cutoff][:topk - len(expected)])
    indices = torch.empty(1, topk, dtype=torch.int32, device="cuda")
    args = (logits[None].cuda(), topk, False, None, None, 1.0, "softmax", None,
            int(tex.NVTERoutingMapFormat.BYTEMAP))
    probs, mask, saved = tex.fused_topk_with_score_function_fwd(*args, None)
    index_probs, _, index_saved = tex.fused_topk_with_score_function_fwd(*args, indices)
    assert torch.equal(probs.view(torch.int32), index_probs.view(torch.int32))
    assert torch.equal(saved.view(torch.int32), index_saved.view(torch.int32))
    assert torch.equal(mask, torch.zeros_like(mask).scatter_(1, indices.long(), True))
    assert indices.cpu().tolist() == [expected]

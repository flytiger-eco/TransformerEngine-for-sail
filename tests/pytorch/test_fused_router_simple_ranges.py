import pytest
import torch
import transformer_engine_torch as tex


@pytest.mark.parametrize("topk", range(1, 16))
@pytest.mark.parametrize("experts", [33, 128, 193, 256, 257])
def test_simple_topk_range_matches_cpu(experts, topk):
    columns = torch.arange(experts)
    tokens = torch.arange(5)[:, None]
    logits = ((37 * columns + 13 * tokens) % experts).float() / experts
    groups, group_topk = (4, 2 if topk % 2 == 0 else 1) if experts % 4 == 0 else (None, None)
    if groups:
        logits += (columns // (experts // groups)) * 2
    expected = logits.topk(topk, dim=1).indices
    indices = torch.empty(5, topk, dtype=torch.int32, device="cuda")
    args = (logits.cuda(), topk, False, groups, group_topk, 1.2, "softmax", None,
            int(tex.NVTERoutingMapFormat.BYTEMAP))
    probs, mask, _ = tex.fused_topk_with_score_function_fwd(*args, None)
    tex.fused_topk_with_score_function_fwd(*args, indices)
    assert torch.equal(mask.cpu(), torch.zeros_like(logits, dtype=torch.bool).scatter_(1, expected, True))
    assert torch.equal(indices.cpu().long(), expected)
    weights = logits.double().gather(1, expected).softmax(1) * 1.2
    reference = torch.zeros_like(logits).scatter_(1, expected, weights.float())
    torch.testing.assert_close(probs.cpu(), reference, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("topk", [2, 7, 8, 9, 12, 15])
def test_simple_ties_and_first_lane_nan(topk):
    logits = torch.tensor([float((column * 13) % 7) for column in range(193)])
    logits[0], logits[32], logits[65] = torch.nan, -torch.inf, torch.nan
    scores = logits.tolist()
    expected = []
    for _ in range(topk):
        winners = list(range(32))
        values = [-float("inf")] * 32
        for lane in range(32):
            for column in range(lane, len(scores), 32):
                if column in expected:
                    continue
                if (topk > 8 and column == lane) or scores[column] > values[lane]:
                    values[lane], winners[lane] = scores[column], column
        for stride in (16, 8, 4, 2, 1):
            source = [lane ^ stride if values[lane ^ stride] > values[lane] else lane for lane in range(32)]
            values, winners = [values[pos] for pos in source], [winners[pos] for pos in source]
        expected.append(winners[0])
    indices = torch.empty(1, topk, dtype=torch.int64, device="cuda")
    args = (logits[None].cuda(), topk, False, None, None, 1.0, "softmax", None,
            int(tex.NVTERoutingMapFormat.BYTEMAP))
    probs, mask, saved = tex.fused_topk_with_score_function_fwd(*args, None)
    index_probs, _, index_saved = tex.fused_topk_with_score_function_fwd(*args, indices)
    assert torch.equal(probs.view(torch.int32), index_probs.view(torch.int32))
    assert torch.equal(saved.view(torch.int32), index_saved.view(torch.int32))
    assert torch.equal(mask, torch.zeros_like(mask).scatter_(1, indices, True))
    assert indices.cpu().tolist() == [expected]

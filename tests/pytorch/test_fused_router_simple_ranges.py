import os

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


@pytest.mark.skipif(int(os.getenv("NVTE_RADIX_TOPK_THRESHOLD", "16")) <= 1, reason="tests naive tie order")
@pytest.mark.parametrize("experts", [8, 33])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_singleton_groups_top1(experts, dtype):
    logits = torch.full((6, experts), -torch.inf, dtype=dtype, device="cuda")
    logits[0, -1] = 3
    logits[1, 1:3] = 2
    logits[2, 1], logits[2, 2] = -0.0, 0.0
    logits[3] = torch.nan
    logits[5] = torch.nan
    logits[5, 1] = 1
    expected = torch.tensor([experts - 1, 2, 2, 0, 0, 1], device="cuda").unsqueeze(1)
    args = (logits, 1, False, experts, 1, 1.25, "softmax", None)
    byte = int(tex.NVTERoutingMapFormat.BYTEMAP)
    probs, mask, saved = tex.fused_topk_with_score_function_fwd(*args, byte, None)
    wanted = torch.zeros_like(mask).scatter_(1, expected, True)
    assert torch.equal(mask, wanted)
    finite = [0, 1, 2, 5]
    assert torch.equal(probs[finite], wanted[finite].to(dtype) * 1.25)
    assert torch.equal(saved[finite], torch.where(wanted[finite], 1.0, -torch.inf))
    for index_dtype in (torch.int16, torch.int32, torch.int64):
        indices = torch.empty_like(expected, dtype=index_dtype)
        tex.fused_topk_with_score_function_fwd(*args, byte, indices)
        assert torch.equal(indices.long(), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_small_group_scores_select_by_top_values_in_rank_order(dtype):
    fmt = int(tex.NVTERoutingMapFormat.BYTEMAP)
    logits = torch.full((1, 24), -100.0, dtype=dtype, device="cuda")
    logits[0, :3] = torch.tensor([32768, -32768, 2.0**-10], dtype=dtype, device="cuda")
    logits[0, 3:6] = 2.0**-12
    probs, mask, _ = tex.fused_topk_with_score_function_fwd(
        logits, 3, False, 8, 1, 1.0, "softmax", None, fmt, None
    )
    assert mask.nonzero()[:, 1].tolist() == [3, 4, 5]
    torch.testing.assert_close(probs[mask], torch.full_like(probs[mask], 1 / 3))

    logits = torch.full((1, 16), -100.0, dtype=dtype, device="cuda")
    logits[0, 0], logits[0, 2], logits[0, 4:6] = 10, 8, 7
    _, mask, _ = tex.fused_topk_with_score_function_fwd(
        logits, 2, False, 8, 2, 1.0, "softmax", None, fmt, None
    )
    assert mask.nonzero()[:, 1].tolist() == [0, 2]

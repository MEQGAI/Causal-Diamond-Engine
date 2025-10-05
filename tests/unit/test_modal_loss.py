import torch

from fm_core.projection import make_view_mask
from fm_train.objectives import compute_modal_penalty


def test_make_view_mask_shapes_and_causality():
    batch = 2
    seq_len = 8
    slot_len = 2
    centers = torch.tensor([1, 2])
    mask = make_view_mask(
        batch, seq_len, slot_len, centers, window=1, device=torch.device("cpu")
    )
    assert mask.shape == (batch, 1, seq_len, seq_len)
    # Future positions should be masked
    assert torch.isinf(mask[0, 0, 0, 1])
    # Allowed slot remains zero
    assert mask[0, 0, 3, 2] == 0
    # Slot outside window masked
    assert torch.isinf(mask[0, 0, 0, 6])


def test_modal_penalty_zero_on_identical_logits():
    torch.manual_seed(0)
    batch, seq_len, vocab = 2, 6, 4
    planner_logits = torch.randn(batch, seq_len, vocab)
    plan_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    plan_mask[:, :3] = True
    centers = torch.tensor([0, 1])
    result = compute_modal_penalty(
        planner_logits=planner_logits,
        planner_logits_view=planner_logits.clone(),
        token_logits=None,
        token_logits_view=None,
        plan_pos_mask=plan_mask,
        plan_span_mask=None,
        slot_len=2,
        center_slots=centers,
        lambda_planner=1.0,
        lambda_token=0.0,
    )
    assert torch.allclose(result["l_mod_total"], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(result["l_mod_planner"], torch.tensor(0.0), atol=1e-6)


def test_modal_penalty_positive_for_mismatch():
    planner_logits = torch.zeros(1, 4, 3)
    planner_logits_view = planner_logits.clone()
    planner_logits_view[:, :, 0] = 2.0
    plan_mask = torch.tensor([[True, True, False, False]])
    centers = torch.tensor([0])
    result = compute_modal_penalty(
        planner_logits=planner_logits,
        planner_logits_view=planner_logits_view,
        token_logits=None,
        token_logits_view=None,
        plan_pos_mask=plan_mask,
        plan_span_mask=None,
        slot_len=2,
        center_slots=centers,
        lambda_planner=1.0,
        lambda_token=0.0,
    )
    assert result["l_mod_planner"] > 0

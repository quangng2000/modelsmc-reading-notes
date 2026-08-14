from __future__ import annotations

import math

import torch

from research.local_repair_importance import repair_distribution


def test_repair_distribution_is_softmax_with_uniform_floor() -> None:
    scores = (-2.0, -2.2, 1.0, -1.8)
    actual = repair_distribution(scores, temperature=1.0, epsilon=0.05)
    logits = torch.tensor(scores, dtype=torch.float64)
    expected = 0.95 * torch.softmax(logits, dim=0) + 0.05 / 4
    assert torch.allclose(actual, expected, atol=1e-15, rtol=0.0)
    assert math.isclose(float(actual.sum().item()), 1.0, abs_tol=1e-15)
    assert bool(torch.all(actual > 0.0).item())


def test_path_probability_is_product_of_conditional_repairs() -> None:
    q_mapper = repair_distribution((0.0, 1.0), temperature=1.0, epsilon=0.1)
    q_predicate = repair_distribution((2.0, -1.0), temperature=1.0, epsilon=0.1)
    joint = torch.outer(q_mapper, q_predicate)
    assert math.isclose(float(joint.sum().item()), 1.0, abs_tol=1e-15)
    assert joint[1, 0] == q_mapper[1] * q_predicate[0]

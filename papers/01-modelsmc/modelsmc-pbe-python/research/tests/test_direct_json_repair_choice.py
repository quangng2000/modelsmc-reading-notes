from __future__ import annotations

import math

import torch

from research.direct_json_repair_choice import calibrated_distribution


def test_calibrated_distribution_uses_smoothed_counts_and_uniform_floor() -> None:
    actual = calibrated_distribution((0, 0, 8, 0), alpha=1.0, epsilon=0.05)
    predictive = torch.tensor((1, 1, 9, 1), dtype=torch.float64) / 12
    expected = 0.95 * predictive + 0.05 / 4
    assert torch.allclose(actual, expected, atol=1e-15, rtol=0.0)
    assert math.isclose(float(actual.sum().item()), 1.0, abs_tol=1e-15)
    assert bool(torch.all(actual > 0.0).item())


def test_symmetric_prior_counts_produce_uniform_distribution() -> None:
    actual = calibrated_distribution((0, 0, 0, 0), alpha=1.0, epsilon=0.05)
    assert torch.equal(actual, torch.full((4,), 0.25, dtype=torch.float64))

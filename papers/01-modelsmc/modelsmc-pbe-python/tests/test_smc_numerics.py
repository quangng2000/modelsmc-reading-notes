from __future__ import annotations

import math

import pytest
import torch

from modelsmc_pbe.runtime import make_cpu_generator
from modelsmc_pbe.smc import (
    categorical_sample,
    effective_sample_size,
    gibbs_log_target,
    normalize_log_weights,
    normalize_weights,
    systematic_resample,
    tempered_log_potential,
)


def test_log_normalization_is_stable_and_cpu_float64() -> None:
    normalized = normalize_log_weights(torch.tensor([-1000.0, -1001.0, -1002.0]))

    assert normalized.weights.device.type == "cpu"
    assert normalized.weights.dtype is torch.float64
    assert normalized.weights.sum().item() == pytest.approx(1.0)
    assert normalized.weights.tolist() == pytest.approx([0.66524096, 0.24472847, 0.09003057])
    assert normalized.log_mean_weight == pytest.approx(
        normalized.log_normalizer - math.log(3)
    )


def test_positive_infinite_log_weights_share_all_mass() -> None:
    normalized = normalize_log_weights(torch.tensor([float("inf"), 3.0, float("inf")]))

    assert normalized.weights.tolist() == [0.5, 0.0, 0.5]
    assert normalized.log_normalizer == math.inf


@pytest.mark.parametrize(
    "values, message",
    [
        ([float("nan"), 0.0], "NaN"),
        ([float("-inf"), float("-inf")], "zero mass"),
    ],
)
def test_invalid_log_weight_populations_fail(values: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_log_weights(torch.tensor(values))


def test_ess_has_expected_extremes() -> None:
    assert effective_sample_size(torch.ones(4)) == pytest.approx(4.0)
    assert effective_sample_size(torch.tensor([1.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_normalization_never_applies_negative_residual_to_zero_bin() -> None:
    probabilities = normalize_weights(torch.tensor([1.0, 1.0, 1.0, 0.0]))

    assert probabilities.sum().item() == pytest.approx(1.0)
    assert bool(torch.all(probabilities >= 0).item())
    assert probabilities[-1].item() == 0.0


def test_normalization_avoids_overflow_in_finite_weight_sum() -> None:
    probabilities = normalize_weights(
        torch.tensor([torch.finfo(torch.float64).max] * 2, dtype=torch.float64)
    )

    assert probabilities.tolist() == [0.5, 0.5]


def test_systematic_resampling_is_seeded_and_monotone() -> None:
    weights = torch.tensor([0.7, 0.2, 0.1])
    first = systematic_resample(weights, generator=make_cpu_generator(19))
    second = systematic_resample(weights, generator=make_cpu_generator(19))

    assert torch.equal(first, second)
    assert first.device.type == "cpu"
    assert first.dtype is torch.int64
    assert bool(torch.all(first[1:] >= first[:-1]).item())
    assert first.min().item() >= 0
    assert first.max().item() < 3


def test_systematic_resampling_uses_right_open_cdf_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def zero_offset(*_args: object, **_kwargs: object) -> torch.Tensor:
        return torch.tensor(0.0, dtype=torch.float64)

    monkeypatch.setattr(torch, "rand", zero_offset)

    balanced = systematic_resample(
        torch.tensor([0.5, 0.5]), generator=make_cpu_generator(1)
    )
    leading_zero = systematic_resample(
        torch.tensor([0.0, 0.5, 0.5]), generator=make_cpu_generator(1)
    )

    assert balanced.tolist() == [0, 1]
    assert leading_zero.tolist() == [1, 1, 2]


def test_categorical_sampling_is_seeded() -> None:
    weights = torch.tensor([0.1, 0.2, 0.7])
    first = categorical_sample(weights, 20, generator=make_cpu_generator(123))
    second = categorical_sample(weights, 20, generator=make_cpu_generator(123))

    assert torch.equal(first, second)
    assert first.shape == (20,)


def test_tempered_potential_and_target_stay_on_compute_device() -> None:
    losses = torch.tensor([0.0, 2.0, 5.0])
    costs = torch.tensor([3.0, 4.0, 1.0])

    potential = tempered_log_potential(
        losses,
        beta_previous=0.25,
        beta_current=0.5,
        loss_scale=2.0,
    )
    target = gibbs_log_target(
        losses,
        costs,
        beta=0.5,
        loss_scale=2.0,
        cost_scale=0.1,
    )

    assert potential.device == losses.device
    assert potential.tolist() == pytest.approx([0.0, -1.0, -2.5])
    assert target.tolist() == pytest.approx([-0.3, -2.4, -5.1])


def test_accelerator_potentials_normalize_on_cpu_when_available() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        pytest.skip("no accelerator available")

    losses = torch.tensor([0.0, 1.0, 2.0], device=device)
    log_potential = tempered_log_potential(
        losses,
        beta_previous=0.0,
        beta_current=1.0,
        loss_scale=1.0,
    )
    normalized = normalize_log_weights(log_potential)

    assert log_potential.device.type == device.type
    assert normalized.weights.device.type == "cpu"
    assert normalized.weights.dtype is torch.float64

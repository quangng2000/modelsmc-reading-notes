"""Torch SMC numerics with deterministic CPU-float64 sampling decisions.

Loss and potential batches can be computed on CUDA or MPS.  Normalization,
ESS, and discrete random draws are intentionally moved to CPU float64 so the
same seed follows the same population-control path on every accelerator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class NormalizedWeights:
    weights: Tensor
    log_normalizer: float
    log_mean_weight: float


def _one_dimensional(values: Tensor, name: str) -> Tensor:
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {tuple(values.shape)}")
    if values.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    # Keep these as two transfers: MPS cannot cast to float64 even when the
    # destination in the same ``to`` call is CPU.
    return values.detach().to(device="cpu").to(dtype=torch.float64)


def _potential_dtype(values: Tensor) -> torch.dtype:
    # MPS does not implement float64.  Its potentials are promoted when they
    # cross the normalization boundary below.
    return torch.float32 if values.device.type == "mps" else torch.float64


def tempered_log_potential(
    losses: Tensor,
    *,
    beta_previous: float,
    beta_current: float,
    loss_scale: float,
) -> Tensor:
    """Compute ``log G_t = -(beta_t-beta_(t-1))*scale*loss`` on-device."""

    if not 0.0 <= beta_previous <= beta_current:
        raise ValueError("beta values must satisfy 0 <= beta_previous <= beta_current")
    if not math.isfinite(beta_current) or not math.isfinite(loss_scale) or loss_scale < 0.0:
        raise ValueError(
            "beta_current and loss_scale must be finite; loss_scale must be nonnegative"
        )
    typed = losses.to(dtype=_potential_dtype(losses))
    if bool(torch.any(~torch.isfinite(typed)).item()):
        raise ValueError("losses must be finite")
    if bool(torch.any(typed < 0).item()):
        raise ValueError("losses must be nonnegative")
    return typed * (-(beta_current - beta_previous) * loss_scale)


def gibbs_log_target(
    losses: Tensor,
    costs: Tensor,
    *,
    beta: float,
    loss_scale: float,
    cost_scale: float,
) -> Tensor:
    """Compute the unnormalized Occam-prior/Gibbs target on-device."""

    if losses.shape != costs.shape:
        raise ValueError("losses and costs must have the same shape")
    if beta < 0.0 or not all(math.isfinite(value) for value in (beta, loss_scale, cost_scale)):
        raise ValueError("beta and scales must be finite and nonnegative")
    if loss_scale < 0.0 or cost_scale < 0.0:
        raise ValueError("beta and scales must be finite and nonnegative")
    dtype = _potential_dtype(losses)
    typed_losses = losses.to(dtype=dtype)
    typed_costs = costs.to(device=losses.device, dtype=dtype)
    if bool(torch.any(~torch.isfinite(typed_losses)).item()) or bool(
        torch.any(~torch.isfinite(typed_costs)).item()
    ):
        raise ValueError("losses and costs must be finite")
    if bool(torch.any(typed_losses < 0).item()) or bool(torch.any(typed_costs < 0).item()):
        raise ValueError("losses and costs must be nonnegative")
    return -(beta * loss_scale * typed_losses) - (cost_scale * typed_costs)


def normalize_log_weights(log_weights: Tensor) -> NormalizedWeights:
    """Stable log-weight normalization on CPU float64.

    Positive infinity is interpreted as a limiting distribution uniform over
    the positive-infinite entries.  NaN and an all-negative-infinity population
    are errors because they do not define a probability distribution.
    """

    values = _one_dimensional(log_weights, "log_weights")
    if bool(torch.any(torch.isnan(values)).item()):
        raise ValueError("log_weights contain NaN")
    positive_infinite = torch.isposinf(values)
    if bool(torch.any(positive_infinite).item()):
        count = int(positive_infinite.sum().item())
        weights = positive_infinite.to(torch.float64) / count
        return NormalizedWeights(weights=weights, log_normalizer=math.inf, log_mean_weight=math.inf)
    if bool(torch.all(torch.isneginf(values)).item()):
        raise ValueError("all log_weights are -inf; the particle population has zero mass")

    log_normalizer_tensor = torch.logsumexp(values, dim=0)
    weights = torch.exp(values - log_normalizer_tensor)
    weights = weights / weights.sum()  # Correct the last few ulps deterministically.
    log_normalizer = float(log_normalizer_tensor.item())
    return NormalizedWeights(
        weights=weights,
        log_normalizer=log_normalizer,
        log_mean_weight=log_normalizer - math.log(values.numel()),
    )


def normalize_weights(weights: Tensor) -> Tensor:
    """Validate and normalize nonnegative probabilities on CPU float64."""

    values = _one_dimensional(weights, "weights")
    if bool(torch.any(~torch.isfinite(values)).item()):
        raise ValueError("weights must be finite")
    if bool(torch.any(values < 0).item()):
        raise ValueError("weights must be nonnegative")
    maximum = values.max()
    if float(maximum.item()) <= 0.0:
        raise ValueError("at least one weight must be positive")
    # Scaling first avoids overflow when several individually finite weights
    # have a non-finite direct sum.
    scaled = values / maximum
    probabilities = scaled / scaled.sum()
    # Correct the rounding residual on the largest bin. Applying a negative
    # residual to a zero/tiny final bin can create an invalid negative weight.
    largest = int(torch.argmax(probabilities).item())
    probabilities[largest] += 1.0 - probabilities.sum()
    if bool(torch.any(probabilities < 0).item()):
        raise ValueError("normalization produced a negative probability")
    return probabilities


def effective_sample_size(weights: Tensor) -> float:
    probabilities = normalize_weights(weights)
    return float((1.0 / torch.sum(probabilities.square())).item())


def relative_effective_sample_size(weights: Tensor) -> float:
    return effective_sample_size(weights) / weights.numel()


def systematic_resample(weights: Tensor, *, generator: torch.Generator) -> Tensor:
    """Draw systematic ancestor indices using one seeded CPU random value."""

    probabilities = normalize_weights(weights)
    count = probabilities.numel()
    offset = torch.rand((), dtype=torch.float64, generator=generator) / count
    positions = offset + torch.arange(count, dtype=torch.float64) / count
    cumulative = torch.cumsum(probabilities, dim=0)
    cumulative[-1] = 1.0
    # CDF intervals are right-open: a position exactly on a boundary belongs
    # to the following positive-mass bin, and a zero-mass leading bin is never
    # selected even if the random offset is exactly zero.
    return torch.searchsorted(cumulative, positions, right=True).to(dtype=torch.int64)


def categorical_sample(
    weights: Tensor,
    count: int,
    *,
    generator: torch.Generator,
    replacement: bool = True,
) -> Tensor:
    """Sample categorical indices from CPU float64 normalized weights."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    probabilities = normalize_weights(weights)
    if not replacement and count > int(torch.count_nonzero(probabilities).item()):
        raise ValueError("cannot sample more nonzero categories than exist without replacement")
    return torch.multinomial(
        probabilities,
        num_samples=count,
        replacement=replacement,
        generator=generator,
    ).to(dtype=torch.int64)

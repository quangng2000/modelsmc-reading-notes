"""Shared exact finite-subtree normalization."""

from __future__ import annotations

import torch

from modelsmc_pbe.smc import NormalizedWeights, normalize_log_weights


def normalized_subtree_distribution(
    log_weights: torch.Tensor,
    groups: tuple[tuple[int, ...], ...],
    *,
    label: str,
) -> NormalizedWeights:
    """Normalize log-sum-exp masses over disjoint nonempty state groups."""

    if log_weights.ndim != 1 or log_weights.numel() == 0:
        raise ValueError(f"{label} log weights must be a nonempty vector")
    if not groups or any(not group for group in groups):
        raise ValueError(f"{label} subtree groups must be nonempty")
    flattened = tuple(index for group in groups for index in group)
    if len(flattened) != len(set(flattened)):
        raise ValueError(f"{label} subtree groups must be disjoint")
    if any(index < 0 or index >= log_weights.numel() for index in flattened):
        raise IndexError(f"{label} subtree state index is outside the finite support")
    log_masses = torch.stack(
        [
            torch.logsumexp(
                log_weights[torch.tensor(group, dtype=torch.int64)],
                dim=0,
            )
            for group in groups
        ]
    )
    return normalize_log_weights(log_masses)


def require_strictly_positive_weights(weights: torch.Tensor, *, label: str) -> None:
    """Fail rather than silently treating float underflow as zero proposal support."""

    if bool(torch.any(weights <= 0.0).item()):
        raise FloatingPointError(
            f"{label} float64 normalization underflowed a positive finite-state mass "
            "to zero; rescale the target before using this proposal"
        )

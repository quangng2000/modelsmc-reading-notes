"""Exact finite Gibbs targets used by importance-corrected SMC."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from modelsmc_pbe.config import SMCConfig
from modelsmc_pbe.runtime import DeviceInfo
from modelsmc_pbe.search.importance.records import ImportanceState, ImportanceSupport
from modelsmc_pbe.smc import NormalizedWeights, normalize_log_weights

from .subtree import normalized_subtree_distribution


def equal_family_occam_log_prior(
    support: ImportanceSupport,
    *,
    cost_scale: float,
) -> Tensor:
    """Return the normalized equal-family, within-family Occam log prior."""

    costs = torch.tensor(
        [state.score.cost for state in support.states],
        dtype=torch.float64,
    )
    log_prior = torch.full((len(support.states),), -math.inf, dtype=torch.float64)
    log_family_mass = -math.log(len(support.families))
    for family in support.families:
        indices = torch.tensor(family.state_indices, dtype=torch.int64)
        local_logits = -cost_scale * costs[indices]
        local_log_normalizer = torch.logsumexp(local_logits, dim=0)
        log_prior[indices] = log_family_mass + local_logits - local_log_normalizer
    if bool(torch.any(~torch.isfinite(log_prior)).item()):
        raise ValueError("every support state must belong to one nonempty family")
    return log_prior


@dataclass(frozen=True, slots=True)
class FiniteImportanceTarget:
    """Equal-family/within-family-Occam targets plus exact CPU references."""

    losses_device: Tensor
    costs_device: Tensor
    losses_cpu: Tensor
    costs_cpu: Tensor
    log_prior_device: Tensor
    log_prior_cpu: Tensor
    exact_mask: Tensor
    smc: SMCConfig

    @classmethod
    def build(
        cls,
        support: ImportanceSupport,
        *,
        smc: SMCConfig,
        device: DeviceInfo,
    ) -> FiniteImportanceTarget:
        states: Sequence[ImportanceState] = support.states
        dtype = torch.float32 if device.resolved == "mps" else torch.float64
        losses = [state.score.total_loss for state in states]
        costs = [state.score.cost for state in states]
        costs_cpu = torch.tensor(costs, dtype=torch.float64)
        log_prior = equal_family_occam_log_prior(
            support,
            cost_scale=float(smc.cost_scale),
        )
        return cls(
            losses_device=torch.tensor(losses, dtype=dtype, device=device.torch_device),
            costs_device=torch.tensor(costs, dtype=dtype, device=device.torch_device),
            losses_cpu=torch.tensor(losses, dtype=torch.float64),
            costs_cpu=costs_cpu,
            log_prior_device=log_prior.to(device=device.torch_device, dtype=dtype),
            log_prior_cpu=log_prior,
            exact_mask=torch.tensor(
                [state.score.exact_program for state in states], dtype=torch.bool
            ),
            smc=smc,
        )

    def log_unnormalized(self, beta: float) -> Tensor:
        """Return the exact CPU-float64 target in fixed support order.

        The device tensors are retained for future batched proposal diagnostics,
        while normalization and reference comparisons deliberately use the same
        CPU-float64 boundary as the shared SMC numerics.
        """

        if not math.isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be finite and nonnegative")
        return self.log_prior_cpu - beta * float(self.smc.loss_scale) * self.losses_cpu

    def distribution(self, beta: float) -> NormalizedWeights:
        """Exactly normalize the finite target at one inverse temperature."""

        return normalize_log_weights(self.log_unnormalized(beta))

    def subtree_probabilities(
        self,
        groups: tuple[tuple[int, ...], ...],
        *,
        beta: float,
    ) -> Tensor:
        """Normalize exact target mass over a partition of a construction subtree."""

        return self.subtree_distribution(groups, beta=beta).weights

    def subtree_distribution(
        self,
        groups: tuple[tuple[int, ...], ...],
        *,
        beta: float,
    ) -> NormalizedWeights:
        """Return probabilities and normalizer for a target-subtree partition."""

        return normalized_subtree_distribution(
            self.log_unnormalized(beta),
            groups,
            label="target",
        )

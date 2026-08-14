"""Exact subtree-marginal guide induced by sound derived hole examples."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from modelsmc_pbe.config import SMCConfig
from modelsmc_pbe.smc import normalize_log_weights

from .proposal_distribution import deduction_mismatch_counts
from .records import ImportanceSupport
from .subtree import normalized_subtree_distribution
from .target import equal_family_occam_log_prior


@dataclass(frozen=True, slots=True)
class FiniteDeductionGuide:
    """A normalized Occam/deduction distribution on the complete support."""

    support: ImportanceSupport
    log_prior: torch.Tensor
    violations: torch.Tensor
    strength: float
    beta_max: float

    @classmethod
    def build(
        cls,
        support: ImportanceSupport,
        *,
        smc: SMCConfig,
        strength: float,
        beta_max: float,
    ) -> FiniteDeductionGuide:
        """Evaluate each distinct filling once and assemble state violations."""

        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError("deduction strength must be finite and nonnegative")
        if not math.isfinite(beta_max) or beta_max <= 0.0:
            raise ValueError("beta_max must be finite and greater than zero")
        families = {
            family.hypothesis_index: family for family in support.families
        }
        cached: dict[tuple[int, str, str], int] = {}
        state_violations: list[float] = []
        for state in support.states:
            family = families[state.hypothesis_index]
            total = 0
            for hole in family.hypothesis.holes:
                filling = state.filling(hole.name)
                key = (state.hypothesis_index, hole.name, filling.key)
                mismatch = cached.get(key)
                if mismatch is None:
                    mismatch = deduction_mismatch_counts(
                        (filling.expression,),
                        family.deduction.examples_for(hole.name),
                    )[0]
                    cached[key] = mismatch
                total += mismatch
            state_violations.append(float(total))
        return cls(
            support=support,
            log_prior=equal_family_occam_log_prior(
                support,
                cost_scale=float(smc.cost_scale),
            ),
            violations=torch.tensor(state_violations, dtype=torch.float64),
            strength=strength,
            beta_max=beta_max,
        )

    def distribution(self, beta: float) -> torch.Tensor:
        """Return the normalized complete-state deduction guide at ``beta``."""

        return normalize_log_weights(self.log_weights(beta)).weights

    def log_weights(self, beta: float) -> torch.Tensor:
        """Return unnormalized complete-state guide log weights."""

        if not math.isfinite(beta) or not 0.0 <= beta <= self.beta_max:
            raise ValueError("guide beta must be finite and in [0, beta_max]")
        scale = self.strength * beta / self.beta_max
        return self.log_prior - scale * self.violations

    def subtree_probabilities(
        self,
        groups: tuple[tuple[int, ...], ...],
        *,
        beta: float,
    ) -> torch.Tensor:
        """Normalize exact guide mass over a partition of a proposal subtree."""

        return normalized_subtree_distribution(
            self.log_weights(beta),
            groups,
            label="deduction",
        ).weights

    def family_probabilities(self, *, beta: float) -> torch.Tensor:
        """Return exact deduction-guide mass for each support family."""

        return self.subtree_probabilities(
            tuple(family.state_indices for family in self.support.families),
            beta=beta,
        )

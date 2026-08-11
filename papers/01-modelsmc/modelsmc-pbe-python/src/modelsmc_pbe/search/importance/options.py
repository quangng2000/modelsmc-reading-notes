"""Validated options, public claims, and boundary errors for importance-SMC."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from modelsmc_pbe.grammar import SkeletonName, available_skeletons
from modelsmc_pbe.proposals import LLMEnergyNormalization

IMPORTANCE_SMC_CLAIM = (
    "calibrated SMC for an explicit finite, typed, deduction-refuted program support "
    "with a normalized within-family Occam prior; the deduction/Qwen defensive proposal "
    "is evaluated completely and included in the importance denominator"
)
JOINT_TARGET_IMPORTANCE_SMC_CLAIM = (
    "oracle finite-support control whose proposal is the exactly normalized "
    "training-loss/Occam target; every complete program is executed before sampling, "
    "and exact target subtree marginals define the sequential construction law"
)

type ImportanceProposalStrategy = Literal["guided", "joint-target"]


class EmptyImportanceSupportError(RuntimeError):
    """Raised when no complete program survives construction and semantic checks."""


class ImportanceSupportLimitExceeded(RuntimeError):
    """Raised instead of silently truncating complete construction support."""


class ImportanceProposalBudgetExceeded(RuntimeError):
    """Raised before candidate scoring would exceed the declared run budget."""


@dataclass(frozen=True, slots=True)
class ImportanceSMCOptions:
    """Bounds and proposal controls unique to importance-corrected SMC."""

    hole_max_cost: int = 3
    hole_state_limit: int = 250_000
    support_limit: int = 250_000
    score_batch_size: int = 512
    proposal_temperature: float = 0.7
    proposal_epsilon: float = 0.05
    deduction_mix: float = 0.5
    family_deduction_mix: float | None = None
    hole_deduction_mix: float | None = None
    deduction_strength: float = 2.0
    beta_max: float = 1.0
    max_scored_candidates: int = 1_000_000
    conditioned_skeleton: SkeletonName | None = None
    multi_family: bool = False
    llm_energy_normalization: LLMEnergyNormalization = (
        LLMEnergyNormalization.TOTAL_FULL_PROMPT_LOGPROB
    )
    proposal_strategy: ImportanceProposalStrategy = "guided"

    def __post_init__(self) -> None:
        for name, value in (
            ("hole_max_cost", self.hole_max_cost),
            ("hole_state_limit", self.hole_state_limit),
            ("support_limit", self.support_limit),
            ("score_batch_size", self.score_batch_size),
            ("max_scored_candidates", self.max_scored_candidates),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.score_batch_size > 10_000:
            raise ValueError("score_batch_size must not exceed 10000")
        if not math.isfinite(self.proposal_temperature) or self.proposal_temperature <= 0:
            raise ValueError("proposal_temperature must be finite and greater than zero")
        if not math.isfinite(self.proposal_epsilon) or not 0.0 < self.proposal_epsilon <= 1.0:
            raise ValueError("proposal_epsilon must be finite and in (0, 1]")
        if not math.isfinite(self.deduction_mix) or not 0.0 <= self.deduction_mix <= 1.0:
            raise ValueError("deduction_mix must be finite and in [0, 1]")
        for name, optional_value in (
            ("family_deduction_mix", self.family_deduction_mix),
            ("hole_deduction_mix", self.hole_deduction_mix),
        ):
            if optional_value is not None and (
                not math.isfinite(optional_value) or not 0.0 <= optional_value <= 1.0
            ):
                raise ValueError(f"{name} must be None or finite and in [0, 1]")
        if not math.isfinite(self.deduction_strength) or self.deduction_strength < 0.0:
            raise ValueError("deduction_strength must be finite and nonnegative")
        if not math.isfinite(self.beta_max) or self.beta_max <= 0:
            raise ValueError("beta_max must be finite and greater than zero")
        if (
            self.conditioned_skeleton is not None
            and self.conditioned_skeleton not in available_skeletons()
        ):
            choices = ", ".join(available_skeletons())
            raise ValueError(
                f"unknown conditioned skeleton {self.conditioned_skeleton!r}; "
                f"expected one of: {choices}"
            )
        if not isinstance(self.multi_family, bool):
            raise TypeError("multi_family must be a boolean")
        if self.multi_family and self.conditioned_skeleton is not None:
            raise ValueError(
                "multi-family support and a single conditioned skeleton are mutually exclusive"
            )
        if self.proposal_strategy not in {"guided", "joint-target"}:
            raise ValueError("proposal_strategy must be guided or joint-target")
        if not isinstance(self.llm_energy_normalization, LLMEnergyNormalization):
            raise TypeError("llm_energy_normalization must be an LLMEnergyNormalization")

    @property
    def resolved_family_deduction_mix(self) -> float:
        """Return the family-wave mix, falling back to the legacy shared value."""

        if self.family_deduction_mix is None:
            return self.deduction_mix
        return self.family_deduction_mix

    @property
    def resolved_hole_deduction_mix(self) -> float:
        """Return the hole-wave mix, falling back to the legacy shared value."""

        if self.hole_deduction_mix is None:
            return self.deduction_mix
        return self.hole_deduction_mix

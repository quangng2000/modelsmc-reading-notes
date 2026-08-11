"""Immutable records for finite-support importance-corrected SMC."""

from __future__ import annotations

import math
from dataclasses import dataclass

from modelsmc_pbe.core import ScoredProgram
from modelsmc_pbe.deduction import DeductionReport
from modelsmc_pbe.domain import ProgramAst
from modelsmc_pbe.domain.ast import AstNode
from modelsmc_pbe.grammar import SkeletonName, available_skeletons
from modelsmc_pbe.induction import InductionReport, TypedSkeleton
from modelsmc_pbe.proposals import LLMEnergyNormalization

from .score_ledger import LLMScoreWaveLedger

IMPORTANCE_SMC_CLAIM = (
    "calibrated SMC for an explicit finite, typed, deduction-refuted program support "
    "with a normalized within-family Occam prior; the deduction/Qwen defensive proposal "
    "is evaluated completely and included in the importance denominator"
)


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


@dataclass(frozen=True, slots=True)
class HoleFilling:
    """One canonical expression assigned to one named skeleton hole."""

    hole_name: str
    expression: AstNode
    key: str


@dataclass(frozen=True, slots=True)
class ImportanceState:
    """One complete scorer-approved state and its unique construction trace."""

    state_index: int
    hypothesis_index: int
    family: str
    fillings: tuple[HoleFilling, ...]
    program: ProgramAst
    key: str
    score: ScoredProgram

    def filling(self, hole_name: str) -> HoleFilling:
        """Return one named filling from the unique construction trace."""

        for filling in self.fillings:
            if filling.hole_name == hole_name:
                return filling
        raise KeyError(f"state has no filling for hole {hole_name!r}")


@dataclass(frozen=True, slots=True)
class FamilySupport:
    """All valid complete states reachable through one viable hypothesis."""

    hypothesis_index: int
    hypothesis: TypedSkeleton
    deduction: DeductionReport
    state_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HoleCatalogSummary:
    """Auditable size information for one fully enumerated hole catalog."""

    family: str
    hole_name: str
    returned_expressions: int
    generated_states: int
    max_cost: int


@dataclass(frozen=True, slots=True)
class ImportanceSupport:
    """Fixed finite support used by both the proposal trie and exact target."""

    induction: InductionReport
    deductions: tuple[DeductionReport, ...]
    families: tuple[FamilySupport, ...]
    states: tuple[ImportanceState, ...]
    hole_catalogs: tuple[HoleCatalogSummary, ...]
    constructed_programs: int
    rejected_programs: int
    conditioned_skeleton: SkeletonName | None
    multi_family: bool
    aliased_programs: int

    @property
    def exact_programs(self) -> int:
        return sum(state.score.exact_program for state in self.states)


@dataclass(frozen=True, slots=True)
class ProposedState:
    """One exact draw from the clone/finite-guided mixture kernel."""

    state_index: int
    ancestor_state_index: int
    log_q_construct: float
    log_q_mixture: float
    cloned: bool
    family: str


@dataclass(frozen=True, slots=True)
class ImportancePopulation:
    """Tensor-free population record used between stages."""

    state_indices: tuple[int, ...]
    weights: tuple[float, ...]
    ancestor_state_indices: tuple[int, ...]
    log_q_mixture: tuple[float, ...]
    log_incremental_weight: tuple[float, ...]
    cloned: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class ImportanceStageDiagnostic:
    """Population and probability diagnostics for one Feynman--Kac stage."""

    stage: int
    beta: float
    ess_before: float
    relative_ess_before: float
    resampled: bool
    clones: int
    unique_programs: int
    exact_programs: int
    ess_after: float
    mean_log_q: float
    min_log_q: float
    max_log_importance_ratio: float
    log_path_z_estimate: float
    log_path_z_reference: float


@dataclass(frozen=True, slots=True)
class ImportanceReferenceMetrics:
    """Final particle projection compared with exact finite enumeration."""

    particle_exact_mass: float
    enumeration_exact_mass: float
    particle_mean_loss: float
    enumeration_mean_loss: float
    particle_mean_cost: float
    enumeration_mean_cost: float
    total_variation_distance: float
    log_path_z_estimate: float
    log_path_z_enumeration: float
    log_path_z_error: float
    absolute_log_path_z_error: float


@dataclass(frozen=True, slots=True)
class ImportanceFamilySummary:
    """Prior, exact-target, and particle mass for one surviving family."""

    family: str
    states: int
    exact_programs: int
    prior_mass: float
    deduction_guide_mass: float
    posterior_mass: float
    particle_mass: float


@dataclass(frozen=True, slots=True)
class ImportanceHypothesisSummary:
    """Persisted induction/deduction outcome for one generated family."""

    family: str
    viable: bool
    states: int
    refutation_kind: str | None
    refutation_sources: tuple[int, ...]
    refutation_detail: str | None


@dataclass(frozen=True, slots=True)
class ImportanceStateSummary:
    """Best program present in the final weighted population."""

    state_index: int
    family: str
    program: ProgramAst
    total_loss: float
    cost: int
    exact_program: bool
    empirical_mass: float
    enumeration_probability: float


@dataclass(frozen=True, slots=True)
class ImportanceParticle:
    """One final program particle persisted for audit and replay."""

    particle_index: int
    state_index: int
    ancestor_state_index: int
    family: str
    program: ProgramAst
    total_loss: float
    cost: int
    exact_program: bool
    weight: float
    log_q_mixture: float
    log_incremental_weight: float
    cloned: bool


@dataclass(frozen=True, slots=True)
class ImportanceSMCResult:
    """Complete output of one finite guided importance-SMC run."""

    mode: str
    probabilistic_claim: str
    proposal_source: str
    deduction_mix: float
    family_deduction_mix: float
    hole_deduction_mix: float
    deduction_strength: float
    llm_energy_normalization: LLMEnergyNormalization
    conditioned_skeleton: SkeletonName | None
    multi_family: bool
    aliased_programs: int
    support_states: int
    generated_hypotheses: int
    viable_hypotheses: int
    refuted_hypotheses: int
    exact_programs: int
    deduction_guide_exact_mass: float
    hole_catalogs: tuple[HoleCatalogSummary, ...]
    hypotheses: tuple[ImportanceHypothesisSummary, ...]
    families: tuple[ImportanceFamilySummary, ...]
    sampled_best: ImportanceStateSummary
    reference: ImportanceReferenceMetrics
    stages: tuple[ImportanceStageDiagnostic, ...]
    final_particles: tuple[ImportanceParticle, ...]
    scored_candidates: int
    max_scored_candidates: int
    score_ledger: tuple[LLMScoreWaveLedger, ...]

    @property
    def exact(self) -> bool:
        return self.sampled_best.exact_program

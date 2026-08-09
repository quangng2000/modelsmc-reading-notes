"""Immutable records for finite-support importance-corrected SMC."""

from __future__ import annotations

import math
from dataclasses import dataclass

from modelsmc_pbe.core import ScoredProgram
from modelsmc_pbe.deduction import DeductionReport
from modelsmc_pbe.domain import ProgramAst
from modelsmc_pbe.domain.ast import AstNode
from modelsmc_pbe.induction import InductionReport, TypedSkeleton

IMPORTANCE_SMC_CLAIM = (
    "calibrated SMC for an explicit finite, typed, deduction-refuted program support; "
    "the complete finite proposal law is evaluated and included in the importance "
    "denominator"
)


class EmptyImportanceSupportError(RuntimeError):
    """Raised when no complete program survives construction and semantic checks."""


class ImportanceSupportLimitExceeded(RuntimeError):
    """Raised instead of silently truncating complete construction support."""


@dataclass(frozen=True, slots=True)
class ImportanceSMCOptions:
    """Bounds and proposal controls unique to importance-corrected SMC."""

    hole_max_cost: int = 3
    hole_state_limit: int = 250_000
    support_limit: int = 250_000
    score_batch_size: int = 512
    proposal_temperature: float = 0.7
    proposal_epsilon: float = 0.05
    beta_max: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("hole_max_cost", self.hole_max_cost),
            ("hole_state_limit", self.hole_state_limit),
            ("support_limit", self.support_limit),
            ("score_batch_size", self.score_batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.score_batch_size > 10_000:
            raise ValueError("score_batch_size must not exceed 10000")
        if not math.isfinite(self.proposal_temperature) or self.proposal_temperature <= 0:
            raise ValueError("proposal_temperature must be finite and greater than zero")
        if (
            not math.isfinite(self.proposal_epsilon)
            or not 0.0 < self.proposal_epsilon <= 1.0
        ):
            raise ValueError("proposal_epsilon must be finite and in (0, 1]")
        if not math.isfinite(self.beta_max) or self.beta_max <= 0:
            raise ValueError("beta_max must be finite and greater than zero")


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

    @property
    def exact_programs(self) -> int:
        return sum(state.score.exact_program for state in self.states)


@dataclass(frozen=True, slots=True)
class ProposedState:
    """One exact draw from the clone/finite-Qwen mixture kernel."""

    state_index: int
    ancestor_state_index: int
    log_q_llm: float
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
    """Complete output of one finite Qwen-energy importance-SMC run."""

    mode: str
    probabilistic_claim: str
    proposal_source: str
    support_states: int
    generated_hypotheses: int
    viable_hypotheses: int
    refuted_hypotheses: int
    exact_programs: int
    hole_catalogs: tuple[HoleCatalogSummary, ...]
    sampled_best: ImportanceStateSummary
    reference: ImportanceReferenceMetrics
    stages: tuple[ImportanceStageDiagnostic, ...]
    final_particles: tuple[ImportanceParticle, ...]

    @property
    def exact(self) -> bool:
        return self.sampled_best.exact_program

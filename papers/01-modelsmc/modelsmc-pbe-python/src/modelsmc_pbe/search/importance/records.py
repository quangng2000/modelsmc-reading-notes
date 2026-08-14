"""Immutable records for finite-support importance-corrected SMC."""

from __future__ import annotations

from dataclasses import dataclass

from modelsmc_pbe.core import ScoredProgram
from modelsmc_pbe.deduction import DeductionReport
from modelsmc_pbe.domain import ProgramAst
from modelsmc_pbe.domain.ast import AstNode
from modelsmc_pbe.grammar import SkeletonName
from modelsmc_pbe.induction import InductionReport, TypedSkeleton
from modelsmc_pbe.proposals import LLMEnergyNormalization

from .options import (
    IMPORTANCE_SMC_CLAIM as IMPORTANCE_SMC_CLAIM,
)
from .options import (
    JOINT_SEMANTIC_IMPORTANCE_SMC_CLAIM as JOINT_SEMANTIC_IMPORTANCE_SMC_CLAIM,
)
from .options import (
    JOINT_TARGET_IMPORTANCE_SMC_CLAIM as JOINT_TARGET_IMPORTANCE_SMC_CLAIM,
)
from .options import (
    EmptyImportanceSupportError as EmptyImportanceSupportError,
)
from .options import (
    ImportanceProposalBudgetExceeded as ImportanceProposalBudgetExceeded,
)
from .options import (
    ImportanceProposalStrategy as ImportanceProposalStrategy,
)
from .options import (
    ImportanceSMCOptions as ImportanceSMCOptions,
)
from .options import (
    ImportanceSupportLimitExceeded as ImportanceSupportLimitExceeded,
)
from .score_ledger import LLMScoreWaveLedger


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
    deduction_guide_mass: float | None
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
    deduction_mix: float | None
    family_deduction_mix: float | None
    hole_deduction_mix: float | None
    deduction_strength: float | None
    llm_energy_normalization: LLMEnergyNormalization | None
    conditioned_skeleton: SkeletonName | None
    multi_family: bool
    aliased_programs: int
    support_states: int
    generated_hypotheses: int
    viable_hypotheses: int
    refuted_hypotheses: int
    exact_programs: int
    deduction_guide_exact_mass: float | None
    hole_catalogs: tuple[HoleCatalogSummary, ...]
    hypotheses: tuple[ImportanceHypothesisSummary, ...]
    families: tuple[ImportanceFamilySummary, ...]
    sampled_best: ImportanceStateSummary
    reference: ImportanceReferenceMetrics
    stages: tuple[ImportanceStageDiagnostic, ...]
    final_particles: tuple[ImportanceParticle, ...]
    scored_candidates: int
    max_scored_candidates: int | None
    score_ledger: tuple[LLMScoreWaveLedger, ...]
    proposal_strategy: ImportanceProposalStrategy = "guided"

    @property
    def exact(self) -> bool:
        return self.sampled_best.exact_program

"""Records for factorized, visit-only importance-SMC execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelsmc_pbe.core import ScoredProgram
from modelsmc_pbe.deduction import DeductionReport
from modelsmc_pbe.domain import ProgramAst
from modelsmc_pbe.induction import HoleSpec, InductionReport, TypedSkeleton
from modelsmc_pbe.proposals import LLMEnergyNormalization

from .records import HoleCatalogSummary, HoleFilling, ImportanceProposalStrategy
from .score_ledger import LLMScoreWaveLedger

if TYPE_CHECKING:
    from .joint_semantic.ledger import SemanticProposalLedger

LAZY_IMPORTANCE_SMC_CLAIM = (
    "importance-corrected SMC on a finite factorized construction-trace support; "
    "typed hole catalogs and exact support combinatorics are materialized, while only "
    "sampled complete programs are assembled and semantically evaluated; no exact "
    "posterior reference is claimed"
)


@dataclass(frozen=True, slots=True)
class FactorizedHoleCatalog:
    """One complete typed hole catalog with local target/guide statistics."""

    hole: HoleSpec
    fillings: tuple[HoleFilling, ...]
    costs: tuple[int, ...]
    deduction_mismatches: tuple[int, ...]

    def __post_init__(self) -> None:
        size = len(self.fillings)
        if size < 1 or len(self.costs) != size or len(self.deduction_mismatches) != size:
            raise ValueError("factorized hole arrays must be nonempty and aligned")


@dataclass(frozen=True, slots=True)
class FactorizedFamily:
    """One skeleton and its catalogs, without a Cartesian complete-state table."""

    hypothesis_index: int
    hypothesis: TypedSkeleton
    deduction: DeductionReport
    catalogs: tuple[FactorizedHoleCatalog, ...]
    base_cost: int
    combination_budget: int
    support_count: int

    def __post_init__(self) -> None:
        if len(self.catalogs) != len(self.hypothesis.holes):
            raise ValueError("one factorized catalog is required for every skeleton hole")
        if self.support_count < 1:
            raise ValueError("factorized family support must be nonempty")


@dataclass(frozen=True, slots=True)
class ConstructionTrace:
    """A compact complete state: family plus one index per typed hole."""

    hypothesis_index: int
    filling_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FactorizedImportanceSupport:
    """Symbolic support whose complete Cartesian product is never materialized."""

    induction: InductionReport
    deductions: tuple[DeductionReport, ...]
    families: tuple[FactorizedFamily, ...]
    hole_catalogs: tuple[HoleCatalogSummary, ...]
    support_states: int
    conditioned_skeleton: str | None
    multi_family: bool

    def family(self, hypothesis_index: int) -> FactorizedFamily:
        for family in self.families:
            if family.hypothesis_index == hypothesis_index:
                return family
        raise KeyError(f"no factorized family for hypothesis {hypothesis_index}")


@dataclass(frozen=True, slots=True)
class LazyImportanceState:
    """One sampled trace after on-demand assembly, checking, and execution."""

    trace: ConstructionTrace
    family: str
    fillings: tuple[HoleFilling, ...]
    program: ProgramAst
    key: str
    score: ScoredProgram
    log_prior: float


@dataclass(frozen=True, slots=True)
class LazyProposedTrace:
    """One proposal draw with its exactly evaluated construction probability."""

    trace: ConstructionTrace
    ancestor_trace: ConstructionTrace
    log_q_construct: float
    log_q_mixture: float
    cloned: bool
    family: str


@dataclass(frozen=True, slots=True)
class LazyImportancePopulation:
    """Weighted sampled traces retained between SMC stages."""

    traces: tuple[ConstructionTrace, ...]
    weights: tuple[float, ...]
    ancestor_traces: tuple[ConstructionTrace, ...]
    log_q_mixture: tuple[float, ...]
    log_incremental_weight: tuple[float, ...]
    cloned: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class LazyStageDiagnostic:
    """Search diagnostics; deliberately contains no enumeration comparison."""

    stage: int
    beta: float
    ess_before: float
    relative_ess_before: float
    resampled: bool
    clones: int
    unique_programs: int
    exact_particles: int
    exact_particle_mass: float
    ess_after: float
    newly_evaluated_programs: int
    cumulative_evaluated_programs: int
    mean_log_q: float
    min_log_q: float
    max_log_importance_ratio: float
    log_path_z_estimate: float


@dataclass(frozen=True, slots=True)
class LazyFamilySummary:
    """Factorized support and sampled mass for one viable family."""

    family: str
    support_states: int
    prior_mass: float
    deduction_guide_mass: float | None
    particle_mass: float
    proposal_mass: float | None = None


@dataclass(frozen=True, slots=True)
class LazyStateSummary:
    """Best complete program actually visited by the search."""

    family: str
    program: ProgramAst
    total_loss: float
    cost: int
    exact_program: bool
    particle_mass: float


@dataclass(frozen=True, slots=True)
class LazyImportanceParticle:
    """One final sampled program persisted for audit and replay."""

    particle_index: int
    trace: ConstructionTrace
    ancestor_trace: ConstructionTrace
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
class LazySearchMetrics:
    """Discovery metrics available without evaluating unvisited programs."""

    exact_found: bool
    final_exact_particle_mass: float
    exact_sampled_programs: int
    evaluated_programs: int
    evaluated_fraction_of_support: float
    first_exact_stage: int | None
    log_path_z_estimate: float


@dataclass(frozen=True, slots=True)
class LazyImportanceSMCResult:
    """Visit-only result with search metrics separated from absent references."""

    mode: str
    execution: str
    probabilistic_claim: str
    proposal_source: str
    deduction_mix: float | None
    family_deduction_mix: float | None
    hole_deduction_mix: float | None
    deduction_strength: float | None
    llm_energy_normalization: LLMEnergyNormalization | None
    conditioned_skeleton: str | None
    multi_family: bool
    support_semantics: str
    support_materialized: bool
    support_states: int
    generated_hypotheses: int
    viable_hypotheses: int
    refuted_hypotheses: int
    hole_catalogs: tuple[HoleCatalogSummary, ...]
    families: tuple[LazyFamilySummary, ...]
    best_visited: LazyStateSummary
    sampled_best: LazyStateSummary
    search: LazySearchMetrics
    reference: None
    stages: tuple[LazyStageDiagnostic, ...]
    final_particles: tuple[LazyImportanceParticle, ...]
    scored_candidates: int
    max_scored_candidates: int
    score_ledger: tuple[LLMScoreWaveLedger, ...]
    proposal_strategy: ImportanceProposalStrategy = "guided"
    proposal_epsilon: float | None = None
    semantic_scale: float | None = None
    semantic_slate_size: int | None = None
    semantic_score_kind: str | None = None
    semantic_score_ledger: SemanticProposalLedger | None = None

    @property
    def exact(self) -> bool:
        """Whether search ever visited an exact program, even if it was later lost."""

        return self.best_visited.exact_program


if not TYPE_CHECKING:
    from .joint_semantic.ledger import SemanticProposalLedger

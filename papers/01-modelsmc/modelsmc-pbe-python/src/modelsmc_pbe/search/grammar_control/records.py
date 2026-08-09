"""Configuration and result records for finite-grammar SMC."""

from __future__ import annotations

import math
from dataclasses import dataclass

from modelsmc_pbe.core import ScoredProgram
from modelsmc_pbe.domain.ast import ProgramAst
from modelsmc_pbe.grammar import SkeletonName

CALIBRATED_CLAIM = (
    "calibrated finite-grammar SMC for the declared bounded support; "
    "exact enumeration is the reference distribution"
)


class EmptyGrammarSupportError(RuntimeError):
    """Raised when no enumerated AST survives semantic scoring and the cost bound."""


@dataclass(frozen=True, slots=True)
class GrammarSMCOptions:
    """Finite-support controls not shared with the paper-style LLM search."""

    skeleton: SkeletonName
    state_limit: int = 250_000
    beta_max: float = 1.0
    moves_per_stage: int = 1
    score_batch_size: int = 512

    def __post_init__(self) -> None:
        if (
            isinstance(self.state_limit, bool)
            or not isinstance(self.state_limit, int)
            or self.state_limit < 1
        ):
            raise ValueError("state_limit must be a positive integer")
        if not math.isfinite(self.beta_max) or self.beta_max <= 0.0:
            raise ValueError("beta_max must be finite and greater than zero")
        if (
            isinstance(self.moves_per_stage, bool)
            or not isinstance(self.moves_per_stage, int)
            or self.moves_per_stage < 0
        ):
            raise ValueError("moves_per_stage must be a nonnegative integer")
        if (
            isinstance(self.score_batch_size, bool)
            or not isinstance(self.score_batch_size, int)
            or self.score_batch_size < 1
        ):
            raise ValueError("score_batch_size must be a positive integer")


@dataclass(frozen=True, slots=True)
class GrammarStageDiagnostic:
    """One annealing stage, including the population-control decisions."""

    stage: int
    beta_previous: float
    beta_current: float
    ess: float
    relative_ess: float
    resampled: bool
    mh_accepted: int
    mh_attempts: int
    log_z_estimate: float
    particle_exact_mass: float
    particle_mean_loss: float


@dataclass(frozen=True, slots=True)
class GrammarReferenceMetrics:
    """Particle estimates compared with the exactly normalized finite target."""

    particle_exact_mass: float
    enumeration_exact_mass: float
    particle_mean_loss: float
    enumeration_mean_loss: float
    log_z_estimate: float
    log_z_enumeration: float
    log_z_error: float
    absolute_log_z_error: float
    total_variation_distance: float


@dataclass(frozen=True, slots=True)
class GrammarStateSummary:
    """The best state present in the final particle population."""

    state_index: int
    program: ProgramAst
    total_loss: float
    target_loss: float
    cost: int
    exact_program: bool
    empirical_mass: float
    enumeration_probability: float


@dataclass(frozen=True, slots=True)
class GrammarParticle:
    """One weighted final particle written to the run artifacts."""

    particle_index: int
    state_index: int
    program: ProgramAst
    total_loss: float
    target_loss: float
    cost: int
    exact_program: bool
    weight: float


@dataclass(frozen=True, slots=True)
class GrammarSMCResult:
    """Complete result of one calibrated finite-grammar run."""

    mode: str
    probabilistic_claim: str
    skeleton: SkeletonName
    grammar_states: int
    enumerated_asts: int
    rejected_asts: int
    over_cost_asts: int
    exact_programs: int
    beta_max: float
    sampled_best: GrammarStateSummary
    reference: GrammarReferenceMetrics
    stages: tuple[GrammarStageDiagnostic, ...]
    final_particles: tuple[GrammarParticle, ...]

    @property
    def exact(self) -> bool:
        """Whether the final population contains an exact program."""

        return self.sampled_best.exact_program


@dataclass(frozen=True, slots=True)
class GrammarState:
    """One scorer-approved state in the finite grammar support."""

    program: ProgramAst
    score: ScoredProgram
    target_loss: float
    key: str


@dataclass(frozen=True, slots=True)
class GrammarSupport:
    """Complete scorer-approved support and its filtering counts."""

    states: tuple[GrammarState, ...]
    enumerated_asts: int
    rejected_asts: int
    over_cost_asts: int

    @property
    def exact_programs(self) -> int:
        """Count exact programs in the accepted support."""

        return sum(state.score.exact_program for state in self.states)

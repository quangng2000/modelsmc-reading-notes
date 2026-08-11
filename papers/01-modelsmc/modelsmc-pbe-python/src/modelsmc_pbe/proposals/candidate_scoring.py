"""Provider-neutral interface for teacher-forced finite-candidate scores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from modelsmc_pbe.domain.ast import AstNode
from modelsmc_pbe.proposals.hole import HoleSpecification


class CandidateLogprobSemantics(StrEnum):
    """Meaning of scores returned by a finite-candidate scorer."""

    TEACHER_FORCED_FULL_PROMPT = "teacher-forced-full-prompt"
    EXPLICIT_UNIFORM = "explicit-uniform"


class LLMEnergyNormalization(StrEnum):
    """Reduction from a scored full-prompt token path to one finite energy.

    The total and mean choices reduce the complete teacher-forced
    ``P || candidate`` path. The label marker instead identifies a separate,
    boundary-verified contrastive reduction and is never passed to
    :func:`llm_energy`.
    """

    TOTAL_FULL_PROMPT_LOGPROB = "total-full-prompt-logprob"
    MEAN_FULL_PROMPT_CONDITIONAL_LOGPROB = "mean-full-prompt-conditional-logprob"
    SYMMETRIZED_FINAL_LABEL_LOG_ODDS = "symmetrized-final-label-log-odds"


class CandidateKind(StrEnum):
    """Validation rule for one finite candidate catalog."""

    EXPRESSION = "expression"
    SKELETON = "skeleton"
    LABEL = "label"


class CandidateScoreOrigin(StrEnum):
    """Where one finite score batch was obtained in this run."""

    PROVIDER = "provider"
    CACHE = "cache"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True, slots=True)
class CandidateScoreProvenance:
    """Immutable cache disposition attached directly to a returned batch."""

    origin: CandidateScoreOrigin
    cache_key_sha256: str | None = None
    cache_hit: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.origin, CandidateScoreOrigin):
            raise TypeError("origin must be a CandidateScoreOrigin")
        if self.origin is CandidateScoreOrigin.CACHE:
            if self.cache_hit is not True or self.cache_key_sha256 is None:
                raise ValueError("cache-origin scores require a hit and cache key")
        elif self.origin is CandidateScoreOrigin.SYNTHETIC:
            if self.cache_key_sha256 is not None or self.cache_hit is not None:
                raise ValueError("synthetic scores cannot have cache provenance")
        elif self.cache_key_sha256 is None:
            if self.cache_hit is not None:
                raise ValueError("direct provider scores cannot have a cache disposition")
        elif self.cache_hit is not False:
            raise ValueError("cached cold provider scores require cache_hit=false")
        if self.cache_key_sha256 is not None and (
            len(self.cache_key_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.cache_key_sha256)
        ):
            raise ValueError("cache_key_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ProviderScoreMetrics:
    """Actual provider-I/O telemetry, independent of scientific score budgets."""

    http_requests: int
    http_failures: int
    scored_token_positions: int
    http_request_seconds_sum: float


@runtime_checkable
class ProviderMetricSource(Protocol):
    """Optional telemetry surface implemented by remote candidate scorers."""

    def provider_metrics(self) -> ProviderScoreMetrics: ...


@dataclass(frozen=True, slots=True)
class CandidateScoreRequest:
    """One common prefix and a complete finite canonical choice set."""

    prompt_prefix: str
    candidates: tuple[str, ...]
    hole: HoleSpecification | None
    integer_constants: tuple[int, ...]
    request_index: int = 0
    max_depth: int = 32
    max_nodes: int = 256
    candidate_kind: CandidateKind = CandidateKind.EXPRESSION

    def __post_init__(self) -> None:
        if not self.prompt_prefix:
            raise ValueError("prompt_prefix must not be empty")
        if not self.candidates:
            raise ValueError("candidates must not be empty")
        if any(not candidate for candidate in self.candidates):
            raise ValueError("candidate strings must not be empty")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("candidate strings must be unique")
        if not isinstance(self.candidate_kind, CandidateKind):
            raise TypeError("candidate_kind must be a CandidateKind")
        if self.candidate_kind is CandidateKind.EXPRESSION:
            if not isinstance(self.hole, HoleSpecification):
                raise TypeError("expression candidates require a HoleSpecification")
        elif self.hole is not None:
            raise TypeError("non-expression candidates must not declare an expression hole")
        if self.request_index < 0:
            raise ValueError("request_index must be nonnegative")
        if self.max_depth < 1 or self.max_nodes < 1:
            raise ValueError("expression depth and node limits must be positive")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.integer_constants
        ):
            raise TypeError("integer_constants must contain only integers")


@dataclass(frozen=True, slots=True)
class CandidateSequenceScore:
    """Teacher-forced energy for one prompt ending in a canonical expression."""

    candidate: str
    expression: AstNode | None
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    sequence_logprob: float

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.token_logprobs):
            raise ValueError("candidate token IDs and logprobs must be aligned")
        if any(not math.isfinite(value) or value > 0 for value in self.token_logprobs):
            raise ValueError("candidate token logprobs must be finite and nonpositive")
        if not math.isclose(
            self.sequence_logprob,
            math.fsum(self.token_logprobs),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("sequence_logprob must equal the sum of token logprobs")


@dataclass(frozen=True, slots=True)
class CandidateScoreBatch:
    """Scores in the same order as the request's finite candidate set."""

    scores: tuple[CandidateSequenceScore, ...]
    source: str
    model: str
    semantics: CandidateLogprobSemantics = CandidateLogprobSemantics.TEACHER_FORCED_FULL_PROMPT
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    provenance: CandidateScoreProvenance | None = None

    def __post_init__(self) -> None:
        if not self.scores:
            raise ValueError("candidate score batch must not be empty")
        if not isinstance(self.semantics, CandidateLogprobSemantics):
            raise ValueError("unsupported candidate score semantics")
        if self.provenance is not None and not isinstance(
            self.provenance, CandidateScoreProvenance
        ):
            raise TypeError("provenance must be CandidateScoreProvenance")
        for name, value in (
            ("model_revision", self.model_revision),
            ("tokenizer_revision", self.tokenizer_revision),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string when provided")


def llm_energy(
    score: CandidateSequenceScore,
    normalization: LLMEnergyNormalization,
) -> float:
    """Reduce one full teacher-forced path to the configured finite energy.

    ``MEAN_FULL_PROMPT_CONDITIONAL_LOGPROB`` averages the observed conditional
    token log probabilities over all *scored* positions in the full prompt.
    It is length-normalized energy, not a candidate-only continuation
    probability.  The token-free uniform control has energy zero in either
    mode.
    """

    if normalization is LLMEnergyNormalization.TOTAL_FULL_PROMPT_LOGPROB:
        return score.sequence_logprob
    if normalization is LLMEnergyNormalization.MEAN_FULL_PROMPT_CONDITIONAL_LOGPROB:
        if not score.token_logprobs:
            if score.sequence_logprob != 0.0:  # pragma: no cover - dataclass invariant
                raise ValueError("a token-free candidate score must have zero total logprob")
            return 0.0
        return score.sequence_logprob / len(score.token_logprobs)
    raise ValueError(f"unsupported LLM energy normalization: {normalization!r}")


@runtime_checkable
class CandidateScorer(Protocol):
    """Scores a finite catalog without sampling free-form output text."""

    name: str

    async def score_candidates(self, request: CandidateScoreRequest) -> CandidateScoreBatch: ...

    async def score_many(
        self, requests: list[CandidateScoreRequest]
    ) -> list[CandidateScoreBatch]: ...

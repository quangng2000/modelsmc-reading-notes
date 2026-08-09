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


@dataclass(frozen=True, slots=True)
class CandidateScoreRequest:
    """One common prefix and a complete finite set of canonical expressions."""

    prompt_prefix: str
    candidates: tuple[str, ...]
    hole: HoleSpecification
    integer_constants: tuple[int, ...]
    request_index: int = 0
    max_depth: int = 32
    max_nodes: int = 256

    def __post_init__(self) -> None:
        if not self.prompt_prefix:
            raise ValueError("prompt_prefix must not be empty")
        if not self.candidates:
            raise ValueError("candidates must not be empty")
        if any(not candidate for candidate in self.candidates):
            raise ValueError("candidate strings must not be empty")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("candidate strings must be unique")
        if not isinstance(self.hole, HoleSpecification):
            raise TypeError("hole must be a HoleSpecification")
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
    expression: AstNode
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
    semantics: CandidateLogprobSemantics = (
        CandidateLogprobSemantics.TEACHER_FORCED_FULL_PROMPT
    )

    def __post_init__(self) -> None:
        if not self.scores:
            raise ValueError("candidate score batch must not be empty")
        if not isinstance(self.semantics, CandidateLogprobSemantics):
            raise ValueError("unsupported candidate score semantics")


@runtime_checkable
class CandidateScorer(Protocol):
    """Scores a finite catalog without sampling free-form output text."""

    name: str

    async def score_candidates(self, request: CandidateScoreRequest) -> CandidateScoreBatch: ...

    async def score_many(
        self, requests: list[CandidateScoreRequest]
    ) -> list[CandidateScoreBatch]: ...

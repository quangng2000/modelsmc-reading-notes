"""Stable scorer records shared with SMC engines and artifact writers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExampleEvaluation:
    input: str
    expected: str
    predicted: str
    exact: bool
    loss: float


@dataclass(frozen=True, slots=True)
class ScoredProgram:
    kind: str
    inferred_type: str
    total_loss: float
    exact_matches: int
    cost: int
    log_target: float
    exact_program: bool
    evaluations: tuple[ExampleEvaluation, ...]


@dataclass(frozen=True, slots=True)
class RejectedProgram:
    kind: str
    reason: str
    inferred_type: str | None = None
    cost: int | None = None


type ScoreResult = ScoredProgram | RejectedProgram

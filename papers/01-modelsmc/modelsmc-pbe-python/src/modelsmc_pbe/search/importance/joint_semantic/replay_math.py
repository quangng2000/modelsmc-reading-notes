"""Numerically stable primitives shared by semantic-ledger replay checks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ledger import SemanticTraceProbabilityLedger

ABSOLUTE_TOLERANCE = 1e-12


def close(left: float, right: float) -> bool:
    """Compare independently replayed log-domain quantities."""

    return math.isclose(left, right, rel_tol=0.0, abs_tol=ABSOLUTE_TOLERANCE)


def _logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def _logsumexp(values: tuple[float, ...]) -> float:
    maximum = max(values)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def semantic_log_probabilities(
    records: tuple[SemanticTraceProbabilityLedger, ...],
    *,
    semantic_scale: float,
) -> tuple[float, ...]:
    """Normalize the recorded ``log prior + eta * semantic score`` logits."""

    maximum_score = max(record.compatibility_log_score for record in records)
    logits = tuple(
        record.log_prior
        + (
            0.0
            if semantic_scale == 0.0
            else semantic_scale * (record.compatibility_log_score - maximum_score)
        )
        for record in records
    )
    normalizer = _logsumexp(logits)
    if not math.isfinite(normalizer):
        raise ValueError("semantic trace ledger has no finite normalized mass")
    return tuple(logit - normalizer for logit in logits)


def defensive_log_probability(
    *,
    epsilon: float,
    log_prior: float,
    log_q_semantic: float | None,
) -> float:
    """Replay ``log(epsilon*pi + (1-epsilon)*q_A)`` at one trace."""

    semantic = -math.inf if log_q_semantic is None else log_q_semantic
    semantic_mix = -math.inf if epsilon == 1.0 else math.log1p(-epsilon)
    return _logaddexp(math.log(epsilon) + log_prior, semantic_mix + semantic)

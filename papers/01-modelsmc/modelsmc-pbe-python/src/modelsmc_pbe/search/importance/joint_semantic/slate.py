"""Compact construction-trace slates for joint semantic proposals."""

from __future__ import annotations

import hashlib
import heapq
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from ..factorized import trace_log_prior, valid_choice_indices
from ..lazy_records import (
    ConstructionTrace,
    FactorizedFamily,
    FactorizedImportanceSupport,
)


@dataclass(frozen=True, slots=True)
class SemanticSlateEntry:
    """One compact trace with its exact prior and execution-free semantic score."""

    trace: ConstructionTrace
    log_prior: float
    semantic_score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.log_prior):
            raise ValueError("semantic slate log priors must be finite")
        if not math.isfinite(self.semantic_score):
            raise ValueError("semantic slate scores must be finite")


def trace_sort_key(trace: ConstructionTrace) -> tuple[int, tuple[int, ...]]:
    """Return the canonical integer-only ordering key for a construction trace."""

    return trace.hypothesis_index, trace.filling_indices


def _iter_family_traces(
    family: FactorizedFamily,
    *,
    hole_index: int,
    prefix: tuple[int, ...],
    prefix_cost: int,
) -> Iterator[ConstructionTrace]:
    if hole_index == len(family.catalogs):
        yield ConstructionTrace(family.hypothesis_index, prefix)
        return
    choices = valid_choice_indices(
        family,
        hole_index=hole_index,
        prefix_cost=prefix_cost,
    )
    for choice in choices:
        yield from _iter_family_traces(
            family,
            hole_index=hole_index + 1,
            prefix=(*prefix, choice),
            prefix_cost=prefix_cost + family.catalogs[hole_index].costs[choice],
        )


def iter_construction_traces(
    support: FactorizedImportanceSupport,
) -> Iterator[ConstructionTrace]:
    """Yield every budget-valid compact trace without assembling any program."""

    for family in support.families:
        yield from _iter_family_traces(
            family,
            hole_index=0,
            prefix=(),
            prefix_cost=0,
        )


def enumerate_construction_traces(
    support: FactorizedImportanceSupport,
) -> tuple[ConstructionTrace, ...]:
    """Materialize compact traces only and verify the symbolic support count."""

    traces = tuple(iter_construction_traces(support))
    if len(traces) != support.support_states:
        raise RuntimeError("compact trace enumeration disagrees with symbolic support count")
    return traces


def _stable_hash_key(
    trace: ConstructionTrace,
    *,
    seed: int,
) -> tuple[bytes, tuple[int, tuple[int, ...]]]:
    identity = ":".join(
        (
            str(seed),
            str(trace.hypothesis_index),
            *(str(index) for index in trace.filling_indices),
        )
    )
    return hashlib.sha256(identity.encode("ascii")).digest(), trace_sort_key(trace)


def select_deterministic_slate(
    traces: Iterable[ConstructionTrace],
    *,
    size: int | None,
    seed: int = 0,
) -> tuple[ConstructionTrace, ...]:
    """Select a stable hash-ranked trace subset, then restore canonical order."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("semantic slate seed must be an integer")
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 1):
        raise ValueError("semantic slate size must be None or a positive integer")
    if size is None:
        selected = tuple(traces)
    else:
        selected = tuple(
            heapq.nsmallest(size, traces, key=lambda trace: _stable_hash_key(trace, seed=seed))
        )
    if not selected:
        raise ValueError("semantic slate must contain at least one trace")
    if len(set(selected)) != len(selected):
        raise ValueError("semantic slate traces must be unique")
    return tuple(sorted(selected, key=trace_sort_key))


def attach_semantic_scores(
    support: FactorizedImportanceSupport,
    traces: Sequence[ConstructionTrace],
    semantic_scores: Sequence[float],
    *,
    cost_scale: float,
) -> tuple[SemanticSlateEntry, ...]:
    """Attach exact prior log masses to an aligned semantic-score slate."""

    if len(traces) != len(semantic_scores):
        raise ValueError("semantic traces and scores must be aligned")
    if not traces:
        raise ValueError("semantic slate must contain at least one trace")
    return tuple(
        SemanticSlateEntry(
            trace=trace,
            log_prior=trace_log_prior(support, trace, cost_scale=cost_scale),
            semantic_score=float(score),
        )
        for trace, score in zip(traces, semantic_scores, strict=True)
    )

"""Exact combinatorics for factorized program-construction supports."""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from modelsmc_pbe.smc import categorical_sample, normalize_log_weights

from .lazy_records import (
    ConstructionTrace,
    FactorizedFamily,
    FactorizedImportanceSupport,
)
from .records import HoleFilling


def _costs(family: FactorizedFamily) -> tuple[tuple[int, ...], ...]:
    return tuple(catalog.costs for catalog in family.catalogs)


def _mismatches(family: FactorizedFamily) -> tuple[tuple[int, ...], ...]:
    return tuple(catalog.deduction_mismatches for catalog in family.catalogs)


@lru_cache(maxsize=16_384)
def _suffix_count(
    costs: tuple[tuple[int, ...], ...],
    hole_index: int,
    remaining_budget: int,
) -> int:
    if remaining_budget < 0:
        return 0
    if hole_index == len(costs):
        return 1
    return sum(
        _suffix_count(costs, hole_index + 1, remaining_budget - cost)
        for cost in costs[hole_index]
        if cost <= remaining_budget
    )


def support_count(family: FactorizedFamily) -> int:
    """Count complete traces with dynamic programming, not a Cartesian product."""

    return _suffix_count(_costs(family), 0, family.combination_budget)


def count_cost_constrained_products(
    costs: tuple[tuple[int, ...], ...],
    budget: int,
) -> int:
    """Count a catalog product under an additive budget without constructing tuples."""

    return _suffix_count(costs, 0, budget)


def _logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


@lru_cache(maxsize=65_536)
def _suffix_log_partition(
    costs: tuple[tuple[int, ...], ...],
    mismatches: tuple[tuple[int, ...], ...],
    hole_index: int,
    remaining_budget: int,
    cost_scale: float,
    violation_scale: float,
) -> float:
    if remaining_budget < 0:
        return -math.inf
    if hole_index == len(costs):
        return 0.0
    total = -math.inf
    for choice, cost in enumerate(costs[hole_index]):
        if cost > remaining_budget:
            continue
        suffix = _suffix_log_partition(
            costs,
            mismatches,
            hole_index + 1,
            remaining_budget - cost,
            cost_scale,
            violation_scale,
        )
        if suffix == -math.inf:
            continue
        local = -cost_scale * cost - violation_scale * mismatches[hole_index][choice]
        total = _logaddexp(total, local + suffix)
    return total


def family_log_partition(
    family: FactorizedFamily,
    *,
    cost_scale: float,
    violation_scale: float,
) -> float:
    """Return the exact bounded hole-product log partition for one family."""

    value = _suffix_log_partition(
        _costs(family),
        _mismatches(family),
        0,
        family.combination_budget,
        cost_scale,
        violation_scale,
    )
    if not math.isfinite(value):
        raise ValueError("factorized family has no complete trace under its budget")
    return value


def valid_choice_indices(
    family: FactorizedFamily,
    *,
    hole_index: int,
    prefix_cost: int,
) -> tuple[int, ...]:
    """Return choices that retain at least one budget-valid suffix."""

    costs = _costs(family)
    remaining = family.combination_budget - prefix_cost
    return tuple(
        choice
        for choice, cost in enumerate(costs[hole_index])
        if cost <= remaining
        and _suffix_count(costs, hole_index + 1, remaining - cost) > 0
    )


def choice_guide_probabilities(
    family: FactorizedFamily,
    *,
    hole_index: int,
    prefix_cost: int,
    choice_indices: tuple[int, ...],
    cost_scale: float,
    violation_scale: float,
) -> torch.Tensor:
    """Return exact guide conditionals after one construction prefix."""

    costs = _costs(family)
    mismatches = _mismatches(family)
    remaining = family.combination_budget - prefix_cost
    logits: list[float] = []
    for choice in choice_indices:
        cost = costs[hole_index][choice]
        suffix = _suffix_log_partition(
            costs,
            mismatches,
            hole_index + 1,
            remaining - cost,
            cost_scale,
            violation_scale,
        )
        logits.append(
            -cost_scale * cost
            - violation_scale * mismatches[hole_index][choice]
            + suffix
        )
    return normalize_log_weights(torch.tensor(logits, dtype=torch.float64)).weights


def family_guide_probabilities(
    support: FactorizedImportanceSupport,
    *,
    cost_scale: float,
    violation_scale: float,
) -> torch.Tensor:
    """Return exact guide family masses relative to the equal-family prior."""

    evidence = []
    for family in support.families:
        log_prior_partition = family_log_partition(
            family,
            cost_scale=cost_scale,
            violation_scale=0.0,
        )
        log_guide_partition = family_log_partition(
            family,
            cost_scale=cost_scale,
            violation_scale=violation_scale,
        )
        evidence.append(log_guide_partition - log_prior_partition)
    return normalize_log_weights(torch.tensor(evidence, dtype=torch.float64)).weights


def trace_fillings(
    family: FactorizedFamily,
    trace: ConstructionTrace,
) -> tuple[HoleFilling, ...]:
    """Resolve a compact trace to its named immutable hole fillings."""

    if trace.hypothesis_index != family.hypothesis_index:
        raise ValueError("construction trace belongs to a different family")
    if len(trace.filling_indices) != len(family.catalogs):
        raise ValueError("construction trace has the wrong number of hole choices")
    return tuple(
        catalog.fillings[index]
        for catalog, index in zip(family.catalogs, trace.filling_indices, strict=True)
    )


def trace_hole_cost(family: FactorizedFamily, trace: ConstructionTrace) -> int:
    """Return the additive hole cost of one trace."""

    if len(trace.filling_indices) != len(family.catalogs):
        raise ValueError("construction trace has the wrong number of hole choices")
    return sum(
        catalog.costs[index]
        for catalog, index in zip(family.catalogs, trace.filling_indices, strict=True)
    )


def trace_log_prior(
    support: FactorizedImportanceSupport,
    trace: ConstructionTrace,
    *,
    cost_scale: float,
) -> float:
    """Evaluate the normalized equal-family/within-family Occam prior exactly."""

    family = support.family(trace.hypothesis_index)
    hole_cost = trace_hole_cost(family, trace)
    if hole_cost > family.combination_budget:
        raise ValueError("construction trace exceeds the complete-program budget")
    return (
        -math.log(len(support.families))
        - cost_scale * hole_cost
        - family_log_partition(
            family,
            cost_scale=cost_scale,
            violation_scale=0.0,
        )
    )


def sample_prior_trace(
    support: FactorizedImportanceSupport,
    *,
    cost_scale: float,
    generator: torch.Generator,
) -> ConstructionTrace:
    """Draw one exact prior trace using family and suffix conditionals."""

    family_position = int(
        categorical_sample(
            torch.full(
                (len(support.families),),
                1.0 / len(support.families),
                dtype=torch.float64,
            ),
            1,
            generator=generator,
        )[0].item()
    )
    family = support.families[family_position]
    selected: list[int] = []
    prefix_cost = 0
    for hole_index in range(len(family.catalogs)):
        choices = valid_choice_indices(
            family,
            hole_index=hole_index,
            prefix_cost=prefix_cost,
        )
        probabilities = choice_guide_probabilities(
            family,
            hole_index=hole_index,
            prefix_cost=prefix_cost,
            choice_indices=choices,
            cost_scale=cost_scale,
            violation_scale=0.0,
        )
        position = int(
            categorical_sample(probabilities, 1, generator=generator)[0].item()
        )
        choice = choices[position]
        selected.append(choice)
        prefix_cost += family.catalogs[hole_index].costs[choice]
    return ConstructionTrace(
        hypothesis_index=family.hypothesis_index,
        filling_indices=tuple(selected),
    )

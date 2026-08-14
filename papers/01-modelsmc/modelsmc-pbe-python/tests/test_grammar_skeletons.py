from __future__ import annotations

import pytest

from modelsmc_pbe.domain.ast import canonical_key
from modelsmc_pbe.grammar import (
    EnumerationLimitExceeded,
    arithmetic_expressions,
    available_skeletons,
    bounded_square_target,
    enumerate_skeleton,
    filter_predicates,
    signed_piecewise_expression_count,
    signed_piecewise_expressions,
    signed_transforms,
    signed_window_target,
)


def test_small_arithmetic_catalogs_are_finite_unique_and_deterministic() -> None:
    expression = enumerate_skeleton("expression-arithmetic", [1, 0, 1], 100)
    reordered = enumerate_skeleton("expression-arithmetic", [0, 1], 100)
    mapped = enumerate_skeleton("map-arithmetic", [0, 1], 100)

    assert len(expression) == 18
    assert [canonical_key(program) for program in expression] == [
        canonical_key(program) for program in reordered
    ]
    assert len({canonical_key(program) for program in expression}) == len(expression)
    assert len(mapped) == 18
    assert {program["kind"] for program in expression} == {"ExpressionProgram"}
    assert {program["kind"] for program in mapped} == {"MapProgram"}


def test_foldr_filter_map_contains_the_bounded_square_target() -> None:
    programs = enumerate_skeleton("foldr-filter-map", [-2, 3], 1_000)
    keys = {canonical_key(program) for program in programs}

    assert len(programs) == 756
    assert canonical_key(bounded_square_target()) in keys


def test_foldr_filter_map_exposes_exact_factorized_hole_catalogs() -> None:
    assert len(filter_predicates([-2, 3])) == 42
    assert len(arithmetic_expressions("Item", [-2, 3])) == 18
    assert len(filter_predicates(range(-3, 5))) == 600
    assert len(arithmetic_expressions("Item", range(-3, 5))) == 60


def test_piecewise_filter_map_contains_the_signed_window_target() -> None:
    constants = [-3, 0, 3]
    programs = enumerate_skeleton("foldr-filter-piecewise-map", constants, 6_000)
    keys = {canonical_key(program) for program in programs}

    assert len(signed_transforms(constants)) == 6
    assert signed_piecewise_expression_count(constants) == 60
    assert len(signed_piecewise_expressions(constants)) == 60
    assert len(programs) == 5_400
    assert canonical_key(signed_window_target()) in keys


def test_piecewise_catalog_requires_a_declared_zero_literal() -> None:
    assert signed_piecewise_expression_count([-3, 3]) == 0
    assert signed_piecewise_expressions([-3, 3]) == ()


def test_enumeration_never_silently_truncates_support() -> None:
    with pytest.raises(EnumerationLimitExceeded, match="more than 755 states"):
        enumerate_skeleton("foldr-filter-map", [-2, 3], 755)


def test_available_skeletons_are_stable() -> None:
    assert available_skeletons() == (
        "expression-arithmetic",
        "map-arithmetic",
        "foldr-filter-map",
        "foldr-filter-piecewise-map",
    )

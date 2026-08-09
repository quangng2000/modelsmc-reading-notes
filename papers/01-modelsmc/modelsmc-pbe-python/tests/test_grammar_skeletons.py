from __future__ import annotations

import pytest

from modelsmc_pbe.domain.ast import canonical_key
from modelsmc_pbe.grammar import (
    EnumerationLimitExceeded,
    available_skeletons,
    bounded_square_target,
    enumerate_skeleton,
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


def test_enumeration_never_silently_truncates_support() -> None:
    with pytest.raises(EnumerationLimitExceeded, match="more than 755 states"):
        enumerate_skeleton("foldr-filter-map", [-2, 3], 755)


def test_available_skeletons_are_stable() -> None:
    assert available_skeletons() == (
        "expression-arithmetic",
        "map-arithmetic",
        "foldr-filter-map",
    )

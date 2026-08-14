from __future__ import annotations

import math

import pytest

from research.adaptive_shortlist_experiment import (
    decode_structured_candidate,
    select_first_hole,
    top_level_mixture_probability,
)


def test_selects_predicate_when_predicate_evidence_exists() -> None:
    hole, audit = select_first_hole({"predicate_toggles": [{"item": 0}], "mapper_constraints": []})
    assert hole == "predicate"
    assert audit["selected_first_hole"] == "predicate"


def test_selects_mapper_when_only_mapper_evidence_exists() -> None:
    hole, _ = select_first_hole({"predicate_toggles": [], "mapper_constraints": [{"item": 2}]})
    assert hole == "mapper"


def test_accepts_tuple_evidence_from_feedback_dataclass() -> None:
    hole, audit = select_first_hole({"predicate_toggles": ({"item": 0},), "mapper_constraints": ()})
    assert hole == "predicate"
    assert audit["predicate_toggle_count"] == 1


def test_top_level_mixture_normalizes_over_full_program_space() -> None:
    full_size = 100
    tree_size = 4
    probabilities = [
        top_level_mixture_probability(
            index < tree_size,
            tree_size=tree_size,
            base_probability=1 / full_size,
            epsilon=0.05,
        )
        for index in range(full_size)
    ]
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-15)
    assert probabilities[0] == pytest.approx(0.2375 + 0.0005)
    assert probabilities[-1] == pytest.approx(0.0005)


def test_decodes_typed_mapper_without_enumerated_catalog() -> None:
    assert (
        decode_structured_candidate("mapper", {"template": "mul(item,item)", "constant": "-3"})
        == "mul(item,item)"
    )
    assert (
        decode_structured_candidate("mapper", {"template": "sub(c,item)", "constant": "-2"})
        == "sub(-2,item)"
    )


def test_decodes_typed_predicate() -> None:
    actual = decode_structured_candidate(
        "predicate",
        {
            "combine": "and",
            "left": {"template": "lt(c,item)", "constant": "-2"},
            "right": {"template": "lt(item,c)", "constant": "3"},
        },
    )
    assert actual == "and(lt(-2,item),lt(item,3))"

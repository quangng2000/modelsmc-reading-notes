from __future__ import annotations

from collections.abc import Iterable, Mapping

import pytest

from modelsmc_pbe.core.cost import expression_cost
from modelsmc_pbe.domain import PBESpec, ValueType, canonical_key
from modelsmc_pbe.enumeration import (
    HoleEnumerationLimitExceeded,
    HoleEnumerationOptions,
    enumerate_hole_expressions,
)
from modelsmc_pbe.induction import (
    HoleSpec,
    SkeletonKind,
    TypedVariable,
    induce_hypotheses,
)


def _node_keys(expressions: Iterable[Mapping[str, object]]) -> set[str]:
    return {canonical_key(expression) for expression in expressions}


def _contains_kind(expression: Mapping[str, object], wanted: str) -> bool:
    if expression.get("kind") == wanted:
        return True
    return any(
        _contains_kind(value, wanted)
        for value in expression.values()
        if isinstance(value, Mapping)
    )


def _spec(
    signature: dict[str, str],
    examples: list[dict[str, object]],
) -> PBESpec:
    return PBESpec.model_validate(
        {
            "name": "enumeration-test",
            "signature": signature,
            "examples": examples,
            "integerConstants": [-1, 0, 1],
        }
    )


def test_arithmetic_hole_is_enumerated_by_exact_structural_cost() -> None:
    hole = HoleSpec(
        name="body",
        parameters=(TypedVariable("x", ValueType.INT),),
        output_type=ValueType.INT,
    )
    catalog = enumerate_hole_expressions(
        hole,
        [-1, 0, 1, 1],
        HoleEnumerationOptions(max_cost=3, state_limit=10_000),
    )

    expected = {
        "kind": "Add",
        "left": {"kind": "Input"},
        "right": {"kind": "IntLiteral", "intValue": "1"},
    }
    assert canonical_key(expected) in _node_keys(catalog.expressions_at_cost(3))
    assert all(entry.cost == expression_cost(entry.expression) for entry in catalog.entries)
    assert [entry.cost for entry in catalog.entries] == sorted(
        entry.cost for entry in catalog.entries
    )
    for cost in range(1, 4):
        assert [entry.canonical_key for entry in catalog.entries_at_cost(cost)] == sorted(
            entry.canonical_key for entry in catalog.entries_at_cost(cost)
        )


def test_predicate_enumeration_uses_only_allowed_integer_constants() -> None:
    hole = HoleSpec(
        name="predicate",
        parameters=(TypedVariable("item", ValueType.INT),),
        output_type=ValueType.BOOL,
    )
    catalog = enumerate_hole_expressions(
        hole,
        [0, 2],
        HoleEnumerationOptions(max_cost=3, state_limit=10_000),
    )

    expected = {
        "kind": "LessThan",
        "left": {"kind": "Item"},
        "right": {"kind": "IntLiteral", "intValue": "0"},
    }
    assert canonical_key(expected) in _node_keys(catalog.expressions_at_cost(3))
    serialized = "\n".join(entry.canonical_key for entry in catalog.entries)
    assert '"intValue":"0"' in serialized
    assert '"intValue":"2"' in serialized
    assert '"intValue":"1"' not in serialized
    assert not any(_contains_kind(entry.expression, "Input") for entry in catalog.entries)
    assert not any(_contains_kind(entry.expression, "Accumulator") for entry in catalog.entries)


def test_induced_map_mapper_scope_exposes_item_but_not_outer_input() -> None:
    spec = _spec(
        {"input": "List<Int>", "output": "List<Int>"},
        [{"input": [], "output": []}, {"input": [1], "output": [2]}],
    )
    mapper = induce_hypotheses(spec).hypothesis(SkeletonKind.MAP).hole("mapper")

    catalog = enumerate_hole_expressions(
        mapper,
        spec.integer_constants,
        HoleEnumerationOptions(max_cost=3, state_limit=10_000),
    )

    assert {"kind": "Item"} in catalog.expressions_at_cost(1)
    assert not any(_contains_kind(entry.expression, "Input") for entry in catalog.entries)
    assert not any(_contains_kind(entry.expression, "Accumulator") for entry in catalog.entries)


def test_induced_foldr_reducer_can_combine_item_and_accumulator() -> None:
    spec = _spec(
        {"input": "List<Int>", "output": "List<Int>"},
        [{"input": [], "output": []}, {"input": [1], "output": [1]}],
    )
    reducer = induce_hypotheses(spec).hypothesis(SkeletonKind.FOLD_RIGHT).hole("reducer")

    catalog = enumerate_hole_expressions(
        reducer,
        spec.integer_constants,
        HoleEnumerationOptions(max_cost=3, state_limit=10_000),
    )

    expected = {
        "kind": "PrependInt",
        "head": {"kind": "Item"},
        "tail": {"kind": "Accumulator"},
    }
    assert canonical_key(expected) in _node_keys(catalog.expressions_at_cost(3))
    assert {"kind": "Accumulator"} in catalog.expressions_at_cost(1)
    assert not any(_contains_kind(entry.expression, "Input") for entry in catalog.entries)


def test_enumeration_fails_instead_of_returning_a_state_prefix() -> None:
    hole = HoleSpec(
        name="body",
        parameters=(TypedVariable("x", ValueType.INT),),
        output_type=ValueType.INT,
    )

    with pytest.raises(HoleEnumerationLimitExceeded) as raised:
        enumerate_hole_expressions(
            hole,
            [0, 1],
            HoleEnumerationOptions(max_cost=3, state_limit=1),
        )

    assert raised.value.state_limit == 1
    assert raised.value.attempted_states == 2
    assert raised.value.cost == 1


@pytest.mark.parametrize(
    "options",
    [
        HoleEnumerationOptions(max_cost=1, state_limit=10),
        HoleEnumerationOptions(max_cost=2, state_limit=20),
    ],
)
def test_catalog_never_exceeds_requested_cost(options: HoleEnumerationOptions) -> None:
    hole = HoleSpec(name="constant", parameters=(), output_type=ValueType.BOOL)
    catalog = enumerate_hole_expressions(hole, [], options)

    assert all(1 <= entry.cost <= options.max_cost for entry in catalog.entries)


def test_invalid_bounds_and_constants_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_cost"):
        HoleEnumerationOptions(max_cost=0, state_limit=1)
    with pytest.raises(ValueError, match="state_limit"):
        HoleEnumerationOptions(max_cost=1, state_limit=0)

    hole = HoleSpec(name="constant", parameters=(), output_type=ValueType.INT)
    with pytest.raises(TypeError, match="integer constants"):
        enumerate_hole_expressions(
            hole,
            [True],
            HoleEnumerationOptions(max_cost=1, state_limit=10),
        )

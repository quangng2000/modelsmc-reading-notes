from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from modelsmc_pbe.domain import PBESpec, ValueType
from modelsmc_pbe.induction import (
    SkeletonKind,
    StructuralRelation,
    assemble_program,
    induce_hypotheses,
    render_hypothesis,
    render_induction_trace,
)


def _spec(payload: dict[str, object]) -> PBESpec:
    return PBESpec.model_validate({"name": "test", "integerConstants": [-1, 0, 1], **payload})


def test_list_signature_generates_several_typed_hypotheses_in_stable_order() -> None:
    spec = _spec(
        {
            "examples": [
                {"input": [], "output": []},
                {"input": [1], "output": [2]},
                {"input": [1, 2], "output": [2, 3]},
            ],
            "signature": {"input": "List<Int>", "output": "List<Int>"},
        }
    )

    report = induce_hypotheses(spec)

    assert tuple(hypothesis.kind for hypothesis in report.hypotheses) == (
        SkeletonKind.EXPRESSION,
        SkeletonKind.MAP,
        SkeletonKind.FOLD_RIGHT,
    )
    mapper = report.hypothesis(SkeletonKind.MAP).hole("mapper")
    assert mapper.parameters[0].value_type is ValueType.INT
    assert report.hypothesis(SkeletonKind.MAP).hole("mapper").output_type is ValueType.INT
    reducer = report.hypothesis(SkeletonKind.FOLD_RIGHT).hole("reducer")
    assert tuple(parameter.value_type for parameter in reducer.parameters) == (
        ValueType.INT,
        ValueType.INT_LIST,
    )
    assert tuple(fact.relation for fact in report.facts) == (
        StructuralRelation.LENGTH_PRESERVED,
        StructuralRelation.EMPTY_INPUT_PRODUCES_EMPTY_OUTPUT,
    )
    assert all(fact.holds for fact in report.facts)


def test_hypothesis_generation_is_signature_based_not_false_fact_pruning() -> None:
    spec = _spec(
        {
            "examples": [{"input": [1, 2], "output": [3]}],
            "signature": {"input": "List<Int>", "output": "List<Int>"},
        }
    )

    report = induce_hypotheses(spec)

    assert report.facts[0].holds is False
    assert report.facts[0].contradicting_examples == (1,)
    assert SkeletonKind.MAP in {hypothesis.kind for hypothesis in report.hypotheses}


@pytest.mark.parametrize(
    ("signature", "kinds"),
    [
        (
            {"input": "Int", "output": "Bool"},
            (SkeletonKind.EXPRESSION,),
        ),
        (
            {"input": "List<Int>", "output": "Int"},
            (SkeletonKind.EXPRESSION, SkeletonKind.FOLD_RIGHT),
        ),
    ],
)
def test_signature_controls_available_wrappers(
    signature: dict[str, str],
    kinds: tuple[SkeletonKind, ...],
) -> None:
    example = {"input": 1, "output": True}
    if signature["input"] == "List<Int>":
        example = {"input": [1], "output": 1}
    report = induce_hypotheses(_spec({"examples": [example], "signature": signature}))

    assert tuple(hypothesis.kind for hypothesis in report.hypotheses) == kinds


def test_map_hypothesis_supports_element_type_change_and_assembly() -> None:
    spec = _spec(
        {
            "examples": [
                {"input": [], "output": []},
                {"input": [0, 1], "output": [True, False]},
            ],
            "signature": {"input": "List<Int>", "output": "List<Bool>"},
        }
    )
    hypothesis = induce_hypotheses(spec).hypothesis(SkeletonKind.MAP)

    program = assemble_program(
        hypothesis,
        {
            "mapper": {
                "kind": "EqualInt",
                "left": {"kind": "Item"},
                "right": {"kind": "IntLiteral", "intValue": "0"},
            }
        },
        allowed_integer_constants=[-1, 0, 1],
    )

    assert program["kind"] == "MapProgram"
    assert hypothesis.hole("mapper").output_type is ValueType.BOOL
    with pytest.raises(ValueError, match="hole fillings mismatch"):
        assemble_program(hypothesis, {})
    with pytest.raises(ValueError, match="expected BoolListType"):
        assemble_program(hypothesis, {"mapper": {"kind": "Item"}})


def test_hypotheses_are_immutable_and_traces_are_stable() -> None:
    spec = _spec(
        {
            "examples": [{"input": [], "output": []}, {"input": [1], "output": [2]}],
            "signature": {"input": "List<Int>", "output": "List<Int>"},
        }
    )
    report = induce_hypotheses(spec)
    mapper = report.hypothesis(SkeletonKind.MAP)

    with pytest.raises(FrozenInstanceError):
        mapper.output_type = ValueType.BOOL_LIST  # type: ignore[misc]

    assert render_hypothesis(mapper) == (
        "(xs: List<Int>) => map((item: Int) => ?mapper: Int, xs)"
    )
    assert render_induction_trace(report) == (
        "[generalize] signature List<Int> -> List<Int>",
        "[generalize] relation length-preserved: true (support=1,2; contradict=none)",
        "[generalize] relation empty-input-produces-empty-output: true "
        "(support=1; contradict=none)",
        "[generalize] hypothesis expression: (xs: List<Int>) => ?body: List<Int>",
        "[generalize] hypothesis map: (xs: List<Int>) => "
        "map((item: Int) => ?mapper: Int, xs)",
        "[generalize] hypothesis foldr: (xs: List<Int>) => "
        "foldr((item: Int, acc: List<Int>) => ?reducer: List<Int>, "
        "?initial: List<Int>, xs)",
    )

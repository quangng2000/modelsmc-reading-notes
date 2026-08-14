from __future__ import annotations

from pathlib import Path

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.deduction import (
    DeductionFactKind,
    RefutationKind,
    deduce_hypotheses,
    deduce_hypothesis,
    render_deduction_guidance,
    render_deduction_trace,
    render_hole_example,
)
from modelsmc_pbe.domain import PBESpec
from modelsmc_pbe.induction import SkeletonKind, induce_hypotheses

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _spec(examples: list[dict[str, object]], output: str) -> PBESpec:
    return PBESpec.model_validate(
        {
            "name": "test",
            "signature": {"input": "List<Int>", "output": output},
            "examples": examples,
            "integerConstants": [-1, 0, 1, 2, 3, 4],
        }
    )


def _report(spec: PBESpec, kind: SkeletonKind):
    hypothesis = induce_hypotheses(spec).hypothesis(kind)
    return deduce_hypothesis(spec, hypothesis)


def test_expression_deduction_turns_top_level_examples_into_body_examples() -> None:
    spec = _spec(
        [
            {"input": [], "output": 0},
            {"input": [1, 2], "output": 3},
        ],
        "Int",
    )

    report = _report(spec, SkeletonKind.EXPRESSION)

    assert report.viable is True
    assert len(report.examples_for("body")) == 2
    assert tuple(render_hole_example(example) for example in report.hole_examples) == (
        "?body: [] -> 0 (examples=1)",
        "?body: [1, 2] -> 3 (examples=2)",
    )
    assert report.facts[0].kind is DeductionFactKind.BODY_EXAMPLES


def test_map_deduction_derives_consistent_pointwise_examples() -> None:
    spec = _spec(
        [
            {"input": [], "output": []},
            {"input": [1, 1], "output": [2, 2]},
            {"input": [2], "output": [3]},
        ],
        "List<Int>",
    )

    report = _report(spec, SkeletonKind.MAP)

    assert report.viable is True
    assert tuple(render_hole_example(example) for example in report.examples_for("mapper")) == (
        "?mapper: 1 -> 2 (examples=2)",
        "?mapper: 2 -> 3 (examples=3)",
    )
    assert tuple(fact.kind for fact in report.facts) == (
        DeductionFactKind.MAP_LENGTH_PRESERVED,
        DeductionFactKind.MAPPER_EXAMPLES,
    )


def test_map_refutes_length_changes_before_inventing_mapper_examples() -> None:
    spec = _spec([{"input": [1, 2], "output": [3]}], "List<Int>")

    report = _report(spec, SkeletonKind.MAP)

    assert report.viable is False
    assert report.refutation is not None
    assert report.refutation.kind is RefutationKind.MAP_LENGTH_MISMATCH
    assert report.hole_examples == ()
    assert "map preserves list length" in report.refutation.detail


def test_map_refutes_one_input_item_requiring_two_outputs() -> None:
    spec = _spec([{"input": [1, 1], "output": [2, 3]}], "List<Int>")

    report = _report(spec, SkeletonKind.MAP)

    assert report.viable is False
    assert report.refutation is not None
    assert report.refutation.kind is RefutationKind.MAP_FUNCTION_CONFLICT
    assert report.refutation.source_examples == (1,)
    assert report.refutation.detail == "a deterministic mapper cannot map 1 to both 2 and 3"


def test_foldr_deduces_initial_and_only_observed_suffix_reducer_examples() -> None:
    spec = _spec(
        [
            {"input": [], "output": 0},
            {"input": [1], "output": 1},
            {"input": [2], "output": 2},
            {"input": [1, 2], "output": 3},
            {"input": [3, 1, 2], "output": 6},
        ],
        "Int",
    )

    report = _report(spec, SkeletonKind.FOLD_RIGHT)

    assert report.viable is True
    assert tuple(render_hole_example(example) for example in report.examples_for("initial")) == (
        "?initial: () -> 0 (examples=1)",
    )
    assert tuple(render_hole_example(example) for example in report.examples_for("reducer")) == (
        "?reducer: (1, 0) -> 1 (examples=2,1)",
        "?reducer: (2, 0) -> 2 (examples=3,1)",
        "?reducer: (1, 2) -> 3 (examples=4,3)",
        "?reducer: (3, 3) -> 6 (examples=5,4)",
    )


def test_foldr_marks_unobserved_holes_as_underconstrained() -> None:
    spec = _spec([{"input": [1, 2], "output": 3}], "Int")

    report = _report(spec, SkeletonKind.FOLD_RIGHT)

    assert report.viable is True
    assert report.hole_examples == ()
    assert [fact.hole.name for fact in report.facts if fact.hole is not None][-2:] == [
        "initial",
        "reducer",
    ]
    assert tuple(fact.kind for fact in report.facts[-2:]) == (
        DeductionFactKind.HOLE_UNDERCONSTRAINED,
        DeductionFactKind.HOLE_UNDERCONSTRAINED,
    )


def test_foldr_refutes_conflicting_reducer_requirements() -> None:
    spec = _spec(
        [
            {"input": [2], "output": 2},
            {"input": [3], "output": 2},
            {"input": [1, 2], "output": 3},
            {"input": [1, 3], "output": 4},
        ],
        "Int",
    )

    report = _report(spec, SkeletonKind.FOLD_RIGHT)

    assert report.viable is False
    assert report.refutation is not None
    assert report.refutation.kind is RefutationKind.FOLDR_REDUCER_CONFLICT
    assert report.refutation.source_examples == (3, 1, 4, 2)
    assert report.refutation.detail == (
        "a deterministic fold reducer cannot map (1, 2) to both 3 and 4"
    )


def test_foldr_refutes_conflicting_empty_input_as_initial_conflict() -> None:
    spec = _spec(
        [
            {"input": [], "output": 0},
            {"input": [], "output": 1},
        ],
        "Int",
    )

    report = _report(spec, SkeletonKind.FOLD_RIGHT)

    assert report.viable is False
    assert report.refutation is not None
    assert report.refutation.kind is RefutationKind.FOLDR_INITIAL_CONFLICT
    assert report.refutation.source_examples == (1, 2)


def test_foldr_filter_map_derives_predicate_and_mapped_value_suffix_examples() -> None:
    spec = load_experiment_config(PROJECT_DIR / "examples" / "foldr-bounded-square.json").spec

    report = _report(spec, SkeletonKind.FOLD_RIGHT_FILTER_MAP)

    assert report.viable is True
    predicate_examples = {
        int(example.inputs[0].value): bool(example.output.value)
        for example in report.examples_for("predicate")
    }
    mapped_examples = {
        int(example.inputs[0].value): int(example.output.value)
        for example in report.examples_for("mapped_value")
    }
    assert predicate_examples == {
        -3: False,
        -2: False,
        -1: True,
        0: True,
        1: True,
        2: True,
        3: False,
        4: False,
    }
    assert mapped_examples == {-1: 1, 0: 0, 1: 1, 2: 4}
    assert tuple(fact.kind for fact in report.facts) == (
        DeductionFactKind.FOLDR_FILTER_PREDICATE_EXAMPLES,
        DeductionFactKind.FOLDR_FILTER_MAPPED_VALUE_EXAMPLES,
    )


def test_piecewise_filter_map_derives_direct_mapper_examples_without_branch_guessing() -> None:
    spec = load_experiment_config(PROJECT_DIR / "examples" / "foldr-signed-window.json").spec

    report = _report(spec, SkeletonKind.FOLD_RIGHT_FILTER_PIECEWISE_MAP)

    assert report.viable is True
    predicate_examples = {
        int(example.inputs[0].value): bool(example.output.value)
        for example in report.examples_for("predicate")
    }
    mapped_examples = {
        int(example.inputs[0].value): int(example.output.value)
        for example in report.examples_for("piecewise_mapped_value")
    }
    assert predicate_examples == {
        -3: False,
        -2: True,
        -1: True,
        0: True,
        1: True,
        2: True,
        3: False,
        4: False,
    }
    assert mapped_examples == {-2: 2, -1: 1, 0: 0, 1: 1, 2: 4}
    assert tuple(fact.kind for fact in report.facts) == (
        DeductionFactKind.FOLDR_FILTER_PREDICATE_EXAMPLES,
        DeductionFactKind.FOLDR_FILTER_PIECEWISE_MAPPED_VALUE_EXAMPLES,
    )


def test_foldr_filter_map_refutes_impossible_shapes_and_context_dependence() -> None:
    shape = _report(
        _spec([{"input": [], "output": [1]}], "List<Int>"),
        SkeletonKind.FOLD_RIGHT_FILTER_MAP,
    )
    predicate = _report(
        _spec(
            [
                {"input": [], "output": []},
                {"input": [1], "output": []},
                {"input": [2], "output": []},
                {"input": [1, 2], "output": [7]},
            ],
            "List<Int>",
        ),
        SkeletonKind.FOLD_RIGHT_FILTER_MAP,
    )
    mapped_value = _report(
        _spec(
            [
                {"input": [], "output": []},
                {"input": [1], "output": [2]},
                {"input": [2], "output": []},
                {"input": [1, 2], "output": [3]},
            ],
            "List<Int>",
        ),
        SkeletonKind.FOLD_RIGHT_FILTER_MAP,
    )

    assert shape.refutation is not None
    assert shape.refutation.kind is RefutationKind.FOLDR_FILTER_MAP_SHAPE_MISMATCH
    assert predicate.refutation is not None
    assert predicate.refutation.kind is RefutationKind.FOLDR_FILTER_PREDICATE_CONFLICT
    assert mapped_value.refutation is not None
    assert mapped_value.refutation.kind is RefutationKind.FOLDR_FILTER_MAPPED_VALUE_CONFLICT


def test_duplicate_top_level_input_refutes_every_hypothesis() -> None:
    spec = _spec(
        [
            {"input": [1], "output": 1},
            {"input": [1], "output": 2},
        ],
        "Int",
    )
    induction = induce_hypotheses(spec)

    reports = deduce_hypotheses(spec, induction.hypotheses)

    assert len(reports) == 2
    assert all(report.refutation is not None for report in reports)
    assert {report.refutation.kind for report in reports if report.refutation is not None} == {
        RefutationKind.INCONSISTENT_SPEC
    }


def test_deduction_trace_and_prompt_guidance_are_stable() -> None:
    spec = _spec(
        [
            {"input": [], "output": []},
            {"input": [1], "output": [2]},
        ],
        "List<Int>",
    )
    report = _report(spec, SkeletonKind.MAP)

    assert render_deduction_trace(report) == (
        "[deduce] hypothesis map: viable",
        "[deduce] fact map-length-preserved: derived=0; examples=1,2",
        "[deduce] fact mapper-examples ?mapper: derived=1; examples=2",
        "[deduce] ?mapper: 1 -> 2 (examples=2)",
    )
    assert render_deduction_guidance(report) == (
        "Hypothesis: (xs: List<Int>) => map((item: Int) => ?mapper: Int, xs)\n"
        "Status: viable on all sound deductions performed\n"
        "Derived hole specifications:\n"
        "- ?mapper: 1 -> 2 (examples=2)"
    )

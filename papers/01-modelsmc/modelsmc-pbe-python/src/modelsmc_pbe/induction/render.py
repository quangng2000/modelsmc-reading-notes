"""Stable rendering for hypotheses and inductive-generalization traces."""

from __future__ import annotations

from .records import HoleSpec, InductionReport, SkeletonKind, StructuralFact, TypedSkeleton


def render_hole_type(hole: HoleSpec) -> str:
    """Render one hole as a curried first-order type."""

    types = [parameter.value_type.value for parameter in hole.parameters]
    types.append(hole.output_type.value)
    return " -> ".join(types)


def render_hypothesis(hypothesis: TypedSkeleton) -> str:
    """Render a skeleton without pretending that its holes are programs."""

    input_variable = hypothesis.input_variable
    if hypothesis.kind is SkeletonKind.EXPRESSION:
        return (
            f"({input_variable.name}: {input_variable.value_type.value}) => "
            f"?body: {hypothesis.output_type.value}"
        )
    if hypothesis.kind is SkeletonKind.MAP:
        mapper = hypothesis.hole("mapper")
        item = mapper.parameters[0]
        return (
            f"({input_variable.name}: {input_variable.value_type.value}) => "
            f"map(({item.name}: {item.value_type.value}) => ?mapper: "
            f"{mapper.output_type.value}, {input_variable.name})"
        )
    if hypothesis.kind in {
        SkeletonKind.FOLD_RIGHT_FILTER_MAP,
        SkeletonKind.FOLD_RIGHT_FILTER_PIECEWISE_MAP,
    }:
        predicate = hypothesis.hole("predicate")
        mapped_label = (
            "piecewise_mapped_value"
            if hypothesis.kind is SkeletonKind.FOLD_RIGHT_FILTER_PIECEWISE_MAP
            else "mapped_value"
        )
        mapped_value = hypothesis.hole(mapped_label)
        item = predicate.parameters[0]
        return (
            f"({input_variable.name}: {input_variable.value_type.value}) => "
            f"foldr(({item.name}: {item.value_type.value}, acc: List<Int>) => "
            f"if ?predicate: {predicate.output_type.value} then "
            f"?{mapped_label}: {mapped_value.output_type.value} :: acc else acc, "
            f"[], {input_variable.name})"
        )
    initial = hypothesis.hole("initial")
    reducer = hypothesis.hole("reducer")
    item, accumulator = reducer.parameters
    return (
        f"({input_variable.name}: {input_variable.value_type.value}) => "
        f"foldr(({item.name}: {item.value_type.value}, "
        f"{accumulator.name}: {accumulator.value_type.value}) => ?reducer: "
        f"{reducer.output_type.value}, ?initial: {initial.output_type.value}, "
        f"{input_variable.name})"
    )


def _examples(indices: tuple[int, ...]) -> str:
    return "none" if not indices else ",".join(str(index) for index in indices)


def render_structural_fact(fact: StructuralFact) -> str:
    """Render one aggregate relationship with explicit evidence indices."""

    truth = "true" if fact.holds else "false"
    return (
        f"relation {fact.relation.value}: {truth} "
        f"(support={_examples(fact.supporting_examples)}; "
        f"contradict={_examples(fact.contradicting_examples)})"
    )


def render_induction_trace(report: InductionReport) -> tuple[str, ...]:
    """Return deterministic console-ready lines for one induction report."""

    lines = [
        f"[generalize] signature {report.signature.input_type.value} -> "
        f"{report.signature.output_type.value}"
    ]
    lines.extend(f"[generalize] {render_structural_fact(fact)}" for fact in report.facts)
    lines.extend(
        f"[generalize] hypothesis {hypothesis.kind.value}: {render_hypothesis(hypothesis)}"
        for hypothesis in report.hypotheses
    )
    return tuple(lines)

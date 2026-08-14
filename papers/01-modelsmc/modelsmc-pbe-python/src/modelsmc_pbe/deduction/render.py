"""Stable rendering for deduction evidence, traces, and LLM guidance."""

from __future__ import annotations

from typing import cast

from modelsmc_pbe.domain.models import ValueType
from modelsmc_pbe.induction import render_hypothesis

from .records import DeductionFact, DeductionReport, HoleExample, Refutation, TypedValue


def render_typed_value(value: TypedValue) -> str:
    """Render an immutable typed value without losing empty-list meaning."""

    raw = value.value
    if value.value_type is ValueType.BOOL:
        return "true" if raw else "false"
    if value.value_type is ValueType.INT:
        return str(raw)
    values = cast(tuple[int, ...] | tuple[bool, ...], raw)
    if value.value_type is ValueType.BOOL_LIST:
        return "[" + ", ".join("true" if item else "false" for item in values) + "]"
    return "[" + ", ".join(str(item) for item in values) + "]"


def _indices(indices: tuple[int, ...]) -> str:
    return ",".join(str(index) for index in indices)


def render_hole_example(example: HoleExample) -> str:
    """Render one derived hole-level PBE constraint."""

    arguments = tuple(render_typed_value(value) for value in example.inputs)
    if not arguments:
        left = "()"
    elif len(arguments) == 1:
        left = arguments[0]
    else:
        left = "(" + ", ".join(arguments) + ")"
    return (
        f"?{example.hole.name}: {left} -> {render_typed_value(example.output)} "
        f"(examples={_indices(example.source_examples)})"
    )


def render_deduction_fact(fact: DeductionFact) -> str:
    """Render a fact with its derivation count and sources."""

    hole = "" if fact.hole is None else f" ?{fact.hole.name}"
    sources = "none" if not fact.source_examples else _indices(fact.source_examples)
    return (
        f"fact {fact.kind.value}{hole}: derived={fact.derived_examples}; "
        f"examples={sources}"
    )


def render_refutation(refutation: Refutation) -> str:
    """Render one sound refutation and its evidence locations."""

    return (
        f"{refutation.kind.value} (examples={_indices(refutation.source_examples)}): "
        f"{refutation.detail}"
    )


def render_deduction_trace(report: DeductionReport) -> tuple[str, ...]:
    """Return deterministic console-ready lines for one deduction report."""

    kind = report.hypothesis.kind.value
    if report.refutation is None:
        status = "viable"
    else:
        status = f"refuted: {render_refutation(report.refutation)}"
    lines = [f"[deduce] hypothesis {kind}: {status}"]
    lines.extend(f"[deduce] {render_deduction_fact(fact)}" for fact in report.facts)
    lines.extend(f"[deduce] {render_hole_example(example)}" for example in report.hole_examples)
    return tuple(lines)


def render_deduction_guidance(report: DeductionReport) -> str:
    """Render a compact deterministic block suitable for an LLM prompt."""

    lines = [f"Hypothesis: {render_hypothesis(report.hypothesis)}"]
    if report.refutation is not None:
        lines.append(f"Status: refuted by {render_refutation(report.refutation)}")
        return "\n".join(lines)
    lines.append("Status: viable on all sound deductions performed")
    if report.hole_examples:
        lines.append("Derived hole specifications:")
        lines.extend(f"- {render_hole_example(example)}" for example in report.hole_examples)
    else:
        lines.append("Derived hole specifications: none")
    return "\n".join(lines)

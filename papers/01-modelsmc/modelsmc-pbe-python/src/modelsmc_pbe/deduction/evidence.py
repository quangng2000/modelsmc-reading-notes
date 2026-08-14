"""Shared operations for merging and checking derived hole examples."""

from __future__ import annotations

from collections.abc import Iterable

from .records import HoleExample, Refutation, RefutationKind, TypedValue
from .render import render_typed_value


def source_union(*groups: tuple[int, ...]) -> tuple[int, ...]:
    """Combine one-based source indices while preserving first-seen order."""

    return tuple(dict.fromkeys(index for group in groups for index in group))


def deduplicate_examples(examples: Iterable[HoleExample]) -> tuple[HoleExample, ...]:
    """Merge identical typed constraints and retain all supporting sources."""

    unique: dict[tuple[tuple[TypedValue, ...], TypedValue], HoleExample] = {}
    for example in examples:
        key = (example.inputs, example.output)
        previous = unique.get(key)
        if previous is None:
            unique[key] = example
        else:
            unique[key] = HoleExample(
                hole=example.hole,
                inputs=example.inputs,
                output=example.output,
                source_examples=source_union(
                    previous.source_examples,
                    example.source_examples,
                ),
            )
    return tuple(unique.values())


def conflicting_examples(
    examples: tuple[HoleExample, ...],
    *,
    kind: RefutationKind,
    subject: str,
) -> Refutation | None:
    """Refute two deterministic constraints with equal inputs and unequal outputs."""

    outputs: dict[tuple[TypedValue, ...], HoleExample] = {}
    for example in examples:
        previous = outputs.get(example.inputs)
        if previous is None:
            outputs[example.inputs] = example
            continue
        if previous.output == example.output:
            continue
        arguments = tuple(render_typed_value(value) for value in example.inputs)
        if not arguments:
            rendered_arguments = "()"
        elif len(arguments) == 1:
            rendered_arguments = arguments[0]
        else:
            rendered_arguments = "(" + ", ".join(arguments) + ")"
        return Refutation(
            kind=kind,
            source_examples=source_union(
                previous.source_examples,
                example.source_examples,
            ),
            detail=(
                f"a deterministic {subject} cannot map {rendered_arguments} to both "
                f"{render_typed_value(previous.output)} and "
                f"{render_typed_value(example.output)}"
            ),
        )
    return None

"""Sound example deduction for expression, map, and right-fold skeletons."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from modelsmc_pbe.domain.models import PBESpec, TypeSignature
from modelsmc_pbe.induction import SkeletonKind, TypedSkeleton

from .records import (
    DeductionFact,
    DeductionFactKind,
    DeductionReport,
    HoleExample,
    Refutation,
    RefutationKind,
    TypedValue,
)
from .render import render_typed_value


def _required_signature(spec: PBESpec) -> TypeSignature:
    signature = spec.signature
    if signature is None:  # pragma: no cover - PBESpec's validator always supplies it
        raise ValueError("PBE specification has no inferred or explicit signature")
    return signature


def _validate_hypothesis(spec: PBESpec, hypothesis: TypedSkeleton) -> TypeSignature:
    signature = _required_signature(spec)
    if (
        hypothesis.input_variable.value_type is not signature.input_type
        or hypothesis.output_type is not signature.output_type
    ):
        raise ValueError("hypothesis signature does not match the PBE specification")
    return signature


def _source_union(*groups: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(index for group in groups for index in group))


def _deduplicate(examples: Iterable[HoleExample]) -> tuple[HoleExample, ...]:
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
                source_examples=_source_union(previous.source_examples, example.source_examples),
            )
    return tuple(unique.values())


def _conflict(
    examples: tuple[HoleExample, ...],
    *,
    kind: RefutationKind,
    subject: str,
) -> Refutation | None:
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
            source_examples=_source_union(previous.source_examples, example.source_examples),
            detail=(
                f"a deterministic {subject} cannot map {rendered_arguments} to both "
                f"{render_typed_value(previous.output)} and {render_typed_value(example.output)}"
            ),
        )
    return None


def _inconsistent_top_level(
    spec: PBESpec,
    signature: TypeSignature,
    hypothesis: TypedSkeleton,
) -> Refutation | None:
    observations: dict[TypedValue, tuple[TypedValue, int]] = {}
    for index, example in enumerate(spec.examples, start=1):
        input_value = TypedValue.freeze(example.input_value, signature.input_type)
        output = TypedValue.freeze(example.output_value, signature.output_type)
        previous = observations.get(input_value)
        if previous is None:
            observations[input_value] = (output, index)
            continue
        previous_output, previous_index = previous
        if previous_output != output:
            is_fold_initial = (
                hypothesis.kind is SkeletonKind.FOLD_RIGHT
                and isinstance(input_value.value, tuple)
                and not input_value.value
            )
            kind = (
                RefutationKind.FOLDR_INITIAL_CONFLICT
                if is_fold_initial
                else RefutationKind.INCONSISTENT_SPEC
            )
            subject = "fold initial expression" if is_fold_initial else "deterministic program"
            return Refutation(
                kind=kind,
                source_examples=(previous_index, index),
                detail=(
                    f"a {subject} cannot map {render_typed_value(input_value)} "
                    f"to both {render_typed_value(previous_output)} and "
                    f"{render_typed_value(output)}"
                ),
            )
    return None


def _expression(
    spec: PBESpec,
    hypothesis: TypedSkeleton,
    signature: TypeSignature,
) -> DeductionReport:
    body = hypothesis.hole("body")
    examples = _deduplicate(
        HoleExample(
            hole=body,
            inputs=(TypedValue.freeze(example.input_value, signature.input_type),),
            output=TypedValue.freeze(example.output_value, signature.output_type),
            source_examples=(index,),
        )
        for index, example in enumerate(spec.examples, start=1)
    )
    return DeductionReport(
        hypothesis=hypothesis,
        facts=(
            DeductionFact(
                kind=DeductionFactKind.BODY_EXAMPLES,
                hole=body,
                source_examples=tuple(range(1, len(spec.examples) + 1)),
                derived_examples=len(examples),
            ),
        ),
        hole_examples=examples,
    )


def _map(spec: PBESpec, hypothesis: TypedSkeleton, signature: TypeSignature) -> DeductionReport:
    mapper = hypothesis.hole("mapper")
    length_mismatches: list[tuple[int, int, int]] = []
    raw_examples: list[HoleExample] = []
    for index, example in enumerate(spec.examples, start=1):
        inputs = cast(list[int] | list[bool], example.input_value)
        outputs = cast(list[int] | list[bool], example.output_value)
        if len(inputs) != len(outputs):
            length_mismatches.append((index, len(inputs), len(outputs)))
            continue
        for input_item, output_item in zip(inputs, outputs, strict=True):
            raw_examples.append(
                HoleExample(
                    hole=mapper,
                    inputs=(TypedValue.freeze(input_item, mapper.parameters[0].value_type),),
                    output=TypedValue.freeze(output_item, mapper.output_type),
                    source_examples=(index,),
                )
            )

    if length_mismatches:
        detail = "; ".join(
            f"example {index} has input length {input_length} and output length {output_length}"
            for index, input_length, output_length in length_mismatches
        )
        return DeductionReport(
            hypothesis=hypothesis,
            facts=(),
            hole_examples=(),
            refutation=Refutation(
                kind=RefutationKind.MAP_LENGTH_MISMATCH,
                source_examples=tuple(index for index, _, _ in length_mismatches),
                detail=f"map preserves list length, but {detail}",
            ),
        )

    unmerged = tuple(raw_examples)
    conflict = _conflict(
        unmerged,
        kind=RefutationKind.MAP_FUNCTION_CONFLICT,
        subject="mapper",
    )
    examples = _deduplicate(unmerged)
    facts = [
        DeductionFact(
            kind=DeductionFactKind.MAP_LENGTH_PRESERVED,
            hole=None,
            source_examples=tuple(range(1, len(spec.examples) + 1)),
            derived_examples=0,
        ),
        DeductionFact(
            kind=DeductionFactKind.MAPPER_EXAMPLES,
            hole=mapper,
            source_examples=_source_union(
                *(example.source_examples for example in examples)
            ),
            derived_examples=len(examples),
        ),
    ]
    if not examples:
        facts.append(
            DeductionFact(
                kind=DeductionFactKind.HOLE_UNDERCONSTRAINED,
                hole=mapper,
                source_examples=(),
                derived_examples=0,
            )
        )
    return DeductionReport(
        hypothesis=hypothesis,
        facts=tuple(facts),
        hole_examples=examples,
        refutation=conflict,
    )


def _foldr(spec: PBESpec, hypothesis: TypedSkeleton, signature: TypeSignature) -> DeductionReport:
    initial = hypothesis.hole("initial")
    reducer = hypothesis.hole("reducer")
    observed: dict[TypedValue, tuple[TypedValue, tuple[int, ...]]] = {}
    initial_examples: list[HoleExample] = []

    for index, example in enumerate(spec.examples, start=1):
        input_value = TypedValue.freeze(example.input_value, signature.input_type)
        output = TypedValue.freeze(example.output_value, signature.output_type)
        previous = observed.get(input_value)
        if previous is None:
            observed[input_value] = (output, (index,))
        else:
            observed[input_value] = (output, _source_union(previous[1], (index,)))
        sequence = cast(list[int] | list[bool], example.input_value)
        if not sequence:
            initial_examples.append(
                HoleExample(
                    hole=initial,
                    inputs=(),
                    output=output,
                    source_examples=(index,),
                )
            )

    unmerged_initial = tuple(initial_examples)
    merged_initial = _deduplicate(unmerged_initial)

    reducer_examples: list[HoleExample] = []
    for input_value, (output, sources) in observed.items():
        frozen_sequence = cast(tuple[int, ...] | tuple[bool, ...], input_value.value)
        if not frozen_sequence:
            continue
        suffix = TypedValue(value_type=signature.input_type, value=frozen_sequence[1:])
        suffix_observation = observed.get(suffix)
        if suffix_observation is None:
            continue
        suffix_output, suffix_sources = suffix_observation
        reducer_examples.append(
            HoleExample(
                hole=reducer,
                inputs=(
                    TypedValue.freeze(
                        frozen_sequence[0],
                        reducer.parameters[0].value_type,
                    ),
                    suffix_output,
                ),
                output=output,
                source_examples=_source_union(sources, suffix_sources),
            )
        )

    unmerged_reducer = tuple(reducer_examples)
    reducer_conflict = _conflict(
        unmerged_reducer,
        kind=RefutationKind.FOLDR_REDUCER_CONFLICT,
        subject="fold reducer",
    )
    merged_reducer = _deduplicate(unmerged_reducer)
    facts = [
        DeductionFact(
            kind=DeductionFactKind.FOLDR_INITIAL_EXAMPLES,
            hole=initial,
            source_examples=_source_union(
                *(example.source_examples for example in merged_initial)
            ),
            derived_examples=len(merged_initial),
        ),
        DeductionFact(
            kind=DeductionFactKind.FOLDR_SUFFIX_EXAMPLES,
            hole=reducer,
            source_examples=_source_union(
                *(example.source_examples for example in merged_reducer)
            ),
            derived_examples=len(merged_reducer),
        ),
    ]
    for hole, examples in ((initial, merged_initial), (reducer, merged_reducer)):
        if not examples:
            facts.append(
                DeductionFact(
                    kind=DeductionFactKind.HOLE_UNDERCONSTRAINED,
                    hole=hole,
                    source_examples=(),
                    derived_examples=0,
                )
            )
    return DeductionReport(
        hypothesis=hypothesis,
        facts=tuple(facts),
        hole_examples=(*merged_initial, *merged_reducer),
        refutation=reducer_conflict,
    )


def deduce_hypothesis(spec: PBESpec, hypothesis: TypedSkeleton) -> DeductionReport:
    """Refute one skeleton or derive sound PBE constraints for its holes."""

    signature = _validate_hypothesis(spec, hypothesis)
    inconsistency = _inconsistent_top_level(spec, signature, hypothesis)
    if inconsistency is not None:
        return DeductionReport(
            hypothesis=hypothesis,
            facts=(),
            hole_examples=(),
            refutation=inconsistency,
        )
    if hypothesis.kind is SkeletonKind.EXPRESSION:
        return _expression(spec, hypothesis, signature)
    if hypothesis.kind is SkeletonKind.MAP:
        return _map(spec, hypothesis, signature)
    return _foldr(spec, hypothesis, signature)


def deduce_hypotheses(
    spec: PBESpec,
    hypotheses: Iterable[TypedSkeleton],
) -> tuple[DeductionReport, ...]:
    """Run deduction for several hypotheses without reordering them."""

    return tuple(deduce_hypothesis(spec, hypothesis) for hypothesis in hypotheses)

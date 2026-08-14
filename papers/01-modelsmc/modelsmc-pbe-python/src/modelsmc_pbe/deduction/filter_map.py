"""Deduction for the conditioned right-fold filter/map skeleton."""

from __future__ import annotations

from typing import cast

from modelsmc_pbe.domain.models import PBESpec, TypeSignature, ValueType
from modelsmc_pbe.induction import SkeletonKind, TypedSkeleton

from .evidence import conflicting_examples, deduplicate_examples, source_union
from .records import (
    DeductionFact,
    DeductionFactKind,
    DeductionReport,
    HoleExample,
    Refutation,
    RefutationKind,
    TypedValue,
)


def _shape_refutation(index: int, detail: str) -> Refutation:
    return Refutation(
        kind=RefutationKind.FOLDR_FILTER_MAP_SHAPE_MISMATCH,
        source_examples=(index,),
        detail=detail,
    )


def deduce_foldr_filter_map(
    spec: PBESpec,
    hypothesis: TypedSkeleton,
    signature: TypeSignature,
) -> DeductionReport:
    """Derive local predicate/value examples from observed suffix transitions.

    For a known suffix result ``acc``, this skeleton can only return ``acc``
    (drop the head) or ``mapped_head :: acc`` (retain it).  This makes each
    observed adjacent suffix pair a sound specification for the two holes.
    """

    if (
        signature.input_type is not ValueType.INT_LIST
        or signature.output_type is not ValueType.INT_LIST
    ):
        raise ValueError("foldr-filter-map deduction requires List<Int> -> List<Int>")
    predicate = hypothesis.hole("predicate")
    piecewise = hypothesis.kind is SkeletonKind.FOLD_RIGHT_FILTER_PIECEWISE_MAP
    mapped_value = hypothesis.hole(
        "piecewise_mapped_value" if piecewise else "mapped_value"
    )
    observed: dict[tuple[int, ...], tuple[tuple[int, ...], tuple[int, ...]]] = {}

    for index, example in enumerate(spec.examples, start=1):
        inputs = tuple(cast(list[int], example.input_value))
        outputs = tuple(cast(list[int], example.output_value))
        if len(outputs) > len(inputs):
            return DeductionReport(
                hypothesis=hypothesis,
                facts=(),
                hole_examples=(),
                refutation=_shape_refutation(
                    index,
                    "a filter/map fold emits at most one output for each input item, "
                    f"but input length {len(inputs)} produced output length {len(outputs)}",
                ),
            )
        if not inputs and outputs:
            return DeductionReport(
                hypothesis=hypothesis,
                facts=(),
                hole_examples=(),
                refutation=_shape_refutation(
                    index,
                    "the conditioned fold has fixed empty initial state, but empty input "
                    "produced a nonempty output",
                ),
            )
        previous = observed.get(inputs)
        sources = (index,) if previous is None else source_union(previous[1], (index,))
        observed[inputs] = (outputs, sources)

    raw_predicate: list[HoleExample] = []
    raw_mapped: list[HoleExample] = []
    for inputs, (outputs, sources) in observed.items():
        if not inputs:
            continue
        suffix = inputs[1:]
        if not suffix:
            suffix_outputs: tuple[int, ...] = ()
            suffix_sources: tuple[int, ...] = ()
        else:
            suffix_observation = observed.get(suffix)
            if suffix_observation is None:
                continue
            suffix_outputs, suffix_sources = suffix_observation
        evidence_sources = source_union(sources, suffix_sources)
        item = TypedValue.freeze(inputs[0], ValueType.INT)
        if outputs == suffix_outputs:
            raw_predicate.append(
                HoleExample(
                    hole=predicate,
                    inputs=(item,),
                    output=TypedValue.freeze(False, ValueType.BOOL),
                    source_examples=evidence_sources,
                )
            )
            continue
        if len(outputs) == len(suffix_outputs) + 1 and outputs[1:] == suffix_outputs:
            raw_predicate.append(
                HoleExample(
                    hole=predicate,
                    inputs=(item,),
                    output=TypedValue.freeze(True, ValueType.BOOL),
                    source_examples=evidence_sources,
                )
            )
            raw_mapped.append(
                HoleExample(
                    hole=mapped_value,
                    inputs=(item,),
                    output=TypedValue.freeze(outputs[0], ValueType.INT),
                    source_examples=evidence_sources,
                )
            )
            continue
        return DeductionReport(
            hypothesis=hypothesis,
            facts=(),
            hole_examples=(),
            refutation=Refutation(
                kind=RefutationKind.FOLDR_FILTER_MAP_SHAPE_MISMATCH,
                source_examples=evidence_sources,
                detail=(
                    f"for head {inputs[0]}, the observed result is neither the suffix "
                    "result nor one value prepended to the suffix result"
                ),
            ),
        )

    predicate_unmerged = tuple(raw_predicate)
    mapped_unmerged = tuple(raw_mapped)
    conflict = conflicting_examples(
        predicate_unmerged,
        kind=RefutationKind.FOLDR_FILTER_PREDICATE_CONFLICT,
        subject="filter predicate",
    )
    if conflict is None:
        conflict = conflicting_examples(
            mapped_unmerged,
            kind=RefutationKind.FOLDR_FILTER_MAPPED_VALUE_CONFLICT,
            subject="mapped-value function",
        )
    predicate_examples = deduplicate_examples(predicate_unmerged)
    mapped_examples = deduplicate_examples(mapped_unmerged)
    facts = [
        DeductionFact(
            kind=DeductionFactKind.FOLDR_FILTER_PREDICATE_EXAMPLES,
            hole=predicate,
            source_examples=source_union(
                *(example.source_examples for example in predicate_examples)
            ),
            derived_examples=len(predicate_examples),
        ),
        DeductionFact(
            kind=(
                DeductionFactKind.FOLDR_FILTER_PIECEWISE_MAPPED_VALUE_EXAMPLES
                if piecewise
                else DeductionFactKind.FOLDR_FILTER_MAPPED_VALUE_EXAMPLES
            ),
            hole=mapped_value,
            source_examples=source_union(*(example.source_examples for example in mapped_examples)),
            derived_examples=len(mapped_examples),
        ),
    ]
    for hole, examples in (
        (predicate, predicate_examples),
        (mapped_value, mapped_examples),
    ):
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
        hole_examples=(*predicate_examples, *mapped_examples),
        refutation=conflict,
    )

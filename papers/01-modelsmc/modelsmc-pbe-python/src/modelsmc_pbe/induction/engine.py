"""Signature-driven hypothesis generation and structural inspection."""

from __future__ import annotations

from typing import cast

from modelsmc_pbe.domain.models import PBESpec, TypeSignature

from .records import (
    HoleSpec,
    InductionReport,
    SkeletonKind,
    StructuralFact,
    StructuralRelation,
    TypedSkeleton,
    TypedVariable,
)
from .signature import list_element_type


def _required_signature(spec: PBESpec) -> TypeSignature:
    signature = spec.signature
    if signature is None:  # pragma: no cover - PBESpec's validator always supplies it
        raise ValueError("PBE specification has no inferred or explicit signature")
    return signature


def _input_name(signature: TypeSignature) -> str:
    return "xs" if list_element_type(signature.input_type) is not None else "x"


def _expression_hypothesis(signature: TypeSignature) -> TypedSkeleton:
    input_variable = TypedVariable(_input_name(signature), signature.input_type)
    return TypedSkeleton(
        kind=SkeletonKind.EXPRESSION,
        input_variable=input_variable,
        output_type=signature.output_type,
        holes=(
            HoleSpec(
                name="body",
                parameters=(input_variable,),
                output_type=signature.output_type,
            ),
        ),
    )


def _map_hypothesis(signature: TypeSignature) -> TypedSkeleton | None:
    input_item = list_element_type(signature.input_type)
    output_item = list_element_type(signature.output_type)
    if input_item is None or output_item is None:
        return None
    return TypedSkeleton(
        kind=SkeletonKind.MAP,
        input_variable=TypedVariable("xs", signature.input_type),
        output_type=signature.output_type,
        holes=(
            HoleSpec(
                name="mapper",
                parameters=(TypedVariable("item", input_item),),
                output_type=output_item,
            ),
        ),
    )


def _foldr_hypothesis(signature: TypeSignature) -> TypedSkeleton | None:
    input_item = list_element_type(signature.input_type)
    if input_item is None:
        return None
    return TypedSkeleton(
        kind=SkeletonKind.FOLD_RIGHT,
        input_variable=TypedVariable("xs", signature.input_type),
        output_type=signature.output_type,
        holes=(
            HoleSpec(name="initial", parameters=(), output_type=signature.output_type),
            HoleSpec(
                name="reducer",
                parameters=(
                    TypedVariable("item", input_item),
                    TypedVariable("acc", signature.output_type),
                ),
                output_type=signature.output_type,
            ),
        ),
    )


def _structural_facts(spec: PBESpec, signature: TypeSignature) -> tuple[StructuralFact, ...]:
    if (
        list_element_type(signature.input_type) is None
        or list_element_type(signature.output_type) is None
    ):
        return ()

    supporting_lengths: list[int] = []
    contradicting_lengths: list[int] = []
    empty_support: list[int] = []
    empty_contradictions: list[int] = []
    for index, example in enumerate(spec.examples, start=1):
        input_value = cast(list[int] | list[bool], example.input_value)
        output_value = cast(list[int] | list[bool], example.output_value)
        destination = (
            supporting_lengths
            if len(input_value) == len(output_value)
            else contradicting_lengths
        )
        destination.append(index)
        if not input_value:
            destination = empty_support if not output_value else empty_contradictions
            destination.append(index)

    facts = [
        StructuralFact(
            relation=StructuralRelation.LENGTH_PRESERVED,
            holds=not contradicting_lengths,
            supporting_examples=tuple(supporting_lengths),
            contradicting_examples=tuple(contradicting_lengths),
        )
    ]
    if empty_support or empty_contradictions:
        facts.append(
            StructuralFact(
                relation=StructuralRelation.EMPTY_INPUT_PRODUCES_EMPTY_OUTPUT,
                holds=not empty_contradictions,
                supporting_examples=tuple(empty_support),
                contradicting_examples=tuple(empty_contradictions),
            )
        )
    return tuple(facts)


def induce_hypotheses(spec: PBESpec) -> InductionReport:
    """Generate all wrapper forms allowed by the inferred first-order signature.

    Generation is intentionally signature-based.  Observed relationships are
    recorded as facts, but deduction—not induction—decides whether a generated
    skeleton is inconsistent with the examples.
    """

    signature = _required_signature(spec)
    candidates = (
        _expression_hypothesis(signature),
        _map_hypothesis(signature),
        _foldr_hypothesis(signature),
    )
    return InductionReport(
        signature=signature,
        facts=_structural_facts(spec, signature),
        hypotheses=tuple(candidate for candidate in candidates if candidate is not None),
    )

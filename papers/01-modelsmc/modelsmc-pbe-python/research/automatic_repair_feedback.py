"""Derive conservative filter/map repair evidence from interpreter traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast

from modelsmc_pbe.core.evaluate import evaluate_expression
from modelsmc_pbe.core.types import Node
from modelsmc_pbe.domain.models import PBESpec


@dataclass(frozen=True, slots=True)
class MapperConstraint:
    """A mapped-value constraint conditional on freezing the current predicate."""

    item: int
    expected_value: int
    source_examples: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PredicateToggle:
    """A one-item counterfactual change that exactly repairs one or more examples."""

    item: int
    keep: bool
    source_examples: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExampleTrace:
    """Observable execution result for one training example."""

    source_example: int
    expected: tuple[int, ...]
    actual: tuple[int, ...]
    exact: bool


@dataclass(frozen=True, slots=True)
class AutomaticRepairFeedback:
    """Conservative evidence for repairing one filter/map candidate."""

    examples: tuple[ExampleTrace, ...]
    mapper_constraints: tuple[MapperConstraint, ...]
    predicate_toggles: tuple[PredicateToggle, ...]

    @property
    def zero_loss(self) -> bool:
        return all(example.exact for example in self.examples)

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self)) | {"zero_loss": self.zero_loss}


def _evaluate_components(
    inputs: list[int],
    predicate: Node,
    mapper: Node,
) -> tuple[list[bool], list[int]]:
    decisions: list[bool] = []
    mapped: list[int] = []
    for item in inputs:
        decision = evaluate_expression(predicate, inputs, item=item)
        value = evaluate_expression(mapper, inputs, item=item)
        if not isinstance(decision, bool):
            raise TypeError("predicate did not evaluate to Bool")
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("mapper did not evaluate to Int")
        decisions.append(decision)
        mapped.append(value)
    return decisions, mapped


def derive_automatic_feedback(
    spec: PBESpec,
    predicate: Node,
    mapper: Node,
) -> AutomaticRepairFeedback:
    """Return evidence that does not assume a hidden program or output alignment.

    Conditional on freezing the current predicate, mapper constraints are emitted
    only when retained-item count equals expected output count, making the
    order-preserving positional alignment exact for that subproblem. Predicate
    evidence is emitted only for a single semantic item-decision toggle whose
    concrete counterfactual execution exactly matches the complete expected output.
    """

    traces: list[ExampleTrace] = []
    raw_mapper: dict[tuple[int, int], list[int]] = {}
    raw_toggles: dict[tuple[int, bool], list[int]] = {}

    for source, example in enumerate(spec.examples, start=1):
        inputs = cast(list[int], example.input_value)
        expected = cast(list[int], example.output_value)
        decisions, mapped = _evaluate_components(inputs, predicate, mapper)
        actual = [value for value, keep in zip(mapped, decisions, strict=True) if keep]
        traces.append(
            ExampleTrace(
                source_example=source,
                expected=tuple(expected),
                actual=tuple(actual),
                exact=actual == expected,
            )
        )

        retained_items = [item for item, keep in zip(inputs, decisions, strict=True) if keep]
        if len(retained_items) == len(expected):
            for item, expected_value in zip(retained_items, expected, strict=True):
                raw_mapper.setdefault((item, expected_value), []).append(source)

        if actual == expected:
            continue
        for item in dict.fromkeys(inputs):
            positions = [position for position, value in enumerate(inputs) if value == item]
            item_decisions = {decisions[position] for position in positions}
            if len(item_decisions) != 1:
                raise ValueError("a deterministic item predicate changed across equal inputs")
            revised_decision = not item_decisions.pop()
            counterfactual = list(decisions)
            for position in positions:
                counterfactual[position] = revised_decision
            revised = [value for value, keep in zip(mapped, counterfactual, strict=True) if keep]
            if revised == expected:
                raw_toggles.setdefault((item, revised_decision), []).append(source)

    mapper_constraints = tuple(
        MapperConstraint(
            item=item,
            expected_value=value,
            source_examples=tuple(dict.fromkeys(sources)),
        )
        for (item, value), sources in sorted(raw_mapper.items())
    )
    predicate_toggles = tuple(
        PredicateToggle(
            item=item,
            keep=keep,
            source_examples=tuple(dict.fromkeys(sources)),
        )
        for (item, keep), sources in sorted(raw_toggles.items())
    )
    return AutomaticRepairFeedback(
        examples=tuple(traces),
        mapper_constraints=mapper_constraints,
        predicate_toggles=predicate_toggles,
    )

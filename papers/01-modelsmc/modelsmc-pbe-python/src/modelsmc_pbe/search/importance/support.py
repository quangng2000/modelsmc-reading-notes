"""Build the fixed typed support and its unique program-construction trie."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any

from modelsmc_pbe.config import SMCConfig
from modelsmc_pbe.core import ProgramScorer, RejectedProgram, ScoredProgram
from modelsmc_pbe.deduction import (
    DeductionReport,
    deduce_hypotheses,
    render_deduction_trace,
)
from modelsmc_pbe.domain import PBESpec, ProgramAst, canonical_key, clone_program
from modelsmc_pbe.enumeration import HoleEnumerationOptions, enumerate_hole_expressions
from modelsmc_pbe.grammar import arithmetic_expressions
from modelsmc_pbe.induction import (
    InductionReport,
    SkeletonKind,
    assemble_program,
    induce_hypotheses,
    render_induction_trace,
)

from .conditioning import (
    condition_induction,
    conditioned_hole_catalog,
)
from .proposal_distribution import deduction_mismatch_counts
from .records import (
    EmptyImportanceSupportError,
    FamilySupport,
    HoleCatalogSummary,
    HoleFilling,
    ImportanceSMCOptions,
    ImportanceState,
    ImportanceSupport,
    ImportanceSupportLimitExceeded,
)

type EventEmitter = Callable[..., None]


@dataclass(frozen=True, slots=True)
class _PendingState:
    hypothesis_index: int
    family: str
    fillings: tuple[HoleFilling, ...]
    program: ProgramAst
    key: str


def _emit(emit: EventEmitter | None, name: str, **data: Any) -> None:
    if emit is not None:
        emit(name, **data)


class ImportanceSupportBuilder:
    """Enumerate every bounded construction before any stochastic proposal."""

    def __init__(
        self,
        *,
        spec: PBESpec,
        smc: SMCConfig,
        options: ImportanceSMCOptions,
        scorer: ProgramScorer,
        emit: EventEmitter | None = None,
    ) -> None:
        self._spec = spec
        self._smc = smc
        self._options = options
        self._scorer = scorer
        self._emit = emit

    def build(self) -> ImportanceSupport:
        """Return fixed support or fail rather than truncate or alias it."""

        induction = condition_induction(
            induce_hypotheses(self._spec),
            self._options.conditioned_skeleton,
            multi_family=self._options.multi_family,
        )
        deductions = deduce_hypotheses(self._spec, induction.hypotheses)
        self._emit_symbolic_traces(induction, deductions)
        excluded_hypotheses = self._auto_specialized_exclusions(deductions)

        pending: list[_PendingState] = []
        summaries: list[HoleCatalogSummary] = []
        seen_programs: dict[str, tuple[int, tuple[HoleFilling, ...]]] = {}
        aliased_programs = 0
        attempted_traces = 0
        for hypothesis_index, report in enumerate(deductions):
            if not report.viable or hypothesis_index in excluded_hypotheses:
                continue
            catalogs: list[tuple[HoleFilling, ...]] = []
            for hole in report.hypothesis.holes:
                conditioned = conditioned_hole_catalog(
                    self._options.conditioned_skeleton,
                    report.hypothesis,
                    hole,
                    tuple(self._spec.integer_constants),
                    multi_family=self._options.multi_family,
                    state_limit=self._options.hole_state_limit,
                )
                if conditioned is None:
                    catalog = enumerate_hole_expressions(
                        hole,
                        self._spec.integer_constants,
                        HoleEnumerationOptions(
                            max_cost=self._options.hole_max_cost,
                            state_limit=self._options.hole_state_limit,
                        ),
                    )
                    expressions = tuple(
                        (entry.expression, entry.canonical_key) for entry in catalog.entries
                    )
                    generated_states = catalog.generated_states
                    catalog_max_cost = self._options.hole_max_cost
                    catalog_source = "typed-cost-enumerator"
                else:
                    expressions = tuple(
                        (expression, canonical_key(expression))
                        for expression in conditioned.expressions
                    )
                    generated_states = conditioned.generated_states
                    catalog_max_cost = conditioned.max_cost
                    catalog_source = "conditioned-skeleton"
                fillings = tuple(
                    HoleFilling(
                        hole_name=hole.name,
                        expression=clone_program(expression),
                        key=key,
                    )
                    for expression, key in expressions
                )
                if not fillings:
                    break
                catalogs.append(fillings)
                summaries.append(
                    HoleCatalogSummary(
                        family=report.hypothesis.kind.value,
                        hole_name=hole.name,
                        returned_expressions=len(fillings),
                        generated_states=generated_states,
                        max_cost=catalog_max_cost,
                    )
                )
                _emit(
                    self._emit,
                    "importance.hole.enumerated",
                    message="complete bounded hole catalog enumerated",
                    level="debug",
                    family=report.hypothesis.kind.value,
                    hole=hole.name,
                    expressions=len(fillings),
                    generated_states=generated_states,
                    max_cost=catalog_max_cost,
                    catalog_source=catalog_source,
                )
            if len(catalogs) != len(report.hypothesis.holes):
                continue
            family_constructions = math.prod(len(catalog) for catalog in catalogs)
            if attempted_traces + family_constructions > self._options.support_limit:
                raise ImportanceSupportLimitExceeded(
                    "complete typed construction support exceeds "
                    f"{self._options.support_limit} programs; lower --hole-max-cost "
                    "or raise --support-limit"
                )
            for combination in product(*catalogs):
                attempted_traces += 1
                fillings_by_name = {
                    filling.hole_name: filling.expression for filling in combination
                }
                program = assemble_program(
                    report.hypothesis,
                    fillings_by_name,
                    allowed_integer_constants=self._spec.integer_constants,
                )
                key = canonical_key(program)
                previous = seen_programs.get(key)
                if previous is not None:
                    previous_kind = deductions[previous[0]].hypothesis.kind
                    if (
                        self._options.multi_family
                        and previous_kind
                        in {
                            SkeletonKind.FOLD_RIGHT_FILTER_MAP,
                            SkeletonKind.FOLD_RIGHT_FILTER_PIECEWISE_MAP,
                        }
                        and report.hypothesis.kind is SkeletonKind.FOLD_RIGHT
                    ):
                        aliased_programs += 1
                        continue
                    raise RuntimeError(
                        "two construction traces produce the same canonical program; "
                        f"first={previous[0]}:{previous[1]!r}, "
                        f"second={hypothesis_index}:{combination!r}"
                    )
                seen_programs[key] = (hypothesis_index, combination)
                pending.append(
                    _PendingState(
                        hypothesis_index=hypothesis_index,
                        family=report.hypothesis.kind.value,
                        fillings=combination,
                        program=program,
                        key=key,
                    )
                )

        states, rejected = self._score(pending)
        if not states:
            raise EmptyImportanceSupportError(
                "no complete program survived the declared hole, cost, depth, and node bounds"
            )
        family_indices: dict[int, list[int]] = {}
        for state in states:
            family_indices.setdefault(state.hypothesis_index, []).append(state.state_index)
        families = tuple(
            FamilySupport(
                hypothesis_index=index,
                hypothesis=deductions[index].hypothesis,
                deduction=deductions[index],
                state_indices=tuple(family_indices[index]),
            )
            for index in sorted(family_indices)
        )
        support = ImportanceSupport(
            induction=induction,
            deductions=deductions,
            families=families,
            states=states,
            hole_catalogs=tuple(summaries),
            constructed_programs=attempted_traces,
            rejected_programs=rejected,
            conditioned_skeleton=self._options.conditioned_skeleton,
            multi_family=self._options.multi_family,
            aliased_programs=aliased_programs,
        )
        _emit(
            self._emit,
            "importance.support.ready",
            message="finite typed proposal and target support is fixed",
            support_states=len(states),
            exact_programs=support.exact_programs,
            viable_hypotheses=len(families),
            refuted_hypotheses=sum(not report.viable for report in deductions),
            rejected_programs=rejected,
            aliased_programs=aliased_programs,
            multi_family=self._options.multi_family,
        )
        return support

    def _auto_specialized_exclusions(
        self,
        deductions: tuple[DeductionReport, ...],
    ) -> frozenset[int]:
        """Choose the smallest adequate specialized catalog in auto mode.

        The abstract filter/map and piecewise filter/map skeletons have the same
        type-level shape.  Their *bounded catalogs* differ: the first admits
        simple arithmetic mappings, while the second admits genuine signed
        conditionals.  When sound mapped-value examples are available, retain
        the simple catalog exactly when one of its expressions satisfies every
        derived example; otherwise retain the piecewise refinement.  This is a
        declared support-selection policy, not a logical refutation of the
        abstract higher-order skeleton.
        """

        if not self._options.multi_family:
            return frozenset()
        by_kind = {
            report.hypothesis.kind: (index, report)
            for index, report in enumerate(deductions)
        }
        simple = by_kind.get(SkeletonKind.FOLD_RIGHT_FILTER_MAP)
        piecewise = by_kind.get(SkeletonKind.FOLD_RIGHT_FILTER_PIECEWISE_MAP)
        if simple is None or piecewise is None:
            return frozenset()
        simple_index, simple_report = simple
        piecewise_index, piecewise_report = piecewise
        if not simple_report.viable or not piecewise_report.viable:
            return frozenset()

        constants = tuple(self._spec.integer_constants)
        mapped_examples = simple_report.examples_for("mapped_value")
        arithmetic_is_adequate = False
        reason: str
        if not mapped_examples:
            arithmetic_is_adequate = True
            reason = "mapped-value evidence is underconstrained; prefer the smaller catalog"
        else:
            arithmetic = arithmetic_expressions("Item", constants)
            mismatches = deduction_mismatch_counts(arithmetic, mapped_examples)
            arithmetic_is_adequate = any(mismatch == 0 for mismatch in mismatches)
            reason = (
                "a compact arithmetic mapping satisfies every derived mapped-value example"
                if arithmetic_is_adequate
                else "no compact arithmetic mapping satisfies all derived mapped-value examples"
            )
        if 0 not in constants:
            arithmetic_is_adequate = True
            reason = "the signed piecewise catalog requires declared constant 0"

        excluded = piecewise_index if arithmetic_is_adequate else simple_index
        retained = simple_index if arithmetic_is_adequate else piecewise_index
        _emit(
            self._emit,
            "importance.family.catalog_selected",
            message="auto mode selected the smallest adequate bounded mapping family",
            level="info",
            retained_family=deductions[retained].hypothesis.kind.value,
            excluded_family=deductions[excluded].hypothesis.kind.value,
            reason=reason,
        )
        return frozenset((excluded,))

    def _score(self, pending: list[_PendingState]) -> tuple[tuple[ImportanceState, ...], int]:
        states: list[ImportanceState] = []
        rejected = 0
        size = self._options.score_batch_size
        for start in range(0, len(pending), size):
            batch = pending[start : start + size]
            results = self._scorer.score_batch([item.program for item in batch])
            for item, result in zip(batch, results, strict=True):
                if isinstance(result, RejectedProgram):
                    rejected += 1
                    continue
                if not isinstance(result, ScoredProgram):  # pragma: no cover
                    raise TypeError("semantic scorer returned an unknown record")
                states.append(
                    ImportanceState(
                        state_index=len(states),
                        hypothesis_index=item.hypothesis_index,
                        family=item.family,
                        fillings=item.fillings,
                        program=clone_program(item.program),
                        key=item.key,
                        score=result,
                    )
                )
        return tuple(states), rejected

    def _emit_symbolic_traces(
        self,
        induction: InductionReport,
        deductions: tuple[DeductionReport, ...],
    ) -> None:
        if self._emit is None:
            return
        for line in render_induction_trace(induction):
            _emit(
                self._emit,
                "importance.induction.trace",
                message=line,
                level="trace",
            )
        for report in deductions:
            for line in render_deduction_trace(report):
                _emit(
                    self._emit,
                    "importance.deduction.trace",
                    message=line,
                    level="trace",
                )

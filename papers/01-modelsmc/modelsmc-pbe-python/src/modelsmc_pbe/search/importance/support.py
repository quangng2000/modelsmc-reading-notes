"""Build the fixed typed support and its unique program-construction trie."""

from __future__ import annotations

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
from modelsmc_pbe.induction import (
    InductionReport,
    assemble_program,
    induce_hypotheses,
    render_induction_trace,
)

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

        induction = induce_hypotheses(self._spec)
        deductions = deduce_hypotheses(self._spec, induction.hypotheses)
        self._emit_symbolic_traces(induction, deductions)

        pending: list[_PendingState] = []
        summaries: list[HoleCatalogSummary] = []
        seen_programs: dict[str, tuple[int, tuple[HoleFilling, ...]]] = {}
        for hypothesis_index, report in enumerate(deductions):
            if not report.viable:
                continue
            catalogs: list[tuple[HoleFilling, ...]] = []
            for hole in report.hypothesis.holes:
                catalog = enumerate_hole_expressions(
                    hole,
                    self._spec.integer_constants,
                    HoleEnumerationOptions(
                        max_cost=self._options.hole_max_cost,
                        state_limit=self._options.hole_state_limit,
                    ),
                )
                fillings = tuple(
                    HoleFilling(
                        hole_name=hole.name,
                        expression=clone_program(entry.expression),
                        key=entry.canonical_key,
                    )
                    for entry in catalog.entries
                )
                if not fillings:
                    break
                catalogs.append(fillings)
                summaries.append(
                    HoleCatalogSummary(
                        family=report.hypothesis.kind.value,
                        hole_name=hole.name,
                        returned_expressions=len(fillings),
                        generated_states=catalog.generated_states,
                        max_cost=self._options.hole_max_cost,
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
                    generated_states=catalog.generated_states,
                    max_cost=self._options.hole_max_cost,
                )
            if len(catalogs) != len(report.hypothesis.holes):
                continue
            for combination in product(*catalogs):
                attempted = len(pending) + 1
                if attempted > self._options.support_limit:
                    raise ImportanceSupportLimitExceeded(
                        "complete typed construction support exceeds "
                        f"{self._options.support_limit} programs; lower --hole-max-cost "
                        "or raise --support-limit"
                    )
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
            constructed_programs=len(pending),
            rejected_programs=rejected,
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
        )
        return support

    def _score(
        self, pending: list[_PendingState]
    ) -> tuple[tuple[ImportanceState, ...], int]:
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

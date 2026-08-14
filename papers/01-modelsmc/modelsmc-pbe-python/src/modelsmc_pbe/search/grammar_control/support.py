"""Construction of the complete scorer-approved grammar support."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Protocol

from modelsmc_pbe.config import SMCConfig
from modelsmc_pbe.core import ProgramScorer, RejectedProgram, ScoredProgram
from modelsmc_pbe.domain.ast import canonical_key, clone_program
from modelsmc_pbe.domain.models import PBESpec
from modelsmc_pbe.grammar import enumerate_skeleton
from modelsmc_pbe.search.grammar_control.records import (
    EmptyGrammarSupportError,
    GrammarSMCOptions,
    GrammarState,
    GrammarSupport,
)


class EventEmitter(Protocol):
    """Structural event sink used without coupling support logic to a logger."""

    def __call__(
        self,
        name: str,
        *,
        message: str,
        level: str = "info",
        **data: Any,
    ) -> None: ...


class ScoredSupportBuilder:
    """Enumerate a skeleton and retain only states accepted by the core."""

    def __init__(
        self,
        *,
        spec: PBESpec,
        smc: SMCConfig,
        options: GrammarSMCOptions,
        scorer: ProgramScorer,
        emit: EventEmitter,
    ) -> None:
        self._spec = spec
        self._smc = smc
        self._options = options
        self._scorer = scorer
        self._emit = emit

    def build(self) -> GrammarSupport:
        """Build the finite support, preserving deterministic catalog order."""

        programs = enumerate_skeleton(
            self._options.skeleton,
            self._spec.integer_constants,
            self._options.state_limit,
        )
        self._emit(
            "grammar.enumerated",
            message="complete finite skeleton enumerated",
            skeleton=self._options.skeleton,
            enumerated_asts=len(programs),
            state_limit=self._options.state_limit,
        )

        states: list[GrammarState] = []
        rejection_reasons: Counter[str] = Counter()
        rejected = 0
        over_cost = 0
        batch_count = math.ceil(len(programs) / self._options.score_batch_size)
        for batch_index, start in enumerate(
            range(0, len(programs), self._options.score_batch_size),
            start=1,
        ):
            batch = programs[start : start + self._options.score_batch_size]
            self._emit_batch_started(batch_index, batch_count, len(batch))
            results = self._scorer.score_batch(batch)
            accepted_delta = rejected_delta = over_cost_delta = 0
            for program, result in zip(batch, results, strict=True):
                if isinstance(result, RejectedProgram):
                    if result.cost is not None and result.cost > self._smc.max_cost:
                        over_cost += 1
                        over_cost_delta += 1
                    else:
                        rejected += 1
                        rejected_delta += 1
                    rejection_reasons[result.reason] += 1
                    continue
                self._validate_score(result)
                if result.cost > self._smc.max_cost:
                    over_cost += 1
                    over_cost_delta += 1
                    continue
                states.append(
                    GrammarState(
                        program=clone_program(program),
                        score=result,
                        target_loss=result.total_loss,
                        key=canonical_key(program),
                    )
                )
                accepted_delta += 1
            self._emit(
                "grammar.score_batch.completed",
                message="semantic scoring batch completed",
                level="debug",
                batch=batch_index,
                batches=batch_count,
                accepted=accepted_delta,
                rejected=rejected_delta,
                over_cost=over_cost_delta,
            )

        if not states:
            self._raise_empty(rejection_reasons)
        support = GrammarSupport(
            states=tuple(states),
            enumerated_asts=len(programs),
            rejected_asts=rejected,
            over_cost_asts=over_cost,
        )
        self._emit_ready(support, rejection_reasons)
        return support

    def _emit_batch_started(self, index: int, count: int, size: int) -> None:
        self._emit(
            "grammar.score_batch.started",
            message="semantic scoring batch started",
            level="debug",
            batch=index,
            batches=count,
            size=size,
        )

    @staticmethod
    def _validate_score(result: ScoredProgram) -> None:
        if not math.isfinite(result.total_loss) or result.total_loss < 0.0:
            raise ValueError("semantic scorer returned a non-finite or negative loss")
        if result.cost < 0:
            raise ValueError("semantic scorer returned a negative cost")

    def _raise_empty(self, rejection_reasons: Counter[str]) -> None:
        details = "; ".join(
            f"{count}x {reason}" for reason, count in rejection_reasons.most_common(3)
        )
        suffix = f"; rejection summary: {details}" if details else ""
        raise EmptyGrammarSupportError(
            f"skeleton {self._options.skeleton!r} has no accepted states at "
            f"max_cost={self._smc.max_cost}{suffix}"
        )

    def _emit_ready(
        self,
        support: GrammarSupport,
        rejection_reasons: Counter[str],
    ) -> None:
        self._emit(
            "grammar.support.ready",
            message="bounded grammar support is ready",
            grammar_states=len(support.states),
            exact_programs=support.exact_programs,
            rejected_asts=support.rejected_asts,
            over_cost_asts=support.over_cost_asts,
            max_cost=self._smc.max_cost,
            rejection_reasons=dict(rejection_reasons.most_common(10)),
        )

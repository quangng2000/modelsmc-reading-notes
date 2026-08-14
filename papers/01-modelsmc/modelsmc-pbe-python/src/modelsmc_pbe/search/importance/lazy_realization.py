"""On-demand semantic execution for factorized construction traces."""

from __future__ import annotations

from collections.abc import MutableMapping

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.core import ProgramScorer, RejectedProgram, ScoredProgram
from modelsmc_pbe.domain import canonical_key
from modelsmc_pbe.induction import assemble_program

from .factorized import trace_fillings, trace_hole_cost, trace_log_prior
from .lazy_records import (
    ConstructionTrace,
    FactorizedImportanceSupport,
    LazyImportanceState,
)
from .records import ImportanceSMCOptions


def realize_traces(
    *,
    config: ExperimentConfig,
    options: ImportanceSMCOptions,
    scorer: ProgramScorer,
    support: FactorizedImportanceSupport,
    traces: tuple[ConstructionTrace, ...],
    states: MutableMapping[ConstructionTrace, LazyImportanceState],
) -> tuple[tuple[LazyImportanceState, ...], int]:
    """Assemble and execute only traces absent from the visited-state cache."""

    missing = tuple(dict.fromkeys(trace for trace in traces if trace not in states))
    programs = []
    metadata = []
    for trace in missing:
        family = support.family(trace.hypothesis_index)
        fillings = trace_fillings(family, trace)
        program = assemble_program(
            family.hypothesis,
            {filling.hole_name: filling.expression for filling in fillings},
            allowed_integer_constants=config.spec.integer_constants,
        )
        programs.append(program)
        metadata.append((trace, family, fillings, program))
    results: list[ScoredProgram | RejectedProgram] = []
    for start in range(0, len(programs), options.score_batch_size):
        results.extend(scorer.score_batch(programs[start : start + options.score_batch_size]))
    for (trace, family, fillings, program), score in zip(metadata, results, strict=True):
        if isinstance(score, RejectedProgram):
            raise RuntimeError(
                "factorized support admitted a sampled program rejected by the semantic "
                f"core: {score.reason}"
            )
        expected_cost = family.base_cost + trace_hole_cost(family, trace)
        if score.cost != expected_cost:
            raise RuntimeError(
                "factorized structural cost disagrees with semantic scorer: "
                f"factorized={expected_cost}, scorer={score.cost}"
            )
        states[trace] = LazyImportanceState(
            trace=trace,
            family=family.hypothesis.kind.value,
            fillings=fillings,
            program=program,
            key=canonical_key(program),
            score=score,
            log_prior=trace_log_prior(
                support,
                trace,
                cost_scale=float(config.smc.cost_scale),
            ),
        )
    return tuple(states[trace] for trace in traces), len(missing)

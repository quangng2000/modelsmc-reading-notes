"""Execution-free preparation of complete programs for semantic scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.domain import canonical_key
from modelsmc_pbe.induction import assemble_program
from modelsmc_pbe.proposals.labels import CompatibilityProgram

from ..factorized import trace_fillings
from ..lazy_records import ConstructionTrace, FactorizedImportanceSupport


@dataclass(frozen=True, slots=True)
class PreparedSemanticPrograms:
    """Unique canonical programs plus the key owned by every slate trace."""

    traces: tuple[ConstructionTrace, ...]
    trace_program_keys: tuple[str, ...]
    programs: tuple[CompatibilityProgram, ...]

    def __post_init__(self) -> None:
        if not self.traces or len(self.traces) != len(self.trace_program_keys):
            raise ValueError("semantic traces and program keys must be nonempty and aligned")
        if len({program.program_key for program in self.programs}) != len(self.programs):
            raise ValueError("prepared semantic programs must be unique")


def dataset_context(config: ExperimentConfig) -> str:
    """Render the frozen PBE task without any candidate execution outcome."""

    return json.dumps(
        config.spec.model_dump(mode="json", by_alias=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def prepare_semantic_programs(
    *,
    config: ExperimentConfig,
    support: FactorizedImportanceSupport,
    traces: tuple[ConstructionTrace, ...],
) -> PreparedSemanticPrograms:
    """Assemble slate ASTs for prompting without executing their examples."""

    if not traces or len(set(traces)) != len(traces):
        raise ValueError("semantic slate traces must be nonempty and unique")
    trace_program_keys: list[str] = []
    unique: dict[str, CompatibilityProgram] = {}
    for trace in traces:
        family = support.family(trace.hypothesis_index)
        fillings = trace_fillings(family, trace)
        program = assemble_program(
            family.hypothesis,
            {filling.hole_name: filling.expression for filling in fillings},
            allowed_integer_constants=config.spec.integer_constants,
        )
        key = canonical_key(program)
        trace_program_keys.append(key)
        unique.setdefault(key, CompatibilityProgram(program_key=key, program_text=key))
    programs = tuple(unique[key] for key in sorted(unique))
    return PreparedSemanticPrograms(traces, tuple(trace_program_keys), programs)

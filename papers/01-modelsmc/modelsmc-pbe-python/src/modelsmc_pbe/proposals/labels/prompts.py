"""Versioned prompts for symmetrized binary compatibility scoring."""

from __future__ import annotations

from modelsmc_pbe.proposals.labels.contracts import (
    CompatibilityLabel,
    CompatibilityMapping,
    CompatibilityPathSpec,
    CompatibilityProgram,
)

COMPATIBILITY_TEMPLATE_VERSION = "joint-semantic-compatibility-v2"

_INSTRUCTIONS = """You are judging semantic program compatibility.
Given the task and input/output examples, decide whether the candidate program can be
consistent with those examples. Judge program meaning, not prose fluency. The answer-label
mapping is supplied beside each candidate. The final answer is exactly one mapped label.

Task and examples:
{dataset_context}
"""


def build_compatibility_prompt(dataset_context: str) -> str:
    """Build the single shared prefix used for every program and label path."""

    if not dataset_context.strip():
        raise ValueError("dataset_context must not be empty")
    return _INSTRUCTIONS.format(dataset_context=dataset_context.rstrip())


def build_program_paths(program: CompatibilityProgram) -> tuple[CompatibilityPathSpec, ...]:
    """Render the four swapped-mapping paths in their canonical audit order."""

    return (
        _path(program, CompatibilityMapping.A_IS_COMPATIBLE, CompatibilityLabel.A),
        _path(program, CompatibilityMapping.A_IS_COMPATIBLE, CompatibilityLabel.B),
        _path(program, CompatibilityMapping.B_IS_COMPATIBLE, CompatibilityLabel.A),
        _path(program, CompatibilityMapping.B_IS_COMPATIBLE, CompatibilityLabel.B),
    )


def _path(
    program: CompatibilityProgram,
    mapping: CompatibilityMapping,
    label: CompatibilityLabel,
) -> CompatibilityPathSpec:
    mapping_text = (
        "A = compatible; B = incompatible"
        if mapping is CompatibilityMapping.A_IS_COMPATIBLE
        else "A = incompatible; B = compatible"
    )
    candidate = (
        "\n\nCandidate program:\n"
        f"{program.program_text}\n"
        f"Answer-label mapping: {mapping_text}\n"
        f"Answer: {label.value}"
    )
    return CompatibilityPathSpec(mapping=mapping, label=label, candidate=candidate)

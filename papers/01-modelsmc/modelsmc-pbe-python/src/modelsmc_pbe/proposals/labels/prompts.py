"""Versioned prompts for symmetrized binary compatibility scoring."""

from __future__ import annotations

from enum import StrEnum

from modelsmc_pbe.proposals.labels.contracts import (
    CompatibilityLabel,
    CompatibilityMapping,
    CompatibilityPathSpec,
    CompatibilityProgram,
)

COMPATIBILITY_TEMPLATE_VERSION = "joint-semantic-compatibility-v2"
HARMONY_GPT_OSS_TEMPLATE_VERSION = "joint-semantic-compatibility-harmony-gpt-oss-v1"


class SemanticPromptProtocol(StrEnum):
    """Versioned wire format for teacher-forced semantic label paths."""

    RAW_V2 = "raw-v2"
    HARMONY_GPT_OSS_V1 = "harmony-gpt-oss-v1"


def compatibility_template_version(protocol: SemanticPromptProtocol) -> str:
    """Return the immutable template identity for one prompt protocol."""

    if protocol is SemanticPromptProtocol.RAW_V2:
        return COMPATIBILITY_TEMPLATE_VERSION
    if protocol is SemanticPromptProtocol.HARMONY_GPT_OSS_V1:
        return HARMONY_GPT_OSS_TEMPLATE_VERSION
    raise ValueError(f"unknown semantic prompt protocol: {protocol!r}")


def compatibility_add_special_tokens(protocol: SemanticPromptProtocol) -> bool:
    """Whether vLLM should add tokenizer special tokens to rendered paths."""

    return protocol is SemanticPromptProtocol.RAW_V2


_SEMANTIC_INSTRUCTIONS = """You are judging semantic program compatibility.
Given the task and input/output examples, decide whether the candidate program can be
consistent with those examples. Judge program meaning, not prose fluency. The answer-label
mapping is supplied beside each candidate. The final answer is exactly one mapped label."""

_INSTRUCTIONS = """{semantic_instructions}

Task and examples:
{dataset_context}
"""

_HARMONY_PREFIX = (
    "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: 2026-08-11\n\n"
    "Reasoning: low\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every "
    "message.<|end|><|start|>developer<|message|># Instructions\n\n"
    "{semantic_instructions}<|end|><|start|>user<|message|>Task and examples:\n"
    "{dataset_context}"
)


def build_compatibility_prompt(
    dataset_context: str,
    protocol: SemanticPromptProtocol = SemanticPromptProtocol.RAW_V2,
) -> str:
    """Build the single shared prefix used for every program and label path."""

    if not dataset_context.strip():
        raise ValueError("dataset_context must not be empty")
    template = _INSTRUCTIONS if protocol is SemanticPromptProtocol.RAW_V2 else _HARMONY_PREFIX
    compatibility_template_version(protocol)
    return template.format(
        semantic_instructions=_SEMANTIC_INSTRUCTIONS,
        dataset_context=dataset_context.rstrip(),
    )


def build_program_paths(
    program: CompatibilityProgram,
    protocol: SemanticPromptProtocol = SemanticPromptProtocol.RAW_V2,
) -> tuple[CompatibilityPathSpec, ...]:
    """Render the four swapped-mapping paths in their canonical audit order."""

    return (
        _path(program, CompatibilityMapping.A_IS_COMPATIBLE, CompatibilityLabel.A, protocol),
        _path(program, CompatibilityMapping.A_IS_COMPATIBLE, CompatibilityLabel.B, protocol),
        _path(program, CompatibilityMapping.B_IS_COMPATIBLE, CompatibilityLabel.A, protocol),
        _path(program, CompatibilityMapping.B_IS_COMPATIBLE, CompatibilityLabel.B, protocol),
    )


def _path(
    program: CompatibilityProgram,
    mapping: CompatibilityMapping,
    label: CompatibilityLabel,
    protocol: SemanticPromptProtocol,
) -> CompatibilityPathSpec:
    mapping_text = (
        "A = compatible; B = incompatible"
        if mapping is CompatibilityMapping.A_IS_COMPATIBLE
        else "A = incompatible; B = compatible"
    )
    if protocol is SemanticPromptProtocol.RAW_V2:
        candidate = (
            "\n\nCandidate program:\n"
            f"{program.program_text}\n"
            f"Answer-label mapping: {mapping_text}\n"
            f"Answer: {label.value}"
        )
    elif protocol is SemanticPromptProtocol.HARMONY_GPT_OSS_V1:
        candidate = (
            "\n\nCandidate program:\n"
            f"{program.program_text}\n"
            f"Answer-label mapping: {mapping_text}\n"
            "Return exactly one mapped label."
            "<|end|><|start|>assistant<|channel|>final<|message|>"
            f"{label.value}"
        )
    else:
        raise ValueError(f"unknown semantic prompt protocol: {protocol!r}")
    return CompatibilityPathSpec(mapping=mapping, label=label, candidate=candidate)

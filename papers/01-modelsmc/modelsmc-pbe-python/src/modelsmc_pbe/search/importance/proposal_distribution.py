"""Normalized Qwen/deduction mixtures for finite proposal choices."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch

from modelsmc_pbe.core.evaluate import evaluate_expression
from modelsmc_pbe.core.types import Node, RuntimeValue
from modelsmc_pbe.deduction import HoleExample
from modelsmc_pbe.domain.ast import AstNode
from modelsmc_pbe.domain.models import RuntimeValue as DomainRuntimeValue
from modelsmc_pbe.domain.models import value_has_type
from modelsmc_pbe.smc import normalize_log_weights


@dataclass(frozen=True, slots=True)
class CandidateDistribution:
    """One auditable finite categorical distribution and its components."""

    probabilities: torch.Tensor
    q_llm: torch.Tensor
    q_deduction: torch.Tensor


def _runtime_value(value: int | bool | tuple[int, ...] | tuple[bool, ...]) -> RuntimeValue:
    if isinstance(value, tuple):
        return list(value)
    return value


def _satisfies(expression: AstNode, example: HoleExample) -> bool:
    bindings = {
        parameter.name: _runtime_value(value.value)
        for parameter, value in zip(example.hole.parameters, example.inputs, strict=True)
    }
    input_value = bindings.get("x", bindings.get("xs", 0))
    keyword_arguments: dict[str, RuntimeValue] = {}
    if "item" in bindings:
        keyword_arguments["item"] = bindings["item"]
    if "acc" in bindings:
        keyword_arguments["accumulator"] = bindings["acc"]
    result = evaluate_expression(
        cast(Node, expression),
        input_value,
        **keyword_arguments,
    )
    expected = _runtime_value(example.output.value)
    return (
        value_has_type(cast(DomainRuntimeValue, result), example.output.value_type)
        and result == expected
    )


def deduction_mismatch_counts(
    expressions: tuple[AstNode | None, ...],
    examples: tuple[HoleExample, ...],
) -> tuple[int, ...]:
    """Count violated sound hole constraints for every finite choice.

    A ``None`` expression denotes a structural-family choice. Deduction has
    already removed refuted families, so the deduction component is uniform
    over those remaining choices.
    """

    if not examples:
        return tuple(0 for _ in expressions)
    if any(expression is None for expression in expressions):
        raise ValueError("hole deduction examples require expression candidates")
    return tuple(
        sum(
            not _satisfies(cast(AstNode, expression), example)
            for example in examples
        )
        for expression in expressions
    )


def candidate_distribution(
    *,
    sequence_logprobs: tuple[float, ...],
    q_deduction: torch.Tensor,
    temperature: float,
    epsilon: float,
    deduction_mix: float,
) -> CandidateDistribution:
    """Return the exact supported mixture used for sampling and correction.

    ``q_llm`` is the locally normalized teacher-forced Qwen energy. ``q_D`` is
    a Boltzmann distribution over violations of sound derived examples (or a
    balanced distribution over viable families). The final proposal is

    ``epsilon / K + (1-epsilon) * ((1-rho) q_llm + rho q_D)``.
    """

    # The historical argument name is retained for internal-call compatibility;
    # callers may now pass either total or mean full-prompt energies.
    llm_energies = sequence_logprobs
    count = len(llm_energies)
    if count < 1 or q_deduction.ndim != 1 or int(q_deduction.numel()) != count:
        raise ValueError("finite proposal components must have one nonempty aligned catalog")
    if any(not math.isfinite(value) for value in llm_energies):
        raise ValueError("finite proposal LLM energies must be finite")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and greater than zero")
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be finite and in (0, 1]")
    if not math.isfinite(deduction_mix) or not 0.0 <= deduction_mix <= 1.0:
        raise ValueError("deduction_mix must be finite and in [0, 1]")
    if bool(torch.any(~torch.isfinite(q_deduction)).item()) or bool(
        torch.any(q_deduction < 0.0).item()
    ):
        raise ValueError("deduction probabilities must be finite and nonnegative")
    deduction_total = float(q_deduction.sum().item())
    if deduction_total <= 0.0:
        raise ValueError("deduction probabilities must have positive total mass")

    llm_logits = torch.tensor(llm_energies, dtype=torch.float64) / temperature
    q_llm = normalize_log_weights(llm_logits).weights
    normalized_deduction = q_deduction.to(dtype=torch.float64) / deduction_total
    guided = (1.0 - deduction_mix) * q_llm + deduction_mix * normalized_deduction
    probabilities = (1.0 - epsilon) * guided + epsilon / count
    probabilities = probabilities / probabilities.sum()
    return CandidateDistribution(
        probabilities=probabilities,
        q_llm=q_llm,
        q_deduction=normalized_deduction,
    )

"""Importance sample local repairs from direct-JSON boundary probabilities.

The generated reasoning/final prefix is an auxiliary variable.  At the finite
repair-ID boundary, vLLM returns the processed categorical probabilities used
for sampling.  The application mixes those probabilities with a uniform floor,
draws the repair independently, and records the resulting conditional q.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import httpx
import torch

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.core.results import ScoredProgram
from modelsmc_pbe.core.scorer import ProgramScorer
from modelsmc_pbe.core.types import Node
from modelsmc_pbe.domain.ast import AstNode, clone_program
from modelsmc_pbe.grammar import arithmetic_expressions, filter_predicates
from modelsmc_pbe.smc import categorical_sample, effective_sample_size, normalize_log_weights
from research.automatic_repair_feedback import derive_automatic_feedback
from research.direct_json_repair_choice import _canonical_bytes, _choice_payload, _write_json
from research.execution_guided_repair import _assemble_program, _dsl_catalog
from research.local_repair_importance import (
    MAPPER_REPAIRS,
    PREDICATE_REPAIRS,
    _resolve_repairs,
)

_DIGIT_TOKEN_IDS = (15, 16, 17, 18)
_CHOICE_PATTERN = re.compile(rb'"selected_repair_id"\s*:\s*"([mp])([0-3])"')


def mix_with_uniform(probabilities: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Add an exact finite uniform exploration component."""

    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise ValueError("probabilities must be a nonempty vector")
    if not bool(torch.all(torch.isfinite(probabilities)).item()) or bool(
        torch.any(probabilities < 0.0).item()
    ):
        raise ValueError("probabilities must be finite and nonnegative")
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    total = float(probabilities.sum().item())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError("probabilities must sum to one")
    mixed = (1.0 - epsilon) * probabilities + epsilon / probabilities.numel()
    return mixed / mixed.sum()


def extract_boundary_distribution(
    body: object,
    repair_ids: tuple[str, ...],
) -> tuple[str, torch.Tensor, dict[str, object]]:
    """Extract q(repair | generated pre-ID trace) from one chat response."""

    if len(repair_ids) != 4:
        raise ValueError("this protocol requires exactly four repair IDs")
    prefix = repair_ids[0][0]
    if tuple(repair_ids) != tuple(f"{prefix}{index}" for index in range(4)):
        raise ValueError("repair IDs must be one common prefix plus digits 0..3")
    if not isinstance(body, dict):
        raise ValueError("response body must be an object")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        raise ValueError("choice did not finish with stop")
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("choice has no final content")
    decoded = json.loads(cast(str, message["content"]))
    if not isinstance(decoded, dict) or set(decoded) != {"selected_repair_id"}:
        raise ValueError("final JSON must contain only selected_repair_id")
    selected = decoded["selected_repair_id"]
    if selected not in repair_ids:
        raise ValueError("selected repair is outside the allowed slate")

    raw_logprobs = choice.get("logprobs")
    entries = raw_logprobs.get("content") if isinstance(raw_logprobs, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("response has no token logprobs")
    message_positions = [
        index
        for index, entry in enumerate(entries)
        if isinstance(entry, dict) and entry.get("token") == "<|message|>"
    ]
    if not message_positions:
        raise ValueError("could not locate the final message boundary")
    final_start = message_positions[-1] + 1
    final_entries = entries[final_start:]
    chunks: list[bytes] = []
    spans: list[tuple[int, int, int]] = []
    offset = 0
    for absolute_index, entry in enumerate(final_entries, start=final_start):
        if not isinstance(entry, dict):
            raise ValueError("token logprob entry must be an object")
        byte_values = entry.get("bytes")
        if not isinstance(byte_values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255
            for value in byte_values
        ):
            raise ValueError("token logprob entry has invalid bytes")
        chunk = bytes(byte_values)
        chunks.append(chunk)
        spans.append((offset, offset + len(chunk), absolute_index))
        offset += len(chunk)
    final_bytes = b"".join(chunks)
    matches = tuple(_CHOICE_PATTERN.finditer(final_bytes))
    if len(matches) != 1:
        raise ValueError("final token trace must contain one repair-ID field")
    matched_id = (matches[0].group(1) + matches[0].group(2)).decode()
    if matched_id != selected:
        raise ValueError("final JSON and token trace disagree on the selected repair")
    digit_offset = matches[0].start(2)
    digit_positions = [
        absolute_index for start, end, absolute_index in spans if start <= digit_offset < end
    ]
    if len(digit_positions) != 1:
        raise ValueError("could not identify the repair digit token")
    digit_position = digit_positions[0]
    digit_entry = entries[digit_position]
    if not isinstance(digit_entry, dict) or digit_entry.get("token") != selected[1]:
        raise ValueError("repair suffix is not one digit token")
    alternatives = digit_entry.get("top_logprobs")
    if not isinstance(alternatives, list):
        raise ValueError("repair digit has no alternative logprobs")
    by_digit: dict[str, float] = {}
    for alternative in alternatives:
        if not isinstance(alternative, dict):
            continue
        token = alternative.get("token")
        value = alternative.get("logprob")
        if (
            token in {"0", "1", "2", "3"}
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            logprob = float(value)
            if math.isfinite(logprob) and logprob <= 0.0:
                by_digit[cast(str, token)] = logprob
    if set(by_digit) != {"0", "1", "2", "3"}:
        raise ValueError("repair boundary does not expose all four digit logprobs")
    values = torch.tensor(
        [math.exp(by_digit[str(index)]) for index in range(4)],
        dtype=torch.float64,
    )
    total = float(values.sum().item())
    if not math.isclose(total, 1.0, abs_tol=1e-5):
        raise ValueError("processed repair probabilities do not normalize")
    probabilities = values / values.sum()
    proof = {
        "selected_repair_id": selected,
        "digit_entry_index": digit_position,
        "digit_logprobs": {f"{prefix}{index}": by_digit[str(index)] for index in range(4)},
        "conditional_probabilities": dict(zip(repair_ids, probabilities.tolist(), strict=True)),
    }
    return cast(str, selected), probabilities, proof


def _choice_only_payload(payload: dict[str, object]) -> dict[str, object]:
    messages = cast(list[dict[str, str]], payload["messages"])
    document = json.loads(messages[1]["content"])
    document["output_schema"] = {"selected_repair_id": "one allowed repair ID"}
    messages[0]["content"] = (
        "Choose the single best allowed local repair. Change exactly the declared hole. "
        "Treat interpreter outputs as authoritative. Return only the selected repair ID "
        "in the required JSON object."
    )
    messages[1]["content"] = _canonical_bytes(document).decode()
    schema = cast(dict[str, Any], payload["response_format"])["json_schema"]["schema"]
    selected_schema = schema["properties"]["selected_repair_id"]
    cast(dict[str, Any], payload["response_format"])["json_schema"]["schema"] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["selected_repair_id"],
        "properties": {"selected_repair_id": selected_schema},
    }
    payload.update(
        {
            "include_reasoning": True,
            "logprobs": True,
            "top_logprobs": 0,
            "logprob_token_ids": list(_DIGIT_TOKEN_IDS),
        }
    )
    return payload


async def _call_distribution(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, object],
    repair_ids: tuple[str, ...],
    stage_dir: Path,
) -> dict[str, object]:
    stage_dir.mkdir(parents=True)
    request = _canonical_bytes(payload)
    (stage_dir / "request.json").write_bytes(request)
    started = time.perf_counter()
    try:
        response = await client.post(
            base_url.rstrip("/") + "/chat/completions",
            content=request,
            headers={"Content-Type": "application/json"},
        )
        (stage_dir / "response.json").write_bytes(response.content)
        response.raise_for_status()
        body = response.json()
        selected, probabilities, proof = extract_boundary_distribution(body, repair_ids)
        result: dict[str, object] = {
            "status": "valid",
            "selected_repair_id": selected,
            "model_probabilities": probabilities.tolist(),
            "boundary_proof": proof,
            "finish_reason": body["choices"][0].get("finish_reason"),
            "usage": body.get("usage"),
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as error:
        result = {
            "status": "invalid",
            "error_type": type(error).__name__,
            "detail": str(error),
            "elapsed_seconds": time.perf_counter() - started,
        }
    _write_json(stage_dir / "result.json", result)
    return result


async def _run_stage(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    payloads: tuple[dict[str, object], ...],
    repair_ids: tuple[str, ...],
    stage_dirs: tuple[Path, ...],
) -> tuple[dict[str, object], ...]:
    results = tuple(
        await asyncio.gather(
            *(
                _call_distribution(
                    client=client,
                    base_url=base_url,
                    payload=payload,
                    repair_ids=repair_ids,
                    stage_dir=stage_dir,
                )
                for payload, stage_dir in zip(payloads, stage_dirs, strict=True)
            )
        )
    )
    if any(result["status"] != "valid" for result in results):
        raise RuntimeError("a direct-JSON trajectory was invalid; no retry was attempted")
    return results


async def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    config = load_experiment_config(args.task)
    scorer = ProgramScorer(config)
    constants = tuple(config.spec.integer_constants)
    mapper_repairs = _resolve_repairs(
        MAPPER_REPAIRS,
        _dsl_catalog(arithmetic_expressions("Item", constants)),
    )
    predicate_repairs = _resolve_repairs(
        PREDICATE_REPAIRS,
        _dsl_catalog(filter_predicates(constants)),
    )
    mapper_ids = tuple(repair_id for repair_id, _, _ in mapper_repairs)
    predicate_ids = tuple(repair_id for repair_id, _, _ in predicate_repairs)
    initial_predicate = clone_program(predicate_repairs[0][2])
    initial_mapper = clone_program(mapper_repairs[0][2])
    examples = [example.model_dump(mode="json", by_alias=True) for example in config.spec.examples]

    def make_payload(
        *,
        predicate: AstNode,
        mapper: AstNode,
        hole: str,
        repairs: tuple[tuple[str, str, AstNode], ...],
        seed: int,
    ) -> dict[str, object]:
        score = scorer.score(_assemble_program(predicate, mapper))
        if not isinstance(score, ScoredProgram):
            raise ValueError(f"current program was rejected: {score.reason}")
        feedback = derive_automatic_feedback(
            config.spec,
            cast(Node, predicate),
            cast(Node, mapper),
        ).to_dict()
        return _choice_only_payload(
            _choice_payload(
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=seed,
                examples=examples,
                predicate=predicate,
                mapper=mapper,
                score=score,
                hole=hole,
                repairs=repairs,
                feedback=feedback,
            )
        )

    started = time.perf_counter()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.sampling_seed)
    particle_dirs = tuple(output / f"particle-{index:03d}" for index in range(args.particles))
    async with httpx.AsyncClient(timeout=args.timeout_seconds) as client:
        mapper_results = await _run_stage(
            client=client,
            base_url=args.base_url,
            payloads=tuple(
                make_payload(
                    predicate=initial_predicate,
                    mapper=initial_mapper,
                    hole="mapper",
                    repairs=mapper_repairs,
                    seed=args.model_seed + index,
                )
                for index in range(args.particles)
            ),
            repair_ids=mapper_ids,
            stage_dirs=tuple(directory / "mapper" for directory in particle_dirs),
        )
        mapper_choices: list[int] = []
        mapper_q: list[torch.Tensor] = []
        for result in mapper_results:
            mixed = mix_with_uniform(
                torch.tensor(result["model_probabilities"], dtype=torch.float64),
                args.epsilon,
            )
            mapper_q.append(mixed)
            mapper_choices.append(int(categorical_sample(mixed, 1, generator=generator)[0].item()))

        predicate_results = await _run_stage(
            client=client,
            base_url=args.base_url,
            payloads=tuple(
                make_payload(
                    predicate=initial_predicate,
                    mapper=mapper_repairs[mapper_choice][2],
                    hole="predicate",
                    repairs=predicate_repairs,
                    seed=args.model_seed + 10_000 + index,
                )
                for index, mapper_choice in enumerate(mapper_choices)
            ),
            repair_ids=predicate_ids,
            stage_dirs=tuple(directory / "predicate" for directory in particle_dirs),
        )

    predicate_choices: list[int] = []
    predicate_q: list[torch.Tensor] = []
    for result in predicate_results:
        mixed = mix_with_uniform(
            torch.tensor(result["model_probabilities"], dtype=torch.float64),
            args.epsilon,
        )
        predicate_q.append(mixed)
        predicate_choices.append(int(categorical_sample(mixed, 1, generator=generator)[0].item()))

    particles: list[dict[str, object]] = []
    log_weights: list[float] = []
    state_indices: list[int] = []
    for index, (mapper_position, predicate_position) in enumerate(
        zip(mapper_choices, predicate_choices, strict=True)
    ):
        score = scorer.score(
            _assemble_program(
                predicate_repairs[predicate_position][2],
                mapper_repairs[mapper_position][2],
            )
        )
        if not isinstance(score, ScoredProgram):
            raise ValueError(f"sampled program was rejected: {score.reason}")
        q_mapper = float(mapper_q[index][mapper_position].item())
        q_predicate = float(predicate_q[index][predicate_position].item())
        log_q = math.log(q_mapper) + math.log(q_predicate)
        log_weight = score.log_target - log_q
        state_index = 4 * mapper_position + predicate_position
        state_indices.append(state_index)
        log_weights.append(log_weight)
        particles.append(
            {
                "particle": index,
                "model_selected_mapper": mapper_results[index]["selected_repair_id"],
                "applied_mapper": mapper_ids[mapper_position],
                "q_mapper_given_auxiliary_prefix": q_mapper,
                "model_selected_predicate": predicate_results[index]["selected_repair_id"],
                "applied_predicate": predicate_ids[predicate_position],
                "q_predicate_given_auxiliary_prefix": q_predicate,
                "q_path": q_mapper * q_predicate,
                "log_importance_weight": log_weight,
                "score": asdict(score),
            }
        )

    normalized = normalize_log_weights(torch.tensor(log_weights, dtype=torch.float64))
    local_scores: list[ScoredProgram] = []
    for mapper_position in range(4):
        for predicate_position in range(4):
            score = scorer.score(
                _assemble_program(
                    predicate_repairs[predicate_position][2],
                    mapper_repairs[mapper_position][2],
                )
            )
            if not isinstance(score, ScoredProgram):
                raise ValueError(f"local program was rejected: {score.reason}")
            local_scores.append(score)
    exact_mask = torch.tensor([score.exact_program for score in local_scores], dtype=torch.bool)
    aggregate = torch.zeros(16, dtype=torch.float64)
    for state_index, weight in zip(state_indices, normalized.weights.tolist(), strict=True):
        aggregate[state_index] += weight
    exact_target = normalize_log_weights(
        torch.tensor([score.log_target for score in local_scores], dtype=torch.float64)
    ).weights
    for particle, weight in zip(particles, normalized.weights.tolist(), strict=True):
        particle["normalized_weight"] = weight
    result = {
        "schema": "direct-json-auxiliary-local-repair-importance-v1",
        "claim": (
            "auxiliary-variable importance sampling over the declared 4x4 local repair "
            "neighborhood; each denominator is the processed repair-ID probability "
            "conditional on that particle's generated pre-ID trace, mixed with a frozen "
            "uniform exploration component"
        ),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "epsilon": args.epsilon,
        "particles": args.particles,
        "model_seed": args.model_seed,
        "sampling_seed": args.sampling_seed,
        "elapsed_seconds": time.perf_counter() - started,
        "particles_detail": particles,
        "diagnostics": {
            "ess": effective_sample_size(normalized.weights),
            "relative_ess": effective_sample_size(normalized.weights) / args.particles,
            "sampled_exact_programs": sum(
                local_scores[state_index].exact_program for state_index in state_indices
            ),
            "weighted_exact_mass": float(aggregate[exact_mask].sum().item()),
            "local_target_exact_mass": float(exact_target[exact_mask].sum().item()),
            "total_variation_to_exact_local_target": 0.5
            * float(torch.abs(aggregate - exact_target).sum().item()),
        },
    }
    _write_json(output / "result.json", result)
    print(json.dumps(result["diagnostics"], indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--model-seed", type=int, default=71000)
    parser.add_argument("--sampling-seed", type=int, default=24601)
    args = parser.parse_args()
    if args.particles < 1:
        parser.error("--particles must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

"""Calibrate a finite repair proposal from repeated direct JSON choices."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from collections import Counter
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
from research.execution_guided_repair import (
    MODEL_REVISION,
    _assemble_program,
    _dsl_catalog,
    render_expression_dsl,
)
from research.local_repair_importance import (
    MAPPER_REPAIRS,
    PREDICATE_REPAIRS,
    _resolve_repairs,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def calibrated_distribution(
    counts: tuple[int, ...],
    *,
    alpha: float,
    epsilon: float,
) -> torch.Tensor:
    """Define a positive categorical proposal from frozen choice counts."""

    if not counts or any(count < 0 for count in counts):
        raise ValueError("counts must be nonempty and nonnegative")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    values = torch.tensor(counts, dtype=torch.float64)
    predictive = (values + alpha) / (float(values.sum().item()) + alpha * len(counts))
    probabilities = (1.0 - epsilon) * predictive + epsilon / len(counts)
    return probabilities / probabilities.sum()


def _choice_payload(
    *,
    model: str,
    reasoning_effort: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    examples: list[dict[str, object]],
    predicate: AstNode,
    mapper: AstNode,
    score: ScoredProgram,
    hole: str,
    repairs: tuple[tuple[str, str, AstNode], ...],
    feedback: dict[str, object],
) -> dict[str, object]:
    repair_ids = [repair_id for repair_id, _, _ in repairs]
    document = {
        "task": "Repair exactly one hole in the current program.",
        "examples": examples,
        "current_program": {
            "predicate": render_expression_dsl(predicate),
            "mapper": render_expression_dsl(mapper),
        },
        "execution": {
            "total_loss": score.total_loss,
            "exact_examples": score.exact_matches,
            "example_count": len(examples),
            "evaluations": [asdict(evaluation) for evaluation in score.evaluations],
            "structured_interpreter_feedback": feedback,
        },
        "repairable_hole": hole,
        "allowed_repairs": [{"id": repair_id, "expression": dsl} for repair_id, dsl, _ in repairs],
        "output_schema": {
            "selected_repair_id": "one allowed repair ID",
            "diagnosis": "brief explanation grounded in interpreter feedback",
        },
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["selected_repair_id", "diagnosis"],
        "properties": {
            "selected_repair_id": {"type": "string", "enum": repair_ids},
            "diagnosis": {"type": "string", "minLength": 1, "maxLength": 300},
        },
    }
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Choose the single best allowed local repair. Change exactly the declared "
                    "hole. Treat concrete interpreter outputs as authoritative. Return only "
                    "the required JSON object in the final answer."
                ),
            },
            {"role": "user", "content": _canonical_bytes(document).decode()},
        ],
        "temperature": temperature,
        "top_p": 1,
        "max_tokens": max_tokens,
        "seed": seed,
        "reasoning_effort": reasoning_effort,
        "include_reasoning": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"direct_{hole}_repair_choice",
                "strict": True,
                "schema": schema,
            },
        },
    }


async def _one_choice(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, object],
    sample_dir: Path,
) -> dict[str, object]:
    sample_dir.mkdir()
    request = _canonical_bytes(payload)
    (sample_dir / "request.json").write_bytes(request)
    started = time.perf_counter()
    try:
        response = await client.post(
            base_url.rstrip("/") + "/chat/completions",
            content=request,
            headers={"Content-Type": "application/json"},
        )
        elapsed = time.perf_counter() - started
        (sample_dir / "response.json").write_bytes(response.content)
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("provider returned the wrong number of choices")
        choice = choices[0]
        if choice.get("finish_reason") != "stop":
            raise ValueError(f"finish_reason={choice.get('finish_reason')}")
        content = choice.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("provider returned no final JSON content")
        decoded = json.loads(content)
        if not isinstance(decoded, dict) or set(decoded) != {
            "selected_repair_id",
            "diagnosis",
        }:
            raise ValueError("final JSON has the wrong fields")
        repair_id = decoded["selected_repair_id"]
        if not isinstance(repair_id, str):
            raise ValueError("selected repair ID is not a string")
        result = {
            "status": "valid",
            "selected_repair_id": repair_id,
            "diagnosis": decoded["diagnosis"],
            "elapsed_seconds": elapsed,
            "usage": body.get("usage"),
            "finish_reason": choice.get("finish_reason"),
        }
    except Exception as error:
        result = {
            "status": "invalid",
            "error_type": type(error).__name__,
            "detail": str(error),
            "elapsed_seconds": time.perf_counter() - started,
        }
    _write_json(sample_dir / "result.json", result)
    return result


async def _calibrate(
    *,
    output: Path,
    stage: str,
    base_url: str,
    timeout_seconds: float,
    payloads: tuple[dict[str, object], ...],
    repair_ids: tuple[str, ...],
    alpha: float,
    epsilon: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    stage_dir = output / stage
    stage_dir.mkdir()
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        results = await asyncio.gather(
            *(
                _one_choice(
                    client=client,
                    base_url=base_url,
                    payload=payload,
                    sample_dir=stage_dir / f"sample-{index:03d}",
                )
                for index, payload in enumerate(payloads)
            )
        )
    valid_ids = [
        cast(str, result["selected_repair_id"])
        for result in results
        if result["status"] == "valid" and result.get("selected_repair_id") in repair_ids
    ]
    if not valid_ids:
        failure = {
            "stage": stage,
            "status": "invalid",
            "reason": "no-valid-direct-json-choices",
            "samples_attempted": len(results),
            "samples_valid": 0,
            "samples_invalid": len(results),
            "results": results,
        }
        _write_json(stage_dir / "calibration.json", failure)
        raise RuntimeError(f"{stage} produced no valid direct JSON choices")
    counts_by_id = Counter(valid_ids)
    counts = tuple(counts_by_id[repair_id] for repair_id in repair_ids)
    probabilities = calibrated_distribution(counts, alpha=alpha, epsilon=epsilon)
    summary = {
        "stage": stage,
        "samples_attempted": len(results),
        "samples_valid": len(valid_ids),
        "samples_invalid": len(results) - len(valid_ids),
        "alpha": alpha,
        "epsilon": epsilon,
        "counts": dict(zip(repair_ids, counts, strict=True)),
        "probabilities": dict(zip(repair_ids, probabilities.tolist(), strict=True)),
        "results": results,
    }
    _write_json(stage_dir / "calibration.json", summary)
    return probabilities, summary


def _aggregate(
    state_count: int,
    indices: tuple[int, ...],
    weights: torch.Tensor,
) -> torch.Tensor:
    result = torch.zeros(state_count, dtype=torch.float64)
    for index, weight in zip(indices, weights.tolist(), strict=True):
        result[index] += weight
    return result


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
    initial_predicate = clone_program(predicate_repairs[0][2])
    initial_mapper = clone_program(mapper_repairs[0][2])
    examples = [example.model_dump(mode="json", by_alias=True) for example in config.spec.examples]

    def payloads_for(
        *,
        predicate: AstNode,
        mapper: AstNode,
        hole: str,
        repairs: tuple[tuple[str, str, AstNode], ...],
        seed_offset: int,
    ) -> tuple[dict[str, object], ...]:
        score = scorer.score(_assemble_program(predicate, mapper))
        if not isinstance(score, ScoredProgram):
            raise ValueError(f"current program was rejected: {score.reason}")
        feedback = derive_automatic_feedback(
            config.spec,
            cast(Node, predicate),
            cast(Node, mapper),
        ).to_dict()
        return tuple(
            _choice_payload(
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.calibration_seed + seed_offset + index,
                examples=examples,
                predicate=predicate,
                mapper=mapper,
                score=score,
                hole=hole,
                repairs=repairs,
                feedback=feedback,
            )
            for index in range(args.calibration_samples)
        )

    started = time.perf_counter()
    mapper_ids = tuple(repair_id for repair_id, _, _ in mapper_repairs)
    q_mapper, mapper_calibration = await _calibrate(
        output=output,
        stage="mapper",
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        payloads=payloads_for(
            predicate=initial_predicate,
            mapper=initial_mapper,
            hole="mapper",
            repairs=mapper_repairs,
            seed_offset=0,
        ),
        repair_ids=mapper_ids,
        alpha=args.alpha,
        epsilon=args.epsilon,
    )

    async def calibrate_predicate(
        mapper_position: int,
    ) -> tuple[int, torch.Tensor, dict[str, object]]:
        mapper_id, _, mapper = mapper_repairs[mapper_position]
        probabilities, summary = await _calibrate(
            output=output,
            stage=f"predicate-given-{mapper_id}",
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            payloads=payloads_for(
                predicate=initial_predicate,
                mapper=mapper,
                hole="predicate",
                repairs=predicate_repairs,
                seed_offset=1_000 * (mapper_position + 1),
            ),
            repair_ids=tuple(repair_id for repair_id, _, _ in predicate_repairs),
            alpha=args.alpha,
            epsilon=args.epsilon,
        )
        return mapper_position, probabilities, summary

    predicate_calibrations = await asyncio.gather(
        *(calibrate_predicate(position) for position in range(len(mapper_repairs)))
    )
    q_predicate = {position: values for position, values, _ in predicate_calibrations}

    states: list[dict[str, object]] = []
    log_q: list[float] = []
    log_target: list[float] = []
    for mapper_position, (mapper_id, mapper_dsl, mapper) in enumerate(mapper_repairs):
        for predicate_position, (predicate_id, predicate_dsl, predicate) in enumerate(
            predicate_repairs
        ):
            score = scorer.score(_assemble_program(predicate, mapper))
            if not isinstance(score, ScoredProgram):
                raise ValueError(f"local program was rejected: {score.reason}")
            probability = float(
                q_mapper[mapper_position].item()
                * q_predicate[mapper_position][predicate_position].item()
            )
            states.append(
                {
                    "state_index": len(states),
                    "mapper_id": mapper_id,
                    "mapper": mapper_dsl,
                    "predicate_id": predicate_id,
                    "predicate": predicate_dsl,
                    "q_mapper": float(q_mapper[mapper_position].item()),
                    "q_predicate_given_mapper": float(
                        q_predicate[mapper_position][predicate_position].item()
                    ),
                    "q_path": probability,
                    "score": asdict(score),
                }
            )
            log_q.append(math.log(probability))
            log_target.append(score.log_target)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.sampling_seed)
    joint_q = torch.exp(torch.tensor(log_q, dtype=torch.float64))
    sampled_tensor = categorical_sample(joint_q, args.particles, generator=generator)
    sampled = tuple(int(value) for value in sampled_tensor.tolist())
    particle_log_weights = torch.tensor(
        [log_target[index] - log_q[index] for index in sampled], dtype=torch.float64
    )
    normalized = normalize_log_weights(particle_log_weights)
    aggregate = _aggregate(len(states), sampled, normalized.weights)
    exact_mask = torch.tensor(
        [cast(dict[str, Any], state["score"])["exact_program"] for state in states],
        dtype=torch.bool,
    )
    exact_target = normalize_log_weights(torch.tensor(log_target, dtype=torch.float64)).weights
    result = {
        "schema": "direct-json-local-repair-importance-v1",
        "claim": (
            "conditional on frozen direct-choice calibration responses, the application "
            "defines a positive 4x4 categorical repair proposal and performs self-normalized "
            "importance sampling over that local neighborhood"
        ),
        "task_sha256": hashlib.sha256(args.task.resolve().read_bytes()).hexdigest(),
        "model": args.model,
        "model_revision": MODEL_REVISION,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "calibration_samples": args.calibration_samples,
        "alpha": args.alpha,
        "epsilon": args.epsilon,
        "particles": args.particles,
        "calibration_seed": args.calibration_seed,
        "sampling_seed": args.sampling_seed,
        "elapsed_seconds": time.perf_counter() - started,
        "mapper_calibration": mapper_calibration,
        "predicate_calibrations": {
            mapper_repairs[position][0]: summary for position, _, summary in predicate_calibrations
        },
        "states": states,
        "particles_detail": [
            {
                "particle": particle,
                "state_index": state_index,
                "q_path": math.exp(log_q[state_index]),
                "log_target": log_target[state_index],
                "log_importance_weight": float(particle_log_weights[particle].item()),
                "normalized_weight": float(normalized.weights[particle].item()),
                "exact_program": bool(exact_mask[state_index]),
            }
            for particle, state_index in enumerate(sampled)
        ],
        "diagnostics": {
            "ess": effective_sample_size(normalized.weights),
            "relative_ess": effective_sample_size(normalized.weights) / args.particles,
            "unique_sampled_programs": len(set(sampled)),
            "sampled_exact_programs": sum(bool(exact_mask[index]) for index in sampled),
            "weighted_exact_mass": float(aggregate[exact_mask].sum().item()),
            "local_target_exact_mass": float(exact_target[exact_mask].sum().item()),
            "total_variation_to_exact_local_target": 0.5
            * float(torch.abs(aggregate - exact_target).sum().item()),
        },
    }
    _write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "diagnostics": result["diagnostics"],
                "elapsed_seconds": result["elapsed_seconds"],
                "mapper": mapper_calibration,
                "predicate_given_square": result["predicate_calibrations"]["m2"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--calibration-samples", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--calibration-seed", type=int, default=41000)
    parser.add_argument("--sampling-seed", type=int, default=24601)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

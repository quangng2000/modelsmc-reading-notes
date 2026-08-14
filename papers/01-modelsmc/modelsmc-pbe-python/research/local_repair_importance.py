"""Importance sample a finite two-hole execution-guided repair neighborhood."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.core.results import ScoredProgram
from modelsmc_pbe.core.scorer import ProgramScorer
from modelsmc_pbe.core.types import Node
from modelsmc_pbe.domain.ast import AstNode, clone_program
from modelsmc_pbe.grammar import arithmetic_expressions, filter_predicates
from modelsmc_pbe.proposals import VLLMPromptLogprobConfig, VLLMPromptLogprobScorer
from modelsmc_pbe.proposals.labels import (
    CompatibilityProgram,
    CompatibilityScoringRequest,
    SemanticPromptProtocol,
    SymmetrizedLabelCompatibilityScorer,
)
from modelsmc_pbe.search.importance.proposal_distribution import candidate_distribution
from modelsmc_pbe.smc import (
    categorical_sample,
    effective_sample_size,
    normalize_log_weights,
    systematic_resample,
)
from research.automatic_repair_feedback import derive_automatic_feedback
from research.execution_guided_repair import (
    MODEL_REVISION,
    _assemble_program,
    _dsl_catalog,
    render_expression_dsl,
)

MAPPER_REPAIRS = (
    ("m0", "item"),
    ("m1", "sub(0,item)"),
    ("m2", "mul(item,item)"),
    ("m3", "add(item,1)"),
)
PREDICATE_REPAIRS = (
    ("p0", "and(lt(-1,item),lt(item,3))"),
    ("p1", "and(lt(-2,item),lt(item,3))"),
    ("p2", "and(lt(-3,item),lt(item,3))"),
    ("p3", "lt(item,3)"),
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _resolve_repairs(
    specifications: tuple[tuple[str, str], ...],
    catalog: dict[str, AstNode],
) -> tuple[tuple[str, str, AstNode], ...]:
    resolved = []
    for repair_id, dsl in specifications:
        try:
            expression = catalog[dsl]
        except KeyError as error:
            raise ValueError(f"repair {repair_id} is outside the finite grammar: {dsl}") from error
        resolved.append((repair_id, dsl, clone_program(expression)))
    if len({repair_id for repair_id, _, _ in resolved}) != len(resolved):
        raise ValueError("repair IDs must be unique")
    if len({dsl for _, dsl, _ in resolved}) != len(resolved):
        raise ValueError("repair expressions must be unique")
    return tuple(resolved)


def repair_distribution(
    scores: tuple[float, ...],
    *,
    temperature: float,
    epsilon: float,
) -> torch.Tensor:
    """Return the application-defined softmax plus local uniform floor."""

    distribution = candidate_distribution(
        sequence_logprobs=scores,
        q_deduction=torch.ones(len(scores), dtype=torch.float64),
        temperature=temperature,
        epsilon=epsilon,
        deduction_mix=0.0,
    )
    return distribution.probabilities


def _examples(config: Any) -> list[dict[str, object]]:
    return [example.model_dump(mode="json", by_alias=True) for example in config.spec.examples]


def _repair_context(
    *,
    config: Any,
    predicate: AstNode,
    mapper: AstNode,
    hole: str,
    repairs: tuple[tuple[str, str, AstNode], ...],
    score: ScoredProgram,
) -> str:
    feedback = derive_automatic_feedback(
        config.spec,
        cast(Node, predicate),
        cast(Node, mapper),
    )
    document = {
        "task": "Repair exactly one hole in the current program.",
        "examples": _examples(config),
        "current_program": {
            "predicate": render_expression_dsl(predicate),
            "mapper": render_expression_dsl(mapper),
        },
        "execution": {
            "total_loss": score.total_loss,
            "exact_examples": score.exact_matches,
            "example_count": len(config.spec.examples),
            "evaluations": [asdict(evaluation) for evaluation in score.evaluations],
            "structured_feedback": feedback.to_dict(),
        },
        "repairable_hole": hole,
        "allowed_repairs": [{"id": repair_id, "expression": dsl} for repair_id, dsl, _ in repairs],
        "instruction": (
            "Judge each allowed repair after applying it to the current program. "
            "The interpreter feedback is authoritative. Change only the declared hole."
        ),
    }
    return _canonical(document)


def _compatibility_programs(
    *,
    repairs: tuple[tuple[str, str, AstNode], ...],
    predicate: AstNode,
    mapper: AstNode,
    hole: str,
) -> tuple[CompatibilityProgram, ...]:
    programs = []
    for repair_id, dsl, expression in repairs:
        revised_predicate = expression if hole == "predicate" else predicate
        revised_mapper = expression if hole == "mapper" else mapper
        programs.append(
            CompatibilityProgram(
                program_key=repair_id,
                program_text=(
                    f"Selected repair ID: {repair_id}\n"
                    f"Selected {hole} expression: {dsl}\n"
                    "Resulting complete program: "
                    f"predicate={render_expression_dsl(revised_predicate)}; "
                    f"mapper={render_expression_dsl(revised_mapper)}"
                ),
            )
        )
    return tuple(programs)


async def _score_repairs(
    *,
    semantic_scorer: SymmetrizedLabelCompatibilityScorer,
    config: Any,
    predicate: AstNode,
    mapper: AstNode,
    hole: str,
    repairs: tuple[tuple[str, str, AstNode], ...],
    request_index: int,
    program_scorer: ProgramScorer,
) -> tuple[tuple[float, ...], dict[str, object]]:
    current = program_scorer.score(_assemble_program(predicate, mapper))
    if not isinstance(current, ScoredProgram):
        raise ValueError(f"current program was rejected: {current.reason}")
    context = _repair_context(
        config=config,
        predicate=predicate,
        mapper=mapper,
        hole=hole,
        repairs=repairs,
        score=current,
    )
    batch = await semantic_scorer.score(
        CompatibilityScoringRequest(
            dataset_context=context,
            programs=_compatibility_programs(
                repairs=repairs,
                predicate=predicate,
                mapper=mapper,
                hole=hole,
            ),
            integer_constants=tuple(config.spec.integer_constants),
            request_index=request_index,
            raw_candidate_batch_size=128,
            raw_request_batch_size=8,
        )
    )
    by_id = {score.program_key: score for score in batch.scores}
    ordered = tuple(by_id[repair_id].compatibility_log_score for repair_id, _, _ in repairs)
    artifact = {
        "hole": hole,
        "context": json.loads(context),
        "prompt_sha256": batch.prompt_sha256,
        "template_version": batch.template_version,
        "scores": [asdict(by_id[repair_id]) for repair_id, _, _ in repairs],
    }
    return ordered, artifact


def _aggregate_particles(
    state_count: int,
    state_indices: tuple[int, ...],
    weights: torch.Tensor,
) -> torch.Tensor:
    aggregate = torch.zeros(state_count, dtype=torch.float64)
    for state_index, weight in zip(state_indices, weights.tolist(), strict=True):
        aggregate[state_index] += weight
    return aggregate


async def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    config = load_experiment_config(args.task)
    program_scorer = ProgramScorer(config)
    constants = tuple(config.spec.integer_constants)
    mapper_catalog = _dsl_catalog(arithmetic_expressions("Item", constants))
    predicate_catalog = _dsl_catalog(filter_predicates(constants))
    mapper_repairs = _resolve_repairs(MAPPER_REPAIRS, mapper_catalog)
    predicate_repairs = _resolve_repairs(PREDICATE_REPAIRS, predicate_catalog)
    initial_predicate = clone_program(predicate_catalog[PREDICATE_REPAIRS[0][1]])
    initial_mapper = clone_program(mapper_catalog[MAPPER_REPAIRS[0][1]])

    raw_scorer = VLLMPromptLogprobScorer(
        VLLMPromptLogprobConfig(
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            max_concurrency=8,
            max_batch_size=128,
            add_special_tokens=False,
            model_revision=MODEL_REVISION,
            tokenizer_revision=MODEL_REVISION,
        )
    )
    semantic_scorer = SymmetrizedLabelCompatibilityScorer(
        raw_scorer,
        prompt_protocol=SemanticPromptProtocol.HARMONY_GPT_OSS_V1,
    )
    started = time.perf_counter()
    mapper_scores, mapper_artifact = await _score_repairs(
        semantic_scorer=semantic_scorer,
        config=config,
        predicate=initial_predicate,
        mapper=initial_mapper,
        hole="mapper",
        repairs=mapper_repairs,
        request_index=0,
        program_scorer=program_scorer,
    )
    q_mapper = repair_distribution(
        mapper_scores,
        temperature=args.temperature,
        epsilon=args.epsilon,
    )

    async def score_predicates(
        mapper_position: int,
    ) -> tuple[int, tuple[float, ...], dict[str, object]]:
        mapper = mapper_repairs[mapper_position][2]
        scores, artifact = await _score_repairs(
            semantic_scorer=semantic_scorer,
            config=config,
            predicate=initial_predicate,
            mapper=mapper,
            hole="predicate",
            repairs=predicate_repairs,
            request_index=1 + mapper_position,
            program_scorer=program_scorer,
        )
        return mapper_position, scores, artifact

    predicate_results = await asyncio.gather(
        *(score_predicates(index) for index in range(len(mapper_repairs)))
    )
    predicate_scores: dict[int, tuple[float, ...]] = {}
    predicate_artifacts: dict[int, dict[str, object]] = {}
    q_predicate: dict[int, torch.Tensor] = {}
    for mapper_position, scores, artifact in predicate_results:
        predicate_scores[mapper_position] = scores
        predicate_artifacts[mapper_position] = artifact
        q_predicate[mapper_position] = repair_distribution(
            scores,
            temperature=args.temperature,
            epsilon=args.epsilon,
        )

    states: list[dict[str, object]] = []
    log_gamma: list[float] = []
    log_q: list[float] = []
    for mapper_position, (mapper_id, mapper_dsl, mapper) in enumerate(mapper_repairs):
        for predicate_position, (predicate_id, predicate_dsl, predicate) in enumerate(
            predicate_repairs
        ):
            score = program_scorer.score(_assemble_program(predicate, mapper))
            if not isinstance(score, ScoredProgram):
                raise ValueError(f"local repair program was rejected: {score.reason}")
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
            log_gamma.append(score.log_target)
            log_q.append(math.log(probability))
    if not math.isclose(sum(math.exp(value) for value in log_q), 1.0, abs_tol=1e-12):
        raise ValueError("joint repair proposal does not normalize to one")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    joint_q = torch.exp(torch.tensor(log_q, dtype=torch.float64))
    sampled = categorical_sample(joint_q, args.particles, generator=generator)
    sampled_indices = tuple(int(value) for value in sampled.tolist())
    particle_log_weights = torch.tensor(
        [log_gamma[index] - log_q[index] for index in sampled_indices],
        dtype=torch.float64,
    )
    normalized = normalize_log_weights(particle_log_weights)
    ess = effective_sample_size(normalized.weights)
    aggregate = _aggregate_particles(len(states), sampled_indices, normalized.weights)

    target = normalize_log_weights(torch.tensor(log_gamma, dtype=torch.float64)).weights
    exact_mask = torch.tensor(
        [cast(dict[str, Any], state["score"])["exact_program"] for state in states],
        dtype=torch.bool,
    )
    weighted_exact_mass = float(aggregate[exact_mask].sum().item())
    target_exact_mass = float(target[exact_mask].sum().item())
    total_variation = 0.5 * float(torch.abs(aggregate - target).sum().item())
    discovered_exact = sum(bool(exact_mask[index]) for index in sampled_indices)

    relative_ess = ess / args.particles
    resampled = relative_ess < args.ess_threshold
    resampled_indices: tuple[int, ...] = ()
    if resampled:
        slots = systematic_resample(normalized.weights, generator=generator)
        resampled_indices = tuple(sampled_indices[int(slot)] for slot in slots.tolist())

    elapsed = time.perf_counter() - started
    mapper_score_records = []
    for position, (repair_id, dsl, _) in enumerate(mapper_repairs):
        mapper_score_records.append(
            {
                "id": repair_id,
                "expression": dsl,
                "score": mapper_scores[position],
                "q": float(q_mapper[position].item()),
            }
        )
    predicate_score_records = {}
    for mapper_position, (mapper_id, _, _) in enumerate(mapper_repairs):
        predicate_score_records[mapper_id] = [
            {
                "id": repair_id,
                "expression": dsl,
                "score": predicate_scores[mapper_position][position],
                "q": float(q_predicate[mapper_position][position].item()),
            }
            for position, (repair_id, dsl, _) in enumerate(predicate_repairs)
        ]
    result = {
        "schema": "local-execution-guided-repair-importance-v1",
        "claim": (
            "self-normalized importance sampling over the declared 4x4 local repair "
            "neighborhood; not the full 36,000-program grammar"
        ),
        "task_sha256": hashlib.sha256(args.task.resolve().read_bytes()).hexdigest(),
        "model": args.model,
        "model_revision": MODEL_REVISION,
        "temperature": args.temperature,
        "epsilon": args.epsilon,
        "particles": args.particles,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "provider_metrics": asdict(raw_scorer.provider_metrics()),
        "mapper_distribution": mapper_score_records,
        "predicate_distributions": predicate_score_records,
        "states": states,
        "particles_detail": [
            {
                "particle": particle,
                "state_index": state_index,
                "q_path": math.exp(log_q[state_index]),
                "log_target": log_gamma[state_index],
                "log_importance_weight": float(particle_log_weights[particle].item()),
                "normalized_weight": float(normalized.weights[particle].item()),
                "exact_program": bool(exact_mask[state_index]),
            }
            for particle, state_index in enumerate(sampled_indices)
        ],
        "diagnostics": {
            "ess": ess,
            "relative_ess": relative_ess,
            "resampled": resampled,
            "resampled_state_indices": resampled_indices,
            "unique_sampled_programs": len(set(sampled_indices)),
            "sampled_exact_programs": discovered_exact,
            "weighted_exact_mass": weighted_exact_mass,
            "local_target_exact_mass": target_exact_mass,
            "total_variation_to_exact_local_target": total_variation,
        },
    }
    _write_json(output / "mapper-score-artifact.json", mapper_artifact)
    _write_json(output / "predicate-score-artifacts.json", predicate_artifacts)
    _write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "diagnostics": result["diagnostics"],
                "elapsed_seconds": elapsed,
                "mapper_distribution": mapper_score_records,
                "predicate_distribution_after_square": predicate_score_records["m2"],
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
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--seed", type=int, default=24601)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

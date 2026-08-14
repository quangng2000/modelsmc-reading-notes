"""Evaluate LLM-generated local repair shortlists against random shortlists.

The provider sees only examples, a frozen current program, interpreter feedback,
and the public finite grammar.  It must generate four canonical DSL expressions
per hole.  Raw provider artifacts are sealed before exhaustive evaluator-only
analysis joins exact-program labels.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any, cast

import httpx

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.core.results import ScoredProgram
from modelsmc_pbe.core.scorer import ProgramScorer
from modelsmc_pbe.core.types import Node
from modelsmc_pbe.domain.ast import AstNode, clone_program
from modelsmc_pbe.grammar import arithmetic_expressions, filter_predicates
from research.automatic_repair_feedback import derive_automatic_feedback
from research.direct_json_repair_choice import _canonical_bytes, _write_json
from research.execution_guided_repair import (
    MODEL_REVISION,
    _assemble_program,
    _dsl_catalog,
    render_expression_dsl,
)


def smoothed_shortlist_probability(
    candidate: str,
    shortlist: tuple[str, ...],
    *,
    catalog_size: int,
    epsilon: float,
) -> float:
    """Return (1-epsilon) Uniform(shortlist) + epsilon Uniform(catalog)."""

    if catalog_size < 1:
        raise ValueError("catalog_size must be positive")
    if not 0.0 < epsilon <= 1.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and in (0, 1]")
    if len(set(shortlist)) != len(shortlist):
        raise ValueError("shortlist must not contain duplicates")
    floor = epsilon / catalog_size
    if not shortlist:
        return 1.0 / catalog_size
    return floor + ((1.0 - epsilon) / len(shortlist) if candidate in shortlist else 0.0)


def validate_generated_shortlist(
    raw_candidates: object,
    catalog: dict[str, AstNode],
    *,
    expected_size: int,
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Validate exact canonical DSL strings without normalizing or repairing them."""

    if not isinstance(raw_candidates, list) or any(
        not isinstance(candidate, str) for candidate in raw_candidates
    ):
        raise ValueError("generated candidates must be an array of strings")
    accepted: list[str] = []
    invalid: list[str] = []
    duplicates: list[str] = []
    for raw_candidate in cast(list[str], raw_candidates):
        if raw_candidate not in catalog:
            invalid.append(raw_candidate)
        elif raw_candidate in accepted:
            duplicates.append(raw_candidate)
        else:
            accepted.append(raw_candidate)
    audit = {
        "raw_candidates": raw_candidates,
        "accepted": accepted,
        "invalid": invalid,
        "duplicates": duplicates,
        "expected_size": expected_size,
        "complete": len(accepted) == expected_size and not invalid and not duplicates,
    }
    return tuple(accepted), audit


def _grammar_text(hole: str, constants: tuple[int, ...]) -> str:
    constant_text = ", ".join(str(value) for value in constants)
    if hole == "mapper":
        return (
            f"Constants: [{constant_text}]. Canonical mapper DSL is exactly one of: item; "
            "a listed integer constant; or add(a,b), sub(a,b), mul(a,b), where (a,b) is "
            "exactly (item,item), (item,c), or (c,item) for one listed constant c. "
            "Do not nest arithmetic operations."
        )
    return (
        f"Constants: [{constant_text}]. A predicate atom is exactly lt(item,c), "
        "lt(c,item), or eq(item,c) for one listed constant c. A predicate is exactly one "
        "atom or and(atom,atom). Do not use <=, >=, >, or nested conjunctions."
    )


def _shortlist_payload(
    *,
    model: str,
    reasoning_effort: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    shortlist_size: int,
    examples: list[dict[str, object]],
    predicate: AstNode,
    mapper: AstNode,
    score: ScoredProgram,
    feedback: dict[str, object],
    hole: str,
    constants: tuple[int, ...],
) -> dict[str, object]:
    document = {
        "task": (
            f"Generate exactly {shortlist_size} distinct grammar-valid {hole} repairs likely "
            "to reduce the interpreter loss. Do not change the other hole."
        ),
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
        "grammar": _grammar_text(hole, constants),
        "requirements": [
            "Return canonical DSL strings only.",
            "Return distinct expressions.",
            "Every expression must be inside the stated grammar.",
            "Order candidates from most to least promising.",
        ],
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": shortlist_size,
                "maxItems": shortlist_size,
                "items": {"type": "string", "minLength": 1, "maxLength": 120},
            }
        },
    }
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Generate a small semantic repair shortlist from concrete execution "
                    "feedback. Never invent new operators or constants. Return only the "
                    "required JSON object in the final answer."
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
                "name": f"generated_{hole}_shortlist",
                "strict": True,
                "schema": schema,
            },
        },
    }


async def _call_shortlist(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, object],
    stage_dir: Path,
    catalog: dict[str, AstNode],
    expected_size: int,
) -> tuple[str, ...]:
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
        elapsed = time.perf_counter() - started
        (stage_dir / "response.json").write_bytes(response.content)
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("provider returned the wrong number of choices")
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
            raise ValueError(f"finish_reason={choice.get('finish_reason')}")
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError("provider returned no final JSON content")
        decoded = json.loads(content)
        if not isinstance(decoded, dict) or set(decoded) != {"candidates"}:
            raise ValueError("final JSON must contain exactly candidates")
        shortlist, validation = validate_generated_shortlist(
            decoded["candidates"],
            catalog,
            expected_size=expected_size,
        )
        result: dict[str, object] = {
            "status": "valid" if validation["complete"] else "invalid-shortlist",
            "elapsed_seconds": elapsed,
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage"),
            "validation": validation,
        }
    except Exception as error:
        shortlist = ()
        result = {
            "status": "invalid-response",
            "error_type": type(error).__name__,
            "detail": str(error),
            "elapsed_seconds": time.perf_counter() - started,
        }
    _write_json(stage_dir / "result.json", result)
    if result["status"] != "valid":
        raise RuntimeError(f"invalid generated shortlist at {stage_dir}: {result['status']}")
    return shortlist


def _provider_inventory(output: Path) -> dict[str, object]:
    records = []
    for path in sorted(output.glob("provider/**/*.json")):
        payload = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    digest = hashlib.sha256(_canonical_bytes(records)).hexdigest()
    return {"records": records, "inventory_sha256": digest}


def _exact_mass(
    exact_paths: tuple[tuple[str, str], ...],
    mapper_shortlist: tuple[str, ...],
    predicate_shortlists: dict[str, tuple[str, ...]],
    *,
    mapper_catalog_size: int,
    predicate_catalog_size: int,
    epsilon: float,
) -> float:
    mass = 0.0
    for mapper, predicate in exact_paths:
        q_mapper = smoothed_shortlist_probability(
            mapper,
            mapper_shortlist,
            catalog_size=mapper_catalog_size,
            epsilon=epsilon,
        )
        if mapper in predicate_shortlists:
            q_predicate = smoothed_shortlist_probability(
                predicate,
                predicate_shortlists[mapper],
                catalog_size=predicate_catalog_size,
                epsilon=epsilon,
            )
        else:
            q_predicate = 1.0 / predicate_catalog_size
        mass += q_mapper * q_predicate
    return mass


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(probability * len(ordered))))
    return ordered[position]


async def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    config = load_experiment_config(args.task)
    scorer = ProgramScorer(config)
    constants = tuple(config.spec.integer_constants)
    mapper_nodes = arithmetic_expressions("Item", constants)
    predicate_nodes = filter_predicates(constants)
    mapper_catalog = _dsl_catalog(mapper_nodes)
    predicate_catalog = _dsl_catalog(predicate_nodes)
    mapper_order = tuple(render_expression_dsl(node) for node in mapper_nodes)
    predicate_order = tuple(render_expression_dsl(node) for node in predicate_nodes)

    start_rng = random.Random(args.start_seed)
    initial_mapper_position = start_rng.randrange(len(mapper_nodes))
    initial_predicate_position = start_rng.randrange(len(predicate_nodes))
    initial_mapper = clone_program(mapper_nodes[initial_mapper_position])
    initial_predicate = clone_program(predicate_nodes[initial_predicate_position])
    initial_score = scorer.score(_assemble_program(initial_predicate, initial_mapper))
    if not isinstance(initial_score, ScoredProgram):
        raise ValueError(f"initial program was rejected: {initial_score.reason}")
    examples = [example.model_dump(mode="json", by_alias=True) for example in config.spec.examples]

    protocol = {
        "schema": "automatic-generated-shortlist-protocol-v1",
        "task_sha256": hashlib.sha256(args.task.resolve().read_bytes()).hexdigest(),
        "model": args.model,
        "model_revision": MODEL_REVISION,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "shortlist_size": args.shortlist_size,
        "epsilon": args.epsilon,
        "start_seed": args.start_seed,
        "initial_mapper_position": initial_mapper_position,
        "initial_mapper": render_expression_dsl(initial_mapper),
        "initial_predicate_position": initial_predicate_position,
        "initial_predicate": render_expression_dsl(initial_predicate),
        "provider_seed": args.provider_seed,
        "random_baseline_seed": args.random_baseline_seed,
        "random_baseline_trials": args.random_baseline_trials,
        "retry_policy": "none",
        "post_provider_gold_join": True,
        "provider_visible_catalog_sizes": False,
        "provider_visible_full_catalogs": False,
        "benchmark_base_proposal": "uniform over the finite catalog, evaluator-side only",
        "conditional_full_support_law": {
            "mapper": "q_M=(1-epsilon) Uniform(S_M)+epsilon Uniform(C_M)",
            "predicate_for_shortlisted_mapper": (
                "q_P(.|m)=(1-epsilon) Uniform(S_P,m)+epsilon Uniform(C_P)"
            ),
            "predicate_for_nonshortlisted_mapper": "q_P(.|m)=Uniform(C_P)",
            "interpretation": (
                "counterfactual normalized proposal conditional on the sealed generated "
                "shortlists; distinct from exhaustively executing the 16 shortlist leaves"
            ),
        },
        "large_space_generalization": (
            "replace the finite uniform floor with an evaluable normalized grammar sampler; "
            "the provider still generates only K repairs and need not know support size"
        ),
    }
    _write_json(output / "protocol.json", protocol)
    started = time.perf_counter()
    provider_root = output / "provider"
    async with httpx.AsyncClient(timeout=args.timeout_seconds) as client:
        initial_feedback = derive_automatic_feedback(
            config.spec,
            cast(Node, initial_predicate),
            cast(Node, initial_mapper),
        ).to_dict()
        mapper_shortlist = await _call_shortlist(
            client=client,
            base_url=args.base_url,
            payload=_shortlist_payload(
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.provider_seed,
                shortlist_size=args.shortlist_size,
                examples=examples,
                predicate=initial_predicate,
                mapper=initial_mapper,
                score=initial_score,
                feedback=initial_feedback,
                hole="mapper",
                constants=constants,
            ),
            stage_dir=provider_root / "mapper",
            catalog=mapper_catalog,
            expected_size=args.shortlist_size,
        )

        async def predicate_shortlist_for(
            position: int,
            mapper_dsl: str,
        ) -> tuple[str, tuple[str, ...]]:
            mapper = clone_program(mapper_catalog[mapper_dsl])
            score = scorer.score(_assemble_program(initial_predicate, mapper))
            if not isinstance(score, ScoredProgram):
                raise ValueError(f"mapper branch was rejected: {score.reason}")
            feedback = derive_automatic_feedback(
                config.spec,
                cast(Node, initial_predicate),
                cast(Node, mapper),
            ).to_dict()
            shortlist = await _call_shortlist(
                client=client,
                base_url=args.base_url,
                payload=_shortlist_payload(
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed=args.provider_seed + 1 + position,
                    shortlist_size=args.shortlist_size,
                    examples=examples,
                    predicate=initial_predicate,
                    mapper=mapper,
                    score=score,
                    feedback=feedback,
                    hole="predicate",
                    constants=constants,
                ),
                stage_dir=provider_root / f"predicate-given-m{position}",
                catalog=predicate_catalog,
                expected_size=args.shortlist_size,
            )
            return mapper_dsl, shortlist

        predicate_pairs = await asyncio.gather(
            *(
                predicate_shortlist_for(position, mapper_dsl)
                for position, mapper_dsl in enumerate(mapper_shortlist)
            )
        )
    provider_completed = time.perf_counter()
    predicate_shortlists = dict(predicate_pairs)

    inventory = _provider_inventory(output)
    _write_json(output / "provider-seal.json", inventory)

    llm_candidates: list[dict[str, object]] = []
    for mapper_dsl in mapper_shortlist:
        mapper = mapper_catalog[mapper_dsl]
        for predicate_dsl in predicate_shortlists[mapper_dsl]:
            predicate = predicate_catalog[predicate_dsl]
            score = scorer.score(_assemble_program(predicate, mapper))
            if not isinstance(score, ScoredProgram):
                raise ValueError(f"shortlisted program was rejected: {score.reason}")
            llm_candidates.append(
                {
                    "mapper": mapper_dsl,
                    "predicate": predicate_dsl,
                    "score": asdict(score),
                }
            )

    loss_matrix: list[list[float]] = []
    exact_paths: list[tuple[str, str]] = []
    for mapper_position, mapper in enumerate(mapper_nodes):
        row: list[float] = []
        for predicate_position, predicate in enumerate(predicate_nodes):
            score = scorer.score(_assemble_program(predicate, mapper))
            if not isinstance(score, ScoredProgram):
                raise ValueError(f"evaluator program was rejected: {score.reason}")
            row.append(score.total_loss)
            if score.exact_program:
                exact_paths.append(
                    (mapper_order[mapper_position], predicate_order[predicate_position])
                )
        loss_matrix.append(row)
    exact_paths_tuple = tuple(exact_paths)
    llm_exact_mass = _exact_mass(
        exact_paths_tuple,
        mapper_shortlist,
        predicate_shortlists,
        mapper_catalog_size=len(mapper_nodes),
        predicate_catalog_size=len(predicate_nodes),
        epsilon=args.epsilon,
    )
    llm_hit = any(
        cast(dict[str, Any], record["score"])["exact_program"] for record in llm_candidates
    )
    llm_best_loss = min(
        cast(dict[str, Any], record["score"])["total_loss"] for record in llm_candidates
    )

    baseline_rng = random.Random(args.random_baseline_seed)
    baseline_hits = 0
    baseline_best_losses: list[float] = []
    baseline_exact_masses: list[float] = []
    for _ in range(args.random_baseline_trials):
        mapper_positions = tuple(baseline_rng.sample(range(len(mapper_nodes)), args.shortlist_size))
        random_mapper_shortlist = tuple(mapper_order[position] for position in mapper_positions)
        random_predicate_shortlists: dict[str, tuple[str, ...]] = {}
        trial_hit = False
        trial_best_loss = math.inf
        for mapper_position in mapper_positions:
            predicate_positions = tuple(
                baseline_rng.sample(range(len(predicate_nodes)), args.shortlist_size)
            )
            mapper_dsl = mapper_order[mapper_position]
            random_predicate_shortlists[mapper_dsl] = tuple(
                predicate_order[position] for position in predicate_positions
            )
            for predicate_position in predicate_positions:
                loss = loss_matrix[mapper_position][predicate_position]
                trial_best_loss = min(trial_best_loss, loss)
                if loss == 0.0:
                    trial_hit = True
        baseline_hits += int(trial_hit)
        baseline_best_losses.append(trial_best_loss)
        baseline_exact_masses.append(
            _exact_mass(
                exact_paths_tuple,
                random_mapper_shortlist,
                random_predicate_shortlists,
                mapper_catalog_size=len(mapper_nodes),
                predicate_catalog_size=len(predicate_nodes),
                epsilon=args.epsilon,
            )
        )

    uniform_exact_mass = len(exact_paths_tuple) / (len(mapper_nodes) * len(predicate_nodes))
    final_candidate_executions = len(llm_candidates)
    feedback_state_executions = 1 + len(mapper_shortlist)
    total_execution_budget = final_candidate_executions + feedback_state_executions
    posthoc_oracle_evaluations = len(mapper_nodes) * len(predicate_nodes)

    def hit_probability(mass: float, draws: int) -> float:
        return 1.0 - (1.0 - mass) ** draws

    def n50(mass: float) -> int:
        return math.ceil(math.log(0.5) / math.log1p(-mass))

    result = {
        "schema": "automatic-generated-shortlist-experiment-v1",
        "claim_scope": (
            "one controlled proposal-quality pilot; provider shortlists were generated "
            "without target labels, then sealed before evaluator-only exhaustive analysis"
        ),
        "protocol": protocol,
        "provider_inventory_sha256": inventory["inventory_sha256"],
        "mapper_shortlist": mapper_shortlist,
        "predicate_shortlists": predicate_shortlists,
        "llm_candidates": llm_candidates,
        "evaluator": {
            "catalog_programs": len(mapper_nodes) * len(predicate_nodes),
            "exact_program_count": len(exact_paths_tuple),
            "exact_paths": exact_paths_tuple,
            "note": "exhaustive execution occurred only after the provider artifact was sealed",
        },
        "execution_budget": {
            "proposal_feedback_state_executions": feedback_state_executions,
            "leaf_search_executions": final_candidate_executions,
            "measured_search_executions_including_feedback": total_execution_budget,
            "posthoc_oracle_evaluations_not_part_of_search": posthoc_oracle_evaluations,
        },
        "llm_shortlist_metrics": {
            "contains_exact_program": llm_hit,
            "best_loss": llm_best_loss,
            "conditional_full_support_exact_mass": llm_exact_mass,
            "counterfactual_iid_hit_probability_at_16_leaf_budget": hit_probability(
                llm_exact_mass, final_candidate_executions
            ),
            "counterfactual_iid_n50_draws": n50(llm_exact_mass),
        },
        "uniform_full_grammar": {
            "exact_mass_per_draw": uniform_exact_mass,
            "iid_hit_probability_at_16_leaf_budget": hit_probability(
                uniform_exact_mass, final_candidate_executions
            ),
            "iid_hit_probability_at_21_total_search_execution_budget": hit_probability(
                uniform_exact_mass, total_execution_budget
            ),
            "iid_n50_draws": n50(uniform_exact_mass),
        },
        "random_shortlist_baseline": {
            "trials": args.random_baseline_trials,
            "exact_shortlist_hits": baseline_hits,
            "exact_shortlist_hit_rate": baseline_hits / args.random_baseline_trials,
            "best_loss_mean": sum(baseline_best_losses) / len(baseline_best_losses),
            "best_loss_median": median(baseline_best_losses),
            "best_loss_p05": _quantile(baseline_best_losses, 0.05),
            "fraction_best_loss_at_most_llm": sum(
                loss <= llm_best_loss for loss in baseline_best_losses
            )
            / len(baseline_best_losses),
            "full_support_exact_mass_mean": sum(baseline_exact_masses) / len(baseline_exact_masses),
            "full_support_exact_mass_p95": _quantile(baseline_exact_masses, 0.95),
            "fraction_exact_mass_at_least_llm": sum(
                mass >= llm_exact_mass for mass in baseline_exact_masses
            )
            / len(baseline_exact_masses),
        },
        "timing": {
            "provider_generation_seconds": provider_completed - started,
            "posthoc_evaluator_and_baseline_seconds": time.perf_counter() - provider_completed,
            "total_seconds": time.perf_counter() - started,
        },
    }
    _write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "initial_program": {
                    "mapper": protocol["initial_mapper"],
                    "predicate": protocol["initial_predicate"],
                },
                "mapper_shortlist": mapper_shortlist,
                "predicate_shortlists": predicate_shortlists,
                "llm_shortlist_metrics": result["llm_shortlist_metrics"],
                "random_shortlist_baseline": result["random_shortlist_baseline"],
                "uniform_full_grammar": result["uniform_full_grammar"],
                "execution_budget": result["execution_budget"],
                "timing": result["timing"],
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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--shortlist-size", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--start-seed", type=int, default=17)
    parser.add_argument("--provider-seed", type=int, default=82000)
    parser.add_argument("--random-baseline-seed", type=int, default=91000)
    parser.add_argument("--random-baseline-trials", type=int, default=10000)
    args = parser.parse_args()
    if args.shortlist_size < 1:
        parser.error("--shortlist-size must be positive")
    if args.shortlist_size > 4:
        parser.error("this initial protocol supports shortlist sizes up to four")
    if args.random_baseline_trials < 1:
        parser.error("--random-baseline-trials must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

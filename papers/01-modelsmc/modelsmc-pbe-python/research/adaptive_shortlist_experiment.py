"""Run an evidence-ordered LLM-generated shortlist-tree pilot."""

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
from research.automatic_shortlist_experiment import (
    _provider_inventory,
    _quantile,
    _shortlist_payload,
    validate_generated_shortlist,
)
from research.direct_json_repair_choice import _canonical_bytes, _write_json
from research.execution_guided_repair import (
    MODEL_REVISION,
    _assemble_program,
    _dsl_catalog,
    render_expression_dsl,
)


def select_first_hole(feedback: dict[str, object]) -> tuple[str, dict[str, object]]:
    """Choose a repair order from mechanically available evidence only."""

    toggles = feedback.get("predicate_toggles")
    constraints = feedback.get("mapper_constraints")
    if not isinstance(toggles, (list, tuple)) or not isinstance(constraints, (list, tuple)):
        raise ValueError("feedback has the wrong evidence fields")
    if toggles:
        selected = "predicate"
        reason = "predicate counterfactual evidence is available"
    elif constraints:
        selected = "mapper"
        reason = "mapper constraints are available and no predicate toggle is available"
    else:
        raise ValueError("feedback supports neither hole; the trial is stalled")
    return selected, {
        "policy": (
            "predicate first when any predicate toggle exists; otherwise mapper first when "
            "any mapper constraint exists; otherwise declare stall"
        ),
        "predicate_toggle_count": len(toggles),
        "mapper_constraint_count": len(constraints),
        "selected_first_hole": selected,
        "reason": reason,
    }


def top_level_mixture_probability(
    in_shortlist_tree: bool,
    *,
    tree_size: int,
    base_probability: float,
    epsilon: float,
) -> float:
    """Return (1-epsilon) Uniform(tree) + epsilon q_base for one program."""

    if tree_size < 1:
        raise ValueError("tree_size must be positive")
    if not math.isfinite(base_probability) or not 0.0 <= base_probability <= 1.0:
        raise ValueError("base_probability must be finite and in [0, 1]")
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be finite and in (0, 1]")
    return epsilon * base_probability + ((1.0 - epsilon) / tree_size if in_shortlist_tree else 0.0)


_MAPPER_TEMPLATES = (
    "item",
    "constant",
    "add(item,item)",
    "sub(item,item)",
    "mul(item,item)",
    "add(item,c)",
    "sub(item,c)",
    "mul(item,c)",
    "add(c,item)",
    "sub(c,item)",
    "mul(c,item)",
)
_ATOM_TEMPLATES = ("lt(item,c)", "lt(c,item)", "eq(item,c)")


def decode_structured_candidate(hole: str, candidate: object) -> str:
    """Map one fixed-shape grammar object to the canonical DSL."""

    if not isinstance(candidate, dict):
        raise ValueError("structured candidate must be an object")
    if hole == "mapper":
        if set(candidate) != {"template", "constant"}:
            raise ValueError("mapper candidate has the wrong fields")
        template = candidate["template"]
        constant = candidate["constant"]
        if template not in _MAPPER_TEMPLATES or not isinstance(constant, str):
            raise ValueError("mapper candidate has invalid field values")
        if template == "item":
            return "item"
        if template == "constant":
            return constant
        return cast(str, template).replace("c", constant)
    if hole != "predicate":
        raise ValueError(f"unsupported hole: {hole}")
    if set(candidate) != {"combine", "left", "right"}:
        raise ValueError("predicate candidate has the wrong fields")

    def atom(value: object) -> str:
        if not isinstance(value, dict) or set(value) != {"template", "constant"}:
            raise ValueError("predicate atom has the wrong fields")
        template = value["template"]
        constant = value["constant"]
        if template not in _ATOM_TEMPLATES or not isinstance(constant, str):
            raise ValueError("predicate atom has invalid field values")
        return cast(str, template).replace("c", constant)

    combine = candidate["combine"]
    left = atom(candidate["left"])
    right = atom(candidate["right"])
    if combine == "single":
        return left
    if combine == "and":
        return f"and({left},{right})"
    raise ValueError("predicate combine must be single or and")


def _structured_payload(
    payload: dict[str, object],
    *,
    hole: str,
    constants: tuple[int, ...],
    shortlist_size: int,
) -> dict[str, object]:
    """Replace free-form DSL strings with a compact typed grammar encoding."""

    constant_values = [str(value) for value in constants]
    messages = cast(list[dict[str, str]], payload["messages"])
    document = json.loads(messages[1]["content"])
    if hole == "mapper":
        candidate_schema: dict[str, object] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["template", "constant"],
            "properties": {
                "template": {"type": "string", "enum": list(_MAPPER_TEMPLATES)},
                "constant": {"type": "string", "enum": constant_values},
            },
        }
        document["typed_output_encoding"] = {
            "template": list(_MAPPER_TEMPLATES),
            "constant": constant_values,
            "note": "constant is ignored by templates that do not contain c",
        }
    else:
        atom_schema: dict[str, object] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["template", "constant"],
            "properties": {
                "template": {"type": "string", "enum": list(_ATOM_TEMPLATES)},
                "constant": {"type": "string", "enum": constant_values},
            },
        }
        candidate_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["combine", "left", "right"],
            "properties": {
                "combine": {"type": "string", "enum": ["single", "and"]},
                "left": atom_schema,
                "right": atom_schema,
            },
        }
        document["typed_output_encoding"] = {
            "combine": ["single", "and"],
            "atom_template": list(_ATOM_TEMPLATES),
            "constant": constant_values,
            "note": "right is ignored when combine is single",
        }
    document["requirements"] = [
        "Return exactly four distinct expressions through the typed encoding.",
        "Order candidates from most to least promising.",
        "Vary semantic expressions, not ignored fields.",
    ]
    messages[1]["content"] = _canonical_bytes(document).decode()
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": shortlist_size,
                "maxItems": shortlist_size,
                "items": candidate_schema,
            }
        },
    }
    cast(dict[str, Any], payload["response_format"])["json_schema"]["schema"] = schema
    return payload


async def _call_structured_shortlist(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, object],
    stage_dir: Path,
    catalog: dict[str, AstNode],
    expected_size: int,
    hole: str,
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
        raw_candidates = decoded["candidates"]
        if not isinstance(raw_candidates, list):
            raise ValueError("candidates must be an array")
        rendered = [decode_structured_candidate(hole, candidate) for candidate in raw_candidates]
        shortlist, validation = validate_generated_shortlist(
            rendered,
            catalog,
            expected_size=expected_size,
        )
        validation["raw_structured_candidates"] = raw_candidates
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


def _hit_probability(mass: float, draws: int) -> float:
    return 1.0 - (1.0 - mass) ** draws


def _n50(mass: float) -> int:
    return math.ceil(math.log(0.5) / math.log1p(-mass))


def _tree_exact_mass(
    exact_paths: tuple[tuple[str, str], ...],
    leaves: frozenset[tuple[str, str]],
    *,
    full_program_count: int,
    epsilon: float,
) -> float:
    return sum(
        top_level_mixture_probability(
            path in leaves,
            tree_size=len(leaves),
            base_probability=1.0 / full_program_count,
            epsilon=epsilon,
        )
        for path in exact_paths
    )


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
    catalogs = {"mapper": mapper_catalog, "predicate": predicate_catalog}
    orders = {"mapper": mapper_order, "predicate": predicate_order}

    start_rng = random.Random(args.start_seed)
    initial_mapper_position = start_rng.randrange(len(mapper_nodes))
    initial_predicate_position = start_rng.randrange(len(predicate_nodes))
    initial_mapper = clone_program(mapper_nodes[initial_mapper_position])
    initial_predicate = clone_program(predicate_nodes[initial_predicate_position])
    initial_score = scorer.score(_assemble_program(initial_predicate, initial_mapper))
    if not isinstance(initial_score, ScoredProgram):
        raise ValueError(f"initial program was rejected: {initial_score.reason}")
    examples = [example.model_dump(mode="json", by_alias=True) for example in config.spec.examples]
    initial_feedback = derive_automatic_feedback(
        config.spec,
        cast(Node, initial_predicate),
        cast(Node, initial_mapper),
    ).to_dict()
    first_hole, order_decision = select_first_hole(initial_feedback)
    second_hole = "mapper" if first_hole == "predicate" else "predicate"

    protocol = {
        "schema": "adaptive-generated-typed-shortlist-protocol-v2",
        "task_sha256": hashlib.sha256(args.task.resolve().read_bytes()).hexdigest(),
        "model": args.model,
        "model_revision": MODEL_REVISION,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "shortlist_size": args.shortlist_size,
        "epsilon": args.epsilon,
        "proposal_law": (
            "q(z|T)=(1-epsilon) Uniform(T)+epsilon q_base(z), where T is the sealed "
            "K-by-K shortlist tree and q_base is evaluator-side Uniform over this finite "
            "benchmark grammar"
        ),
        "large_space_generalization": (
            "replace finite Uniform with any normalized grammar sampler whose sampled-program "
            "probability is evaluable; neither support enumeration nor its cardinality is "
            "shown to or required by the LLM"
        ),
        "provider_visible_catalog_sizes": False,
        "provider_visible_full_catalogs": False,
        "order_decision": order_decision,
        "start_seed": args.start_seed,
        "initial_mapper_position": initial_mapper_position,
        "initial_mapper": render_expression_dsl(initial_mapper),
        "initial_predicate_position": initial_predicate_position,
        "initial_predicate": render_expression_dsl(initial_predicate),
        "provider_seed": args.provider_seed,
        "random_baseline_seed": args.random_baseline_seed,
        "random_baseline_trials": args.random_baseline_trials,
        "retry_policy": "none",
        "max_provider_concurrency": args.max_concurrency,
        "output_encoding": (
            "fixed-shape typed grammar objects with enum-constrained operators and declared "
            "constants; the decoder never receives the enumerated expression catalog"
        ),
        "invalid_or_duplicate_policy": "invalidate trial; never repair or backfill",
        "post_provider_gold_join": True,
    }
    _write_json(output / "protocol.json", protocol)

    def payload_for(
        *,
        predicate: AstNode,
        mapper: AstNode,
        hole: str,
        seed: int,
    ) -> dict[str, object]:
        score = scorer.score(_assemble_program(predicate, mapper))
        if not isinstance(score, ScoredProgram):
            raise ValueError(f"feedback program was rejected: {score.reason}")
        feedback = derive_automatic_feedback(
            config.spec,
            cast(Node, predicate),
            cast(Node, mapper),
        ).to_dict()
        return _structured_payload(
            _shortlist_payload(
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=seed,
                shortlist_size=args.shortlist_size,
                examples=examples,
                predicate=predicate,
                mapper=mapper,
                score=score,
                feedback=feedback,
                hole=hole,
                constants=constants,
            ),
            hole=hole,
            constants=constants,
            shortlist_size=args.shortlist_size,
        )

    started = time.perf_counter()
    provider_root = output / "provider"
    async with httpx.AsyncClient(timeout=args.timeout_seconds) as client:
        first_shortlist = await _call_structured_shortlist(
            client=client,
            base_url=args.base_url,
            payload=payload_for(
                predicate=initial_predicate,
                mapper=initial_mapper,
                hole=first_hole,
                seed=args.provider_seed,
            ),
            stage_dir=provider_root / f"first-{first_hole}",
            catalog=catalogs[first_hole],
            expected_size=args.shortlist_size,
            hole=first_hole,
        )

        provider_semaphore = asyncio.Semaphore(args.max_concurrency)

        async def second_shortlist_for(
            position: int,
            first_dsl: str,
        ) -> tuple[str, tuple[str, ...]]:
            predicate = (
                clone_program(predicate_catalog[first_dsl])
                if first_hole == "predicate"
                else clone_program(initial_predicate)
            )
            mapper = (
                clone_program(mapper_catalog[first_dsl])
                if first_hole == "mapper"
                else clone_program(initial_mapper)
            )
            async with provider_semaphore:
                shortlist = await _call_structured_shortlist(
                    client=client,
                    base_url=args.base_url,
                    payload=payload_for(
                        predicate=predicate,
                        mapper=mapper,
                        hole=second_hole,
                        seed=args.provider_seed + 1 + position,
                    ),
                    stage_dir=provider_root / f"second-{second_hole}-given-{first_hole}-{position}",
                    catalog=catalogs[second_hole],
                    expected_size=args.shortlist_size,
                    hole=second_hole,
                )
            return first_dsl, shortlist

        second_pairs = await asyncio.gather(
            *(
                second_shortlist_for(position, first_dsl)
                for position, first_dsl in enumerate(first_shortlist)
            )
        )
    provider_completed = time.perf_counter()
    second_shortlists = dict(second_pairs)
    inventory = _provider_inventory(output)
    _write_json(output / "provider-seal.json", inventory)

    leaves: list[dict[str, object]] = []
    leaf_paths: set[tuple[str, str]] = set()
    for first_dsl in first_shortlist:
        for second_dsl in second_shortlists[first_dsl]:
            predicate_dsl = first_dsl if first_hole == "predicate" else second_dsl
            mapper_dsl = first_dsl if first_hole == "mapper" else second_dsl
            path = (mapper_dsl, predicate_dsl)
            if path in leaf_paths:
                raise ValueError("shortlist tree contains a duplicate complete program")
            leaf_paths.add(path)
            score = scorer.score(
                _assemble_program(predicate_catalog[predicate_dsl], mapper_catalog[mapper_dsl])
            )
            if not isinstance(score, ScoredProgram):
                raise ValueError(f"shortlist leaf was rejected: {score.reason}")
            leaves.append(
                {
                    "mapper": mapper_dsl,
                    "predicate": predicate_dsl,
                    "score": asdict(score),
                }
            )
    frozen_leaf_paths = frozenset(leaf_paths)

    loss_matrix: list[list[float]] = []
    exact_paths: list[tuple[str, str]] = []
    for mapper_position, mapper in enumerate(mapper_nodes):
        row = []
        for predicate_position, predicate in enumerate(predicate_nodes):
            score = scorer.score(_assemble_program(predicate, mapper))
            if not isinstance(score, ScoredProgram):
                raise ValueError(f"posthoc evaluator program was rejected: {score.reason}")
            row.append(score.total_loss)
            if score.exact_program:
                exact_paths.append(
                    (mapper_order[mapper_position], predicate_order[predicate_position])
                )
        loss_matrix.append(row)
    exact_paths_tuple = tuple(exact_paths)
    full_program_count = len(mapper_nodes) * len(predicate_nodes)
    exact_mass = _tree_exact_mass(
        exact_paths_tuple,
        frozen_leaf_paths,
        full_program_count=full_program_count,
        epsilon=args.epsilon,
    )
    contains_exact = any(cast(dict[str, Any], leaf["score"])["exact_program"] for leaf in leaves)
    best_loss = min(cast(dict[str, Any], leaf["score"])["total_loss"] for leaf in leaves)

    baseline_rng = random.Random(args.random_baseline_seed)
    baseline_hits = 0
    baseline_best_losses: list[float] = []
    baseline_exact_masses: list[float] = []
    for _ in range(args.random_baseline_trials):
        first_positions = tuple(
            baseline_rng.sample(range(len(orders[first_hole])), args.shortlist_size)
        )
        random_leaves: set[tuple[str, str]] = set()
        trial_best_loss = math.inf
        trial_hit = False
        for first_position in first_positions:
            second_positions = tuple(
                baseline_rng.sample(range(len(orders[second_hole])), args.shortlist_size)
            )
            for second_position in second_positions:
                predicate_position = (
                    first_position if first_hole == "predicate" else second_position
                )
                mapper_position = first_position if first_hole == "mapper" else second_position
                path = (mapper_order[mapper_position], predicate_order[predicate_position])
                random_leaves.add(path)
                loss = loss_matrix[mapper_position][predicate_position]
                trial_best_loss = min(trial_best_loss, loss)
                trial_hit = trial_hit or loss == 0.0
        if len(random_leaves) != args.shortlist_size**2:
            raise ValueError("random shortlist tree unexpectedly contains duplicate leaves")
        baseline_hits += int(trial_hit)
        baseline_best_losses.append(trial_best_loss)
        baseline_exact_masses.append(
            _tree_exact_mass(
                exact_paths_tuple,
                frozenset(random_leaves),
                full_program_count=full_program_count,
                epsilon=args.epsilon,
            )
        )

    leaf_budget = len(leaves)
    feedback_budget = 1 + len(first_shortlist)
    uniform_exact_mass = len(exact_paths_tuple) / full_program_count
    result = {
        "schema": "adaptive-generated-shortlist-experiment-v1",
        "claim_scope": (
            "one controlled feasibility pilot; shortlists were generated without target "
            "labels and sealed before exhaustive evaluator-only analysis"
        ),
        "protocol": protocol,
        "provider_inventory_sha256": inventory["inventory_sha256"],
        "first_hole": first_hole,
        "second_hole": second_hole,
        "first_shortlist": first_shortlist,
        "second_shortlists": second_shortlists,
        "leaves": leaves,
        "execution_budget": {
            "proposal_feedback_state_executions": feedback_budget,
            "leaf_search_executions": leaf_budget,
            "measured_search_executions_including_feedback": feedback_budget + leaf_budget,
            "posthoc_oracle_evaluations_not_part_of_search": full_program_count,
        },
        "evaluator": {
            "catalog_programs": full_program_count,
            "exact_program_count": len(exact_paths_tuple),
            "exact_paths": exact_paths_tuple,
            "note": "exact labels joined only after provider-seal.json was written",
        },
        "llm_shortlist_metrics": {
            "contains_exact_program": contains_exact,
            "best_loss": best_loss,
            "conditional_full_support_exact_mass": exact_mass,
            "counterfactual_iid_hit_probability_at_leaf_budget": _hit_probability(
                exact_mass, leaf_budget
            ),
            "counterfactual_iid_n50_draws": _n50(exact_mass),
        },
        "uniform_full_grammar": {
            "exact_mass_per_draw": uniform_exact_mass,
            "iid_hit_probability_at_leaf_budget": _hit_probability(uniform_exact_mass, leaf_budget),
            "iid_hit_probability_at_total_search_execution_budget": _hit_probability(
                uniform_exact_mass, feedback_budget + leaf_budget
            ),
            "iid_n50_draws": _n50(uniform_exact_mass),
        },
        "random_shortlist_baseline": {
            "trials": args.random_baseline_trials,
            "exact_shortlist_hits": baseline_hits,
            "exact_shortlist_hit_rate": baseline_hits / args.random_baseline_trials,
            "best_loss_mean": sum(baseline_best_losses) / len(baseline_best_losses),
            "best_loss_median": median(baseline_best_losses),
            "best_loss_p05": _quantile(baseline_best_losses, 0.05),
            "fraction_best_loss_at_most_llm": sum(
                loss <= best_loss for loss in baseline_best_losses
            )
            / len(baseline_best_losses),
            "conditional_full_support_exact_mass_mean": sum(baseline_exact_masses)
            / len(baseline_exact_masses),
            "conditional_full_support_exact_mass_p95": _quantile(baseline_exact_masses, 0.95),
            "fraction_exact_mass_at_least_llm": sum(
                mass >= exact_mass for mass in baseline_exact_masses
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
                "order_decision": order_decision,
                "first_shortlist": first_shortlist,
                "second_shortlists": second_shortlists,
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
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--timeout-seconds", type=float, default=420.0)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--shortlist-size", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--start-seed", type=int, default=17)
    parser.add_argument("--provider-seed", type=int, default=83000)
    parser.add_argument("--random-baseline-seed", type=int, default=92000)
    parser.add_argument("--random-baseline-trials", type=int, default=10000)
    args = parser.parse_args()
    if args.shortlist_size < 1:
        parser.error("--shortlist-size must be positive")
    if args.shortlist_size > 4:
        parser.error("this initial protocol supports shortlist sizes up to four")
    if args.random_baseline_trials < 1:
        parser.error("--random-baseline-trials must be positive")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

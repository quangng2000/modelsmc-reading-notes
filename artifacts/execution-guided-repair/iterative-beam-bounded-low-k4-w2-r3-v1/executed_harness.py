"""Run a typed execution-guided LLM beam search and matched random baseline."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, cast

import httpx

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.core.results import ScoredProgram
from modelsmc_pbe.core.scorer import ProgramScorer
from modelsmc_pbe.core.types import Node
from modelsmc_pbe.domain.ast import AstNode, clone_program
from modelsmc_pbe.grammar import arithmetic_expressions, filter_predicates
from research.adaptive_shortlist_experiment import (
    _call_structured_shortlist,
    _structured_payload,
    select_first_hole,
)
from research.automatic_repair_feedback import derive_automatic_feedback
from research.automatic_shortlist_experiment import (
    _provider_inventory,
    _quantile,
    _shortlist_payload,
)
from research.direct_json_repair_choice import _canonical_bytes, _write_json
from research.execution_guided_repair import (
    MODEL_REVISION,
    _assemble_program,
    _dsl_catalog,
    render_expression_dsl,
)


@dataclass(frozen=True, slots=True)
class RepairStep:
    """One executed edge in a beam-search path."""

    round: int
    hole: str
    repair: str
    loss_after: float


@dataclass(frozen=True, slots=True)
class BeamState:
    """One complete, executable filter-map program in the beam."""

    predicate: str
    mapper: str
    score: ScoredProgram
    history: tuple[RepairStep, ...] = ()


def state_key(state: BeamState) -> str:
    """Return a stable opaque identifier for one complete program state."""

    material = f"{state.predicate}\0{state.mapper}".encode()
    return hashlib.sha256(material).hexdigest()[:16]


def beam_rank_key(state: BeamState, *, tie_seed: int) -> tuple[float, str]:
    """Rank without any hidden target or catalog-position information."""

    tie = hashlib.sha256(
        f"{tie_seed}\0{state.predicate}\0{state.mapper}".encode()
    ).hexdigest()
    return (state.score.total_loss, tie)


def select_beam(
    candidates: list[BeamState],
    *,
    width: int,
    tie_seed: int,
) -> tuple[BeamState, ...]:
    """Deduplicate complete programs and retain the best fixed-width beam."""

    if width < 1:
        raise ValueError("beam width must be positive")
    unique: dict[tuple[str, str], BeamState] = {}
    for candidate in candidates:
        unique.setdefault((candidate.predicate, candidate.mapper), candidate)
    ranked = sorted(unique.values(), key=lambda state: beam_rank_key(state, tie_seed=tie_seed))
    return tuple(ranked[:width])


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a binomial success rate."""

    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("successes and trials are inconsistent")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _state_record(state: BeamState) -> dict[str, object]:
    return {
        "state_id": state_key(state),
        "predicate": state.predicate,
        "mapper": state.mapper,
        "score": asdict(state.score),
        "history": [asdict(step) for step in state.history],
    }


def _evidence_summary(feedback: dict[str, object], hole: str) -> dict[str, object]:
    """Make the semantics of mechanical evidence explicit without adding oracle facts."""

    toggles = feedback.get("predicate_toggles", ())
    constraints = feedback.get("mapper_constraints", ())
    if not isinstance(toggles, (list, tuple)) or not isinstance(constraints, (list, tuple)):
        raise ValueError("feedback evidence has the wrong shape")
    if hole == "predicate":
        return {
            "repair_semantics": (
                "For each supported item decision below, a proposed predicate should return "
                "the stated Boolean on that item. lt(a,b) means a is strictly less than b. "
                "These are local counterfactual supports; execution remains authoritative."
            ),
            "supported_item_decisions": [
                {
                    "item": cast(dict[str, object], toggle)["item"],
                    "predicate_should_keep": cast(dict[str, object], toggle)["keep"],
                }
                for toggle in toggles
            ],
        }
    return {
        "repair_semantics": (
            "Conditional on freezing the current predicate, a proposed mapper should map "
            "each listed retained item to its required output value. Execution remains "
            "authoritative."
        ),
        "required_item_outputs": [
            {
                "item": cast(dict[str, object], constraint)["item"],
                "mapper_should_return": cast(dict[str, object], constraint)["expected_value"],
            }
            for constraint in constraints
        ],
    }


def _augment_beam_prompt(
    payload: dict[str, object],
    *,
    round_number: int,
    max_rounds: int,
    beam_rank: int,
    state: BeamState,
    feedback: dict[str, object],
    hole: str,
) -> dict[str, object]:
    messages = cast(list[dict[str, str]], payload["messages"])
    document = json.loads(messages[1]["content"])
    document["beam_search_context"] = {
        "round": round_number,
        "maximum_rounds": max_rounds,
        "beam_rank": beam_rank,
        "current_loss": state.score.total_loss,
        "previous_repairs_on_this_path": [asdict(step) for step in state.history],
        "selection_rule": (
            "All proposed repairs are executed; the application retains states with the "
            "lowest interpreter loss."
        ),
    }
    document["mechanical_evidence_interpretation"] = _evidence_summary(feedback, hole)
    current_expression = state.predicate if hole == "predicate" else state.mapper
    document["forbidden_no_op_expression"] = current_expression
    document["requirements"] = [
        "Return exactly four distinct typed grammar expressions.",
        "Do not return the current expression; every candidate must change the selected hole.",
        "Use the mechanical evidence literally and respect operand direction.",
        "Prefer repairs likely to reduce complete-program execution loss.",
        "Order alternatives from most to least promising.",
    ]
    messages[1]["content"] = _canonical_bytes(document).decode()
    return payload


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
    catalogs = {"mapper": mapper_catalog, "predicate": predicate_catalog}
    orders = {
        "mapper": tuple(render_expression_dsl(node) for node in mapper_nodes),
        "predicate": tuple(render_expression_dsl(node) for node in predicate_nodes),
    }
    for hole, order in orders.items():
        if len(order) - 1 < args.branching_factor:
            raise ValueError(
                f"{hole} grammar cannot provide K distinct repairs after excluding the no-op"
            )
    examples = [example.model_dump(mode="json", by_alias=True) for example in config.spec.examples]

    score_cache: dict[tuple[str, str], ScoredProgram] = {}
    feedback_cache: dict[tuple[str, str], dict[str, object]] = {}

    def score_pair(predicate: str, mapper: str) -> ScoredProgram:
        key = (predicate, mapper)
        existing = score_cache.get(key)
        if existing is not None:
            return existing
        score = scorer.score(
            _assemble_program(predicate_catalog[predicate], mapper_catalog[mapper])
        )
        if not isinstance(score, ScoredProgram):
            raise ValueError(f"grammar-valid beam state was rejected: {score.reason}")
        score_cache[key] = score
        return score

    def feedback_for(state: BeamState) -> dict[str, object]:
        key = (state.predicate, state.mapper)
        existing = feedback_cache.get(key)
        if existing is not None:
            return existing
        feedback = derive_automatic_feedback(
            config.spec,
            cast(Node, predicate_catalog[state.predicate]),
            cast(Node, mapper_catalog[state.mapper]),
        ).to_dict()
        feedback_cache[key] = feedback
        return feedback

    start_rng = random.Random(args.start_seed)
    initial_mapper = orders["mapper"][start_rng.randrange(len(orders["mapper"]))]
    initial_predicate = orders["predicate"][start_rng.randrange(len(orders["predicate"]))]
    initial = BeamState(
        predicate=initial_predicate,
        mapper=initial_mapper,
        score=score_pair(initial_predicate, initial_mapper),
    )
    protocol = {
        "schema": "iterative-typed-llm-beam-protocol-v1",
        "task_sha256": hashlib.sha256(args.task.resolve().read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": args.model,
        "model_revision": MODEL_REVISION,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "rounds": args.rounds,
        "beam_width": args.beam_width,
        "branching_factor": args.branching_factor,
        "maximum_proposal_slot_budget": (
            1
            + args.branching_factor
            + (args.rounds - 1) * args.beam_width * args.branching_factor
        ),
        "ranking": "ascending total training loss, then SHA256(tie_seed,predicate,mapper)",
        "beam_transition": (
            "elitist pool of parents plus valid children; deduplicate complete programs and "
            "retain the lowest-loss fixed-width beam"
        ),
        "hole_policy": (
            "predicate when any predicate-toggle support exists; otherwise mapper when any "
            "conditional mapper constraint exists; otherwise state stalls"
        ),
        "start_seed": args.start_seed,
        "initial_state": _state_record(initial),
        "provider_seed": args.provider_seed,
        "tie_seed": args.tie_seed,
        "random_baseline_seed": args.random_baseline_seed,
        "random_baseline_trials": args.random_baseline_trials,
        "max_provider_concurrency": args.max_concurrency,
        "retry_policy": "none",
        "invalid_or_duplicate_shortlist_policy": (
            "a semantic invalid, duplicate, or no-op response consumes all K proposal slots; "
            "it produces no children and receives no within-call retry or backfill. Infrastructure "
            "failure aborts and seals the trial"
        ),
        "provider_visible_catalog_sizes": False,
        "provider_visible_full_catalogs": False,
        "output_encoding": "typed grammar objects constrained by operators and constants",
        "claim": "heuristic search benchmark, not importance sampling or posterior inference",
        "large_space_note": (
            "The LLM request generates only K repairs and receives neither component-catalog "
            "cardinalities nor a complete-program list. The finite grammar can still make its "
            "size inferable; the claim is that generation does not require enumeration."
        ),
    }
    _write_json(output / "protocol.json", protocol)

    def provider_payload(
        *,
        state: BeamState,
        hole: str,
        feedback: dict[str, object],
        round_number: int,
        beam_rank: int,
        seed: int,
    ) -> dict[str, object]:
        base = _shortlist_payload(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=seed,
            shortlist_size=args.branching_factor,
            examples=examples,
            predicate=predicate_catalog[state.predicate],
            mapper=mapper_catalog[state.mapper],
            score=state.score,
            feedback=feedback,
            hole=hole,
            constants=constants,
        )
        structured = _structured_payload(
            base,
            hole=hole,
            constants=constants,
            shortlist_size=args.branching_factor,
        )
        return _augment_beam_prompt(
            structured,
            round_number=round_number,
            max_rounds=args.rounds,
            beam_rank=beam_rank,
            state=state,
            feedback=feedback,
            hole=hole,
        )

    started = time.perf_counter()
    beam: tuple[BeamState, ...] = (initial,)
    proposal_slots_consumed = 1
    provider_calls_attempted = 0
    execution_records: list[dict[str, object]] = [
        {
            "state_evaluation_index": 1,
            "proposal_slot": 1,
            "round": 0,
            "role": "initial",
            **_state_record(initial),
        }
    ]
    best_state = initial
    first_exact_slot = 1 if initial.score.exact_program else None
    first_exact_state_evaluation = 1 if initial.score.exact_program else None
    round_records: list[dict[str, object]] = []
    provider_root = output / "provider"
    async with httpx.AsyncClient(timeout=args.timeout_seconds) as client:
        semaphore = asyncio.Semaphore(args.max_concurrency)
        for round_number in range(1, args.rounds + 1):
            if first_exact_slot is not None:
                break
            expansion_specs: list[tuple[int, BeamState, str, dict[str, object]]] = []
            stalled = []
            for beam_rank, state in enumerate(beam):
                feedback = feedback_for(state)
                try:
                    hole, decision = select_first_hole(feedback)
                except ValueError as error:
                    stalled.append({"state_id": state_key(state), "detail": str(error)})
                    continue
                expansion_specs.append((beam_rank, state, hole, decision))
            if not expansion_specs:
                round_records.append(
                    {
                        "round": round_number,
                        "input_beam": [_state_record(state) for state in beam],
                        "stalled": stalled,
                        "children": [],
                        "selected_beam": [_state_record(state) for state in beam],
                    }
                )
                break

            async def request_shortlist(
                specification: tuple[int, BeamState, str, dict[str, object]],
            ) -> tuple[
                int,
                BeamState,
                str,
                dict[str, object],
                tuple[str, ...],
                str,
            ]:
                beam_rank, state, hole, decision = specification
                feedback = feedback_for(state)
                stage_dir = (
                    provider_root
                    / f"round-{round_number:02d}"
                    / f"beam-{beam_rank:02d}-{state_key(state)}-{hole}"
                )
                async with semaphore:
                    try:
                        shortlist = await _call_structured_shortlist(
                            client=client,
                            base_url=args.base_url,
                            payload=provider_payload(
                                state=state,
                                hole=hole,
                                feedback=feedback,
                                round_number=round_number,
                                beam_rank=beam_rank,
                                seed=args.provider_seed + 100 * round_number + beam_rank,
                            ),
                            stage_dir=stage_dir,
                            catalog=catalogs[hole],
                            expected_size=args.branching_factor,
                            hole=hole,
                        )
                    except RuntimeError:
                        result = json.loads((stage_dir / "result.json").read_text())
                        infrastructure_errors = {
                            "ConnectError",
                            "ConnectTimeout",
                            "HTTPStatusError",
                            "NetworkError",
                            "PoolTimeout",
                            "ProtocolError",
                            "ProxyError",
                            "ReadError",
                            "ReadTimeout",
                            "RemoteProtocolError",
                            "TimeoutException",
                            "TooManyRedirects",
                            "TransportError",
                            "WriteError",
                            "WriteTimeout",
                        }
                        if result.get("error_type") in infrastructure_errors:
                            raise
                        return beam_rank, state, hole, decision, (), cast(
                            str, result.get("status", "invalid-response")
                        )
                current_expression = state.predicate if hole == "predicate" else state.mapper
                if current_expression in shortlist:
                    _write_json(
                        stage_dir / "beam-validation.json",
                        {
                            "status": "invalid-no-op",
                            "current_expression": current_expression,
                            "shortlist": shortlist,
                            "policy": "consume K slots; generate no children; never backfill",
                        },
                    )
                    return beam_rank, state, hole, decision, (), "invalid-no-op"
                _write_json(
                    stage_dir / "beam-validation.json",
                    {"status": "valid", "current_expression_excluded": True},
                )
                return beam_rank, state, hole, decision, shortlist, "valid"

            expansions = await asyncio.gather(
                *(request_shortlist(specification) for specification in expansion_specs)
            )
            provider_calls_attempted += len(expansions)
            children: list[BeamState] = []
            expansion_records = []
            for beam_rank, parent, hole, decision, shortlist, validation_status in expansions:
                child_records = []
                slot_start = proposal_slots_consumed + 1
                proposal_slots_consumed += args.branching_factor
                for proposal_rank, repair in enumerate(shortlist):
                    proposal_slot = slot_start + proposal_rank
                    predicate = repair if hole == "predicate" else parent.predicate
                    mapper = repair if hole == "mapper" else parent.mapper
                    score = score_pair(predicate, mapper)
                    child = BeamState(
                        predicate=predicate,
                        mapper=mapper,
                        score=score,
                        history=parent.history
                        + (
                            RepairStep(
                                round=round_number,
                                hole=hole,
                                repair=repair,
                                loss_after=score.total_loss,
                            ),
                        ),
                    )
                    children.append(child)
                    state_evaluation_index = len(execution_records) + 1
                    execution_records.append(
                        {
                            "state_evaluation_index": state_evaluation_index,
                            "proposal_slot": proposal_slot,
                            "round": round_number,
                            "role": "generated-child",
                            "parent_state_id": state_key(parent),
                            "repaired_hole": hole,
                            **_state_record(child),
                        }
                    )
                    child_records.append(_state_record(child))
                    if beam_rank_key(child, tie_seed=args.tie_seed) < beam_rank_key(
                        best_state, tie_seed=args.tie_seed
                    ):
                        best_state = child
                    if score.exact_program and first_exact_slot is None:
                        first_exact_slot = proposal_slot
                        first_exact_state_evaluation = state_evaluation_index
                expansion_records.append(
                    {
                        "beam_rank": beam_rank,
                        "parent": _state_record(parent),
                        "hole_decision": decision,
                        "validation_status": validation_status,
                        "proposal_slots": list(
                            range(slot_start, slot_start + args.branching_factor)
                        ),
                        "shortlist": shortlist,
                        "children": child_records,
                    }
                )
            selected = select_beam(
                [*beam, *children],
                width=args.beam_width,
                tie_seed=args.tie_seed + round_number,
            )
            round_record = {
                "round": round_number,
                "input_beam": [_state_record(state) for state in beam],
                "stalled": stalled,
                "expansions": expansion_records,
                "children_executed": len(children),
                "selected_beam": [_state_record(state) for state in selected],
                "cumulative_best": _state_record(best_state),
                "first_exact_proposal_slot": first_exact_slot,
                "first_exact_state_evaluation": first_exact_state_evaluation,
            }
            round_records.append(round_record)
            _write_json(output / f"round-{round_number:02d}.json", round_record)
            beam = selected
            if first_exact_slot is not None:
                break
    provider_completed = time.perf_counter()
    provider_inventory = _provider_inventory(output)
    _write_json(output / "provider-seal.json", provider_inventory)

    max_budget = cast(int, protocol["maximum_proposal_slot_budget"])

    checkpoints = [
        1,
        1 + args.branching_factor,
        *[
            1 + args.branching_factor + round_index * args.beam_width * args.branching_factor
            for round_index in range(1, args.rounds)
        ],
    ]

    def random_trial(trial_index: int) -> dict[str, object]:
        trial_beam: tuple[BeamState, ...] = (initial,)
        trial_best = initial
        proposal_slots = 1
        state_evaluations = 1
        exact_at = 1 if initial.score.exact_program else None
        rounds_completed = 0
        for round_number in range(1, args.rounds + 1):
            if exact_at is not None:
                break
            trial_children: list[BeamState] = []
            expansion_count = 0
            for parent_rank, state in enumerate(trial_beam):
                try:
                    hole, _ = select_first_hole(feedback_for(state))
                except ValueError:
                    continue
                expansion_count += 1
                current_expression = state.predicate if hole == "predicate" else state.mapper
                available = [
                    expression for expression in orders[hole] if expression != current_expression
                ]
                stream = hashlib.sha256(
                    (
                        f"{args.random_baseline_seed}\0{trial_index}\0{round_number}\0"
                        f"{parent_rank}\0{state_key(state)}\0{hole}"
                    ).encode()
                ).digest()
                rng = random.Random(int.from_bytes(stream[:8], "big"))
                for repair in rng.sample(available, args.branching_factor):
                    proposal_slots += 1
                    predicate = repair if hole == "predicate" else state.predicate
                    mapper = repair if hole == "mapper" else state.mapper
                    score = score_pair(predicate, mapper)
                    child = BeamState(predicate=predicate, mapper=mapper, score=score)
                    trial_children.append(child)
                    state_evaluations += 1
                    if beam_rank_key(child, tie_seed=args.tie_seed) < beam_rank_key(
                        trial_best, tie_seed=args.tie_seed
                    ):
                        trial_best = child
                    if score.exact_program and exact_at is None:
                        exact_at = proposal_slots
            rounds_completed = round_number
            if expansion_count == 0 or exact_at is not None:
                break
            trial_beam = select_beam(
                [*trial_beam, *trial_children],
                width=args.beam_width,
                tie_seed=args.tie_seed + round_number,
            )
        return {
            "success": exact_at is not None,
            "first_exact_execution": exact_at,
            "best_loss": trial_best.score.total_loss,
            "best_exact_examples": trial_best.score.exact_matches,
            "proposal_slots": proposal_slots,
            "state_evaluations": state_evaluations,
            "rounds_completed": rounds_completed,
        }

    baseline_started = time.perf_counter()
    baseline = [random_trial(index) for index in range(args.random_baseline_trials)]
    successes = sum(cast(bool, record["success"]) for record in baseline)
    best_losses = [cast(float, record["best_loss"]) for record in baseline]
    success_executions = [
        cast(int, record["first_exact_execution"])
        for record in baseline
        if record["first_exact_execution"] is not None
    ]
    interval = wilson_interval(successes, args.random_baseline_trials)
    checkpoint_successes = {
        str(checkpoint): sum(
            record["first_exact_execution"] is not None
            and cast(int, record["first_exact_execution"]) <= checkpoint
            for record in baseline
        )
        for checkpoint in checkpoints
    }
    baseline_completed = time.perf_counter()

    unique_search_programs = {
        (cast(str, record["predicate"]), cast(str, record["mapper"]))
        for record in execution_records
    }

    result = {
        "schema": "iterative-typed-llm-beam-experiment-v1",
        "claim_scope": (
            "one exploratory LLM beam trajectory compared with a matched random-beam "
            "simulation; this is not evidence of general speedup without repeated held-out tasks"
        ),
        "protocol": protocol,
        "provider_inventory_sha256": provider_inventory["inventory_sha256"],
        "rounds": round_records,
        "state_evaluations": execution_records,
        "llm_beam_metrics": {
            "success": first_exact_slot is not None,
            "first_exact_proposal_slot": first_exact_slot,
            "first_exact_state_evaluation": first_exact_state_evaluation,
            "proposal_slots_consumed": proposal_slots_consumed,
            "state_evaluations": len(execution_records),
            "unique_programs_evaluated": len(unique_search_programs),
            "provider_calls_attempted": provider_calls_attempted,
            "maximum_budget": max_budget,
            "rounds_completed": len(round_records),
            "best_state": _state_record(best_state),
            "best_loss": best_state.score.total_loss,
            "best_exact_examples": best_state.score.exact_matches,
        },
        "matched_random_beam": {
            "trials": args.random_baseline_trials,
            "successes": successes,
            "success_rate": successes / args.random_baseline_trials,
            "success_rate_wilson_95": interval,
            "successes_by_proposal_slot_checkpoint": checkpoint_successes,
            "best_loss_mean": mean(best_losses),
            "best_loss_median": median(best_losses),
            "best_loss_p05": _quantile(best_losses, 0.05),
            "fraction_best_loss_at_most_llm": sum(
                loss <= best_state.score.total_loss for loss in best_losses
            )
            / len(best_losses),
            "first_exact_execution_median_when_successful": (
                median(success_executions) if success_executions else None
            ),
            "mean_proposal_slots_used": mean(
                cast(int, record["proposal_slots"]) for record in baseline
            ),
            "mean_state_evaluations": mean(
                cast(int, record["state_evaluations"]) for record in baseline
            ),
        },
        "timing": {
            "provider_and_llm_search_seconds": provider_completed - started,
            "matched_random_baseline_seconds": baseline_completed - baseline_started,
            "total_seconds": baseline_completed - started,
        },
        "no_exhaustive_program_enumeration_for_evaluation": True,
    }
    _write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "llm_beam_metrics": result["llm_beam_metrics"],
                "matched_random_beam": result["matched_random_beam"],
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
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--branching-factor", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=17)
    parser.add_argument("--provider-seed", type=int, default=95000)
    parser.add_argument("--tie-seed", type=int, default=95100)
    parser.add_argument("--random-baseline-seed", type=int, default=96000)
    parser.add_argument("--random-baseline-trials", type=int, default=10000)
    args = parser.parse_args()
    for name in ("max_concurrency", "rounds", "beam_width", "branching_factor"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.branching_factor != 4:
        parser.error("the current typed output protocol requires branching factor four")
    if args.random_baseline_trials < 1:
        parser.error("--random-baseline-trials must be positive")
    try:
        asyncio.run(run(args))
    except Exception as error:
        output = args.output.resolve()
        if output.exists():
            inventory = _provider_inventory(output)
            _write_json(output / "provider-seal.json", inventory)
            _write_json(
                output / "failure.json",
                {
                    "status": "aborted",
                    "error_type": type(error).__name__,
                    "detail": str(error),
                    "provider_inventory_sha256": inventory["inventory_sha256"],
                    "policy": "infrastructure failure aborts; no retry or backfill",
                },
            )
        raise


if __name__ == "__main__":
    main()

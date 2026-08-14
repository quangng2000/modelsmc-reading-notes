"""Run a typed execution-guided LLM beam search and matched random baseline."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, median
from typing import cast

import httpx

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.core.evaluate import evaluate_expression
from modelsmc_pbe.core.results import ScoredProgram
from modelsmc_pbe.core.scorer import ProgramScorer
from modelsmc_pbe.core.types import Node
from modelsmc_pbe.domain.models import PBESpec
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

_BLIND_FOUR_RUN_SEEDS = {
    "blind-01": (101000, 101100, 101200),
    "blind-02": (102000, 102100, 102200),
    "blind-03": (103000, 103100, 103200),
    "blind-04": (104000, 104100, 104200),
}


def validate_blind_four_args(args: argparse.Namespace) -> str | None:
    """Reject any blind-pilot invocation that differs from the frozen protocol."""

    if args.protocol_mode is None:
        return None
    if args.protocol_mode != "blind-four-v1":
        raise ValueError(f"unsupported protocol mode: {args.protocol_mode}")
    expected = {
        "model": "gpt-oss-120b",
        "reasoning_effort": "low",
        "temperature": 0.0,
        "max_tokens": 1600,
        "timeout_seconds": 420.0,
        "max_concurrency": 2,
        "rounds": 5,
        "beam_width": 2,
        "branching_factor": 4,
        "start_seed": 17,
        "random_baseline_trials": 10000,
        "selection_policy": "semantic-diverse",
        "stall_policy": "alternate-hole",
        "singleton_evidence": True,
        "primary_checkpoint_round": 4,
    }
    mismatches = {
        name: {"expected": value, "actual": getattr(args, name)}
        for name, value in expected.items()
        if getattr(args, name) != value
    }
    if mismatches:
        raise ValueError(f"blind-four-v1 setting mismatch: {mismatches}")
    if args.blind_manifest is None:
        raise ValueError("blind-four-v1 requires --blind-manifest")
    if args.study_protocol is None:
        raise ValueError("blind-four-v1 requires --study-protocol")
    study_protocol = json.loads(
        args.study_protocol.expanduser().resolve().read_text(encoding="utf-8")
    )
    if not isinstance(study_protocol, dict) or any(
        (
            study_protocol.get("protocol_status") != "frozen",
            study_protocol.get("frozen_before_task_generation") is not True,
            study_protocol.get("frozen_before_provider_calls") is not True,
        )
    ):
        raise ValueError("blind-four-v1 study protocol is not frozen")
    frozen_harness = study_protocol.get("freeze_requirements", {}).get("harness", {}).get("sha256")
    current_harness = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if frozen_harness != current_harness:
        raise ValueError("blind-four-v1 harness hash differs from the frozen protocol")
    manifest_path = args.blind_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("tasks") if isinstance(manifest, dict) else None
    if not isinstance(records, list):
        raise ValueError("blind manifest has no task array")
    task_path = args.task.expanduser().resolve()
    task_sha256 = hashlib.sha256(task_path.read_bytes()).hexdigest()
    matches = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("path") == task_path.name
        and record.get("task_file_sha256") == task_sha256
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("task_id"), str):
        raise ValueError("task path/hash is not uniquely bound by the blind manifest")
    task_id = cast(str, matches[0]["task_id"])
    frozen_task_hashes = study_protocol.get("task_generator", {}).get("public_task_sha256", {})
    if not isinstance(frozen_task_hashes, dict) or frozen_task_hashes.get(task_id) != task_sha256:
        raise ValueError("task hash differs from the frozen study protocol")
    if task_id not in _BLIND_FOUR_RUN_SEEDS:
        raise ValueError(f"unexpected blind task ID: {task_id}")
    actual_seeds = (args.provider_seed, args.tie_seed, args.random_baseline_seed)
    if actual_seeds != _BLIND_FOUR_RUN_SEEDS[task_id]:
        raise ValueError(
            f"seed mismatch for {task_id}: expected "
            f"{_BLIND_FOUR_RUN_SEEDS[task_id]}, got {actual_seeds}"
        )
    return task_id


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
    predicate_signature: tuple[bool, ...] = ()
    mapper_signature: tuple[int, ...] = ()
    singleton_predicate_violations: int = 0
    singleton_mapper_violations: int = 0
    proposal_rank: int = 0
    parent_rank: int = 0


def state_key(state: BeamState) -> str:
    """Return a stable opaque identifier for one complete program state."""

    material = f"{state.predicate}\0{state.mapper}".encode()
    return hashlib.sha256(material).hexdigest()[:16]


def beam_rank_key(state: BeamState, *, tie_seed: int) -> tuple[float, str]:
    """Rank without any hidden target or catalog-position information."""

    tie = hashlib.sha256(f"{tie_seed}\0{state.predicate}\0{state.mapper}".encode()).hexdigest()
    return (state.score.total_loss, tie)


def semantic_key(state: BeamState) -> tuple[tuple[bool, ...], tuple[int, ...]]:
    """Return finite component behavior on the frozen observed-item domain."""

    if len(state.predicate_signature) != len(state.mapper_signature):
        raise ValueError("predicate and mapper signatures must share one probe domain")
    return state.predicate_signature, state.mapper_signature


def semantic_id(state: BeamState) -> str:
    """Hash one finite observational component cell without claiming global equivalence."""

    predicate, mapper = semantic_key(state)
    material = {
        "version": "training-item-component-signature-v1",
        "predicate": predicate,
        "mapper": mapper,
    }
    return hashlib.sha256(_canonical_bytes(material)).hexdigest()


def _semantic_candidate_key(state: BeamState, *, tie_seed: int) -> tuple[float, int, int, str]:
    tie = hashlib.sha256(f"{tie_seed}\0{state.predicate}\0{state.mapper}".encode()).hexdigest()
    return (state.score.total_loss, state.proposal_rank, state.parent_rank, tie)


def _component_distance(left: BeamState, right: BeamState) -> float:
    left_predicate, left_mapper = semantic_key(left)
    right_predicate, right_mapper = semantic_key(right)
    if not left_predicate:
        return 0.0
    predicate_distance = sum(
        a is not b for a, b in zip(left_predicate, right_predicate, strict=True)
    ) / len(left_predicate)
    mapper_distance = sum(a != b for a, b in zip(left_mapper, right_mapper, strict=True)) / len(
        left_mapper
    )
    return (predicate_distance + mapper_distance) / 2.0


def select_semantic_beam(
    candidates: list[BeamState],
    *,
    width: int,
    tie_seed: int,
) -> tuple[BeamState, ...]:
    """Select a loss-gated beam with finite component-behavior diversity."""

    if width < 1:
        raise ValueError("beam width must be positive")
    syntactic: dict[tuple[str, str], BeamState] = {}
    for candidate in candidates:
        key = (candidate.predicate, candidate.mapper)
        incumbent = syntactic.get(key)
        if incumbent is None or _semantic_candidate_key(
            candidate, tie_seed=tie_seed
        ) < _semantic_candidate_key(incumbent, tie_seed=tie_seed):
            syntactic[key] = candidate

    cells: dict[tuple[tuple[bool, ...], tuple[int, ...]], BeamState] = {}
    for candidate in syntactic.values():
        key = semantic_key(candidate)
        incumbent = cells.get(key)
        if incumbent is not None and incumbent.score.total_loss != candidate.score.total_loss:
            raise ValueError("one observed semantic cell produced inconsistent training losses")
        if incumbent is None or _semantic_candidate_key(
            candidate, tie_seed=tie_seed
        ) < _semantic_candidate_key(incumbent, tie_seed=tie_seed):
            cells[key] = candidate
    ranked = sorted(
        cells.values(), key=lambda state: _semantic_candidate_key(state, tie_seed=tie_seed)
    )
    if len(ranked) <= width:
        return tuple(ranked)

    quality_index = min(len(ranked), 2 * width) - 1
    cutoff_loss = ranked[quality_index].score.total_loss
    eligible = [state for state in ranked if state.score.total_loss <= cutoff_loss]
    selected = [min(eligible, key=lambda state: _semantic_candidate_key(state, tie_seed=tie_seed))]
    while len(selected) < width:
        remaining = [state for state in eligible if state not in selected]
        if not remaining:
            break
        predicate_seen = {state.predicate_signature for state in selected}
        mapper_seen = {state.mapper_signature for state in selected}

        def diversity_key(
            state: BeamState,
            predicate_cells: frozenset[tuple[bool, ...]] = frozenset(predicate_seen),
            mapper_cells: frozenset[tuple[int, ...]] = frozenset(mapper_seen),
            selected_states: tuple[BeamState, ...] = tuple(selected),
        ) -> tuple[int, float, float, int, int, str]:
            novelty = int(state.predicate_signature not in predicate_cells) + int(
                state.mapper_signature not in mapper_cells
            )
            minimum_distance = min(
                _component_distance(state, existing) for existing in selected_states
            )
            base = _semantic_candidate_key(state, tie_seed=tie_seed)
            return (-novelty, base[0], -minimum_distance, base[1], base[2], base[3])

        selected.append(min(remaining, key=diversity_key))
    return tuple(selected)


def select_evidence_frontier_beam(
    candidates: list[BeamState],
    *,
    width: int,
    tie_seed: int,
) -> tuple[BeamState, ...]:
    """Retain both the best-loss state and a public-evidence component frontier.

    The frontier is ranked first by violations of sound singleton predicate
    facts, then mapper facts. This mirrors the frozen hole-repair order and lets
    a correct component survive temporarily poor complete-program loss while the
    other component remains wrong.
    """

    if width < 2:
        raise ValueError("evidence-frontier selection requires beam width at least two")
    for candidate in candidates:
        for name, value in (
            ("predicate", candidate.singleton_predicate_violations),
            ("mapper", candidate.singleton_mapper_violations),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"singleton {name} violation count must be a nonnegative int")
    syntactic: dict[tuple[str, str], BeamState] = {}
    for candidate in candidates:
        key = (candidate.predicate, candidate.mapper)
        incumbent = syntactic.get(key)
        if incumbent is None or _semantic_candidate_key(
            candidate, tie_seed=tie_seed
        ) < _semantic_candidate_key(incumbent, tie_seed=tie_seed):
            syntactic[key] = candidate
    cells: dict[tuple[tuple[bool, ...], tuple[int, ...]], BeamState] = {}
    for candidate in syntactic.values():
        key = semantic_key(candidate)
        incumbent = cells.get(key)
        if incumbent is not None and (
            incumbent.score.total_loss != candidate.score.total_loss
            or incumbent.singleton_predicate_violations
            != candidate.singleton_predicate_violations
            or incumbent.singleton_mapper_violations
            != candidate.singleton_mapper_violations
        ):
            raise ValueError("one observed semantic cell produced inconsistent evidence metrics")
        if incumbent is None or _semantic_candidate_key(
            candidate, tie_seed=tie_seed
        ) < _semantic_candidate_key(incumbent, tie_seed=tie_seed):
            cells[key] = candidate
    states = list(cells.values())
    if not states:
        return ()

    best_loss = min(
        states,
        key=lambda state: _semantic_candidate_key(state, tie_seed=tie_seed),
    )

    def evidence_key(state: BeamState) -> tuple[int, int, float, int, int, str]:
        base = _semantic_candidate_key(state, tie_seed=tie_seed)
        return (
            state.singleton_predicate_violations,
            state.singleton_mapper_violations,
            base[0],
            base[1],
            base[2],
            base[3],
        )

    selected = [best_loss]
    evidence_frontier = min(states, key=evidence_key)
    if semantic_key(evidence_frontier) != semantic_key(best_loss) and width > 1:
        selected.append(evidence_frontier)
    if len(selected) < width:
        remaining = [state for state in states if state not in selected]
        predicate_seen = {state.predicate_signature for state in selected}
        mapper_seen = {state.mapper_signature for state in selected}

        def fill_key(state: BeamState) -> tuple[int, int, int, float, int, int, str]:
            novelty = int(state.predicate_signature not in predicate_seen) + int(
                state.mapper_signature not in mapper_seen
            )
            evidence = evidence_key(state)
            return (-novelty, *evidence)

        selected.extend(sorted(remaining, key=fill_key)[: width - len(selected)])
    return tuple(selected[:width])


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


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a binomial success rate."""

    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("successes and trials are inconsistent")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def proposal_slot_checkpoints(
    rounds: int,
    branching_factor: int,
    beam_width: int,
) -> tuple[int, ...]:
    """Return cumulative slot endpoints for one initial state and fixed expansions."""

    if rounds < 1 or branching_factor < 1 or beam_width < 1:
        raise ValueError("rounds, branching factor, and beam width must be positive")
    return (
        1,
        *(
            1 + branching_factor + max(0, round_number - 1) * beam_width * branching_factor
            for round_number in range(1, rounds + 1)
        ),
    )


def scheduled_expansions(
    beam: tuple[BeamState, ...],
    *,
    round_number: int,
    beam_width: int,
) -> tuple[tuple[int, BeamState], ...]:
    """Schedule one first-round expansion and exactly W later expansions.

    If semantic deduplication leaves fewer than W states, states are cycled. The
    expansion index, rather than a possibly repeated beam rank, remains unique.
    """

    if not beam:
        return ()
    if round_number < 1 or beam_width < 1:
        raise ValueError("round number and beam width must be positive")
    count = 1 if round_number == 1 else beam_width
    return tuple((index, beam[index % len(beam)]) for index in range(count))


def derive_singleton_constraints(
    spec: PBESpec,
) -> dict[str, tuple[dict[str, object], ...]]:
    """Derive exact filter/map facts from singleton I/O examples only.

    For this fixed order-preserving filter-map skeleton, ``[x] -> []`` proves
    ``predicate(x)=false`` and ``[x] -> [y]`` proves both
    ``predicate(x)=true`` and ``mapper(x)=y``. No hidden target is consulted.
    """

    predicate_facts: dict[int, tuple[bool, list[int]]] = {}
    mapper_facts: dict[int, tuple[int, list[int]]] = {}
    for source_example, example in enumerate(spec.examples, start=1):
        input_value = example.input_value
        output_value = example.output_value
        if not isinstance(input_value, list) or len(input_value) != 1:
            continue
        if not isinstance(output_value, list) or len(output_value) > 1:
            continue
        item = input_value[0]
        if not isinstance(item, int) or isinstance(item, bool):
            continue
        keep = len(output_value) == 1
        prior_predicate = predicate_facts.get(item)
        if prior_predicate is not None and prior_predicate[0] != keep:
            raise ValueError(f"conflicting singleton predicate facts for item {item}")
        if prior_predicate is None:
            predicate_facts[item] = (keep, [source_example])
        elif source_example not in prior_predicate[1]:
            prior_predicate[1].append(source_example)
        if not keep:
            continue
        expected_value = output_value[0]
        if not isinstance(expected_value, int) or isinstance(expected_value, bool):
            continue
        prior_mapper = mapper_facts.get(item)
        if prior_mapper is not None and prior_mapper[0] != expected_value:
            raise ValueError(f"conflicting singleton mapper facts for item {item}")
        if prior_mapper is None:
            mapper_facts[item] = (expected_value, [source_example])
        elif source_example not in prior_mapper[1]:
            prior_mapper[1].append(source_example)

    return {
        "predicate": tuple(
            {
                "item": item,
                "keep": keep,
                "source_examples": tuple(sources),
            }
            for item, (keep, sources) in sorted(predicate_facts.items())
        ),
        "mapper": tuple(
            {
                "item": item,
                "expected_value": expected_value,
                "source_examples": tuple(sources),
            }
            for item, (expected_value, sources) in sorted(mapper_facts.items())
        ),
    }


def _state_record(state: BeamState) -> dict[str, object]:
    record: dict[str, object] = {
        "state_id": state_key(state),
        "predicate": state.predicate,
        "mapper": state.mapper,
        "score": asdict(state.score),
        "history": [asdict(step) for step in state.history],
        "selection_provenance": {
            "proposal_rank": state.proposal_rank,
            "parent_rank": state.parent_rank,
        },
        "singleton_constraint_violations": {
            "predicate": state.singleton_predicate_violations,
            "mapper": state.singleton_mapper_violations,
        },
    }
    if state.predicate_signature or state.mapper_signature:
        record["finite_component_semantics"] = {
            "version": "training-item-component-signature-v1",
            "semantic_id": semantic_id(state),
            "predicate_keep_mask": state.predicate_signature,
            "mapper_values": state.mapper_signature,
            "scope": "finite observational equivalence only",
        }
    return record


def _semantic_pool_summary(
    candidates: list[BeamState],
    selected: tuple[BeamState, ...],
) -> dict[str, object]:
    syntactic = {(state.predicate, state.mapper) for state in candidates}
    joint = {semantic_key(state) for state in candidates}
    predicates = {state.predicate_signature for state in candidates}
    mappers = {state.mapper_signature for state in candidates}
    return {
        "syntactic_programs": len(syntactic),
        "joint_semantic_cells": len(joint),
        "semantic_duplicates": len(syntactic) - len(joint),
        "predicate_behaviors": len(predicates),
        "mapper_behaviors": len(mappers),
        "selected_syntactic_programs": len({(state.predicate, state.mapper) for state in selected}),
        "selected_joint_semantic_cells": len({semantic_key(state) for state in selected}),
        "selected_predicate_behaviors": len({state.predicate_signature for state in selected}),
        "selected_mapper_behaviors": len({state.mapper_signature for state in selected}),
    }


def _evidence_summary(feedback: dict[str, object], hole: str) -> dict[str, object]:
    """Make the semantics of mechanical evidence explicit without adding oracle facts."""

    toggles = feedback.get("predicate_toggles", ())
    constraints = feedback.get("mapper_constraints", ())
    singleton_predicates = feedback.get("singleton_predicate_constraints", ())
    singleton_mappers = feedback.get("singleton_mapper_constraints", ())
    if any(
        not isinstance(value, (list, tuple))
        for value in (toggles, constraints, singleton_predicates, singleton_mappers)
    ):
        raise ValueError("feedback evidence has the wrong shape")
    if hole == "predicate":
        if singleton_predicates:
            decisions = [
                {
                    "item": cast(dict[str, object], fact)["item"],
                    "predicate_should_keep": cast(dict[str, object], fact)["keep"],
                }
                for fact in singleton_predicates
            ]
            return {
                "repair_semantics": (
                    "For this fixed order-preserving filter-map skeleton, each singleton "
                    "example certifies the listed predicate decision. These facts come only "
                    "from public I/O examples; execution remains authoritative."
                ),
                "supported_item_decisions": decisions,
                "evidence_kind": "sound-singleton-constraints",
            }
        if not toggles:
            return {
                "repair_semantics": (
                    "No sound one-item predicate counterfactual is available. Use the raw "
                    "examples and complete execution traces heuristically; execution of every "
                    "proposal remains authoritative."
                ),
                "supported_item_decisions": [],
            }
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
    if singleton_mappers:
        return {
            "repair_semantics": (
                "For this fixed order-preserving filter-map skeleton, singleton examples "
                "certify each listed mapper output independently of the current program. "
                "These facts come only from public I/O; execution remains authoritative."
            ),
            "required_item_outputs": [
                {
                    "item": cast(dict[str, object], fact)["item"],
                    "mapper_should_return": cast(dict[str, object], fact)["expected_value"],
                }
                for fact in singleton_mappers
            ],
            "evidence_kind": "sound-singleton-constraints",
        }
    if not constraints:
        return {
            "repair_semantics": (
                "No sound mapper-only value constraint is available. Use the raw examples "
                "and complete execution traces heuristically; execution of every proposal "
                "remains authoritative."
            ),
            "required_item_outputs": [],
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


def select_repair_hole(
    feedback: dict[str, object],
    state: BeamState,
    *,
    round_number: int,
    stall_policy: str,
) -> tuple[str, dict[str, object]]:
    """Select an evidence-backed hole or a preregistered heuristic fallback."""

    predicate_violations = feedback.get("singleton_predicate_violations", ())
    mapper_violations = feedback.get("singleton_mapper_violations", ())
    if not isinstance(predicate_violations, (list, tuple)) or not isinstance(
        mapper_violations, (list, tuple)
    ):
        raise ValueError("singleton violation evidence has the wrong shape")
    if predicate_violations:
        return "predicate", {
            "policy": "repair violated singleton predicate constraints before mapper facts",
            "singleton_predicate_violation_count": len(predicate_violations),
            "singleton_mapper_violation_count": len(mapper_violations),
            "selected_first_hole": "predicate",
            "reason": "current predicate violates sound singleton I/O constraints",
            "evidence_backed": True,
            "fallback_used": False,
            "state_id": state_key(state),
        }
    if mapper_violations:
        return "mapper", {
            "policy": "repair violated singleton predicate constraints before mapper facts",
            "singleton_predicate_violation_count": 0,
            "singleton_mapper_violation_count": len(mapper_violations),
            "selected_first_hole": "mapper",
            "reason": "predicate facts hold but mapper violates sound singleton I/O constraints",
            "evidence_backed": True,
            "fallback_used": False,
            "state_id": state_key(state),
        }

    try:
        hole, decision = select_first_hole(feedback)
        return hole, decision | {"evidence_backed": True, "fallback_used": False}
    except ValueError:
        if stall_policy != "alternate-hole":
            raise
        hole = "predicate" if round_number % 2 == 1 else "mapper"
        return hole, {
            "policy": (
                "use conservative evidence when available; otherwise choose predicate on odd "
                "rounds and mapper on even rounds"
            ),
            "predicate_toggle_count": 0,
            "mapper_constraint_count": 0,
            "selected_first_hole": hole,
            "reason": "no local evidence; preregistered heuristic fallback",
            "evidence_backed": False,
            "fallback_used": True,
            "state_id": state_key(state),
        }


def _augment_beam_prompt(
    payload: dict[str, object],
    *,
    round_number: int,
    max_rounds: int,
    state: BeamState,
    feedback: dict[str, object],
    hole: str,
    hole_decision: dict[str, object],
) -> dict[str, object]:
    messages = cast(list[dict[str, str]], payload["messages"])
    document = json.loads(messages[1]["content"])
    document["beam_search_context"] = {
        "round": round_number,
        "maximum_rounds": max_rounds,
        "current_loss": state.score.total_loss,
    }
    fallback_used = cast(bool, hole_decision["fallback_used"])
    document["hole_selection"] = {
        "selected_hole": hole,
        "evidence_backed": cast(bool, hole_decision["evidence_backed"]),
        "fallback_used": fallback_used,
        "label": (
            "unsupported deterministic exploration"
            if fallback_used
            else "mechanically supported hole selection"
        ),
        "reason": hole_decision["reason"],
    }
    document["mechanical_evidence_interpretation"] = _evidence_summary(feedback, hole)
    current_expression = state.predicate if hole == "predicate" else state.mapper
    document["forbidden_no_op_expression"] = current_expression
    evidence_requirement = (
        "No mechanically certified local constraint selected this hole. Use only the public "
        "examples and execution traces as heuristic evidence."
        if fallback_used
        else "Use the mechanically certified evidence literally and respect operand direction."
    )
    document["requirements"] = [
        "Return exactly four distinct typed grammar expressions.",
        "Do not return the current expression; every candidate must change the selected hole.",
        evidence_requirement,
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
    task_sha256 = hashlib.sha256(args.task.resolve().read_bytes()).hexdigest()
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
    observed_items = tuple(
        sorted(
            {
                item
                for example in config.spec.examples
                for item in cast(list[int], example.input_value)
            }
        )
    )
    if not observed_items:
        raise ValueError("semantic beam requires at least one observed scalar input item")

    def predicate_signature(expression: Node) -> tuple[bool, ...]:
        values = tuple(
            evaluate_expression(expression, list(observed_items), item=item)
            for item in observed_items
        )
        if any(not isinstance(value, bool) for value in values):
            raise TypeError("predicate catalog member did not produce Boolean probe values")
        return cast(tuple[bool, ...], values)

    def mapper_signature(expression: Node) -> tuple[int, ...]:
        values = tuple(
            evaluate_expression(expression, list(observed_items), item=item)
            for item in observed_items
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise TypeError("mapper catalog member did not produce integer probe values")
        return cast(tuple[int, ...], values)

    predicate_signatures = {
        dsl: predicate_signature(cast(Node, expression))
        for dsl, expression in predicate_catalog.items()
    }
    mapper_signatures = {
        dsl: mapper_signature(cast(Node, expression)) for dsl, expression in mapper_catalog.items()
    }
    observed_item_index = {item: index for index, item in enumerate(observed_items)}
    singleton_constraints = (
        derive_singleton_constraints(config.spec)
        if args.singleton_evidence
        else {"predicate": (), "mapper": ()}
    )

    score_cache: dict[tuple[str, str], ScoredProgram] = {}
    score_cache_stats = {"hits": 0, "misses": 0}
    feedback_cache: dict[tuple[str, str], dict[str, object]] = {}

    def score_pair(predicate: str, mapper: str) -> ScoredProgram:
        key = (predicate, mapper)
        existing = score_cache.get(key)
        if existing is not None:
            score_cache_stats["hits"] += 1
            return existing
        score_cache_stats["misses"] += 1
        score = scorer.score(
            _assemble_program(predicate_catalog[predicate], mapper_catalog[mapper])
        )
        if not isinstance(score, ScoredProgram):
            raise ValueError(f"grammar-valid beam state was rejected: {score.reason}")
        score_cache[key] = score
        return score

    def make_state(
        predicate: str,
        mapper: str,
        *,
        history: tuple[RepairStep, ...] = (),
        proposal_rank: int,
        parent_rank: int,
    ) -> BeamState:
        predicate_signature_value = predicate_signatures[predicate]
        mapper_signature_value = mapper_signatures[mapper]
        return BeamState(
            predicate=predicate,
            mapper=mapper,
            score=score_pair(predicate, mapper),
            history=history,
            predicate_signature=predicate_signature_value,
            mapper_signature=mapper_signature_value,
            singleton_predicate_violations=sum(
                predicate_signature_value[
                    observed_item_index[cast(int, fact["item"])]
                ]
                != cast(bool, fact["keep"])
                for fact in singleton_constraints["predicate"]
            ),
            singleton_mapper_violations=sum(
                mapper_signature_value[
                    observed_item_index[cast(int, fact["item"])]
                ]
                != cast(int, fact["expected_value"])
                for fact in singleton_constraints["mapper"]
            ),
            proposal_rank=proposal_rank,
            parent_rank=parent_rank,
        )

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
        predicate_violations = tuple(
            fact
            for fact in singleton_constraints["predicate"]
            if state.predicate_signature[observed_item_index[cast(int, fact["item"])]]
            != cast(bool, fact["keep"])
        )
        mapper_violations = tuple(
            fact
            for fact in singleton_constraints["mapper"]
            if state.mapper_signature[observed_item_index[cast(int, fact["item"])]]
            != cast(int, fact["expected_value"])
        )
        feedback |= {
            "singleton_predicate_constraints": singleton_constraints["predicate"],
            "singleton_mapper_constraints": singleton_constraints["mapper"],
            "singleton_predicate_violations": predicate_violations,
            "singleton_mapper_violations": mapper_violations,
        }
        feedback_cache[key] = feedback
        return feedback

    start_rng = random.Random(args.start_seed)
    initial_mapper = orders["mapper"][start_rng.randrange(len(orders["mapper"]))]
    initial_predicate = orders["predicate"][start_rng.randrange(len(orders["predicate"]))]
    initial = make_state(
        initial_predicate,
        initial_mapper,
        proposal_rank=args.branching_factor,
        parent_rank=0,
    )
    if args.protocol_mode == "blind-four-v1":
        expected_initial = (
            "and(lt(item,-2),lt(item,-1))",
            "mul(4,item)",
        )
        actual_initial = (initial.predicate, initial.mapper)
        if actual_initial != expected_initial:
            raise ValueError(
                f"blind-four-v1 initial state changed: expected {expected_initial}, "
                f"got {actual_initial}"
            )
    protocol = {
        "schema": "iterative-typed-llm-beam-protocol-v2",
        "task_sha256": task_sha256,
        "blind_task_id": getattr(args, "blind_task_id", None),
        "protocol_mode": args.protocol_mode,
        "blind_manifest_sha256": (
            None
            if args.blind_manifest is None
            else hashlib.sha256(args.blind_manifest.resolve().read_bytes()).hexdigest()
        ),
        "study_protocol_sha256": (
            None
            if args.study_protocol is None
            else hashlib.sha256(args.study_protocol.resolve().read_bytes()).hexdigest()
        ),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": args.model,
        "model_revision": MODEL_REVISION,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "rounds": args.rounds,
        "beam_width": args.beam_width,
        "branching_factor": args.branching_factor,
        "maximum_proposal_slot_budget": (
            1 + args.branching_factor + (args.rounds - 1) * args.beam_width * args.branching_factor
        ),
        "selection_policy": args.selection_policy,
        "ranking": (
            "finite semantic cells; best complete-program loss anchor plus a public singleton "
            "constraint frontier ranked by predicate violations, mapper violations, loss, "
            "proposal rank, parent rank, and SHA256 tie break"
            if args.selection_policy == "evidence-frontier"
            else "finite component semantic cells; loss rank-2W quality gate; best-loss "
            "anchor; then component novelty, loss, minimum signature distance, proposal rank, "
            "parent rank, and SHA256 tie break"
            if args.selection_policy == "semantic-diverse"
            else "ascending total training loss, then SHA256(tie_seed,predicate,mapper)"
        ),
        "beam_transition": (
            "elitist pool of parents plus valid children; deduplicate finite joint component "
            "behavior, apply a rank-2W loss quality gate, anchor at best loss, and fill by "
            "component novelty"
            if args.selection_policy == "semantic-diverse"
            else "elitist pool of parents plus valid children; deduplicate finite semantic "
            "cells and retain the best-loss anchor plus the public-evidence constraint frontier"
            if args.selection_policy == "evidence-frontier"
            else "elitist pool of parents plus valid children; deduplicate complete programs "
            "and retain the lowest-loss fixed-width beam"
        ),
        "hole_policy": (
            "violated sound singleton predicate constraints first; then violated singleton "
            "mapper constraints; then predicate-toggle support; then conditional mapper "
            "constraints; otherwise follow the configured stall policy"
        ),
        "singleton_evidence": args.singleton_evidence,
        "stall_policy": args.stall_policy,
        "component_signature": {
            "version": "training-item-component-signature-v1",
            "domain": observed_items,
            "domain_sha256": hashlib.sha256(_canonical_bytes(observed_items)).hexdigest(),
            "interpretation": "finite observational equivalence, not global equivalence",
            "predicate_behavior_count_in_catalog": len(set(predicate_signatures.values())),
            "mapper_behavior_count_in_catalog": len(set(mapper_signatures.values())),
        },
        "start_seed": args.start_seed,
        "initial_state": _state_record(initial),
        "provider_seed": args.provider_seed,
        "tie_seed": args.tie_seed,
        "random_baseline_seed": args.random_baseline_seed,
        "random_baseline_trials": args.random_baseline_trials,
        "max_provider_concurrency": args.max_concurrency,
        "maximum_provider_calls": 1 + max(0, args.rounds - 1) * args.beam_width,
        "expansion_schedule": (
            "one expansion in round 1 and exactly beam_width expansions in each later round; "
            "cycle retained states if the semantic beam is underfull"
        ),
        "exact_stop": (
            "finish all scheduled expansions in the active round, then stop before the next round"
        ),
        "retry_policy": "none",
        "invalid_or_duplicate_shortlist_policy": (
            "a semantic invalid, duplicate, or no-op response consumes all K proposal slots; "
            "it produces no children and receives no within-call retry or backfill. Infrastructure "
            "failure aborts and seals the trial"
        ),
        "provider_visible_catalog_sizes": False,
        "provider_visible_full_catalogs": False,
        "provider_candidate_order_contract": "most to least promising; retained as ordinal rank",
        "output_encoding": "typed grammar objects constrained by operators and constants",
        "claim": "heuristic search benchmark, not importance sampling or posterior inference",
        "large_space_note": (
            "The LLM request generates only K repairs and receives neither component-catalog "
            "cardinalities nor a complete-program list. The finite grammar can still make its "
            "size inferable; the claim is that generation does not require enumeration."
        ),
    }
    checkpoint_slots = list(
        proposal_slot_checkpoints(
            args.rounds,
            args.branching_factor,
            args.beam_width,
        )
    )
    primary_checkpoint_slot = (
        None
        if args.primary_checkpoint_round == 0
        else 1
        + args.branching_factor
        + max(0, args.primary_checkpoint_round - 1) * args.beam_width * args.branching_factor
    )
    protocol["proposal_slot_checkpoints"] = checkpoint_slots
    protocol["primary_checkpoint_round"] = args.primary_checkpoint_round or None
    protocol["primary_checkpoint_slot"] = primary_checkpoint_slot
    _write_json(output / "protocol.json", protocol)

    def provider_payload(
        *,
        state: BeamState,
        hole: str,
        feedback: dict[str, object],
        round_number: int,
        hole_decision: dict[str, object],
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
            state=state,
            feedback=feedback,
            hole=hole,
            hole_decision=hole_decision,
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
    evaluated_states = [initial]
    best_state = initial
    first_exact_slot = 1 if initial.score.exact_program else None
    first_exact_state_evaluation = 1 if initial.score.exact_program else None
    llm_best_loss_by_checkpoint: dict[str, float] = {
        "1": initial.score.total_loss,
    }
    round_records: list[dict[str, object]] = []
    provider_root = output / "provider"
    async with httpx.AsyncClient(timeout=args.timeout_seconds) as client:
        semaphore = asyncio.Semaphore(args.max_concurrency)
        for round_number in range(1, args.rounds + 1):
            if first_exact_slot is not None:
                break
            expansion_specs: list[tuple[int, BeamState, str, dict[str, object]]] = []
            stalled = []
            for expansion_index, state in scheduled_expansions(
                beam,
                round_number=round_number,
                beam_width=args.beam_width,
            ):
                feedback = feedback_for(state)
                try:
                    hole, decision = select_repair_hole(
                        feedback,
                        state,
                        round_number=round_number,
                        stall_policy=args.stall_policy,
                    )
                except ValueError as error:
                    stalled.append({"state_id": state_key(state), "detail": str(error)})
                    continue
                expansion_specs.append((expansion_index, state, hole, decision))
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
                active_round: int = round_number,
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
                    / f"round-{active_round:02d}"
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
                                round_number=active_round,
                                hole_decision=decision,
                                seed=args.provider_seed + 100 * active_round + beam_rank,
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
                        return (
                            beam_rank,
                            state,
                            hole,
                            decision,
                            (),
                            cast(str, result.get("status", "invalid-response")),
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
                    child = make_state(
                        predicate,
                        mapper,
                        history=(
                            *parent.history,
                            RepairStep(
                                round=round_number,
                                hole=hole,
                                repair=repair,
                                loss_after=score.total_loss,
                            ),
                        ),
                        proposal_rank=proposal_rank,
                        parent_rank=beam_rank,
                    )
                    score = child.score
                    children.append(child)
                    evaluated_states.append(child)
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
                repaired_signatures = (
                    [
                        (
                            child.predicate_signature
                            if hole == "predicate"
                            else child.mapper_signature
                        )
                        for child in children[-len(shortlist) :]
                    ]
                    if shortlist
                    else []
                )
                parent_signature = (
                    parent.predicate_signature if hole == "predicate" else parent.mapper_signature
                )
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
                        "repaired_hole_unique_behaviors": len(set(repaired_signatures)),
                        "within_call_semantic_duplicates": len(repaired_signatures)
                        - len(set(repaired_signatures)),
                        "semantic_no_ops_relative_to_parent": sum(
                            signature == parent_signature for signature in repaired_signatures
                        ),
                    }
                )
            parent_pool = [
                replace(
                    state,
                    proposal_rank=args.branching_factor,
                    parent_rank=beam_rank,
                )
                for beam_rank, state in enumerate(beam)
            ]
            pool = [*parent_pool, *children]
            if args.selection_policy == "semantic-diverse":
                selected = select_semantic_beam(
                    pool,
                    width=args.beam_width,
                    tie_seed=args.tie_seed + round_number,
                )
            elif args.selection_policy == "evidence-frontier":
                selected = select_evidence_frontier_beam(
                    pool,
                    width=args.beam_width,
                    tie_seed=args.tie_seed + round_number,
                )
            else:
                selected = select_beam(
                    pool,
                    width=args.beam_width,
                    tie_seed=args.tie_seed + round_number,
                )
            round_record = {
                "round": round_number,
                "input_beam": [_state_record(state) for state in beam],
                "stalled": stalled,
                "expansions": expansion_records,
                "children_executed": len(children),
                "selection_pool": _semantic_pool_summary(pool, selected),
                "selected_beam": [_state_record(state) for state in selected],
                "cumulative_best": _state_record(best_state),
                "first_exact_proposal_slot": first_exact_slot,
                "first_exact_state_evaluation": first_exact_state_evaluation,
            }
            round_records.append(round_record)
            _write_json(output / f"round-{round_number:02d}.json", round_record)
            beam = selected
            llm_best_loss_by_checkpoint[str(proposal_slots_consumed)] = best_state.score.total_loss
            if first_exact_slot is not None:
                break
    provider_completed = time.perf_counter()
    llm_score_cache_stats = dict(score_cache_stats)
    provider_inventory = _provider_inventory(output)
    _write_json(output / "provider-seal.json", provider_inventory)

    max_budget = cast(int, protocol["maximum_proposal_slot_budget"])

    checkpoints = checkpoint_slots

    def random_trial(trial_index: int) -> dict[str, object]:
        trial_beam: tuple[BeamState, ...] = (initial,)
        trial_best = initial
        proposal_slots = 1
        state_evaluations = 1
        exact_at = 1 if initial.score.exact_program else None
        rounds_completed = 0
        seen_syntax = {(initial.predicate, initial.mapper)}
        seen_joint = {semantic_key(initial)}
        seen_predicates = {initial.predicate_signature}
        seen_mappers = {initial.mapper_signature}
        semantic_no_ops = 0
        best_loss_by_checkpoint: dict[str, float] = {
            "1": initial.score.total_loss,
        }
        evidence_guided_expansions = 0
        fallback_expansions = 0
        for round_number in range(1, args.rounds + 1):
            if exact_at is not None:
                break
            trial_children: list[BeamState] = []
            expansion_count = 0
            for expansion_index, state in scheduled_expansions(
                trial_beam,
                round_number=round_number,
                beam_width=args.beam_width,
            ):
                try:
                    hole, decision = select_repair_hole(
                        feedback_for(state),
                        state,
                        round_number=round_number,
                        stall_policy=args.stall_policy,
                    )
                except ValueError:
                    continue
                if cast(bool, decision["fallback_used"]):
                    fallback_expansions += 1
                else:
                    evidence_guided_expansions += 1
                expansion_count += 1
                current_expression = state.predicate if hole == "predicate" else state.mapper
                available = [
                    expression for expression in orders[hole] if expression != current_expression
                ]
                stream = hashlib.sha256(
                    (
                        f"{args.random_baseline_seed}\0{trial_index}\0{round_number}\0"
                        f"{task_sha256}\0{expansion_index}\0{state_key(state)}\0{hole}"
                    ).encode()
                ).digest()
                rng = random.Random(int.from_bytes(stream[:8], "big"))
                parent_signature = (
                    state.predicate_signature if hole == "predicate" else state.mapper_signature
                )
                for proposal_rank, repair in enumerate(
                    rng.sample(available, args.branching_factor)
                ):
                    proposal_slots += 1
                    predicate = repair if hole == "predicate" else state.predicate
                    mapper = repair if hole == "mapper" else state.mapper
                    child = make_state(
                        predicate,
                        mapper,
                        proposal_rank=proposal_rank,
                        parent_rank=expansion_index,
                    )
                    score = child.score
                    trial_children.append(child)
                    state_evaluations += 1
                    seen_syntax.add((predicate, mapper))
                    seen_joint.add(semantic_key(child))
                    seen_predicates.add(child.predicate_signature)
                    seen_mappers.add(child.mapper_signature)
                    repaired_signature = (
                        child.predicate_signature if hole == "predicate" else child.mapper_signature
                    )
                    semantic_no_ops += repaired_signature == parent_signature
                    if beam_rank_key(child, tie_seed=args.tie_seed) < beam_rank_key(
                        trial_best, tie_seed=args.tie_seed
                    ):
                        trial_best = child
                    if score.exact_program and exact_at is None:
                        exact_at = proposal_slots
            rounds_completed = round_number
            best_loss_by_checkpoint[str(proposal_slots)] = trial_best.score.total_loss
            if expansion_count == 0 or exact_at is not None:
                break
            parent_pool = [
                replace(
                    state,
                    proposal_rank=args.branching_factor,
                    parent_rank=parent_rank,
                )
                for parent_rank, state in enumerate(trial_beam)
            ]
            pool = [*parent_pool, *trial_children]
            if args.selection_policy == "semantic-diverse":
                trial_beam = select_semantic_beam(
                    pool,
                    width=args.beam_width,
                    tie_seed=args.tie_seed + round_number,
                )
            elif args.selection_policy == "evidence-frontier":
                trial_beam = select_evidence_frontier_beam(
                    pool,
                    width=args.beam_width,
                    tie_seed=args.tie_seed + round_number,
                )
            else:
                trial_beam = select_beam(
                    pool,
                    width=args.beam_width,
                    tie_seed=args.tie_seed + round_number,
                )
        return {
            "success": exact_at is not None,
            "first_exact_execution": exact_at,
            "best_loss": trial_best.score.total_loss,
            "best_exact_examples": trial_best.score.exact_matches,
            "proposal_slots": proposal_slots,
            "logical_candidate_evaluations": state_evaluations,
            "rounds_completed": rounds_completed,
            "unique_syntactic_programs": len(seen_syntax),
            "unique_joint_semantic_cells": len(seen_joint),
            "unique_predicate_behaviors": len(seen_predicates),
            "unique_mapper_behaviors": len(seen_mappers),
            "semantic_no_op_proposals": semantic_no_ops,
            "best_loss_by_checkpoint": best_loss_by_checkpoint,
            "evidence_guided_expansions": evidence_guided_expansions,
            "fallback_expansions": fallback_expansions,
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
    checkpoint_success_rates = {
        checkpoint: successes_at_checkpoint / args.random_baseline_trials
        for checkpoint, successes_at_checkpoint in checkpoint_successes.items()
    }
    checkpoint_success_intervals = {
        checkpoint: wilson_interval(
            successes_at_checkpoint,
            args.random_baseline_trials,
        )
        for checkpoint, successes_at_checkpoint in checkpoint_successes.items()
    }
    checkpoint_best_loss_summaries: dict[str, dict[str, object]] = {}
    for checkpoint in checkpoints:
        losses_at_checkpoint = [
            (
                0.0
                if record["first_exact_execution"] is not None
                and cast(int, record["first_exact_execution"]) <= checkpoint
                else cast(
                    float,
                    cast(dict[str, object], record["best_loss_by_checkpoint"])[str(checkpoint)],
                )
            )
            for record in baseline
            if (
                record["first_exact_execution"] is not None
                and cast(int, record["first_exact_execution"]) <= checkpoint
            )
            or str(checkpoint) in cast(dict[str, object], record["best_loss_by_checkpoint"])
        ]
        checkpoint_best_loss_summaries[str(checkpoint)] = {
            "observed_trials": len(losses_at_checkpoint),
            "mean": mean(losses_at_checkpoint) if losses_at_checkpoint else None,
            "median": median(losses_at_checkpoint) if losses_at_checkpoint else None,
            "p05": _quantile(losses_at_checkpoint, 0.05) if losses_at_checkpoint else None,
            "p95": _quantile(losses_at_checkpoint, 0.95) if losses_at_checkpoint else None,
            "fraction_at_most_llm": (
                None
                if not losses_at_checkpoint or str(checkpoint) not in llm_best_loss_by_checkpoint
                else sum(
                    loss <= llm_best_loss_by_checkpoint[str(checkpoint)]
                    for loss in losses_at_checkpoint
                )
                / len(losses_at_checkpoint)
            ),
        }
    _write_json(output / "matched-random-trials.json", baseline)
    baseline_completed = time.perf_counter()
    baseline_score_cache_stats = {
        "hits": score_cache_stats["hits"] - llm_score_cache_stats["hits"],
        "misses": score_cache_stats["misses"] - llm_score_cache_stats["misses"],
    }

    unique_search_programs = {
        (cast(str, record["predicate"]), cast(str, record["mapper"]))
        for record in execution_records
    }
    unique_joint_semantics = {semantic_key(state) for state in evaluated_states}
    unique_predicate_semantics = {state.predicate_signature for state in evaluated_states}
    unique_mapper_semantics = {state.mapper_signature for state in evaluated_states}

    result = {
        "schema": "iterative-typed-llm-beam-experiment-v2",
        "claim_scope": (
            "one exploratory LLM beam trajectory compared with a matched random-beam "
            "simulation; this is not evidence of general speedup without repeated held-out tasks"
        ),
        "protocol": protocol,
        "provider_inventory_sha256": provider_inventory["inventory_sha256"],
        "rounds": round_records,
        "logical_candidate_evaluations": execution_records,
        "llm_beam_metrics": {
            "success": first_exact_slot is not None,
            "first_exact_proposal_slot": first_exact_slot,
            "first_exact_state_evaluation": first_exact_state_evaluation,
            "success_by_proposal_slot_checkpoint": {
                str(checkpoint): first_exact_slot is not None and first_exact_slot <= checkpoint
                for checkpoint in checkpoints
            },
            "primary_checkpoint_slot": primary_checkpoint_slot,
            "primary_checkpoint_success": (
                None
                if primary_checkpoint_slot is None
                else first_exact_slot is not None and first_exact_slot <= primary_checkpoint_slot
            ),
            "best_loss_by_proposal_slot_checkpoint": {
                str(checkpoint): (
                    0.0
                    if first_exact_slot is not None and first_exact_slot <= checkpoint
                    else llm_best_loss_by_checkpoint.get(str(checkpoint))
                )
                for checkpoint in checkpoints
            },
            "proposal_slots_consumed": proposal_slots_consumed,
            "logical_candidate_evaluations": len(execution_records),
            "physical_scorer_calls": llm_score_cache_stats["misses"],
            "score_cache_hits": llm_score_cache_stats["hits"],
            "unique_programs_evaluated": len(unique_search_programs),
            "unique_joint_semantic_cells_evaluated": len(unique_joint_semantics),
            "unique_predicate_behaviors_evaluated": len(unique_predicate_semantics),
            "unique_mapper_behaviors_evaluated": len(unique_mapper_semantics),
            "semantic_redundancy_rate": 1.0
            - len(unique_joint_semantics) / len(unique_search_programs),
            "semantic_no_op_proposals": sum(
                cast(int, expansion["semantic_no_ops_relative_to_parent"])
                for round_record in round_records
                for expansion in cast(list[dict[str, object]], round_record.get("expansions", []))
            ),
            "within_call_semantic_duplicates": sum(
                cast(int, expansion["within_call_semantic_duplicates"])
                for round_record in round_records
                for expansion in cast(list[dict[str, object]], round_record.get("expansions", []))
            ),
            "provider_calls_attempted": provider_calls_attempted,
            "provider_calls_valid": sum(
                expansion["validation_status"] == "valid"
                for round_record in round_records
                for expansion in cast(list[dict[str, object]], round_record.get("expansions", []))
            ),
            "provider_calls_invalid": sum(
                expansion["validation_status"] != "valid"
                for round_record in round_records
                for expansion in cast(list[dict[str, object]], round_record.get("expansions", []))
            ),
            "evidence_guided_expansions": sum(
                not cast(bool, cast(dict[str, object], expansion["hole_decision"])["fallback_used"])
                for round_record in round_records
                for expansion in cast(list[dict[str, object]], round_record.get("expansions", []))
            ),
            "fallback_expansions": sum(
                cast(bool, cast(dict[str, object], expansion["hole_decision"])["fallback_used"])
                for round_record in round_records
                for expansion in cast(list[dict[str, object]], round_record.get("expansions", []))
            ),
            "maximum_budget": max_budget,
            "rounds_completed": len(round_records),
            "termination_reason": (
                "exact-program"
                if first_exact_slot is not None
                else "round-limit"
                if len(round_records) == args.rounds
                else "all-scheduled-states-stalled"
            ),
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
            "success_rates_by_proposal_slot_checkpoint": checkpoint_success_rates,
            "success_rate_wilson_95_by_proposal_slot_checkpoint": (checkpoint_success_intervals),
            "best_loss_summary_by_proposal_slot_checkpoint": (checkpoint_best_loss_summaries),
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
            "mean_logical_candidate_evaluations": mean(
                cast(int, record["logical_candidate_evaluations"]) for record in baseline
            ),
            "aggregate_physical_scorer_calls": baseline_score_cache_stats["misses"],
            "aggregate_score_cache_hits": baseline_score_cache_stats["hits"],
            "mean_unique_joint_semantic_cells": mean(
                cast(int, record["unique_joint_semantic_cells"]) for record in baseline
            ),
            "mean_unique_predicate_behaviors": mean(
                cast(int, record["unique_predicate_behaviors"]) for record in baseline
            ),
            "mean_unique_mapper_behaviors": mean(
                cast(int, record["unique_mapper_behaviors"]) for record in baseline
            ),
            "mean_semantic_no_op_proposals": mean(
                cast(int, record["semantic_no_op_proposals"]) for record in baseline
            ),
            "mean_evidence_guided_expansions": mean(
                cast(int, record["evidence_guided_expansions"]) for record in baseline
            ),
            "mean_fallback_expansions": mean(
                cast(int, record["fallback_expansions"]) for record in baseline
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
    parser.add_argument("--protocol-mode", choices=("blind-four-v1",))
    parser.add_argument("--blind-manifest", type=Path)
    parser.add_argument("--study-protocol", type=Path)
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
    parser.add_argument(
        "--selection-policy",
        choices=("syntactic-loss", "semantic-diverse", "evidence-frontier"),
        default="syntactic-loss",
    )
    parser.add_argument(
        "--stall-policy",
        choices=("stop", "alternate-hole"),
        default="stop",
    )
    parser.add_argument("--singleton-evidence", action="store_true")
    parser.add_argument("--primary-checkpoint-round", type=int, default=0)
    args = parser.parse_args()
    for name in ("max_concurrency", "rounds", "beam_width", "branching_factor"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.branching_factor != 4:
        parser.error("the current typed output protocol requires branching factor four")
    if args.selection_policy == "evidence-frontier" and args.beam_width < 2:
        parser.error("--selection-policy evidence-frontier requires --beam-width at least two")
    if args.random_baseline_trials < 1:
        parser.error("--random-baseline-trials must be positive")
    if not 0 <= args.primary_checkpoint_round <= args.rounds:
        parser.error("--primary-checkpoint-round must be zero or at most --rounds")
    try:
        args.blind_task_id = validate_blind_four_args(args)
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

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import pytest

from modelsmc_pbe.core.results import ScoredProgram
from modelsmc_pbe.domain.models import PBESpec
from research.iterative_beam_experiment import (
    BeamState,
    RepairStep,
    _augment_beam_prompt,
    derive_singleton_constraints,
    proposal_slot_checkpoints,
    scheduled_expansions,
    select_beam,
    select_evidence_frontier_beam,
    select_repair_hole,
    select_semantic_beam,
    semantic_id,
    validate_blind_four_args,
    wilson_interval,
)


def _score(loss: float, exact_examples: int, cost: int = 10) -> ScoredProgram:
    return ScoredProgram(
        kind="Scored",
        inferred_type="List<Int>",
        total_loss=loss,
        exact_matches=exact_examples,
        cost=cost,
        log_target=-loss,
        exact_program=loss == 0,
        evaluations=(),
    )


def _locked_blind_four_args(tmp_path: Path) -> argparse.Namespace:
    task = tmp_path / "blind-01.json"
    task.write_text('{"name":"blind-01"}\n', encoding="utf-8")
    task_sha256 = hashlib.sha256(task.read_bytes()).hexdigest()
    harness_path = Path(validate_blind_four_args.__code__.co_filename)
    harness_sha256 = hashlib.sha256(harness_path.read_bytes()).hexdigest()
    manifest = tmp_path / "blind-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "blind-01",
                        "path": task.name,
                        "task_file_sha256": task_sha256,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    study_protocol = tmp_path / "study-protocol.json"
    study_protocol.write_text(
        json.dumps(
            {
                "protocol_status": "frozen",
                "frozen_before_task_generation": True,
                "frozen_before_provider_calls": True,
                "freeze_requirements": {
                    "harness": {"sha256": harness_sha256},
                },
                "task_generator": {
                    "public_task_sha256": {"blind-01": task_sha256},
                },
            }
        ),
        encoding="utf-8",
    )
    return argparse.Namespace(
        protocol_mode="blind-four-v1",
        blind_manifest=manifest,
        study_protocol=study_protocol,
        task=task,
        model="gpt-oss-120b",
        reasoning_effort="low",
        temperature=0.0,
        max_tokens=1600,
        timeout_seconds=420.0,
        max_concurrency=2,
        rounds=5,
        beam_width=2,
        branching_factor=4,
        start_seed=17,
        random_baseline_trials=10000,
        selection_policy="semantic-diverse",
        stall_policy="alternate-hole",
        singleton_evidence=True,
        primary_checkpoint_round=4,
        provider_seed=101000,
        tie_seed=101100,
        random_baseline_seed=101200,
    )


def test_blind_four_validation_accepts_exact_locked_task_and_args(tmp_path: Path) -> None:
    assert validate_blind_four_args(_locked_blind_four_args(tmp_path)) == "blind-01"


def test_blind_four_validation_rejects_old_three_round_setting(tmp_path: Path) -> None:
    args = _locked_blind_four_args(tmp_path)
    args.rounds = 3

    with pytest.raises(ValueError, match="rounds"):
        validate_blind_four_args(args)


@pytest.mark.parametrize(
    "seed_name",
    ("provider_seed", "tie_seed", "random_baseline_seed"),
)
def test_blind_four_validation_rejects_seed_drift(
    tmp_path: Path,
    seed_name: str,
) -> None:
    args = _locked_blind_four_args(tmp_path)
    setattr(args, seed_name, getattr(args, seed_name) + 1)

    with pytest.raises(ValueError, match="seed mismatch"):
        validate_blind_four_args(args)


def test_blind_four_validation_rejects_task_hash_drift(tmp_path: Path) -> None:
    args = _locked_blind_four_args(tmp_path)
    args.task.write_text('{"name":"tampered"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="path/hash"):
        validate_blind_four_args(args)


def test_blind_four_validation_rejects_frozen_harness_hash_drift(
    tmp_path: Path,
) -> None:
    args = _locked_blind_four_args(tmp_path)
    protocol = json.loads(args.study_protocol.read_text(encoding="utf-8"))
    protocol["freeze_requirements"]["harness"]["sha256"] = "0" * 64
    args.study_protocol.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="harness hash"):
        validate_blind_four_args(args)


def test_select_beam_prefers_loss_and_deduplicates() -> None:
    candidates = [
        BeamState("p0", "m0", _score(5, 2)),
        BeamState("p1", "m0", _score(3, 1)),
        BeamState("p2", "m0", _score(2, 4)),
        BeamState("p2", "m0", _score(2, 4), (RepairStep(1, "predicate", "p2", 2),)),
    ]
    selected = select_beam(candidates, width=2, tie_seed=7)
    assert [(state.predicate, state.mapper) for state in selected] == [
        ("p2", "m0"),
        ("p1", "m0"),
    ]


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = wilson_interval(12, 100)
    assert lower < 0.12 < upper
    assert 0.0 <= lower <= upper <= 1.0


def test_five_round_proposal_slot_checkpoints_are_frozen() -> None:
    assert proposal_slot_checkpoints(
        rounds=5,
        branching_factor=4,
        beam_width=2,
    ) == (1, 5, 13, 21, 29, 37)


def test_fixed_expansion_schedule_cycles_an_underfilled_beam() -> None:
    first = BeamState("p0", "m0", _score(3, 1))
    second = BeamState("p1", "m1", _score(4, 1))

    assert scheduled_expansions((first, second), round_number=1, beam_width=3) == (
        (0, first),
    )
    assert scheduled_expansions((first, second), round_number=2, beam_width=3) == (
        (0, first),
        (1, second),
        (2, first),
    )
    assert scheduled_expansions((first,), round_number=5, beam_width=2) == (
        (0, first),
        (1, first),
    )


def test_prompt_makes_predicate_toggle_direction_explicit() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": '{"requirements":[]}'},
        ]
    }
    state = BeamState("lt(item,0)", "item", _score(2, 3))
    feedback = {
        "predicate_toggles": ({"item": -3, "keep": False}, {"item": 0, "keep": True}),
        "mapper_constraints": (),
    }
    actual = _augment_beam_prompt(
        payload,
        round_number=2,
        max_rounds=3,
        state=state,
        feedback=feedback,
        hole="predicate",
        hole_decision={
            "fallback_used": False,
            "evidence_backed": True,
            "reason": "predicate counterfactual evidence is available",
        },
    )
    document = json.loads(actual["messages"][1]["content"])
    assert document["mechanical_evidence_interpretation"]["supported_item_decisions"] == [
        {"item": -3, "predicate_should_keep": False},
        {"item": 0, "predicate_should_keep": True},
    ]
    assert document["forbidden_no_op_expression"] == "lt(item,0)"
    assert any("current expression" in requirement for requirement in document["requirements"])


def test_prompt_labels_unsupported_deterministic_exploration() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": '{"requirements":[]}'},
        ]
    }
    state = BeamState("lt(item,0)", "item", _score(2, 3))
    actual = _augment_beam_prompt(
        payload,
        round_number=3,
        max_rounds=5,
        state=state,
        feedback={"predicate_toggles": (), "mapper_constraints": ()},
        hole="predicate",
        hole_decision={
            "fallback_used": True,
            "evidence_backed": False,
            "reason": "no local evidence; preregistered heuristic fallback",
        },
    )
    document = json.loads(actual["messages"][1]["content"])
    assert document["hole_selection"] == {
        "selected_hole": "predicate",
        "evidence_backed": False,
        "fallback_used": True,
        "label": "unsupported deterministic exploration",
        "reason": "no local evidence; preregistered heuristic fallback",
    }
    assert "beam_rank" not in document["beam_search_context"]
    assert "previous_repairs_on_this_path" not in document["beam_search_context"]


def test_elitist_pool_can_retain_a_better_parent() -> None:
    parent = BeamState("parent", "m0", _score(1, 10))
    children = [
        BeamState("child-a", "m0", _score(4, 2)),
        BeamState("child-b", "m0", _score(3, 3)),
    ]
    selected = select_beam([parent, *children], width=2, tie_seed=11)
    assert parent in selected
    assert {state.predicate for state in selected} == {"parent", "child-b"}


def test_repair_hole_fallback_alternates_deterministically_without_evidence() -> None:
    state = BeamState("p0", "m0", _score(3, 1))
    feedback = {"predicate_toggles": (), "mapper_constraints": ()}

    odd_hole, odd_decision = select_repair_hole(
        feedback,
        state,
        round_number=3,
        stall_policy="alternate-hole",
    )
    even_hole, even_decision = select_repair_hole(
        feedback,
        state,
        round_number=4,
        stall_policy="alternate-hole",
    )

    assert (odd_hole, even_hole) == ("predicate", "mapper")
    assert odd_decision["fallback_used"] is True
    assert odd_decision["evidence_backed"] is False
    assert even_decision["fallback_used"] is True
    assert even_decision["evidence_backed"] is False
    assert odd_decision["state_id"] == even_decision["state_id"]


def test_singleton_evidence_overrides_fallback_round_parity() -> None:
    state = BeamState("p0", "m0", _score(3, 1))
    predicate_feedback = {
        "predicate_toggles": ({"item": 2, "keep": True},),
        "mapper_constraints": (),
    }
    mapper_feedback = {
        "predicate_toggles": (),
        "mapper_constraints": ({"item": 2, "expected_value": 4},),
    }

    predicate_hole, predicate_decision = select_repair_hole(
        predicate_feedback,
        state,
        round_number=4,
        stall_policy="alternate-hole",
    )
    mapper_hole, mapper_decision = select_repair_hole(
        mapper_feedback,
        state,
        round_number=3,
        stall_policy="alternate-hole",
    )

    assert predicate_hole == "predicate"
    assert mapper_hole == "mapper"
    for decision in (predicate_decision, mapper_decision):
        assert decision["evidence_backed"] is True
        assert decision["fallback_used"] is False


def test_singleton_constraints_are_exact_sorted_and_source_stable() -> None:
    spec = PBESpec.model_validate(
        {
            "name": "singleton constraints",
            "signature": {"input": "List<Int>", "output": "List<Int>"},
            "examples": [
                {"input": [2], "output": [5]},
                {"input": [-1], "output": []},
                {"input": [0, 1], "output": [3]},
                {"input": [2], "output": [5]},
            ],
            "integerConstants": [-1, 0, 1, 2, 5],
        }
    )

    assert derive_singleton_constraints(spec) == {
        "predicate": (
            {"item": -1, "keep": False, "source_examples": (2,)},
            {"item": 2, "keep": True, "source_examples": (1, 4)},
        ),
        "mapper": (
            {"item": 2, "expected_value": 5, "source_examples": (1, 4)},
        ),
    }


def _semantic_state(
    predicate: str,
    mapper: str,
    loss: float,
    predicate_signature: tuple[bool, ...],
    mapper_signature: tuple[int, ...],
    *,
    proposal_rank: int,
    singleton_predicate_violations: int = 0,
    singleton_mapper_violations: int = 0,
) -> BeamState:
    return BeamState(
        predicate,
        mapper,
        _score(loss, 1),
        predicate_signature=predicate_signature,
        mapper_signature=mapper_signature,
        singleton_predicate_violations=singleton_predicate_violations,
        singleton_mapper_violations=singleton_mapper_violations,
        proposal_rank=proposal_rank,
    )


def test_semantic_selector_collapses_aliases_and_preserves_component_novelty() -> None:
    identity_a = _semantic_state(
        "eq(item,0)", "item", 17, (False, True, False), (-1, 0, 1), proposal_rank=4
    )
    identity_b = _semantic_state(
        "and(eq(item,0),lt(-2,item))",
        "add(item,0)",
        17,
        (False, True, False),
        (-1, 0, 1),
        proposal_rank=2,
    )
    square = _semantic_state(
        "eq(item,0)",
        "mul(item,item)",
        17,
        (False, True, False),
        (1, 0, 1),
        proposal_rank=0,
    )
    worse = _semantic_state(
        "lt(-1,item)", "item", 20, (False, False, True), (-1, 0, 1), proposal_rank=1
    )
    selected = select_semantic_beam(
        [identity_a, identity_b, square, worse], width=2, tie_seed=9
    )
    assert square in selected
    assert worse in selected
    assert identity_a not in selected
    assert identity_b not in selected
    assert len({(state.predicate_signature, state.mapper_signature) for state in selected}) == 2


def test_semantic_selector_is_input_order_invariant_and_never_backfills_aliases() -> None:
    first = _semantic_state("p0", "m0", 2, (True,), (0,), proposal_rank=0)
    alias = _semantic_state("p1", "m1", 2, (True,), (0,), proposal_rank=1)
    forward = select_semantic_beam([first, alias], width=2, tie_seed=5)
    reverse = select_semantic_beam([alias, first], width=2, tie_seed=5)
    assert forward == reverse
    assert len(forward) == 1
    assert semantic_id(forward[0]) == semantic_id(reverse[0])


def test_evidence_frontier_keeps_loss_anchor_and_exact_predicate_state() -> None:
    best_loss = _semantic_state(
        "best-loss-predicate",
        "best-loss-mapper",
        2,
        (False, False, True),
        (-1, 0, 1),
        proposal_rank=0,
        singleton_predicate_violations=2,
        singleton_mapper_violations=0,
    )
    exact_predicate_wrong_mapper = _semantic_state(
        "exact-predicate",
        "wrong-mapper",
        12,
        (False, True, True),
        (7, 7, 7),
        proposal_rank=2,
        singleton_predicate_violations=0,
        singleton_mapper_violations=3,
    )
    lower_loss_but_incomplete = _semantic_state(
        "incomplete-predicate",
        "incomplete-mapper",
        3,
        (True, True, True),
        (1, 1, 1),
        proposal_rank=1,
        singleton_predicate_violations=1,
        singleton_mapper_violations=1,
    )

    selected = select_evidence_frontier_beam(
        [lower_loss_but_incomplete, exact_predicate_wrong_mapper, best_loss],
        width=2,
        tie_seed=23,
    )

    assert selected == (best_loss, exact_predicate_wrong_mapper)


def test_evidence_frontier_collapses_observed_semantic_aliases() -> None:
    alias_late = _semantic_state(
        "alias-late-predicate",
        "alias-late-mapper",
        5,
        (False, True),
        (0, 1),
        proposal_rank=3,
        singleton_predicate_violations=0,
        singleton_mapper_violations=1,
    )
    alias_early = _semantic_state(
        "alias-early-predicate",
        "alias-early-mapper",
        5,
        (False, True),
        (0, 1),
        proposal_rank=1,
        singleton_predicate_violations=0,
        singleton_mapper_violations=1,
    )
    other_cell = _semantic_state(
        "other-predicate",
        "other-mapper",
        2,
        (True, True),
        (2, 2),
        proposal_rank=0,
        singleton_predicate_violations=1,
        singleton_mapper_violations=0,
    )

    selected = select_evidence_frontier_beam(
        [alias_late, other_cell, alias_early],
        width=3,
        tie_seed=29,
    )

    assert selected == (other_cell, alias_early)
    assert alias_late not in selected
    assert len({semantic_id(state) for state in selected}) == len(selected)


def test_evidence_frontier_is_invariant_to_candidate_input_order() -> None:
    candidates = (
        _semantic_state(
            "p-best",
            "m-best",
            1,
            (False, False),
            (0, 0),
            proposal_rank=3,
            singleton_predicate_violations=2,
            singleton_mapper_violations=0,
        ),
        _semantic_state(
            "p-frontier",
            "m-frontier",
            8,
            (False, True),
            (1, 1),
            proposal_rank=2,
            singleton_predicate_violations=0,
            singleton_mapper_violations=2,
        ),
        _semantic_state(
            "p-novel",
            "m-novel",
            4,
            (True, True),
            (2, 2),
            proposal_rank=1,
            singleton_predicate_violations=1,
            singleton_mapper_violations=1,
        ),
        _semantic_state(
            "p-dominated",
            "m-dominated",
            3,
            (True, False),
            (3, 3),
            proposal_rank=0,
            singleton_predicate_violations=2,
            singleton_mapper_violations=1,
        ),
    )
    expected = select_evidence_frontier_beam(
        list(candidates),
        width=3,
        tie_seed=31,
    )

    for ordering in permutations(candidates):
        assert (
            select_evidence_frontier_beam(
                list(ordering),
                width=3,
                tie_seed=31,
            )
            == expected
        )


def test_evidence_frontier_rejects_width_one() -> None:
    candidate = _semantic_state(
        "predicate",
        "mapper",
        1,
        (True,),
        (0,),
        proposal_rank=0,
    )

    with pytest.raises(ValueError, match="beam width at least two"):
        select_evidence_frontier_beam([candidate], width=1, tie_seed=37)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("singleton_predicate_violations", -1),
        ("singleton_mapper_violations", -1),
        ("singleton_predicate_violations", 0.5),
        ("singleton_mapper_violations", 0.5),
        ("singleton_predicate_violations", True),
        ("singleton_mapper_violations", False),
    ),
)
def test_evidence_frontier_rejects_invalid_violation_counts(
    field: str,
    invalid_value: object,
) -> None:
    valid = _semantic_state(
        "predicate",
        "mapper",
        1,
        (True,),
        (0,),
        proposal_rank=0,
    )
    invalid = replace(valid, **{field: invalid_value})

    with pytest.raises(ValueError, match="violation count must be a nonnegative int"):
        select_evidence_frontier_beam([invalid], width=2, tie_seed=41)

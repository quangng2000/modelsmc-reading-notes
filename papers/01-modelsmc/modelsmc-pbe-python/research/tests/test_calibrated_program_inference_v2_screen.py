from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from research.calibrated_program_inference_v2_screen import (
    ANNEAL_CONDITIONAL_ESS,
    ARMS,
    IDENTITY_TOLERANCE,
    MAX_ALIASES_PER_MODE,
    MAX_MODE_COUNT,
    RUN_SCHEMA,
    STUDY_SCHEMA,
    ModeBank,
    Proposal,
    ScreenError,
    _conditional_ess,
    _semantic_signature,
    build_frozen_protocol,
    build_proposal,
    canonical_bytes,
    freeze_protocol,
    independent_mh_move,
    load_frozen_plan,
    next_beta,
    run_repetition,
    sampled_semantic_mode_bank,
)
from research.particle_calibration_terminal_diagnostic_v2 import (
    TaskBinding,
    TerminalTask,
)

PROJECT_ROOT = Path(__file__).parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def bounded_task() -> TerminalTask:
    path = PROJECT_ROOT / "examples/foldr-bounded-square.json"
    return TerminalTask(TaskBinding("foldr-bounded-square", path, _sha256(path)))


@pytest.fixture(scope="module")
def sampled_bank(bounded_task: TerminalTask) -> ModeBank:
    return sampled_semantic_mode_bank(bounded_task)


def test_sampled_mode_bank_is_deterministic_and_semantically_distinct(
    bounded_task: TerminalTask,
) -> None:
    first = sampled_semantic_mode_bank(bounded_task)
    second = sampled_semantic_mode_bank(bounded_task)

    assert first == second
    assert len(first.groups) == MAX_MODE_COUNT
    assert all(1 <= len(group) <= MAX_ALIASES_PER_MODE for group in first.groups)
    representatives = [group[0] for group in first.groups]
    assert len({_semantic_signature(bounded_task, index) for index in representatives}) == len(
        representatives
    )
    assert first.metadata["logical_discovery_draws"] == 4096
    assert first.metadata["retained_syntax_centers"] == sum(
        len(group) for group in first.groups
    )


def test_all_proposals_are_normalized_full_support_and_exactly_accounted(
    bounded_task: TerminalTask,
    sampled_bank: ModeBank,
) -> None:
    proposals = [build_proposal(bounded_task, arm, sampled_bank) for arm in ARMS]

    assert len(proposals) == 14
    for proposal in proposals:
        assert proposal.probabilities.sum() == pytest.approx(1.0, abs=1e-15)
        assert proposal.probabilities.min() > 0.0
        assert proposal.cdf[-1] == 1.0
        assert proposal.census[
            "importance_identity_maximum_absolute_error"
        ] <= IDENTITY_TOLERANCE
        assert proposal.census["importance_identity_weight_mean"] == pytest.approx(
            1.0, abs=IDENTITY_TOLERANCE
        )
        assert proposal.census[
            "importance_identity_exact_numerator"
        ] == pytest.approx(bounded_task.exact_mass, abs=IDENTITY_TOLERANCE)

    by_id = {proposal.arm.arm_id: proposal for proposal in proposals}
    assert by_id["grammar-terminal-is"].bank_indices == ()
    assert by_id["sampled-modes-k8-a025-terminal-is"].semantic_mode_count == 8
    assert by_id["sampled-modes-k16-a025-terminal-is"].semantic_mode_count == 16
    assert len(by_id["sampled-modes-k8-a025-terminal-is"].bank_indices) <= (
        8 * MAX_ALIASES_PER_MODE
    )
    assert len(by_id["sampled-modes-k16-a025-terminal-is"].bank_indices) <= (
        16 * MAX_ALIASES_PER_MODE
    )
    assert len(
        by_id["exhaustive-top64-a050-terminal-positive-control"].bank_indices
    ) == 64
    high_center = by_id["sampled-modes-k8-a025-e075-terminal-is"]
    assert high_center.arm.center_mass == 0.75


def test_adaptive_temperature_hits_conditional_ess_target() -> None:
    weights = np.full(128, 1.0 / 128, dtype=np.float64)
    log_likelihood = np.linspace(-20.0, 0.0, 128, dtype=np.float64)

    beta = next_beta(weights, log_likelihood, 0.0)

    assert 0.0 < beta < 1.0
    assert _conditional_ess(weights, log_likelihood, beta) == pytest.approx(
        ANNEAL_CONDITIONAL_ESS * 128,
        rel=1e-10,
    )


def test_independent_mh_preserves_a_small_declared_target() -> None:
    rng = np.random.Generator(np.random.PCG64(20260814))
    grammar = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    log_likelihood = np.log(np.asarray([1.0, 4.0, 2.0], dtype=np.float64))
    target = grammar * np.exp(log_likelihood)
    target /= target.sum()
    q = np.asarray([0.7, 0.2, 0.1], dtype=np.float64)
    cdf = np.cumsum(q)
    cdf[-1] = 1.0
    arm = ARMS[0]
    proposal = Proposal(arm, q, cdf, (), 0, {})
    states = rng.choice(3, size=200_000, p=target).astype(np.int64)

    moved, accepted = independent_mh_move(
        states,
        log_tempered_target=np.log(grammar) + log_likelihood,
        proposal=proposal,
        rng=rng,
    )
    observed = np.bincount(moved, minlength=3) / moved.size

    assert 0 < accepted < moved.size
    assert observed == pytest.approx(target, abs=0.004)


def test_terminal_and_annealed_repetitions_are_deterministic(
    bounded_task: TerminalTask,
    sampled_bank: ModeBank,
) -> None:
    selected_arms = (ARMS[3], ARMS[5], ARMS[6], ARMS[12])
    for arm in selected_arms:
        proposal = build_proposal(bounded_task, arm, sampled_bank)
        keyword = {
            "particles": 128,
            "repetition": 0,
            "base_seed": 913_001,
        }
        first = run_repetition(bounded_task, proposal, **keyword)
        second = run_repetition(bounded_task, proposal, **keyword)
        assert first == second
        assert first["schema"] == RUN_SCHEMA
        assert first["provider_calls"] == 0
        diagnostics = cast(dict[str, float], first["diagnostics"])
        assert 1.0 <= diagnostics["importance_ess"] <= 128.0
        if arm.algorithm in {"annealed-smc", "proposal-bridge-smc"}:
            stages = cast(list[dict[str, float]], first["stages"])
            assert stages
            assert stages[-1]["beta_current"] == 1.0


def test_protocol_freezes_deterministically_and_fails_closed(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_sha = freeze_protocol(first_path, PROJECT_ROOT)
    second_sha = freeze_protocol(second_path, PROJECT_ROOT)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_sha == second_sha == _sha256(first_path)
    assert first_path.read_bytes() == canonical_bytes(json.loads(first_path.read_text())) + b"\n"
    plan, protocol = load_frozen_plan(
        first_path,
        PROJECT_ROOT,
        expected_protocol_sha256=first_sha,
    )
    assert protocol["schema"] == STUDY_SCHEMA
    assert plan.arms == ARMS
    assert len(plan.tasks) == 4
    with pytest.raises(ScreenError, match="externally expected"):
        load_frozen_plan(
            first_path,
            PROJECT_ROOT,
            expected_protocol_sha256="0" * 64,
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_protocol(first_path, PROJECT_ROOT)


def test_protocol_declares_reused_task_and_positive_control_boundaries() -> None:
    protocol = build_frozen_protocol(PROJECT_ROOT)
    assert protocol["schema"] == STUDY_SCHEMA
    authorization = cast(dict[str, object], protocol["authorization"])
    assert authorization == {
        "provider_calls": False,
        "new_model_responses": False,
        "fresh_tasks": False,
        "exact_reference_enumeration": True,
    }
    assert any(
        "positive control" in statement
        for statement in cast(list[str], protocol["claim_boundary"])
    )

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import cast

import pytest

from research.particle_calibration_terminal_diagnostic_v2 import (
    ANALYSIS_SCHEMA,
    ARMS,
    GLOBAL_SLATE_SIZE,
    IDENTITY_TOLERANCE,
    PARTICLE_COUNTS,
    REPETITION_SEEDS,
    REPETITIONS,
    RUN_SCHEMA,
    STUDY_SCHEMA,
    DiagnosticError,
    TaskBinding,
    TerminalTask,
    analyze_runs,
    build_frozen_protocol,
    canonical_bytes,
    freeze_protocol,
    load_frozen_plan,
    run_repetition,
)

PROJECT_ROOT = Path(__file__).parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def bounded_task() -> TerminalTask:
    path = PROJECT_ROOT / "examples/foldr-bounded-square.json"
    return TerminalTask(TaskBinding("foldr-bounded-square", path, _sha256(path)))


def test_exact_population_and_six_proposals_are_normalized(
    bounded_task: TerminalTask,
) -> None:
    assert len(bounded_task.keys) == 36_000
    assert bounded_task.exact_mass == pytest.approx(0.494441737003204)
    assert bounded_task.exact.sum() == 2
    assert bounded_task.score_key(bounded_task.fixed_parent).exact_program is True
    assert len(ARMS) == 6

    reference = bounded_task.reference_record(ARMS)
    census = cast(dict[str, dict[str, object]], reference["arm_census"])
    for arm in ARMS:
        distribution = bounded_task.arm_distribution(arm)
        assert distribution.probabilities.sum() == pytest.approx(1.0, abs=1e-15)
        assert distribution.probabilities.min() > 0.0
        assert census[arm.arm_id][
            "importance_identity_maximum_absolute_error"
        ] <= IDENTITY_TOLERANCE
        assert census[arm.arm_id]["importance_identity_weight_mean"] == pytest.approx(
            1.0, abs=IDENTITY_TOLERANCE
        )
        assert census[arm.arm_id][
            "importance_identity_exact_numerator"
        ] == pytest.approx(bounded_task.exact_mass, abs=IDENTITY_TOLERANCE)


def test_arms_isolate_sticky_nonsticky_and_global_mechanisms(
    bounded_task: TerminalTask,
) -> None:
    sticky = bounded_task.arm_distribution(ARMS[0])
    grammar_only = bounded_task.arm_distribution(ARMS[3])
    nonsticky = bounded_task.arm_distribution(ARMS[4])
    positive = bounded_task.arm_distribution(ARMS[5])

    assert len(set(sticky.local_indices)) == 1
    assert sticky.local_indices == (bounded_task.fixed_parent_index,) * 4
    assert grammar_only.arm.epsilon == 1.0
    assert len(set(nonsticky.local_indices)) == 4
    assert bounded_task.fixed_parent_index not in nonsticky.local_indices
    assert len(set(positive.local_indices)) == GLOBAL_SLATE_SIZE
    assert positive.arm.role == "posthoc-developmental-positive-control"
    assert positive.census["population_importance_ess_fraction"] > sticky.census[
        "population_importance_ess_fraction"
    ]


def test_terminal_repetition_is_seeded_finite_and_budgeted(
    bounded_task: TerminalTask,
) -> None:
    distribution = bounded_task.arm_distribution(ARMS[0])
    keyword = {
        "particles": 256,
        "repetition": 0,
        "base_seed": REPETITION_SEEDS[0],
    }
    first = run_repetition(bounded_task, distribution, **keyword)
    second = run_repetition(bounded_task, distribution, **keyword)

    assert first == second
    assert first["schema"] == RUN_SCHEMA
    assert first["provider_calls"] == 0
    assert first["terminal_proposal_draws"] == 256
    assert all(
        math.isfinite(float(value))
        for value in cast(dict[str, float], first["estimate"]).values()
    )
    diagnostics = cast(dict[str, float], first["diagnostics"])
    assert 1.0 <= diagnostics["importance_ess"] <= 256.0
    assert 0.0 < diagnostics["maximum_normalized_weight"] <= 1.0


def test_analysis_keeps_positive_control_diagnostic_not_a_gate(
    bounded_task: TerminalTask,
) -> None:
    references = {
        bounded_task.binding.task_id: bounded_task.reference_record(ARMS),
    }
    runs = [
        run_repetition(
            bounded_task,
            bounded_task.arm_distribution(arm),
            particles=512,
            repetition=repetition,
            base_seed=REPETITION_SEEDS[repetition],
        )
        for arm in ARMS
        for repetition in range(2)
    ]
    analysis = analyze_runs(
        runs,
        references,
        task_ids=[bounded_task.binding.task_id],
        arms=ARMS,
        particle_counts=[512],
        repetitions=2,
    )
    assert analysis["schema"] == ANALYSIS_SCHEMA
    conclusion = cast(dict[str, object], analysis["diagnostic_conclusion"])
    assert conclusion["not_a_calibration_gate"] is True
    assert conclusion["v1_gate_remains_failed_and_unchanged"] is True
    assert conclusion["exact_importance_identities_pass"] is True
    assert conclusion[
        "positive_control_population_ess_exceeds_sticky_for_every_task"
    ] is True


def test_protocol_builder_declares_exact_budget_and_posthoc_boundary() -> None:
    protocol = build_frozen_protocol(PROJECT_ROOT)
    assert protocol["schema"] == STUDY_SCHEMA
    design = cast(dict[str, object], protocol["design"])
    assert design["particle_counts"] == list(PARTICLE_COUNTS)
    assert design["repetitions"] == REPETITIONS
    assert design["total_repetitions"] == 6_144
    assert design["total_terminal_proposal_draws"] == 2_359_296
    arm_boundary = cast(dict[str, object], protocol["arm_boundaries"])
    positive = cast(dict[str, object], arm_boundary["global_positive_control"])
    assert positive["classification"] == (
        "posthoc developmental exhaustive positive control"
    )
    assert positive["search_efficiency_claim_authorized"] is False
    assert cast(dict[str, object], protocol["v1_preservation"])[
        "v1_primary_gate_pass"
    ] is False


def test_protocol_freezes_deterministically_and_fails_closed(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first_sha = freeze_protocol(first, PROJECT_ROOT)
    second_sha = freeze_protocol(second, PROJECT_ROOT)
    assert first.read_bytes() == second.read_bytes()
    assert first_sha == second_sha == _sha256(first)
    assert first.read_bytes() == canonical_bytes(json.loads(first.read_text())) + b"\n"

    plan, protocol = load_frozen_plan(
        first,
        PROJECT_ROOT,
        expected_protocol_sha256=first_sha,
    )
    assert protocol["schema"] == STUDY_SCHEMA
    assert len(plan.tasks) == 4
    assert plan.arms == ARMS
    assert plan.particle_counts == PARTICLE_COUNTS
    assert len(plan.repetition_seeds) == REPETITIONS
    with pytest.raises(DiagnosticError, match="externally expected"):
        load_frozen_plan(
            first,
            PROJECT_ROOT,
            expected_protocol_sha256="0" * 64,
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_protocol(first, PROJECT_ROOT)

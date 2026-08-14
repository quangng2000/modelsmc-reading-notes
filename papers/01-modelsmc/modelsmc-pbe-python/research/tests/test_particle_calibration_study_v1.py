from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from research.evidence_shortlist_smc import ProgramKey
from research.particle_calibration_study_v1 import (
    ANALYSIS_SCHEMA,
    ORACLE_ID,
    RUN_SCHEMA,
    STUDY_SCHEMA,
    CalibrationError,
    CalibrationTask,
    StudyPlan,
    TaskBinding,
    analyze_runs,
    canonical_bytes,
    load_frozen_plan,
    run_repetition,
    run_study,
)

PROJECT_ROOT = Path(__file__).parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tiny_task(path: Path) -> TaskBinding:
    task = {
        "name": "tiny provider-free calibration fixture",
        "signature": {"input": "List<Int>", "output": "List<Int>"},
        "examples": [
            {"input": [], "output": []},
            {"input": ["-1"], "output": []},
            {"input": ["0"], "output": []},
            {"input": ["1"], "output": ["2"]},
            {"input": ["-1", "0", "1", "1"], "output": ["2", "2"]},
        ],
        "integerConstants": ["-1", "0", "1"],
        "particles": 4,
        "iterations": 4,
        "cloneProbability": 0,
        "essThreshold": 1,
        "seed": 17,
        "lossScale": 0.75,
        "costScale": 0.02,
        "lossCap": 1000,
        "maxCost": 30,
        "maxDepth": 12,
        "maxNodes": 191,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(task) + b"\n")
    return TaskBinding("tiny", path, _sha256(path))


def _tiny_task(tmp_path: Path) -> CalibrationTask:
    return CalibrationTask(_write_tiny_task(tmp_path / "tiny.json"), start_seed=17)


def test_declared_task_targets_are_in_grammar_and_exact() -> None:
    targets = {
        "examples/foldr-bounded-square.json": ProgramKey(
            "and(lt(-2,item),lt(item,3))",
            "mul(item,item)",
        ),
        "examples/calibration-filter-map-lower-shift-v1.json": ProgramKey(
            "lt(-1,item)",
            "add(item,1)",
        ),
        "examples/calibration-filter-map-upper-negate-v1.json": ProgramKey(
            "lt(item,2)",
            "sub(0,item)",
        ),
        "examples/calibration-filter-map-equality-scale-v1.json": ProgramKey(
            "eq(item,1)",
            "mul(item,3)",
        ),
    }
    for relative, target in targets.items():
        path = PROJECT_ROOT / relative
        task = CalibrationTask(TaskBinding(path.stem, path, _sha256(path)), start_seed=17)
        assert target.predicate in task.predicate_catalog
        assert target.mapper in task.mapper_catalog
        assert task.score_key(target).exact_program is True
        assert "target" not in json.loads(path.read_text(encoding="utf-8"))


def test_exact_reference_and_oracle_are_deterministic(tmp_path: Path) -> None:
    task = _tiny_task(tmp_path)
    reference = task.exact_reference(limit=10_000, evidence_scale=2.0)

    assert reference["program_syntaxes"] == len(task.predicate_order) * len(
        task.mapper_order
    )
    assert reference["reference_complete_program_evaluations"] == reference[
        "program_syntaxes"
    ]
    assert 0.0 < cast(float, reference["exact_target_mass"]) < 1.0
    assert cast(float, reference["target_mean_loss"]) >= 0.0

    hole, _ = task.select_hole(task.initial, round_number=1)
    before = task.work_counters()
    first = task.oracle_slots(task.initial, hole)
    middle = task.work_counters()
    second = task.oracle_slots(task.initial, hole)
    after = task.work_counters()
    assert first == second
    assert len(first) == len(set(first)) == 4
    assert all(slot != task.initial.key for slot in first)
    if hole == "predicate":
        assert all(slot.mapper == task.initial.key.mapper for slot in first)
    else:
        assert all(slot.predicate == task.initial.key.predicate for slot in first)
    assert middle["oracle_rankings"] == before["oracle_rankings"] + 1
    assert after["oracle_slot_cache_hits"] == middle["oracle_slot_cache_hits"] + 1
    assert task.oracle_inventory()["target_weights_or_reference_aggregates_used"] is False


def test_repetition_is_seeded_provider_free_and_budgeted(tmp_path: Path) -> None:
    first_task = _tiny_task(tmp_path / "first")
    second_task = _tiny_task(tmp_path / "second")
    first_reference = first_task.exact_reference(limit=10_000, evidence_scale=2.0)
    second_reference = second_task.exact_reference(limit=10_000, evidence_scale=2.0)

    first = run_repetition(
        first_task,
        first_reference,
        particles=8,
        repetition=0,
        base_seed=741001,
        epsilon=0.05,
        evidence_scale=2.0,
    )
    second = run_repetition(
        second_task,
        second_reference,
        particles=8,
        repetition=0,
        base_seed=741001,
        epsilon=0.05,
        evidence_scale=2.0,
    )

    assert first == second
    assert first["schema"] == RUN_SCHEMA
    assert first["proposal_source"] == ORACLE_ID
    assert first["provider_calls"] == 0
    assert first["logical_complete_program_executions"] == 33
    assert first["sampled_complete_program_slots"] == 33
    assert len(cast(list[object], first["stages"])) == 4
    assert all(
        math.isfinite(cast(float, value))
        for value in cast(dict[str, float], first["estimate"]).values()
    )


def _synthetic_run(
    exact_estimate: float,
    loss_estimate: float,
    *,
    repetition: int,
    particles: int = 256,
) -> dict[str, object]:
    return {
        "task_id": "synthetic",
        "particles": particles,
        "repetition": repetition,
        "estimate": {
            "exact_target_mass": exact_estimate,
            "target_mean_loss": loss_estimate,
        },
        "reference": {"exact_target_mass": 0.3, "target_mean_loss": 3.0},
        "diagnostics": {
            "final_ess": 2.0 + repetition,
            "final_relative_ess": (2.0 + repetition) / particles,
            "maximum_normalized_weight": 0.5,
            "found_exact": repetition == 0,
        },
        "proposal_work": {
            "oracle_candidate_score_lookups": 10 + repetition,
            "oracle_rankings": 2 + repetition,
        },
    }


def test_analysis_computes_signed_bias_rmse_ess_and_scaling() -> None:
    analysis = analyze_runs(
        [
            _synthetic_run(0.2, 2.0, repetition=0),
            _synthetic_run(0.4, 4.0, repetition=1),
        ],
        task_ids=["synthetic"],
        particle_counts=[256],
        repetitions=2,
    )
    assert analysis["schema"] == ANALYSIS_SCHEMA
    per_task = cast(dict[str, object], analysis["per_task"])
    by_n = cast(dict[str, object], cast(dict[str, object], per_task["synthetic"])[
        "by_particle_count"
    ])
    summary = cast(dict[str, object], by_n["256"])
    exact = cast(dict[str, float], summary["exact_mass"])
    loss = cast(dict[str, float], summary["target_mean_loss"])
    assert exact["bias"] == pytest.approx(0.0)
    assert exact["rmse"] == pytest.approx(0.1)
    assert loss["bias"] == pytest.approx(0.0)
    assert loss["rmse"] == pytest.approx(1.0)
    assert summary["exact_discovery_rate"] == pytest.approx(0.5)
    gate = cast(dict[str, object], analysis["primary_gate"])
    assert gate["exact_mass_rmse_pass"] is False  # Strictly less than 0.10.
    assert gate["exact_mass_signed_bias_pass"] is True
    assert gate["overall_pass"] is False
    assert gate["per_task_pass_required"] is False


def test_study_artifact_is_deterministic_sealed_and_fail_closed(tmp_path: Path) -> None:
    binding = _write_tiny_task(tmp_path / "task.json")
    plan = StudyPlan(
        tasks=(binding,),
        particle_counts=(256,),
        repetition_seeds=(741001, 741002),
        epsilon=0.05,
        evidence_scale=2.0,
        start_seed=17,
        exact_reference_limit=10_000,
        protocol_sha256="a" * 64,
        harness_sha256="b" * 64,
    )
    output = tmp_path / "complete"
    result = run_study(plan, {"schema": "test-protocol"}, output)
    assert result["status"] == "complete"
    assert result["run_count"] == 2
    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    records = cast(list[dict[str, object]], inventory["records"])
    assert inventory["file_count"] == len(records)
    for record in records:
        path = output / cast(str, record["path"])
        assert record["sha256"] == _sha256(path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_study(plan, {}, output)

    failing = replace(plan, exact_reference_limit=1)
    failure_output = tmp_path / "failed"
    with pytest.raises(CalibrationError, match="exceeds limit"):
        run_study(failing, {}, failure_output)
    failure = json.loads(
        (failure_output / "failure.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed-closed"
    assert (failure_output / "inventory.json").is_file()


def test_frozen_protocol_and_all_bindings_validate() -> None:
    protocol_path = PROJECT_ROOT / "research/protocol-particle-calibration-v1.json"
    protocol_sha256 = _sha256(protocol_path)
    plan, protocol = load_frozen_plan(
        protocol_path,
        PROJECT_ROOT,
        expected_protocol_sha256=protocol_sha256,
    )
    assert protocol["schema"] == STUDY_SCHEMA
    assert plan.protocol_sha256 == protocol_sha256
    assert plan.particle_counts == (32, 64, 128, 256, 512)
    assert len(plan.tasks) == 4
    assert len(plan.repetition_seeds) == 32

    with pytest.raises(CalibrationError, match="externally expected"):
        load_frozen_plan(
            protocol_path,
            PROJECT_ROOT,
            expected_protocol_sha256="0" * 64,
        )

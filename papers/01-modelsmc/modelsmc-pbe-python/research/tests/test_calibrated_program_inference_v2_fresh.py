from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.calibrated_program_inference_v2_fresh import (
    ALIASES_PER_MODE,
    ARM_PARTICLES,
    BOOTSTRAP_REPLICATES,
    CENTER_MASS,
    DISCOVERY_BUDGET,
    FROZEN_ARMS,
    GRAMMAR_MASS,
    MODE_COUNT,
    PRIMARY_ARM_ID,
    PRIMARY_PARTICLE_COUNT,
    PROTOCOL_STATUS,
    REPETITIONS,
    STUDY_SCHEMA,
    TASK_COUNT,
    TASK_IDS,
    FreshCalibrationError,
    _read_object,
    _target_draw,
    _task_document,
    _task_first_bootstrap,
    analyze,
    build_protocol,
    generate_suite,
    prepare_secret,
    seal_custody,
    seal_method,
    secret_commitment,
    validate_artifact,
)
from research.calibrated_program_inference_v2_screen import canonical_bytes

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_protocol(path: Path) -> str:
    path.write_bytes(canonical_bytes(build_protocol(PROJECT_ROOT)) + b"\n")
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_freezes_selected_method_and_gate() -> None:
    protocol = build_protocol(PROJECT_ROOT)
    assert protocol["schema"] == STUDY_SCHEMA
    assert protocol["status"] == PROTOCOL_STATUS
    assert protocol["authorization"]["provider_calls"] is False
    assert protocol["method"]["mode_count"] == MODE_COUNT
    assert protocol["method"]["maximum_syntax_aliases_per_mode"] == ALIASES_PER_MODE
    assert protocol["method"]["discovery_grammar_draws"] == DISCOVERY_BUDGET
    assert protocol["method"]["grammar_mass"] == GRAMMAR_MASS
    assert protocol["method"]["local_center_mass"] == CENTER_MASS
    assert protocol["design"]["primary_particle_count"] == PRIMARY_PARTICLE_COUNT
    assert "upper bound" in protocol["primary_gate"]["rmse"]
    assert protocol["task_distribution"]["task_count"] == TASK_COUNT


def test_target_draw_is_deterministic_and_informative() -> None:
    secret = bytes(range(32))
    first = _target_draw(secret, TASK_IDS[0])
    second = _target_draw(secret, TASK_IDS[0])
    assert first == second
    document = _task_document(first, secret)
    assert len(document["examples"]) == 13
    singleton_outputs = [example["output"] for example in document["examples"][1:9]]
    retained = [output for output in singleton_outputs if output]
    assert 3 <= len(retained) <= 5
    assert len({tuple(output) for output in retained}) >= 3


def test_task_first_bootstrap_is_deterministic() -> None:
    errors = np.linspace(-0.08, 0.08, TASK_COUNT * REPETITIONS).reshape(
        TASK_COUNT, REPETITIONS
    )
    first = _task_first_bootstrap(errors, replicates=500, seed=123)
    second = _task_first_bootstrap(errors, replicates=500, seed=123)
    assert first == second
    assert first["replicates"] == 500
    assert first["rmse_upper_95"] > 0
    assert len(first["bias_interval_90"]) == 2


def test_full_pre_generation_chain_and_public_boundary(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol_sha = _write_protocol(protocol_path)
    method_seal_path = tmp_path / "method-seal.json"
    method_seal_sha = seal_method(
        PROJECT_ROOT, protocol_path, protocol_sha, method_seal_path
    )
    secret_path = tmp_path / "secret.bin"
    commitment = prepare_secret(
        PROJECT_ROOT,
        protocol_path,
        protocol_sha,
        method_seal_path,
        method_seal_sha,
        secret_path,
    )
    assert commitment == secret_commitment(secret_path.read_bytes())
    custody_path = tmp_path / "custody.json"
    custody_sha = seal_custody(
        PROJECT_ROOT,
        protocol_path,
        protocol_sha,
        method_seal_path,
        method_seal_sha,
        secret_path,
        custody_path,
    )
    suite = tmp_path / "suite"
    manifest = generate_suite(
        PROJECT_ROOT,
        protocol_path,
        protocol_sha,
        method_seal_path,
        method_seal_sha,
        custody_path,
        custody_sha,
        secret_path,
        suite,
    )
    assert manifest["task_count"] == TASK_COUNT
    public_text = "\n".join(
        path.read_text() for path in (suite / "public").glob("*.json")
    )
    for forbidden in ("predicate_dsl", "mapper_dsl", "secret_hex", "rejections"):
        assert forbidden not in public_text
    reveal = _read_object(suite / "private/reveal.json")
    assert reveal["secret_hex"] == secret_path.read_bytes().hex()


def test_prepare_secret_fails_on_unsealed_or_drifted_method(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol_sha = _write_protocol(protocol_path)
    fake_seal = tmp_path / "fake-seal.json"
    fake_seal.write_text("{}\n")
    with pytest.raises(FreshCalibrationError):
        prepare_secret(
            PROJECT_ROOT,
            protocol_path,
            protocol_sha,
            fake_seal,
            "0" * 64,
            tmp_path / "secret.bin",
        )


def test_artifact_inventory_detects_tampering(tmp_path: Path) -> None:
    from research.calibrated_program_inference_v2_fresh import _seal_inventory

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "analysis.json").write_text("{}\n")
    _seal_inventory(artifact)
    validate_artifact(artifact)
    (artifact / "analysis.json").write_text('{"changed":true}\n')
    with pytest.raises(FreshCalibrationError):
        validate_artifact(artifact)


def test_frozen_constants_are_not_accidental_developmental_defaults() -> None:
    protocol = build_protocol(PROJECT_ROOT)
    arms = {record["arm_id"]: record for record in protocol["arms"]}
    assert set(arms[PRIMARY_ARM_ID]["particle_counts"]) == {256, 512, 1024}
    assert protocol["design"]["bootstrap_replicates"] == BOOTSTRAP_REPLICATES
    assert protocol["design"]["repetitions_per_task_arm_particle_count"] == 128
    candidate = json.loads(
        (
            PROJECT_ROOT
            / "research/calibrated-program-inference-v2-method-candidate.json"
        ).read_text()
    )
    assert candidate["status"] == (
        "selected-after-developmental-screen-before-fresh-task-generation"
    )


def test_analysis_reconstructs_complete_synthetic_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "research.calibrated_program_inference_v2_fresh.BOOTSTRAP_REPLICATES", 200
    )
    runs = []
    for task_index, task_id in enumerate(TASK_IDS):
        for arm in FROZEN_ARMS:
            for particles in ARM_PARTICLES[arm.arm_id]:
                for repetition in range(REPETITIONS):
                    signed = 0.01 * (-1 if (task_index + repetition) % 2 else 1)
                    runs.append(
                        {
                            "task_id": task_id,
                            "arm_id": arm.arm_id,
                            "particles": particles,
                            "repetition": repetition,
                            "logical_proposal_draws": particles,
                            "error": {
                                "exact_mass_signed": signed,
                                "target_mean_loss_signed": signed,
                            },
                            "diagnostics": {
                                "relative_importance_ess": 0.75,
                                "maximum_normalized_weight": 0.02,
                            },
                        }
                    )
    references = {
        task_id: {
            "proposal_census": {
                PRIMARY_ARM_ID: {
                    "importance_identity_maximum_absolute_error": 1e-15
                }
            }
        }
        for task_id in TASK_IDS
    }
    result = analyze(runs, references)
    assert result["primary_gate"]["pass"] is True
    assert result["primary_gate"]["checks"]["identity"] is True

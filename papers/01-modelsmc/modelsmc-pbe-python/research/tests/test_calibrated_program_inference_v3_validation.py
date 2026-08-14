from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import research.calibrated_program_inference_v3_validation as validation
from research.calibrated_program_inference_v3_validation import (
    ANALYSIS_SCHEMA,
    ARM_ALGORITHMS,
    ARM_PARTICLES,
    BOOTSTRAP_SEED,
    EXPECTED_CELLS,
    EXPECTED_RUN_COUNT,
    FACTORIZED_IS_ARM_ID,
    METADATA_SCHEMA,
    PARTICLE_COUNTS,
    PRIMARY_ARM_ID,
    PRIMARY_PARTICLE_COUNT,
    REFERENCE_SCHEMA,
    REPETITION_SEEDS,
    REPETITIONS,
    RUN_SCHEMA,
    STICKY_IS_ARM_ID,
    STUDY_SCHEMA,
    TASK_COUNT,
    TASK_IDS,
    V3ValidationError,
    analyze,
    canonical_bytes,
    seal_inventory,
    task_first_bootstrap,
    validate_artifact,
    validate_inventory,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _references() -> dict[str, Mapping[str, object]]:
    return {
        task_id: {
            "schema": REFERENCE_SCHEMA,
            "task_id": task_id,
            "task_sha256": _digest(f"task:{task_id}"),
            "provider_calls": 0,
            "program_syntaxes": 36_000,
            "exact_program_syntaxes": 12,
            "exact_target_mass": 0.5,
            "target_mean_loss": 0.2,
            "program_population_sha256": _digest(f"population:{task_id}"),
            "mode_bank": {
                "acquisition_completed_before_reference_materialization": True,
                "reference_materialized_after_acquisition": True,
                "hidden_target_used": False,
                "complete_program_catalog_ranked": False,
                "predicate_syntaxes_evaluated": 600,
                "mapper_syntaxes_evaluated": 60,
                "complete_representatives_scored": 64,
            },
            "proposal_census": {
                arm_id: {
                    "importance_identity_maximum_absolute_error": 1e-15,
                    **(
                        {
                            "factorized_mass_ledger": {
                                "grammar_component_mass": 0.25,
                                "local_component_mass": 0.75,
                                "pre_normalization_sum_error": 0.0,
                            },
                            "factorized_reconstruction_maximum_absolute_error": 1e-16,
                        }
                        if arm_id != STICKY_IS_ARM_ID
                        else {}
                    ),
                }
                for arm_id in ARM_PARTICLES
            },
        }
        for task_id in TASK_IDS
    }


def _runs(
    references: Mapping[str, Mapping[str, object]],
) -> list[Mapping[str, object]]:
    result: list[Mapping[str, object]] = []
    for task_index, task_id in enumerate(TASK_IDS):
        for arm_id, particle_counts in ARM_PARTICLES.items():
            for particles in particle_counts:
                for repetition, base_seed in enumerate(REPETITION_SEEDS):
                    signed = 0.01 if (task_index + repetition) % 2 == 0 else -0.01
                    importance_ess = 0.75 * particles
                    result.append(
                        {
                            "schema": RUN_SCHEMA,
                            "task_id": task_id,
                            "task_sha256": references[task_id]["task_sha256"],
                            "arm_id": arm_id,
                            "algorithm": ARM_ALGORITHMS[arm_id],
                            "particles": particles,
                            "repetition": repetition,
                            "base_seed": base_seed,
                            "sample_seed": validation._expected_sample_seed(
                                cast(str, references[task_id]["task_sha256"]),
                                arm_id,
                                particles,
                                repetition,
                                base_seed,
                            ),
                            "provider_calls": 0,
                            "logical_proposal_draws": particles,
                            "estimate": {
                                "exact_target_mass": 0.5 + signed,
                                "target_mean_loss": 0.2 + signed,
                            },
                            "reference": {
                                "exact_target_mass": 0.5,
                                "target_mean_loss": 0.2,
                            },
                            "error": {
                                "exact_mass_signed": (0.5 + signed) - 0.5,
                                "target_mean_loss_signed": (0.2 + signed) - 0.2,
                            },
                            "diagnostics": {
                                "importance_ess": importance_ess,
                                "relative_importance_ess": importance_ess / particles,
                                "maximum_normalized_weight": 0.02,
                                "unique_terminal_programs": particles,
                                "exact_terminal_particles": particles // 2,
                                "resampling_events": 0,
                                "mh_attempts": 0,
                                "mh_accepted": 0,
                                "mh_acceptance_rate": 0.0,
                                "annealing_stages": (
                                    1 if arm_id == PRIMARY_ARM_ID else 0
                                ),
                            },
                            "stages": (
                                [
                                    {
                                        "stage": 1,
                                        "beta_previous": 0.0,
                                        "beta_current": 1.0,
                                        "relative_ess_before_optional_resampling": 0.75,
                                        "resampled": False,
                                    }
                                ]
                                if arm_id == PRIMARY_ARM_ID
                                else []
                            ),
                        }
                    )
    return result


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def _artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object]]:
    monkeypatch.setattr(validation, "BOOTSTRAP_REPLICATES", 200)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    references = _references()
    runs = _runs(references)
    analysis = analyze(runs, references)
    protocol = {
        "schema": STUDY_SCHEMA,
        "authorization": {"provider_calls": False},
        "task_distribution": {
            "task_count": TASK_COUNT,
            "task_ids": list(TASK_IDS),
        },
        "design": {
            "arms": list(ARM_PARTICLES),
            "arm_particles": {
                arm_id: list(particle_counts)
                for arm_id, particle_counts in ARM_PARTICLES.items()
            },
            "repetitions": REPETITIONS,
            "repetition_seeds": list(REPETITION_SEEDS),
            "bootstrap_replicates": validation.BOOTSTRAP_REPLICATES,
            "bootstrap_seed": validation.BOOTSTRAP_SEED,
            "expected_run_count": EXPECTED_RUN_COUNT,
            "expected_initial_proposal_draws": (
                validation.EXPECTED_INITIAL_PROPOSAL_DRAWS
            ),
        },
    }
    _write_json(artifact / "protocol.json", protocol)
    protocol_sha256 = hashlib.sha256(
        (artifact / "protocol.json").read_bytes()
    ).hexdigest()
    method_sha256 = _digest("method")
    custody_sha256 = _digest("custody")
    manifest = {
        "schema": f"{STUDY_SCHEMA}-public-suite-v1",
        "protocol_sha256": protocol_sha256,
        "method_seal_sha256": method_sha256,
        "custody_seal_sha256": custody_sha256,
        "secret_commitment_sha256": _digest("v3-secret"),
        "secret_commitment_under_v2_domain_sha256": _digest("v3-under-v2"),
        "prior_v2_secret_commitment_sha256": (
            validation.PRIOR_V2_SECRET_COMMITMENT
        ),
        "same_domain_secret_nonreuse_verified": True,
        "task_count": TASK_COUNT,
        "task_ids": list(TASK_IDS),
        "tasks": [
            {
                "task_id": task_id,
                "path": f"{task_id}.json",
                "task_sha256": references[task_id]["task_sha256"],
                "target_commitment_sha256": _digest(f"target:{task_id}"),
            }
            for task_id in TASK_IDS
        ],
    }
    _write_json(artifact / "public-suite-manifest.json", manifest)
    _write_json(artifact / "runs.json", runs)
    _write_json(artifact / "analysis.json", analysis)
    for task_id, reference in references.items():
        _write_json(artifact / "references" / f"{task_id}.json", reference)
    _write_json(
        artifact / "study-metadata.json",
        {
            "schema": METADATA_SCHEMA,
            "protocol_sha256": protocol_sha256,
            "method_seal_sha256": method_sha256,
            "custody_seal_sha256": custody_sha256,
            "public_manifest_sha256": hashlib.sha256(
                (artifact / "public-suite-manifest.json").read_bytes()
            ).hexdigest(),
            "provider_calls": 0,
            "private_reveal_read": False,
            "exact_support_materialized_after_mode_bank": True,
            "runtime": {},
        },
    )
    (artifact / "SUMMARY.md").write_text("# Synthetic V3 validation fixture\n")
    seal_inventory(artifact)
    return artifact, analysis


def test_frozen_v3_design_is_exactly_32_by_64_by_five() -> None:
    assert TASK_IDS[0] == "fresh-cal-v3-001"
    assert TASK_IDS[-1] == "fresh-cal-v3-032"
    assert len(TASK_IDS) == TASK_COUNT == 32
    assert REPETITIONS == 64
    assert REPETITION_SEEDS == tuple(range(936_001, 936_065))
    assert EXPECTED_CELLS == (
        (PRIMARY_ARM_ID, 256),
        (PRIMARY_ARM_ID, 512),
        (PRIMARY_ARM_ID, 1024),
        (FACTORIZED_IS_ARM_ID, 256),
        (STICKY_IS_ARM_ID, 256),
    )
    assert EXPECTED_RUN_COUNT == 10_240
    assert PARTICLE_COUNTS == (256, 512, 1024)
    assert PRIMARY_PARTICLE_COUNT == 256


def test_analysis_recomputes_equal_task_summaries_and_all_required_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation, "BOOTSTRAP_REPLICATES", 300)
    references = _references()
    result = analyze(_runs(references), references)
    assert result["schema"] == ANALYSIS_SCHEMA
    assert result["run_count"] == 10_240
    assert result["provider_calls"] == 0
    assert result["task_weighting"] == "equal-task"
    assert set(result["per_task"]) == set(TASK_IDS)
    pooled = result["pooled"]
    assert set(pooled[PRIMARY_ARM_ID]["by_particle_count"]) == {
        "256",
        "512",
        "1024",
    }
    assert set(pooled[FACTORIZED_IS_ARM_ID]["by_particle_count"]) == {"256"}
    assert set(pooled[STICKY_IS_ARM_ID]["by_particle_count"]) == {"256"}
    gate = result["primary_gate"]
    assert gate["pass"] is True
    assert all(gate["checks"].values())
    assert gate["maximum_identity_error"] == 1e-15
    assert max(gate["task_rmse"].values()) == pytest.approx(0.01)
    assert result["bootstrap"]["replicates"] == 300


def test_task_first_bootstrap_is_deterministic_and_uses_frozen_seed() -> None:
    errors = np.linspace(-0.02, 0.02, TASK_COUNT * REPETITIONS).reshape(
        TASK_COUNT, REPETITIONS
    )
    first = task_first_bootstrap(errors, replicates=250, seed=BOOTSTRAP_SEED)
    second = task_first_bootstrap(errors, replicates=250, seed=BOOTSTRAP_SEED)
    assert first == second
    assert first["seed"] == 937_001
    assert first["replicates"] == 250
    assert len(first["bias_interval_90"]) == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "old-v2-run", "schema"),
        ("provider_calls", 1, "provider-free"),
        ("base_seed", 1, "base-seed"),
    ],
)
def test_ledger_rejects_schema_provider_or_seed_drift(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.setattr(validation, "BOOTSTRAP_REPLICATES", 20)
    references = _references()
    runs = _runs(references)
    runs[0] = {**runs[0], field: value}
    with pytest.raises(V3ValidationError, match=message):
        analyze(runs, references)


def test_ledger_rejects_a_duplicate_even_when_row_count_is_unchanged() -> None:
    references = _references()
    runs = _runs(references)
    runs[-1] = runs[0]
    with pytest.raises(V3ValidationError, match="duplicate run row"):
        analyze(runs, references)


def test_ledger_rejects_stored_errors_that_do_not_match_endpoints() -> None:
    references = _references()
    runs = _runs(references)
    first = dict(runs[0])
    first["error"] = {
        **cast(Mapping[str, object], first["error"]),
        "exact_mass_signed": 0.123,
    }
    runs[0] = first
    with pytest.raises(V3ValidationError, match="stored run error"):
        analyze(runs, references)


def test_ledger_recomputes_sample_seed_and_rejects_extra_payload() -> None:
    references = _references()
    runs = _runs(references)
    runs[0] = {**runs[0], "sample_seed": cast(int, runs[0]["sample_seed"]) + 1}
    with pytest.raises(V3ValidationError, match="sample seed"):
        analyze(runs, references)

    runs = _runs(references)
    runs[0] = {**runs[0], "provider_payload": {"unexpected": True}}
    with pytest.raises(V3ValidationError, match="run fields differ"):
        analyze(runs, references)


def test_reference_requires_acquisition_and_factorized_mass_ledgers() -> None:
    references = _references()
    first = dict(references[TASK_IDS[0]])
    first["mode_bank"] = {
        **cast(Mapping[str, object], first["mode_bank"]),
        "acquisition_completed_before_reference_materialization": False,
    }
    references[TASK_IDS[0]] = first
    with pytest.raises(V3ValidationError, match="acquisition boundary"):
        analyze(_runs(references), references)

    references = _references()
    first = dict(references[TASK_IDS[0]])
    census = {
        arm_id: dict(cast(Mapping[str, object], value))
        for arm_id, value in cast(
            Mapping[str, Mapping[str, object]], first["proposal_census"]
        ).items()
    }
    del census[PRIMARY_ARM_ID]["factorized_mass_ledger"]
    first["proposal_census"] = census
    references[TASK_IDS[0]] = first
    with pytest.raises(V3ValidationError, match="factorized mass ledger"):
        analyze(_runs(references), references)


def test_full_artifact_validation_recomputes_analysis_byte_for_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, expected = _artifact(tmp_path, monkeypatch)
    assert validate_artifact(artifact) == expected


def test_hash_consistent_analysis_tamper_still_fails_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, analysis = _artifact(tmp_path, monkeypatch)
    changed = {**analysis, "status": "tampered-but-rehashed"}
    _write_json(artifact / "analysis.json", changed)
    (artifact / "inventory.json").unlink()
    seal_inventory(artifact)
    validate_inventory(artifact)
    with pytest.raises(V3ValidationError, match="byte-identical"):
        validate_artifact(artifact)


def test_noncanonical_analysis_bytes_fail_even_with_matching_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, analysis = _artifact(tmp_path, monkeypatch)
    (artifact / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifact / "inventory.json").unlink()
    seal_inventory(artifact)
    with pytest.raises(V3ValidationError, match="byte-identical"):
        validate_artifact(artifact)


def test_artifact_rejects_manifest_or_metadata_binding_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, _ = _artifact(tmp_path, monkeypatch)
    manifest_path = artifact / "public-suite-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["method_seal_sha256"] = _digest("changed-method")
    manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")
    (artifact / "inventory.json").unlink()
    seal_inventory(artifact)
    with pytest.raises(V3ValidationError, match="custody binding"):
        validate_artifact(artifact)


def test_inventory_rejects_safe_path_escape_and_symlink(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "one.txt").write_text("one\n")
    inventory = seal_inventory(artifact)
    entries = cast(list[dict[str, object]], inventory["entries"])
    entries[0]["path"] = "../escape.txt"
    inventory["entries_sha256"] = hashlib.sha256(canonical_bytes(entries)).hexdigest()
    (artifact / "inventory.json").write_bytes(canonical_bytes(inventory) + b"\n")
    with pytest.raises(V3ValidationError, match="unsafe inventory path"):
        validate_inventory(artifact)

    symlink_artifact = tmp_path / "symlink-artifact"
    symlink_artifact.mkdir()
    target = symlink_artifact / "target.txt"
    target.write_text("target\n")
    os.symlink(target, symlink_artifact / "alias.txt")
    with pytest.raises(V3ValidationError, match="symlink"):
        seal_inventory(symlink_artifact)

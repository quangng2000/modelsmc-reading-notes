from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import research.calibrated_program_inference_v3_fresh as v3
from research.calibrated_program_inference_v2_fresh import (
    secret_commitment as v2_secret_commitment,
)
from research.particle_calibration_terminal_diagnostic_v2 import (
    TaskBinding,
    TerminalTask,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[2]
LEGACY_SUITE = (
    WORKSPACE_ROOT / "artifacts/calibrated-program-inference-v2-fresh-suite/public"
)
LEGACY_DIAGNOSTIC_REFERENCES = (
    WORKSPACE_ROOT
    / "artifacts/calibrated-program-inference-v3-factorized-diagnostic/references"
)


def _binding(task_id: str) -> TaskBinding:
    path = LEGACY_SUITE / f"{task_id}.json"
    return TaskBinding(task_id, path, hashlib.sha256(path.read_bytes()).hexdigest())


def _frozen_chain(tmp_path: Path) -> dict[str, object]:
    protocol = tmp_path / "protocol.json"
    protocol_sha = v3.freeze_protocol(PROJECT_ROOT, protocol)
    method = tmp_path / "method.json"
    method_sha = v3.seal_method(PROJECT_ROOT, protocol, protocol_sha, method)
    secret = tmp_path / "secret.bin"
    commitment = v3.prepare_secret(
        PROJECT_ROOT,
        protocol,
        protocol_sha,
        method,
        method_sha,
        secret,
    )
    custody = tmp_path / "custody.json"
    custody_sha = v3.seal_custody(
        PROJECT_ROOT,
        protocol,
        protocol_sha,
        method,
        method_sha,
        secret,
        custody,
    )
    suite = tmp_path / "custody-suite"
    manifest = v3.generate_suite(
        PROJECT_ROOT,
        protocol,
        protocol_sha,
        method,
        method_sha,
        custody,
        custody_sha,
        secret,
        suite,
    )
    return {
        "protocol": protocol,
        "protocol_sha": protocol_sha,
        "method": method,
        "method_sha": method_sha,
        "secret": secret,
        "commitment": commitment,
        "custody": custody,
        "custody_sha": custody_sha,
        "suite": suite,
        "manifest": manifest,
    }


@pytest.fixture(scope="module")
def exact_bundle() -> tuple[TerminalTask, v3.PublicModeBank, v3.ModeBank]:
    binding = _binding("fresh-cal-v2-01")
    public_bank = v3.factorized_public_mode_bank(binding)
    task = TerminalTask(binding)
    return task, public_bank, v3.materialize_mode_bank(task, public_bank)


def test_v3_design_is_native_and_exact() -> None:
    assert v3.DEFAULT_PROTOCOL.endswith("-r2.json")
    assert v3.DEFAULT_METHOD_SEAL.endswith("-r2.method-seal.json")
    assert v3.TASK_IDS[0] == "fresh-cal-v3-001"
    assert v3.TASK_IDS[-1] == "fresh-cal-v3-032"
    assert len(v3.TASK_IDS) == v3.TASK_COUNT == 32
    assert v3.REPETITION_SEEDS == tuple(range(936_001, 936_065))
    assert v3.BOOTSTRAP_SEED == 937_001
    assert v3.PARTICLE_COUNTS == (256, 512, 1024)
    assert v3.EXPECTED_RUN_COUNT == 10_240
    assert v3.EXPECTED_INITIAL_PROPOSAL_DRAWS == 4_718_592
    assert set(v3.ARM_PARTICLES) == {
        v3.PRIMARY_ARM_ID,
        v3.FACTOR_TERMINAL_ARM_ID,
        v3.STICKY_ARM_ID,
    }


def test_protocol_binds_method_tree_and_complete_attempt_chain() -> None:
    protocol = v3.build_protocol(PROJECT_ROOT)
    assert protocol["schema"] == v3.STUDY_SCHEMA
    assert protocol["status"] == v3.STATUS
    assert protocol["authorization"]["provider_calls"] is False
    assert protocol["method"]["acquisition_before_reference_materialization"] is True
    assert protocol["design"]["expected_run_count"] == 10_240
    assert protocol["design"]["expected_initial_proposal_draws"] == 4_718_592
    assert "strictly inside (-0.03,+0.03)" in protocol["primary_gate"]["bias"]
    assert protocol["bindings"]["python_tree"]["file_count"] == 140
    assert set(protocol["bindings"]["attempt_chain"]) == {
        "v2_failed_protocol",
        "v2_failed_method_seal",
        "v2_failed_custody_seal",
        "v2_failed_public_manifest",
        "v2_failed_runs",
        "v2_failed_analysis",
        "v2_failed_inventory",
        "v2_failed_unblind",
        "v3_diagnostic_protocol",
        "v3_diagnostic_analysis",
        "v3_diagnostic_runs",
        "v3_diagnostic_inventory",
        "v3_fresh_r1_protocol_superseded_pre_secret",
        "v3_fresh_r1_method_seal_superseded_pre_secret",
        "v3_fresh_r1_supersession_record",
    }
    failed = json.loads(
        (
            WORKSPACE_ROOT
            / "artifacts/calibrated-program-inference-v2-fresh/analysis.json"
        ).read_text()
    )
    assert failed["primary_gate"]["pass"] is False


def test_public_acquisition_reproduces_all_20_sealed_diagnostic_banks() -> None:
    for index in range(1, 21):
        task_id = f"fresh-cal-v2-{index:02d}"
        bank = v3.factorized_public_mode_bank(_binding(task_id))
        actual = [asdict(key) for group in bank.groups for key in group]
        reference = json.loads(
            (LEGACY_DIAGNOSTIC_REFERENCES / f"{task_id}.json").read_text()
        )
        expected = reference["proposal_census"][
            "sampled-modes-k8-a025-e075-proposal-bridge"
        ]["bank_programs"]
        assert actual == expected
        assert bank.metadata["acquisition_completed_before_reference_materialization"]
        assert bank.metadata["complete_representatives_scored"] <= 64
        assert bank.metadata["predicate_syntaxes_evaluated"] == 600
        assert bank.metadata["mapper_syntaxes_evaluated"] == 60


def test_factorized_q_ledger_and_bridge_telescope(
    exact_bundle: tuple[TerminalTask, v3.PublicModeBank, v3.ModeBank],
) -> None:
    task, public_bank, bank = exact_bundle
    assert not hasattr(public_bank, "target")
    assert not hasattr(public_bank, "exact")
    proposal = v3.build_v3_proposal(task, v3.PRIMARY_ARM, bank)
    ledger = cast(
        dict[str, object], proposal.census["factorized_mass_ledger"]
    )
    assert ledger["grammar_component_mass"] == pytest.approx(0.25, abs=1e-12)
    assert ledger["local_component_mass"] == pytest.approx(0.75, abs=1e-12)
    assert cast(float, ledger["pre_normalization_sum_error"]) == pytest.approx(
        0.0, abs=1e-12
    )
    assert all(
        value == pytest.approx(0.75 / 8, abs=1e-12)
        for value in cast(list[float], ledger["per_mode_component_mass"])
    )
    assert proposal.census[
        "factorized_reconstruction_maximum_absolute_error"
    ] <= 1e-12
    indices = np.arange(0, len(task.keys), 137, dtype=np.int64)
    assert v3.bridge_telescoping_error(
        task, proposal, indices, (0.0, 0.1, 0.37, 0.8, 1.0)
    ) <= 1e-14


def test_v3_repetition_uses_native_seed_and_schema(
    exact_bundle: tuple[TerminalTask, v3.PublicModeBank, v3.ModeBank],
) -> None:
    task, _, bank = exact_bundle
    proposal = v3.build_v3_proposal(task, v3.PRIMARY_ARM, bank)
    run = v3.run_v3_repetition(
        task,
        proposal,
        particles=256,
        repetition=0,
        base_seed=v3.REPETITION_SEEDS[0],
    )
    assert run["schema"] == v3.RUN_SCHEMA
    assert run["arm_id"] == v3.PRIMARY_ARM_ID
    assert run["provider_calls"] == 0
    assert run["logical_proposal_draws"] == 256
    assert cast(int, run["sample_seed"]) == v3._v3_seed(
        "smc",
        task.binding.sha256,
        v3.PRIMARY_ARM_ID,
        256,
        0,
        v3.REPETITION_SEEDS[0],
    )


def test_full_pre_generation_chain_and_public_only_suite(tmp_path: Path) -> None:
    chain = _frozen_chain(tmp_path)
    secret = cast(Path, chain["secret"])
    assert secret.stat().st_mode & 0o077 == 0
    assert chain["commitment"] != v3.PRIOR_SECRET_COMMITMENT
    custody = json.loads(cast(Path, chain["custody"]).read_text())
    assert custody["same_domain_secret_nonreuse_verified"] is True
    assert (
        custody["secret_commitment_under_v2_domain_sha256"]
        != v3.PRIOR_SECRET_COMMITMENT
    )
    suite = cast(Path, chain["suite"])
    with pytest.raises(v3.FreshCalibrationError, match="only the public directory"):
        v3._validate_suite(
            suite,
            cast(str, chain["protocol_sha"]),
            cast(str, chain["method_sha"]),
            cast(str, chain["custody_sha"]),
        )
    public_only = tmp_path / "public-only"
    public_only.mkdir()
    shutil.copytree(suite / "public", public_only / "public")
    bindings, manifest = v3._validate_suite(
        public_only,
        cast(str, chain["protocol_sha"]),
        cast(str, chain["method_sha"]),
        cast(str, chain["custody_sha"]),
    )
    assert len(bindings) == 32
    assert manifest["task_ids"] == list(v3.TASK_IDS)
    public_text = "\n".join(path.read_text() for path in (public_only / "public").iterdir())
    for forbidden in ("secret_hex", "predicate_dsl", "mapper_dsl", "rejections"):
        assert forbidden not in public_text

    (public_only / "secret.bin").write_bytes(b"unexpected")
    with pytest.raises(v3.FreshCalibrationError, match="only the public directory"):
        v3._validate_suite(
            public_only,
            cast(str, chain["protocol_sha"]),
            cast(str, chain["method_sha"]),
            cast(str, chain["custody_sha"]),
        )
    (public_only / "secret.bin").unlink()
    (public_only / "public/unexpected").mkdir()
    with pytest.raises(v3.FreshCalibrationError, match="file set differs"):
        v3._validate_suite(
            public_only,
            cast(str, chain["protocol_sha"]),
            cast(str, chain["method_sha"]),
            cast(str, chain["custody_sha"]),
        )


def test_prepare_secret_rejects_same_secret_under_v2_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repeated = bytes(range(32))
    protocol = tmp_path / "protocol.json"
    protocol_sha = v3.freeze_protocol(PROJECT_ROOT, protocol)
    method = tmp_path / "method.json"
    method_sha = v3.seal_method(PROJECT_ROOT, protocol, protocol_sha, method)
    monkeypatch.setattr(v3, "PRIOR_SECRET_COMMITMENT", v2_secret_commitment(repeated))
    monkeypatch.setattr(v3.secrets, "token_bytes", lambda count: repeated)
    with pytest.raises(v3.FreshCalibrationError, match="reuse the V2 secret"):
        v3.prepare_secret(
            PROJECT_ROOT,
            protocol,
            protocol_sha,
            method,
            method_sha,
            tmp_path / "secret.bin",
        )


def test_suite_rejects_hash_consistent_path_escape(tmp_path: Path) -> None:
    chain = _frozen_chain(tmp_path)
    suite = cast(Path, chain["suite"])
    manifest_path = suite / "public/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tasks"][0]["path"] = "../private/reveal.json"
    manifest_path.write_bytes(v3.canonical_bytes(manifest) + b"\n")
    with pytest.raises(v3.FreshCalibrationError, match="path differs"):
        v3._validate_suite(
            suite,
            cast(str, chain["protocol_sha"]),
            cast(str, chain["method_sha"]),
            cast(str, chain["custody_sha"]),
            require_public_only=False,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_count", 999),
        ("task_ids", ["wrong-task"]),
    ],
)
def test_suite_rejects_declared_task_design_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    chain = _frozen_chain(tmp_path)
    suite = cast(Path, chain["suite"])
    manifest_path = suite / "public/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_bytes(v3.canonical_bytes(manifest) + b"\n")
    with pytest.raises(v3.FreshCalibrationError, match="task design"):
        v3._validate_suite(
            suite,
            cast(str, chain["protocol_sha"]),
            cast(str, chain["method_sha"]),
            cast(str, chain["custody_sha"]),
            require_public_only=False,
        )


def test_method_and_custody_seals_fail_closed(tmp_path: Path) -> None:
    chain = _frozen_chain(tmp_path)
    with pytest.raises(v3.FreshCalibrationError, match="method-seal SHA-256"):
        v3._validate_method_seal(
            PROJECT_ROOT,
            cast(Path, chain["protocol"]),
            cast(str, chain["protocol_sha"]),
            cast(Path, chain["method"]),
            "0" * 64,
        )
    custody = cast(Path, chain["custody"])
    record = json.loads(custody.read_text())
    record["same_domain_secret_nonreuse_verified"] = False
    custody.write_bytes(v3.canonical_bytes(record) + b"\n")
    custody_sha = hashlib.sha256(custody.read_bytes()).hexdigest()
    with pytest.raises(v3.FreshCalibrationError, match="content differs"):
        v3._validate_custody_seal(
            custody,
            custody_sha,
            protocol_sha256=cast(str, chain["protocol_sha"]),
            method_seal_sha256=cast(str, chain["method_sha"]),
        )


def test_unblind_replay_receipt_gate_binds_artifact_and_counts(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = {
        "method_seal_sha256": "1" * 64,
        "custody_seal_sha256": "2" * 64,
    }
    for name, value in (
        ("protocol.json", {"schema": v3.STUDY_SCHEMA}),
        ("public-suite-manifest.json", manifest),
        ("runs.json", []),
        ("analysis.json", {}),
        ("inventory.json", {}),
    ):
        (artifact / name).write_bytes(v3.canonical_bytes(value) + b"\n")
    reference_records = []
    for task_id in v3.TASK_IDS:
        path = artifact / "references" / f"{task_id}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(v3.canonical_bytes({"task_id": task_id}) + b"\n")
        reference_records.append(
            {
                "task_id": task_id,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "schema": v3.REPLAY_SCHEMA,
        "status": "deterministic-public-replay-passed",
        "bindings": {
            "protocol_sha256": sha(artifact / "protocol.json"),
            "method_seal_sha256": "1" * 64,
            "custody_seal_sha256": "2" * 64,
            "public_manifest_sha256": sha(
                artifact / "public-suite-manifest.json"
            ),
            "runs_sha256": sha(artifact / "runs.json"),
            "analysis_sha256": sha(artifact / "analysis.json"),
            "inventory_sha256": sha(artifact / "inventory.json"),
            "reference_set_sha256": hashlib.sha256(
                v3.canonical_bytes(reference_records)
            ).hexdigest(),
        },
        "replay_counts": {
            "tasks": v3.TASK_COUNT,
            "public_banks": v3.TASK_COUNT,
            "references": v3.TASK_COUNT,
            "runs": v3.EXPECTED_RUN_COUNT,
            "logical_proposal_draws": v3.EXPECTED_INITIAL_PROPOSAL_DRAWS,
        },
        "provider_calls": 0,
        "private_reveal_read": False,
        "all_public_banks_acquired_before_reference_materialization": True,
        "artifact_references_runs_analysis_byte_identical": True,
    }
    receipt_path = tmp_path / "replay.json"
    receipt_path.write_bytes(v3.canonical_bytes(receipt) + b"\n")
    receipt_sha = sha(receipt_path)
    assert v3._validate_replay_receipt(artifact, receipt_path, receipt_sha) == receipt

    receipt["replay_counts"]["runs"] = 1
    receipt_path.write_bytes(v3.canonical_bytes(receipt) + b"\n")
    with pytest.raises(v3.FreshCalibrationError, match="counts differ"):
        v3._validate_replay_receipt(artifact, receipt_path, sha(receipt_path))


def test_unblind_regenerates_every_task_and_rejects_reveal_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = _frozen_chain(tmp_path)
    suite = cast(Path, chain["suite"])
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    shutil.copy2(
        suite / "public/manifest.json", artifact / "public-suite-manifest.json"
    )
    for name, value in (
        ("protocol.json", {"schema": v3.STUDY_SCHEMA}),
        ("analysis.json", {"primary_gate": {"pass": True}}),
        ("inventory.json", {"schema": "test"}),
    ):
        (artifact / name).write_bytes(v3.canonical_bytes(value) + b"\n")
    monkeypatch.setattr(
        v3,
        "validate_artifact",
        lambda output: {"primary_gate": {"pass": True}},
    )
    replay_receipt = tmp_path / "replay-receipt.json"
    replay_receipt.write_bytes(v3.canonical_bytes({"test": True}) + b"\n")
    replay_receipt_sha = hashlib.sha256(replay_receipt.read_bytes()).hexdigest()
    monkeypatch.setattr(v3, "_validate_replay_receipt", lambda *args: {})
    result = v3.verify_unblind(
        suite,
        artifact,
        cast(Path, chain["secret"]),
        replay_receipt,
        replay_receipt_sha,
        tmp_path / "unblind.json",
    )
    assert result["passed"] is True
    assert result["task_count"] == 32
    assert result["primary_gate_pass"] is True

    reveal_path = suite / "private/reveal.json"
    reveal = json.loads(reveal_path.read_text())
    reveal["tasks"][0]["accepted_attempt"] += 1
    reveal_path.write_bytes(v3.canonical_bytes(reveal) + b"\n")
    with pytest.raises(v3.FreshCalibrationError, match="regeneration differs"):
        v3.verify_unblind(
            suite,
            artifact,
            cast(Path, chain["secret"]),
            replay_receipt,
            replay_receipt_sha,
            tmp_path / "tampered-unblind.json",
        )


def test_run_cli_has_no_secret_or_reveal_argument(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        v3._parser().parse_args(["run", "--help"])
    help_text = capsys.readouterr().out
    assert "--suite" in help_text
    assert "--secret" not in help_text
    assert "--reveal" not in help_text

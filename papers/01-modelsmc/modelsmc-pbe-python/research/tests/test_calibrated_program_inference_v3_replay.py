from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import research.calibrated_program_inference_v3_replay as replay
from research.calibrated_program_inference_v3_fresh import PRIMARY_ARM
from research.calibrated_program_inference_v3_validation import (
    canonical_bytes,
    seal_inventory,
)
from research.particle_calibration_terminal_diagnostic_v2 import TaskBinding

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MINI_TASK_ID = "fresh-cal-v3-001"
MINI_SEED = 936_001


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def _task_document() -> dict[str, object]:
    domain = tuple(range(-3, 5))
    first = (1, 2, -1, -2, 3, 0, 4, -3)
    second = (3, 2, -2, -1, -3, 4, 0, 1)
    inputs = ((), *((item,) for item in domain), domain, first, second, (*first, *second))

    def encoded(values: tuple[int, ...]) -> list[str]:
        return [str(value) for value in values]

    return {
        "name": "public mini replay task",
        "signature": {"input": "List<Int>", "output": "List<Int>"},
        "examples": [
            {
                "input": encoded(values),
                "output": encoded(tuple(item - 2 for item in values if item > 0)),
            }
            for values in inputs
        ],
        "integerConstants": encoded(domain),
        "particles": 8,
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


def _mini_analysis(
    runs: list[Mapping[str, object]] | tuple[Mapping[str, object], ...],
    references: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema": "mini-real-replay-analysis-v1",
        "task_ids": list(references),
        "run_count": len(runs),
        "exact_mass_signed": [
            float(cast(Mapping[str, object], run["error"])["exact_mass_signed"])
            for run in runs
        ],
    }


def _patch_mini_design(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replay, "TASK_IDS", (MINI_TASK_ID,))
    monkeypatch.setattr(replay, "FROZEN_ARMS", (PRIMARY_ARM,))
    monkeypatch.setattr(replay, "ARM_PARTICLES", {PRIMARY_ARM.arm_id: (8,)})
    monkeypatch.setattr(replay, "REPETITION_SEEDS", (MINI_SEED,))
    monkeypatch.setattr(replay, "EXPECTED_RUN_COUNT", 1)
    monkeypatch.setattr(replay, "EXPECTED_INITIAL_PROPOSAL_DRAWS", 8)
    monkeypatch.setattr(replay, "analyze", _mini_analysis)


def _prepare_mini_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    _patch_mini_design(monkeypatch)
    protocol = tmp_path / "protocol.json"
    method_seal = tmp_path / "method-seal.json"
    custody_seal = tmp_path / "custody-seal.json"
    _write_json(protocol, {"schema": "mini-frozen-protocol-v1"})
    _write_json(method_seal, {"schema": "mini-method-seal-v1"})
    custody = {
        "schema": "mini-custody-seal-v1",
        "secret_commitment_sha256": _digest("mini-v3-secret"),
        "secret_commitment_under_v2_domain_sha256": _digest("mini-v2-domain"),
    }
    _write_json(custody_seal, custody)
    protocol_sha = _sha256(protocol)
    method_sha = _sha256(method_seal)
    custody_sha = _sha256(custody_seal)

    suite = tmp_path / "suite"
    public = suite / "public"
    public.mkdir(parents=True)
    task_path = public / f"{MINI_TASK_ID}.json"
    _write_json(task_path, _task_document())
    task_sha = _sha256(task_path)
    manifest = {
        "schema": f"{replay.STUDY_SCHEMA}-public-suite-v1",
        "protocol_sha256": protocol_sha,
        "method_seal_sha256": method_sha,
        "custody_seal_sha256": custody_sha,
        "secret_commitment_sha256": custody["secret_commitment_sha256"],
        "secret_commitment_under_v2_domain_sha256": custody[
            "secret_commitment_under_v2_domain_sha256"
        ],
        "task_count": 1,
        "task_ids": [MINI_TASK_ID],
        "tasks": [
            {
                "task_id": MINI_TASK_ID,
                "path": task_path.name,
                "task_sha256": task_sha,
                "target_commitment_sha256": _digest("mini-target"),
            }
        ],
    }
    _write_json(public / "manifest.json", manifest)
    binding = TaskBinding(MINI_TASK_ID, task_path, task_sha)

    calls: list[str] = []

    def validate_protocol(
        project_root: Path, path: Path, expected_sha256: str
    ) -> dict[str, object]:
        calls.append("protocol")
        assert project_root == PROJECT_ROOT.resolve()
        assert path == protocol.resolve()
        assert expected_sha256 == protocol_sha
        return cast(dict[str, object], json.loads(path.read_text()))

    def validate_method(*args: object, **kwargs: object) -> None:
        calls.append("method")

    def validate_custody(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append("custody")
        return custody

    def validate_suite(
        suite_path: Path,
        *args: object,
        require_public_only: bool,
        **kwargs: object,
    ) -> tuple[list[TaskBinding], dict[str, object]]:
        calls.append("suite")
        assert suite_path == suite.resolve()
        assert require_public_only is True
        return [binding], manifest

    monkeypatch.setattr(replay, "_validate_protocol", validate_protocol)
    monkeypatch.setattr(replay, "_validate_method_seal", validate_method)
    monkeypatch.setattr(replay, "_validate_custody_seal", validate_custody)
    monkeypatch.setattr(replay, "_validate_suite", validate_suite)

    references, runs = replay.reconstruct_public_ledger([binding])
    analysis = _mini_analysis(runs, references)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "protocol.json").write_bytes(protocol.read_bytes())
    (artifact / "public-suite-manifest.json").write_bytes(
        (public / "manifest.json").read_bytes()
    )
    for task_id, reference in references.items():
        _write_json(artifact / "references" / f"{task_id}.json", reference)
    _write_json(artifact / "runs.json", runs)
    _write_json(artifact / "analysis.json", analysis)
    seal_inventory(artifact)

    def validate_artifact(path: Path) -> dict[str, object]:
        assert path == artifact.resolve()
        return cast(dict[str, object], json.loads((path / "analysis.json").read_text()))

    monkeypatch.setattr(replay, "validate_artifact", validate_artifact)
    return {
        "protocol": protocol,
        "protocol_sha": protocol_sha,
        "method_seal": method_seal,
        "method_sha": method_sha,
        "custody_seal": custody_seal,
        "custody_sha": custody_sha,
        "suite": suite,
        "artifact": artifact,
        "references": references,
        "runs": runs,
        "analysis": analysis,
        "calls": calls,
    }


def _replay(prepared: Mapping[str, Any], receipt: Path) -> dict[str, object]:
    return replay.replay_calibration(
        PROJECT_ROOT,
        prepared["protocol"],
        prepared["protocol_sha"],
        prepared["method_seal"],
        prepared["method_sha"],
        prepared["custody_seal"],
        prepared["custody_sha"],
        prepared["suite"],
        prepared["artifact"],
        receipt,
    )


def test_real_mini_replay_is_deterministic_and_receipt_is_reveal_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_mini_replay(tmp_path, monkeypatch)
    first_path = tmp_path / "receipt-1.json"
    second_path = tmp_path / "receipt-2.json"
    first = _replay(prepared, first_path)
    second = _replay(prepared, second_path)
    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["status"] == "deterministic-public-replay-passed"
    assert first["replay_counts"] == {
        "tasks": 1,
        "public_banks": 1,
        "references": 1,
        "runs": 1,
        "logical_proposal_draws": 8,
    }
    assert first["provider_calls"] == 0
    assert first["private_reveal_read"] is False
    assert prepared["calls"] == [
        "protocol",
        "method",
        "custody",
        "suite",
        "protocol",
        "method",
        "custody",
        "suite",
    ]


def test_coherent_estimate_error_analysis_and_inventory_tamper_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_mini_replay(tmp_path, monkeypatch)
    artifact = cast(Path, prepared["artifact"])
    runs = cast(list[dict[str, Any]], json.loads((artifact / "runs.json").read_text()))
    run = runs[0]
    reference_mass = float(run["reference"]["exact_target_mass"])
    run["estimate"]["exact_target_mass"] = float(
        run["estimate"]["exact_target_mass"]
    ) + 0.001
    run["error"]["exact_mass_signed"] = (
        float(run["estimate"]["exact_target_mass"]) - reference_mass
    )
    references = cast(Mapping[str, Mapping[str, object]], prepared["references"])
    _write_json(artifact / "runs.json", runs)
    _write_json(artifact / "analysis.json", _mini_analysis(runs, references))
    (artifact / "inventory.json").unlink()
    seal_inventory(artifact)

    receipt = tmp_path / "tampered-receipt.json"
    with pytest.raises(replay.V3ReplayError, match="artifact runs structure differs"):
        _replay(prepared, receipt)
    assert not receipt.exists()


def test_external_hash_drift_is_rejected_before_any_replay(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    method = tmp_path / "method.json"
    custody = tmp_path / "custody.json"
    for path in (protocol, method, custody):
        _write_json(path, {})
    with pytest.raises(replay.V3ReplayError, match="external protocol SHA-256 differs"):
        replay.replay_calibration(
            PROJECT_ROOT,
            protocol,
            "0" * 64,
            method,
            _sha256(method),
            custody,
            _sha256(custody),
            tmp_path / "suite",
            tmp_path / "artifact",
            tmp_path / "receipt.json",
        )


def test_public_private_suite_is_rejected_before_frozen_suite_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = tmp_path / "protocol.json"
    method = tmp_path / "method.json"
    custody = tmp_path / "custody.json"
    for path in (protocol, method, custody):
        _write_json(path, {})
    suite = tmp_path / "suite"
    (suite / "public").mkdir(parents=True)
    (suite / "private").mkdir()
    (suite / "private" / "reveal.json").write_text('{"secret_hex":"forbidden"}\n')
    called = False

    def forbidden(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(replay, "_validate_suite", forbidden)
    with pytest.raises(replay.V3ReplayError, match="public-only suite"):
        replay.replay_calibration(
            PROJECT_ROOT,
            protocol,
            _sha256(protocol),
            method,
            _sha256(method),
            custody,
            _sha256(custody),
            suite,
            tmp_path / "artifact",
            tmp_path / "receipt.json",
        )
    assert called is False


def test_existing_receipt_is_rejected_before_input_access(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("preserve me\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        replay.replay_calibration(
            tmp_path,
            tmp_path / "missing-protocol",
            "0" * 64,
            tmp_path / "missing-method",
            "0" * 64,
            tmp_path / "missing-custody",
            "0" * 64,
            tmp_path / "missing-suite",
            tmp_path / "missing-artifact",
            receipt,
        )
    assert receipt.read_text() == "preserve me\n"


def test_all_public_banks_are_built_before_any_terminal_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_ids = ("task-a", "task-b")
    paths = []
    for task_id in task_ids:
        path = tmp_path / f"{task_id}.json"
        path.write_text("{}\n")
        paths.append(path)
    bindings = [
        TaskBinding(task_id, path, _sha256(path))
        for task_id, path in zip(task_ids, paths, strict=True)
    ]
    events: list[str] = []

    def acquire(binding: TaskBinding) -> SimpleNamespace:
        events.append(f"bank:{binding.task_id}")
        return SimpleNamespace(metadata={})

    def terminal(binding: TaskBinding) -> SimpleNamespace:
        assert events[:2] == ["bank:task-a", "bank:task-b"]
        assert len([event for event in events if event.startswith("bank:")]) == 2
        events.append(f"terminal:{binding.task_id}")
        return SimpleNamespace(binding=binding)

    monkeypatch.setattr(replay, "TASK_IDS", task_ids)
    monkeypatch.setattr(replay, "FROZEN_ARMS", ())
    monkeypatch.setattr(replay, "ARM_PARTICLES", {})
    monkeypatch.setattr(replay, "REPETITION_SEEDS", ())
    monkeypatch.setattr(replay, "EXPECTED_RUN_COUNT", 0)
    monkeypatch.setattr(replay, "EXPECTED_INITIAL_PROPOSAL_DRAWS", 0)
    monkeypatch.setattr(replay, "factorized_public_mode_bank", acquire)
    monkeypatch.setattr(replay, "TerminalTask", terminal)
    monkeypatch.setattr(replay, "materialize_mode_bank", lambda task, bank: bank)
    monkeypatch.setattr(
        replay,
        "_native_reference",
        lambda task, proposals, metadata: {"task_id": task.binding.task_id},
    )
    references, runs = replay.reconstruct_public_ledger(bindings)
    assert list(references) == list(task_ids)
    assert runs == []
    assert events == [
        "bank:task-a",
        "bank:task-b",
        "terminal:task-a",
        "terminal:task-b",
    ]

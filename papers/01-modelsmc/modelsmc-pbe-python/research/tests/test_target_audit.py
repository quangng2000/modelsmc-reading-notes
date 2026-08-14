from __future__ import annotations

import json
from pathlib import Path

import pytest

import research.target_audit as target_audit
from research.protocol import Protocol, StageSpec, load_protocol
from research.run_matrix import find_stage

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_DIR / "research" / "protocol-deduction-stress-v2.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _certificate_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Protocol, StageSpec, dict[str, object]]:
    protocol = load_protocol(PROTOCOL_PATH)
    reference = find_stage(protocol, "provider-free-reference-audit")
    paid = find_stage(protocol, "paired-d-qd-32b-pilot")
    assert reference is not None
    assert paid is not None
    matrix = tmp_path / "reference"
    manifest_path = matrix / "matrix_manifest.json"
    cell_path = matrix / "cells" / "cell-1" / "cell.json"
    core_path = matrix / "cells" / "cell-1" / "artifacts" / "run" / "result.json"
    _write_json(
        manifest_path,
        {
            "protocol_sha256": protocol.protocol_sha256,
            "stage_id": reference.stage_id,
            "planned_cells": [{"cell_id": "cell-1"}],
        },
    )
    _write_json(
        cell_path,
        {
            "status": "completed",
            "core_status": "completed",
            "protocol_sha256": protocol.protocol_sha256,
            "core_artifact": "artifacts/run",
            "cell": {
                "cell_id": "cell-1",
                "task_id": "foldr-sparse-bounded-square",
            },
        },
    )
    _write_json(core_path, {"status": "completed", "result": {"support_states": 36_198}})
    executable = tmp_path / "bin" / "modelsmc-pbe"
    executable.parent.mkdir()
    executable.write_text("entry-point-v1", encoding="utf-8")
    task = protocol.tasks[0]
    task_report: dict[str, object] = {
        "task_id": task.task_id,
        "spec": str(task.spec_path),
        "spec_sha256": target_audit.sha256_file(task.spec_path),
        "support_states": 36_198,
        "exact_states": 2,
        "inexact_states": 36_196,
        "loss_scale": 2.0,
        "beta_max": 1.0,
        "minimum_exact_log_target": -11.5,
        "maximum_inexact_log_target": -15.5,
        "exact_over_inexact_margin": 4.0,
        "dominance_passed": True,
    }
    certificate: dict[str, object] = {
        "schema_version": target_audit.SCHEMA_VERSION,
        "audit": "finite-target-dominance",
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "reference_audit_stage": reference.stage_id,
        "source_matrix": str(matrix.resolve()),
        "matrix_manifest": "matrix_manifest.json",
        "matrix_manifest_sha256": target_audit.sha256_file(manifest_path),
        "terminal_dominance_requirement": (
            protocol.target_contract.terminal_dominance_requirement
            if protocol.target_contract is not None
            else None
        ),
        "implementation": target_audit.implementation_identity(
            protocol, str(executable)
        ),
        "tasks": [task_report],
        "evidence": [
            {
                "cell_id": "cell-1",
                "task_id": task.task_id,
                "cell_manifest": str(cell_path.relative_to(matrix)),
                "cell_manifest_sha256": target_audit.sha256_file(cell_path),
                "core_result": str(core_path.relative_to(matrix)),
                "core_result_sha256": target_audit.sha256_file(core_path),
                "support_states": 36_198,
            }
        ],
        "invariants_passed": True,
        "violations": [],
    }
    certificate_path = matrix / target_audit.CERTIFICATE_NAME
    _write_json(certificate_path, certificate)
    return certificate_path, executable, protocol, paid, task_report


def test_certificate_validation_recomputes_claims_and_binds_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    certificate_path, executable, protocol, paid, task_report = _certificate_fixture(
        tmp_path
    )
    monkeypatch.setattr(
        target_audit,
        "_audit_target_task",
        lambda *_args, **_kwargs: (task_report, []),
    )

    validated = target_audit.validate_target_audit_certificate(
        certificate_path,
        protocol,
        paid,
        executable=str(executable),
    )
    assert validated["reference_audit_stage"] == "provider-free-reference-audit"

    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate["tasks"][0]["exact_over_inexact_margin"] = 400.0
    _write_json(certificate_path, certificate)
    with pytest.raises(ValueError, match="recomputation differs"):
        target_audit.validate_target_audit_certificate(
            certificate_path,
            protocol,
            paid,
            executable=str(executable),
        )

    certificate["tasks"] = [task_report]
    _write_json(certificate_path, certificate)
    executable.write_text("entry-point-v2", encoding="utf-8")
    with pytest.raises(ValueError, match="implementation identity mismatch"):
        target_audit.validate_target_audit_certificate(
            certificate_path,
            protocol,
            paid,
            executable=str(executable),
        )

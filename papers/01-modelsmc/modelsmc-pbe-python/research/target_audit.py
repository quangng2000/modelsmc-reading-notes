"""Hash-bound target-dominance certificates for provider-gated research stages."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.core import ProgramScorer
from modelsmc_pbe.runtime import resolve_device
from modelsmc_pbe.search.importance import ImportanceSMCOptions
from modelsmc_pbe.search.importance.support import ImportanceSupportBuilder
from modelsmc_pbe.search.importance.target import FiniteImportanceTarget
from research.heldout import write_json_atomic
from research.protocol import Protocol, StageSpec

SCHEMA_VERSION = 1
CERTIFICATE_NAME = "target_audit_certificate.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_identity(protocol: Protocol, executable: str) -> dict[str, object]:
    """Hash support/target code, dependency locks, and the exact core entry point."""

    root = protocol.project_root.resolve()
    executable_path = Path(executable).expanduser().resolve()
    if not executable_path.is_file():
        raise ValueError(f"synthesizer executable does not exist: {executable_path}")
    source_paths = sorted((root / "src" / "modelsmc_pbe").rglob("*.py"))
    source_paths.extend(
        path
        for path in (
            root / "research" / "heldout.py",
            root / "research" / "protocol.py",
            root / "research" / "run_matrix.py",
            root / "research" / "target_audit.py",
            root / "pyproject.toml",
            root / "uv.lock",
        )
        if path.is_file()
    )
    source_paths = sorted(set(source_paths))
    digest = hashlib.sha256()
    for source_path in source_paths:
        relative = source_path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = source_path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {
        "source_sha256": digest.hexdigest(),
        "source_files": len(source_paths),
        "executable": str(executable_path),
        "executable_sha256": sha256_file(executable_path),
    }


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _safe_evidence_path(matrix_dir: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("audit evidence path must be a nonempty string")
    path = (matrix_dir / relative).resolve()
    if not path.is_relative_to(matrix_dir):
        raise ValueError("audit evidence path escapes its source matrix")
    return path


def _completed_evidence(
    protocol: Protocol, stage: StageSpec, matrix_dir: Path
) -> tuple[list[dict[str, object]], dict[str, set[int]], list[dict[str, str]]]:
    violations: list[dict[str, str]] = []
    evidence: list[dict[str, object]] = []
    declared_support: dict[str, set[int]] = {task_id: set() for task_id in stage.task_ids}
    manifest_path = matrix_dir / "matrix_manifest.json"
    manifest = _object(manifest_path)
    if manifest.get("protocol_sha256") != protocol.protocol_sha256:
        violations.append(
            {"code": "MATRIX_PROTOCOL_HASH_MISMATCH", "detail": str(manifest_path)}
        )
    if manifest.get("stage_id") != stage.stage_id:
        violations.append({"code": "MATRIX_STAGE_MISMATCH", "detail": str(manifest_path)})
    planned = manifest.get("planned_cells")
    if not isinstance(planned, list) or not planned:
        violations.append({"code": "MATRIX_PLAN_MISSING", "detail": str(manifest_path)})
        planned = []
    for raw_plan in planned:
        if not isinstance(raw_plan, dict) or not isinstance(raw_plan.get("cell_id"), str):
            violations.append({"code": "MATRIX_PLAN_INVALID", "detail": str(raw_plan)})
            continue
        cell_id = cast(str, raw_plan["cell_id"])
        task_id = raw_plan.get("task_id")
        cell_path = matrix_dir / "cells" / cell_id / "cell.json"
        try:
            cell = _object(cell_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            violations.append({"code": "CELL_MANIFEST_MISSING", "detail": str(error)})
            continue
        if (
            cell.get("status") != "completed"
            or cell.get("core_status") != "completed"
            or cell.get("protocol_sha256") != protocol.protocol_sha256
        ):
            violations.append({"code": "CELL_NOT_COMPLETED", "detail": cell_id})
            continue
        core_relative = cell.get("core_artifact")
        if not isinstance(core_relative, str):
            violations.append({"code": "CORE_ARTIFACT_MISSING", "detail": cell_id})
            continue
        core_result_path = cell_path.parent / core_relative / "result.json"
        try:
            envelope = _object(core_result_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            violations.append({"code": "CORE_RESULT_MISSING", "detail": str(error)})
            continue
        result = envelope.get("result")
        if envelope.get("status") != "completed" or not isinstance(result, dict):
            violations.append({"code": "CORE_RESULT_NOT_COMPLETED", "detail": cell_id})
            continue
        support_states = result.get("support_states")
        if (
            not isinstance(task_id, str)
            or task_id not in declared_support
            or isinstance(support_states, bool)
            or not isinstance(support_states, int)
        ):
            violations.append({"code": "CORE_SUPPORT_INVALID", "detail": cell_id})
            continue
        declared_support[task_id].add(support_states)
        evidence.append(
            {
                "cell_id": cell_id,
                "task_id": task_id,
                "cell_manifest": str(cell_path.relative_to(matrix_dir)),
                "cell_manifest_sha256": sha256_file(cell_path),
                "core_result": str(core_result_path.relative_to(matrix_dir)),
                "core_result_sha256": sha256_file(core_result_path),
                "support_states": support_states,
            }
        )
    return evidence, declared_support, violations


def _audit_target_task(
    protocol: Protocol,
    task_id: str,
    declared_support: set[int],
) -> tuple[dict[str, object], list[dict[str, str]]]:
    contract = protocol.target_contract
    if contract is None:
        raise ValueError("protocol target contract is missing")
    violations: list[dict[str, str]] = []
    task = next(task for task in protocol.tasks if task.task_id == task_id)
    config = load_experiment_config(task.spec_path)
    options = ImportanceSMCOptions(
        multi_family=True,
        support_limit=protocol.caps.support_limit,
        hole_state_limit=protocol.caps.hole_state_limit,
        hole_max_cost=protocol.caps.hole_max_cost,
    )
    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()
    target = FiniteImportanceTarget.build(
        support,
        smc=config.smc,
        device=resolve_device("cpu"),
    )
    terminal = target.log_unnormalized(beta=contract.beta_max)
    exact_states = int(target.exact_mask.sum().item())
    inexact_states = len(support.states) - exact_states
    minimum_exact = (
        None if exact_states == 0 else float(terminal[target.exact_mask].min().item())
    )
    maximum_inexact = (
        None if inexact_states == 0 else float(terminal[~target.exact_mask].max().item())
    )
    margin = (
        None
        if minimum_exact is None or maximum_inexact is None
        else minimum_exact - maximum_inexact
    )
    loss_scale_matches = math.isclose(
        float(config.smc.loss_scale), contract.loss_scale, rel_tol=0.0, abs_tol=1e-12
    )
    beta_matches = math.isclose(
        float(protocol.shared_arguments.get("beta_max", math.nan)),
        contract.beta_max,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    support_matches = declared_support == {len(support.states)}
    dominance_passed = margin is not None and margin > 0.0
    if not loss_scale_matches:
        violations.append({"code": "LOSS_SCALE_CONTRACT_MISMATCH", "detail": task_id})
    if not beta_matches:
        violations.append({"code": "BETA_CONTRACT_MISMATCH", "detail": task_id})
    if not support_matches:
        violations.append(
            {
                "code": "MATERIALIZED_SUPPORT_MISMATCH",
                "detail": (
                    f"{task_id}: cells={sorted(declared_support)}, "
                    f"audit={len(support.states)}"
                ),
            }
        )
    if not dominance_passed:
        violations.append({"code": "TARGET_DOMINANCE_FAILED", "detail": task_id})
    return (
        {
            "task_id": task_id,
            "spec": str(task.spec_path),
            "spec_sha256": sha256_file(task.spec_path),
            "support_states": len(support.states),
            "exact_states": exact_states,
            "inexact_states": inexact_states,
            "loss_scale": float(config.smc.loss_scale),
            "beta_max": contract.beta_max,
            "minimum_exact_log_target": minimum_exact,
            "maximum_inexact_log_target": maximum_inexact,
            "exact_over_inexact_margin": margin,
            "dominance_passed": dominance_passed,
        },
        violations,
    )


def create_target_audit_certificate(
    protocol: Protocol,
    stage: StageSpec,
    matrix_dir: Path,
    *,
    executable: str,
) -> dict[str, object]:
    """Exhaustively audit the declared target and bind the result to local artifacts."""

    contract = protocol.target_contract
    if contract is None or not contract.reference_audit_required:
        raise ValueError("protocol does not require a target audit certificate")
    if stage.stage_id != contract.reference_audit_stage:
        raise ValueError("certificate can only be emitted by the declared reference audit stage")
    matrix_dir = matrix_dir.expanduser().resolve()
    evidence, declared_support, violations = _completed_evidence(protocol, stage, matrix_dir)
    task_reports: list[dict[str, object]] = []
    for task_id in stage.task_ids:
        report, task_violations = _audit_target_task(
            protocol,
            task_id,
            declared_support.get(task_id, set()),
        )
        task_reports.append(report)
        violations.extend(task_violations)
    certificate: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "audit": "finite-target-dominance",
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "reference_audit_stage": stage.stage_id,
        "source_matrix": str(matrix_dir),
        "matrix_manifest": "matrix_manifest.json",
        "matrix_manifest_sha256": sha256_file(matrix_dir / "matrix_manifest.json"),
        "terminal_dominance_requirement": contract.terminal_dominance_requirement,
        "implementation": implementation_identity(protocol, executable),
        "tasks": task_reports,
        "evidence": evidence,
        "invariants_passed": not violations,
        "violations": violations,
    }
    write_json_atomic(matrix_dir / CERTIFICATE_NAME, certificate)
    return certificate


def validate_target_audit_certificate(
    certificate_path: Path,
    protocol: Protocol,
    stage: StageSpec,
    *,
    executable: str,
) -> dict[str, object]:
    """Validate a prerequisite certificate and all artifact hashes it binds."""

    if stage.requires_audit_stage is None:
        raise ValueError(f"stage {stage.stage_id} has no audit prerequisite")
    path = certificate_path.expanduser().resolve()
    certificate = _object(path)
    if certificate.get("schema_version") != SCHEMA_VERSION or certificate.get("audit") != (
        "finite-target-dominance"
    ):
        raise ValueError("target audit certificate schema or audit kind is invalid")
    if certificate.get("invariants_passed") is not True:
        raise ValueError("target audit certificate did not pass all invariants")
    if certificate.get("violations") != []:
        raise ValueError("target audit certificate contains invariant violations")
    if certificate.get("protocol_sha256") != protocol.protocol_sha256:
        raise ValueError("target audit certificate has a different protocol hash")
    if certificate.get("protocol_id") != protocol.protocol_id:
        raise ValueError("target audit certificate has a different protocol id")
    if certificate.get("reference_audit_stage") != stage.requires_audit_stage:
        raise ValueError("target audit certificate is for the wrong prerequisite stage")
    current_implementation = implementation_identity(protocol, executable)
    if certificate.get("implementation") != current_implementation:
        raise ValueError("target audit certificate implementation identity mismatch")
    source_value = certificate.get("source_matrix")
    if not isinstance(source_value, str):
        raise ValueError("target audit certificate source matrix is missing")
    matrix_dir = Path(source_value).expanduser().resolve()
    manifest = _safe_evidence_path(matrix_dir, certificate.get("matrix_manifest"))
    if sha256_file(manifest) != certificate.get("matrix_manifest_sha256"):
        raise ValueError("target audit matrix manifest hash mismatch")
    manifest_document = _object(manifest)
    if (
        manifest_document.get("protocol_sha256") != protocol.protocol_sha256
        or manifest_document.get("stage_id") != stage.requires_audit_stage
    ):
        raise ValueError("target audit source matrix identity mismatch")
    task_records = certificate.get("tasks")
    if not isinstance(task_records, list):
        raise ValueError("target audit certificate tasks are missing")
    by_task = {
        record.get("task_id"): record for record in task_records if isinstance(record, dict)
    }
    required_stage = next(
        item for item in protocol.stages if item.stage_id == stage.requires_audit_stage
    )
    contract = protocol.target_contract
    if contract is None:
        raise ValueError("protocol target contract is missing")
    if certificate.get("terminal_dominance_requirement") != (
        contract.terminal_dominance_requirement
    ):
        raise ValueError("target audit dominance requirement mismatch")
    if set(by_task) != set(required_stage.task_ids):
        raise ValueError("target audit certificate task coverage is invalid")
    for task_id in required_stage.task_ids:
        task = next(task for task in protocol.tasks if task.task_id == task_id)
        record = by_task.get(task_id)
        if not isinstance(record, dict) or record.get("dominance_passed") is not True:
            raise ValueError(f"target audit task {task_id} did not pass dominance")
        if record.get("spec_sha256") != sha256_file(task.spec_path):
            raise ValueError(f"target audit task {task_id} specification hash mismatch")
        if record.get("loss_scale") != contract.loss_scale:
            raise ValueError(f"target audit task {task_id} loss scale mismatch")
        if record.get("beta_max") != contract.beta_max:
            raise ValueError(f"target audit task {task_id} beta mismatch")
        margin = record.get("exact_over_inexact_margin")
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or margin <= 0:
            raise ValueError(f"target audit task {task_id} dominance margin is invalid")
    evidence = certificate.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("target audit certificate evidence is missing")
    evidence_cell_ids: set[str] = set()
    declared_support: dict[str, set[int]] = {
        task_id: set() for task_id in required_stage.task_ids
    }
    for raw_record in evidence:
        if not isinstance(raw_record, dict):
            raise ValueError("target audit evidence record is invalid")
        for path_key, hash_key in (
            ("cell_manifest", "cell_manifest_sha256"),
            ("core_result", "core_result_sha256"),
        ):
            evidence_path = _safe_evidence_path(matrix_dir, raw_record.get(path_key))
            if sha256_file(evidence_path) != raw_record.get(hash_key):
                raise ValueError(f"target audit evidence hash mismatch: {evidence_path}")
        cell_path = _safe_evidence_path(matrix_dir, raw_record.get("cell_manifest"))
        core_path = _safe_evidence_path(matrix_dir, raw_record.get("core_result"))
        cell = _object(cell_path)
        core = _object(core_path)
        cell_plan = cell.get("cell")
        core_relative = cell.get("core_artifact")
        result = core.get("result")
        if (
            cell.get("status") != "completed"
            or cell.get("core_status") != "completed"
            or cell.get("protocol_sha256") != protocol.protocol_sha256
            or core.get("status") != "completed"
            or not isinstance(cell_plan, dict)
            or not isinstance(core_relative, str)
            or not isinstance(result, dict)
        ):
            raise ValueError("target audit evidence is not a completed protocol cell")
        if (
            raw_record.get("cell_id") != cell_plan.get("cell_id")
            or raw_record.get("task_id") != cell_plan.get("task_id")
            or raw_record.get("support_states") != result.get("support_states")
        ):
            raise ValueError("target audit evidence contents do not match its certificate")
        expected_core_path = (cell_path.parent / core_relative / "result.json").resolve()
        if core_path != expected_core_path:
            raise ValueError("target audit evidence does not match the cell's core artifact")
        cell_id = raw_record.get("cell_id")
        if not isinstance(cell_id, str) or cell_id in evidence_cell_ids:
            raise ValueError("target audit evidence cell ids must be unique strings")
        evidence_cell_ids.add(cell_id)
        evidence_task_id = raw_record.get("task_id")
        support_states = raw_record.get("support_states")
        if (
            not isinstance(evidence_task_id, str)
            or evidence_task_id not in declared_support
            or isinstance(support_states, bool)
            or not isinstance(support_states, int)
        ):
            raise ValueError("target audit evidence task or support is invalid")
        declared_support[evidence_task_id].add(support_states)
    planned = manifest_document.get("planned_cells")
    if not isinstance(planned, list) or any(not isinstance(item, dict) for item in planned):
        raise ValueError("target audit source matrix plan is invalid")
    planned_cell_ids = {item.get("cell_id") for item in planned}
    if any(not isinstance(cell_id, str) for cell_id in planned_cell_ids) or (
        evidence_cell_ids != planned_cell_ids
    ):
        raise ValueError("target audit evidence does not cover the complete source matrix")
    for task_id in required_stage.task_ids:
        recomputed, recompute_violations = _audit_target_task(
            protocol,
            task_id,
            declared_support[task_id],
        )
        if recompute_violations:
            raise ValueError(
                f"target audit recomputation failed for {task_id}: {recompute_violations}"
            )
        if by_task[task_id] != recomputed:
            raise ValueError(f"target audit recomputation differs for {task_id}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "protocol_sha256": protocol.protocol_sha256,
        "reference_audit_stage": stage.requires_audit_stage,
        "source_matrix": str(matrix_dir),
    }

"""Execute the preregistered U/D/Q/QD matrix without dropping failed cells."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from research.heldout import evaluate_result, unavailable_evaluation, write_json_atomic
from research.protocol import (
    AnalysisLabel,
    ArmName,
    ArmSpec,
    Caps,
    ModelSpec,
    Protocol,
    StageSpec,
    TaskSpec,
    effective_caps,
    load_protocol,
)
from research.target_audit import (
    CERTIFICATE_NAME,
    create_target_audit_certificate,
    validate_target_audit_certificate,
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class CellPlan:
    cell_id: str
    task_id: str
    arm: ArmName
    seed: int
    analysis_label: AnalysisLabel
    model_scope: str
    model_id: str | None
    model_alias: str | None
    model_hf_repository: str | None
    model_architecture: str | None
    model_parameterization: str | None
    model_total_parameters_billion: float | None
    model_active_parameters_billion: float | None
    model_dtype: str | None
    model_quantization: str | None
    model_revision: str | None
    tokenizer_revision: str | None


def _require_known(
    requested: set[Any] | None, declared: set[Any], name: str
) -> None:
    if requested is None:
        return
    unknown = requested - declared
    if unknown:
        raise ValueError(f"unknown {name}: {sorted(unknown, key=str)!r}")


def find_stage(protocol: Protocol, stage_id: str | None) -> StageSpec | None:
    if stage_id is None:
        return None
    try:
        return next(stage for stage in protocol.stages if stage.stage_id == stage_id)
    except StopIteration as error:
        choices = ", ".join(stage.stage_id for stage in protocol.stages) or "<none>"
        raise ValueError(f"unknown stage {stage_id!r}; expected one of: {choices}") from error


def build_plan(
    protocol: Protocol,
    *,
    stage: StageSpec | None = None,
    labels: set[str] | None = None,
    task_ids: set[str] | None = None,
    arms: set[str] | None = None,
    seeds: set[int] | None = None,
    model_ids: set[str] | None = None,
) -> tuple[CellPlan, ...]:
    """Build a deterministic matrix without duplicating size-invariant controls."""

    declared_labels = {task.label for task in protocol.tasks}
    declared_tasks = {task.task_id for task in protocol.tasks}
    declared_arms = {arm.name for arm in protocol.arms}
    declared_seeds = set(protocol.seeds)
    declared_models = {model.model_id for model in protocol.models}
    _require_known(labels, declared_labels, "analysis labels")
    _require_known(task_ids, declared_tasks, "tasks")
    _require_known(arms, declared_arms, "arms")
    _require_known(seeds, declared_seeds, "seeds")
    _require_known(model_ids, declared_models, "models")

    stage_tasks = set(stage.task_ids) if stage else None
    stage_arms = set(stage.arms) if stage else None
    stage_seeds = set(stage.seeds) if stage else None
    stage_models = set(stage.model_ids) if stage else None

    selected_tasks = [
        task
        for task in protocol.tasks
        if (labels is None or task.label in labels)
        and (task_ids is None or task.task_id in task_ids)
        and (stage_tasks is None or task.task_id in stage_tasks)
    ]
    selected_arms = [
        arm
        for arm in protocol.arms
        if (arms is None or arm.name in arms)
        and (stage_arms is None or arm.name in stage_arms)
    ]
    selected_seeds = [
        seed
        for seed in protocol.seeds
        if (seeds is None or seed in seeds)
        and (stage_seeds is None or seed in stage_seeds)
    ]
    selected_models = [
        model
        for model in protocol.models
        if (model_ids is None or model.model_id in model_ids)
        and (stage_models is None or model.model_id in stage_models)
    ]
    if not selected_tasks or not selected_arms or not selected_seeds or not selected_models:
        raise ValueError("matrix selection is empty")
    plans: list[CellPlan] = []
    for task in selected_tasks:
        for arm in selected_arms:
            for seed in selected_seeds:
                models: Sequence[ModelSpec | None] = (
                    selected_models if arm.proposal == "vllm" else (None,)
                )
                for model in models:
                    model_token = model.model_id if model is not None else "size-invariant"
                    plans.append(
                        CellPlan(
                            cell_id=(
                                f"{task.task_id}--{arm.name}--{model_token}--seed-{seed}"
                            ),
                            task_id=task.task_id,
                            arm=arm.name,
                            seed=seed,
                            analysis_label=task.label,
                            model_scope=(
                                "model-specific" if model is not None else "size-invariant"
                            ),
                            model_id=model.model_id if model is not None else None,
                            model_alias=model.alias if model is not None else None,
                            model_hf_repository=(
                                model.hf_repository if model is not None else None
                            ),
                            model_architecture=(
                                model.architecture if model is not None else None
                            ),
                            model_parameterization=(
                                model.parameterization if model is not None else None
                            ),
                            model_total_parameters_billion=(
                                model.total_parameters_billion if model is not None else None
                            ),
                            model_active_parameters_billion=(
                                model.active_parameters_billion if model is not None else None
                            ),
                            model_dtype=(model.dtype if model is not None else None),
                            model_quantization=(
                                model.quantization if model is not None else None
                            ),
                            model_revision=(
                                model.model_revision if model is not None else None
                            ),
                            tokenizer_revision=(
                                model.tokenizer_revision if model is not None else None
                            ),
                        )
                    )
    return tuple(plans)


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def command_for(
    protocol: Protocol,
    task: TaskSpec,
    arm: ArmSpec,
    plan: CellPlan,
    *,
    caps: Caps,
    executable: str,
    artifacts_dir: Path,
    base_url: str | None,
) -> list[str]:
    model_alias = plan.model_alias or "size-invariant-catalog-control"
    command = [
        executable,
        "synthesize",
        str(task.spec_path),
        "--mode",
        "importance-smc",
        "--proposal",
        arm.proposal,
        "--model",
        model_alias,
        "--skeleton",
        "auto",
        "--particles",
        str(caps.particles),
        "--iterations",
        str(caps.iterations),
        "--alpha",
        str(caps.alpha),
        "--ess-threshold",
        str(caps.ess_threshold),
        "--seed",
        str(plan.seed),
        "--max-scored-candidates",
        str(caps.max_scored_candidates),
        "--support-limit",
        str(caps.support_limit),
        "--hole-state-limit",
        str(caps.hole_state_limit),
        "--hole-max-cost",
        str(caps.hole_max_cost),
        "--timeout-seconds",
        str(caps.provider_timeout_seconds),
        "--deduction-mix",
        str(arm.deduction_mix),
        "--artifacts-dir",
        str(artifacts_dir),
        "--trace",
    ]
    if arm.family_deduction_mix is not None:
        command.extend(("--family-deduction-mix", str(arm.family_deduction_mix)))
    if arm.hole_deduction_mix is not None:
        command.extend(("--hole-deduction-mix", str(arm.hole_deduction_mix)))
    for name, value in sorted(protocol.shared_arguments.items()):
        command.extend((_flag(name), str(value)))
    if arm.proposal == "vllm":
        if not all(
            (
                plan.model_hf_repository,
                plan.model_revision,
                plan.tokenizer_revision,
            )
        ):
            raise ValueError(f"arm {arm.name} requires pinned model identity")
        command.extend(
            (
                "--model-repository",
                str(plan.model_hf_repository),
                "--model-revision",
                str(plan.model_revision),
                "--tokenizer-revision",
                str(plan.tokenizer_revision),
            )
        )
        if base_url is None:
            raise ValueError(
                f"arm {arm.name} requires --base-url or {protocol.provider.base_url_env}"
            )
        command.extend(("--base-url", base_url))
        if protocol.provider.api_key_env:
            command.extend(("--api-key-env", protocol.provider.api_key_env))
        if protocol.provider.score_cache_mode != "off":
            assert protocol.provider.score_cache_dir is not None
            assert protocol.provider.vllm_server_config is not None
            command.extend(
                (
                    "--score-cache-dir",
                    str(protocol.provider.score_cache_dir),
                    "--score-cache-mode",
                    protocol.provider.score_cache_mode,
                    "--vllm-server-config",
                    protocol.provider.vllm_server_config,
                )
            )
    if protocol.materialize_reference:
        command.append("--materialize-reference")
    return command


def _locate_core_artifact(artifacts_dir: Path) -> Path | None:
    manifests = sorted(artifacts_dir.glob("*/manifest.json"))
    if len(manifests) != 1:
        return None
    return manifests[0].parent


def _terminal_core_status(core_dir: Path | None) -> str | None:
    if core_dir is None:
        return None
    result_path = core_dir / "result.json"
    try:
        document = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    status = document.get("status") if isinstance(document, dict) else None
    return status if isinstance(status, str) else None


def _write_stream(path: Path, content: str | bytes | None) -> None:
    if content is None:
        text = ""
    elif isinstance(content, bytes):
        text = content.decode(errors="replace")
    else:
        text = content
    path.write_text(text, encoding="utf-8")


def execute_cell(
    protocol: Protocol,
    plan: CellPlan,
    *,
    output_dir: Path,
    caps: Caps,
    executable: str,
    base_url: str | None,
) -> dict[str, Any]:
    """Execute one cell and always emit a terminal cell manifest."""

    task = next(task for task in protocol.tasks if task.task_id == plan.task_id)
    arm = next(arm for arm in protocol.arms if arm.name == plan.arm)
    cell_dir = output_dir / "cells" / plan.cell_id
    cell_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir = cell_dir / "artifacts"
    artifacts_dir.mkdir()
    command = command_for(
        protocol,
        task,
        arm,
        plan,
        caps=caps,
        executable=executable,
        artifacts_dir=artifacts_dir,
        base_url=base_url,
    )
    started_at = _now()
    start = monotonic()
    exit_code: int | None = None
    timed_out = False
    exception: str | None = None
    stdout: str | bytes | None = None
    stderr: str | bytes | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=protocol.project_root,
            capture_output=True,
            text=True,
            timeout=caps.wall_time_seconds,
            check=False,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout
        stderr = error.stderr
        exception = f"wall-time cap exceeded ({caps.wall_time_seconds}s)"
    except OSError as error:
        exception = f"could not start synthesizer: {error}"
    wall_time = monotonic() - start
    _write_stream(cell_dir / "stdout.log", stdout)
    _write_stream(cell_dir / "stderr.log", stderr)

    core_dir = _locate_core_artifact(artifacts_dir)
    core_status = _terminal_core_status(core_dir)
    if core_dir is None:
        heldout = unavailable_evaluation(task, "core artifact directory is unavailable")
    else:
        heldout = evaluate_result(task, core_dir / "result.json")
    write_json_atomic(cell_dir / "heldout.json", heldout)
    terminal_status = (
        "completed"
        if exit_code == 0 and core_status == "completed"
        else "timed_out"
        if timed_out
        else "failed"
    )
    cell_manifest: dict[str, Any] = {
        "schema_version": 2,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "cell": asdict(plan),
        "status": terminal_status,
        "started_at": started_at,
        "completed_at": _now(),
        "wall_time_seconds": wall_time,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "exception": exception,
        "command": command,
        "working_directory": str(protocol.project_root),
        "core_artifact": str(core_dir.relative_to(cell_dir)) if core_dir else None,
        "core_status": core_status,
        "files": {
            "stdout": "stdout.log",
            "stderr": "stderr.log",
            "heldout": "heldout.json",
        },
    }
    write_json_atomic(cell_dir / "cell.json", cell_manifest)
    return cell_manifest


def _parse_csv(value: str | None) -> set[str] | None:
    if value is None:
        return None
    result = {part.strip() for part in value.split(",") if part.strip()}
    return result or None


def _parse_seed_csv(value: str | None) -> set[int] | None:
    parsed = _parse_csv(value)
    return None if parsed is None else {int(seed) for seed in parsed}


def _validate_selection(protocol: Protocol, plans: tuple[CellPlan, ...]) -> None:
    declared = {task.task_id for task in protocol.tasks}
    declared_arms = {arm.name for arm in protocol.arms}
    if any(plan.task_id not in declared or plan.arm not in declared_arms for plan in plans):
        raise ValueError("plan contains undeclared cells")


def _initialize_matrix(
    output: Path,
    protocol: Protocol,
    plans: tuple[CellPlan, ...],
    *,
    stage: StageSpec | None,
    caps: Caps,
    audit_certificate: Mapping[str, object] | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(protocol.path, output / "protocol.json")
    write_json_atomic(
        output / "matrix_manifest.json",
        {
            "schema_version": 2,
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.protocol_sha256,
            "protocol_status": protocol.status,
            "created_at": _now(),
            "analysis_population": "intention-to-treat",
            "stage_id": stage.stage_id if stage else None,
            "audit_certificate": audit_certificate,
            "effective_caps": asdict(caps),
            "models": [asdict(model) for model in protocol.models],
            "planned_cells": [asdict(plan) for plan in plans],
        },
    )


def _validate_resume(
    output: Path,
    protocol: Protocol,
    plans: tuple[CellPlan, ...],
    *,
    stage: StageSpec | None,
    caps: Caps,
    audit_certificate: Mapping[str, object] | None = None,
) -> None:
    manifest_path = output / "matrix_manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("protocol_sha256") != (
        protocol.protocol_sha256
    ):
        raise ValueError("resume directory was created from a different protocol hash")
    if document.get("planned_cells") != [asdict(plan) for plan in plans]:
        raise ValueError("resume selection differs from the original planned cells")
    if document.get("stage_id") != (stage.stage_id if stage else None):
        raise ValueError("resume stage differs from the original matrix")
    if document.get("effective_caps") != asdict(caps):
        raise ValueError("resume resource caps differ from the original matrix")
    if document.get("audit_certificate") != audit_certificate:
        raise ValueError("resume target-audit certificate differs from the original matrix")


def _cost_summary(plans: tuple[CellPlan, ...], caps: Caps) -> dict[str, Any]:
    provider_cells = sum(plan.model_scope == "model-specific" for plan in plans)

    def first_seen(values: Sequence[Any]) -> list[Any]:
        return list(dict.fromkeys(values))

    return {
        "cells": len(plans),
        "local_control_cells": len(plans) - provider_cells,
        "provider_cells": provider_cells,
        "provider_candidate_score_cap": provider_cells * caps.max_scored_candidates,
        "models": first_seen(
            [plan.model_id for plan in plans if plan.model_id is not None]
        ),
        "tasks": first_seen([plan.task_id for plan in plans]),
        "arms": first_seen([plan.arm for plan in plans]),
        "seeds": first_seen([plan.seed for plan in plans]),
    }


def _enforce_cost_gates(
    summary: dict[str, Any],
    *,
    stage: StageSpec | None,
    max_provider_cells: int | None,
    max_provider_score_cap: int | None,
) -> None:
    provider_cells = int(summary["provider_cells"])
    score_cap = int(summary["provider_candidate_score_cap"])
    if stage is not None and score_cap > stage.max_provider_scored_candidates:
        raise ValueError(
            f"stage {stage.stage_id} provider candidate-score cap {score_cap} exceeds "
            f"its preregistered gate {stage.max_provider_scored_candidates}"
        )
    if max_provider_cells is not None and provider_cells > max_provider_cells:
        raise ValueError(
            f"selection has {provider_cells} provider cells, above explicit gate "
            f"{max_provider_cells}"
        )
    if max_provider_score_cap is not None and score_cap > max_provider_score_cap:
        raise ValueError(
            f"selection provider candidate-score cap {score_cap} exceeds explicit gate "
            f"{max_provider_score_cap}"
        )


def _base_url_for_plan(
    protocol: Protocol, plan: CellPlan, override: str | None
) -> str | None:
    if plan.model_scope != "model-specific":
        return None
    if override is not None:
        return override
    model = next(model for model in protocol.models if model.model_id == plan.model_id)
    if model.base_url_env:
        model_endpoint = os.environ.get(model.base_url_env)
        if model_endpoint is not None:
            return model_endpoint
    return os.environ.get(protocol.provider.base_url_env)


def _resolve_executable(value: str) -> str:
    """Resolve once so hashing and subprocess execution name the same file."""

    located = shutil.which(value)
    candidate = Path(located if located is not None else value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"synthesizer executable does not exist: {resolved}")
    return str(resolved)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", help="Comma-separated exploratory/confirmatory labels")
    parser.add_argument("--tasks", help="Comma-separated task ids")
    parser.add_argument("--arms", help="Comma-separated U,D,Q,QD arms")
    parser.add_argument("--seeds", help="Comma-separated protocol seeds")
    parser.add_argument("--models", help="Comma-separated model ids; Q/QD only")
    parser.add_argument("--stage", help="Named preregistered cost gate")
    parser.add_argument("--base-url", help="vLLM /v1 URL; otherwise read protocol env name")
    parser.add_argument("--max-provider-cells", type=int)
    parser.add_argument("--max-provider-score-cap", type=int)
    parser.add_argument(
        "--audit-certificate",
        type=Path,
        help="Passing target-audit certificate required by a provider-backed stage",
    )
    parser.add_argument("--executable", default="modelsmc-pbe")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    protocol = load_protocol(args.protocol)
    labels = _parse_csv(args.label)
    tasks = _parse_csv(args.tasks)
    arms = _parse_csv(args.arms)
    seeds = _parse_seed_csv(args.seeds)
    models = _parse_csv(args.models)
    stage = find_stage(protocol, args.stage)
    caps = effective_caps(protocol, stage)
    plans = build_plan(
        protocol,
        stage=stage,
        labels=labels,
        task_ids=tasks,
        arms=arms,
        seeds=seeds,
        model_ids=models,
    )
    _validate_selection(protocol, plans)
    summary = _cost_summary(plans, caps)
    _enforce_cost_gates(
        summary,
        stage=stage,
        max_provider_cells=args.max_provider_cells,
        max_provider_score_cap=args.max_provider_score_cap,
    )
    output = args.output.expanduser().resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol_sha256": protocol.protocol_sha256,
                    "stage_id": stage.stage_id if stage else None,
                    "requires_audit_stage": (
                        stage.requires_audit_stage if stage is not None else None
                    ),
                    "audit_certificate_required": bool(
                        stage is not None and stage.requires_audit_stage is not None
                    ),
                    "effective_caps": asdict(caps),
                    "cost_gate": summary,
                    "cells": [asdict(plan) for plan in plans],
                },
                indent=2,
            )
        )
        return 0
    executable = _resolve_executable(args.executable)
    if (
        int(summary["provider_cells"]) > 0
        and protocol.target_contract is not None
        and protocol.target_contract.reference_audit_required
        and (stage is None or stage.requires_audit_stage is None)
    ):
        raise ValueError(
            "provider-backed selection is forbidden outside a named stage with an "
            "audit prerequisite"
        )
    audit_certificate: Mapping[str, object] | None = None
    if stage is not None and stage.requires_audit_stage is not None:
        if args.audit_certificate is None:
            raise ValueError(
                f"stage {stage.stage_id} requires --audit-certificate from "
                f"{stage.requires_audit_stage}"
            )
        audit_certificate = validate_target_audit_certificate(
            args.audit_certificate,
            protocol,
            stage,
            executable=executable,
        )
    elif args.audit_certificate is not None:
        raise ValueError("--audit-certificate is only valid for a stage that requires one")
    missing_endpoints = [
        plan.cell_id
        for plan in plans
        if plan.model_scope == "model-specific"
        and _base_url_for_plan(protocol, plan, args.base_url) is None
    ]
    if missing_endpoints:
        raise ValueError(
            "Q/QD selection lacks a vLLM endpoint for cells: "
            + ", ".join(missing_endpoints[:4])
        )
    if args.resume:
        _validate_resume(
            output,
            protocol,
            plans,
            stage=stage,
            caps=caps,
            audit_certificate=audit_certificate,
        )
    else:
        _initialize_matrix(
            output,
            protocol,
            plans,
            stage=stage,
            caps=caps,
            audit_certificate=audit_certificate,
        )

    failures = 0
    for index, plan in enumerate(plans, start=1):
        cell_path = output / "cells" / plan.cell_id / "cell.json"
        if cell_path.exists():
            print(f"[matrix] {index}/{len(plans)} retained existing {plan.cell_id}")
            continue
        print(f"[matrix] {index}/{len(plans)} running {plan.cell_id}")
        try:
            record = execute_cell(
                protocol,
                plan,
                output_dir=output,
                caps=caps,
                executable=executable,
                base_url=_base_url_for_plan(protocol, plan, args.base_url),
            )
        except (OSError, ValueError) as error:
            # Configuration errors before subprocess launch are fatal: counting them as
            # treatment failures would conflate a malformed study with an algorithm.
            raise RuntimeError(f"could not materialize {plan.cell_id}: {error}") from error
        failures += int(record["status"] != "completed")
        print(
            f"[matrix] {plan.cell_id} status={record['status']} "
            f"wall={record['wall_time_seconds']:.3f}s"
        )
    print(f"[matrix] finished cells={len(plans)} process_failures={failures}")
    contract = protocol.target_contract
    if (
        contract is not None
        and contract.reference_audit_required
        and stage is not None
        and stage.stage_id == contract.reference_audit_stage
    ):
        certificate = create_target_audit_certificate(
            protocol,
            stage,
            output,
            executable=executable,
        )
        if certificate["invariants_passed"] is not True:
            raise RuntimeError(
                f"target audit failed; see {output / CERTIFICATE_NAME}"
            )
        print(f"[matrix] target audit passed certificate={output / CERTIFICATE_NAME}")
    # Treatment failures are data, so a completed matrix command exits successfully.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

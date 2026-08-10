"""Execute the preregistered U/D/Q/QD matrix without dropping failed cells."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from research.heldout import evaluate_result, unavailable_evaluation, write_json_atomic
from research.protocol import AnalysisLabel, ArmName, ArmSpec, Protocol, TaskSpec, load_protocol


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class CellPlan:
    cell_id: str
    task_id: str
    arm: ArmName
    seed: int
    analysis_label: AnalysisLabel


def build_plan(
    protocol: Protocol,
    *,
    labels: set[str] | None = None,
    task_ids: set[str] | None = None,
    arms: set[str] | None = None,
    seeds: set[int] | None = None,
) -> tuple[CellPlan, ...]:
    """Build the deterministic task-major, arm-major, seed-major matrix."""

    selected_tasks = [
        task
        for task in protocol.tasks
        if (labels is None or task.label in labels)
        and (task_ids is None or task.task_id in task_ids)
    ]
    selected_arms = [arm for arm in protocol.arms if arms is None or arm.name in arms]
    selected_seeds = [seed for seed in protocol.seeds if seeds is None or seed in seeds]
    if not selected_tasks or not selected_arms or not selected_seeds:
        raise ValueError("matrix selection is empty")
    plans = [
        CellPlan(
            cell_id=f"{task.task_id}--{arm.name}--seed-{seed}",
            task_id=task.task_id,
            arm=arm.name,
            seed=seed,
            analysis_label=task.label,
        )
        for task in selected_tasks
        for arm in selected_arms
        for seed in selected_seeds
    ]
    return tuple(plans)


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def command_for(
    protocol: Protocol,
    task: TaskSpec,
    arm: ArmSpec,
    plan: CellPlan,
    *,
    executable: str,
    artifacts_dir: Path,
    base_url: str | None,
) -> list[str]:
    caps = protocol.caps
    command = [
        executable,
        "synthesize",
        str(task.spec_path),
        "--mode",
        "importance-smc",
        "--proposal",
        arm.proposal,
        "--model",
        protocol.provider.model,
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
    for name, value in sorted(protocol.shared_arguments.items()):
        command.extend((_flag(name), str(value)))
    if arm.proposal == "vllm":
        if base_url is None:
            raise ValueError(
                f"arm {arm.name} requires --base-url or {protocol.provider.base_url_env}"
            )
        command.extend(("--base-url", base_url))
        if protocol.provider.api_key_env:
            command.extend(("--api-key-env", protocol.provider.api_key_env))
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
            timeout=protocol.caps.wall_time_seconds,
            check=False,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout
        stderr = error.stderr
        exception = f"wall-time cap exceeded ({protocol.caps.wall_time_seconds}s)"
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
        "schema_version": 1,
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


def _initialize_matrix(output: Path, protocol: Protocol, plans: tuple[CellPlan, ...]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(protocol.path, output / "protocol.json")
    write_json_atomic(
        output / "matrix_manifest.json",
        {
            "schema_version": 1,
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.protocol_sha256,
            "protocol_status": protocol.status,
            "created_at": _now(),
            "analysis_population": "intention-to-treat",
            "planned_cells": [asdict(plan) for plan in plans],
        },
    )


def _validate_resume(output: Path, protocol: Protocol) -> None:
    manifest_path = output / "matrix_manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or document.get("protocol_sha256") != protocol.protocol_sha256
    ):
        raise ValueError("resume directory was created from a different protocol hash")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", help="Comma-separated exploratory/confirmatory labels")
    parser.add_argument("--tasks", help="Comma-separated task ids")
    parser.add_argument("--arms", help="Comma-separated U,D,Q,QD arms")
    parser.add_argument("--seeds", help="Comma-separated protocol seeds")
    parser.add_argument("--base-url", help="vLLM /v1 URL; otherwise read protocol env name")
    parser.add_argument("--executable", default="modelsmc-pbe")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    protocol = load_protocol(args.protocol)
    labels = _parse_csv(args.label)
    tasks = _parse_csv(args.tasks)
    arms = _parse_csv(args.arms)
    seeds = _parse_seed_csv(args.seeds)
    plans = build_plan(protocol, labels=labels, task_ids=tasks, arms=arms, seeds=seeds)
    _validate_selection(protocol, plans)
    base_url = args.base_url or os.environ.get(protocol.provider.base_url_env)
    if any(plan.arm in {"Q", "QD"} for plan in plans) and base_url is None:
        raise ValueError(
            f"Q/QD selection requires --base-url or {protocol.provider.base_url_env}"
        )
    executable = shutil.which(args.executable) or args.executable
    output = args.output.expanduser().resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol_sha256": protocol.protocol_sha256,
                    "cells": [asdict(plan) for plan in plans],
                },
                indent=2,
            )
        )
        return 0
    if args.resume:
        _validate_resume(output, protocol)
    else:
        _initialize_matrix(output, protocol, plans)

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
                executable=executable,
                base_url=base_url,
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
    # Treatment failures are data, so a completed matrix command exits successfully.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

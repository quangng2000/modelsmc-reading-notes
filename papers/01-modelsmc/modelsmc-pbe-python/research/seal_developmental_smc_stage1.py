"""Seal the completed public Stage-1 developmental SMC artifact bundle.

This is a post-run integrity tool.  It does not launch inference, contact a
provider, or change the frozen study protocol.  It separately records the
aborted pre-provider v1 attempt and accepts the corrected v2 matrix only when
all 36 frozen task-arm runs and the fail-closed analysis are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from research.analyze_developmental_smc_benchmark import (
    ANALYSIS_SCHEMA,
    _validate_result,
    analyze,
)
from research.run_developmental_smc_benchmark import (
    ARMS,
    TASK_IDS,
    _index_runs,
    _object,
    canonical_bytes,
    sha256_file,
    validate_protocol,
    validate_public_suite,
)

SCHEMA = "developmental-smc-stage1-public-bundle-seal-v1"
FROZEN_PROTOCOL_SHA256 = (
    "0b7092349c5e240aac378bd04a97020ea15e170d56b8d573890d418217db689d"
)
ABORTED_V1_PROTOCOL_SHA256 = (
    "06a6bba8a4634574a2bc403a9dbf3673f788fa0d143a48979d6e7a43c8543bcc"
)
ABORTED_V1_SCHEDULER_SHA256 = (
    "32cc2ab6a0d80f7df9c3779966e73a5a1d295423fe665c6607ff911942e3be8f"
)
CORRECTED_V2_SCHEDULER_SHA256 = (
    "448e5220761584b37d739185be04af72060e524c812e0c5c5d582d3eef002336"
)
EXECUTION_ORDER = ("evidence-only", "grammar-only", "llm-smc")
INTER_TASK_CONCURRENCY = {
    "evidence-only": 6,
    "grammar-only": 6,
    "llm-smc": 4,
}
V1_PREFLIGHTS = tuple(
    [("evidence-only", task_id) for task_id in TASK_IDS]
    + [("grammar-only", task_id) for task_id in TASK_IDS[:6]]
)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _expect(actual: object, expected: object, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _regular_directory(path: Path, *, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir() or path.is_symlink():
        raise ValueError(f"{name} must be a regular directory")
    return resolved


def _regular_file(path: Path, *, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a regular file")
    return resolved


def _within(root: Path, path: Path, *, name: str) -> Path:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{name} escaped {root}")
    return resolved


def _repo_path(repo_root: Path, path: Path, *, name: str) -> str:
    resolved = _within(repo_root, path, name=name)
    return resolved.relative_to(repo_root).as_posix()


def _inventory(root: Path) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().encode()):
        if path.is_symlink():
            raise ValueError(f"bundle inventory rejects symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"bundle inventory rejects special path: {path}")
        payload = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "schema": "sha256-file-inventory-v1",
        "file_count": len(records),
        "total_bytes": sum(cast(int, record["bytes"]) for record in records),
        "inventory_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
        "records": records,
    }


def _file_binding(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_json_lines(path: Path, *, name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} line {line_number} is not JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{name} line {line_number} is not an object")
        records.append(cast(dict[str, Any], value))
    return records


def _option_map(
    command: object,
    *,
    module: str,
    name: str,
) -> tuple[str, dict[str, str]]:
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        raise ValueError(f"{name} command must be an array of strings")
    values = cast(list[str], command)
    if len(values) < 3 or values[1:3] != ["-m", module]:
        raise ValueError(f"{name} has the wrong module invocation")
    tail = values[3:]
    if len(tail) % 2:
        raise ValueError(f"{name} options must be flag/value pairs")
    options: dict[str, str] = {}
    for flag, value in zip(tail[::2], tail[1::2], strict=True):
        if not flag.startswith("--") or flag in options:
            raise ValueError(f"{name} has an invalid or duplicate option {flag!r}")
        options[flag] = value
    return values[0], options


def _expected_harness_options(
    *,
    repo_root: Path,
    protocol_path: Path,
    suite_dir: Path,
    runs_root: Path,
    arm: str,
    task_id: str,
    run: Mapping[str, Any],
    provider: Mapping[str, Any],
    protocol_sha256: str,
) -> dict[str, str]:
    options = {
        "--task": str((suite_dir / f"{task_id}.json").resolve()),
        "--output": str((runs_root / arm / task_id).resolve()),
        "--study-protocol": str(protocol_path.resolve()),
        "--run-id": f"{task_id}--{arm}",
        "--expected-study-protocol-sha256": protocol_sha256,
        "--base-url": str(provider["base_url"]),
        "--model": str(provider["model"]),
        "--reasoning-effort": str(provider["reasoning_effort"]),
        "--temperature": str(provider["temperature"]),
        "--max-tokens": str(provider["max_output_tokens"]),
        "--timeout-seconds": str(provider["timeout_seconds"]),
        "--epsilon": str(run["epsilon"]),
        "--evidence-scale": str(run["evidence_scale"]),
        "--start-seed": str(run["start_seed"]),
        "--provider-seed": str(run["provider_seed"]),
        "--sample-seed": str(run["sample_seed"]),
        "--resample-seed": str(run["resample_seed"]),
        "--parent-count": str(run["parent_count"]),
        "--offspring-per-parent": str(run["offspring_per_parent"]),
        "--first-round-offspring": str(run["first_round_offspring"]),
        "--max-concurrency": str(run["max_concurrency"]),
        "--exact-reference-limit": str(run["exact_reference_limit"]),
        "--proposal-source": str(run["proposal_source"]),
    }
    if run.get("random_shortlist_seed") is not None:
        options["--random-shortlist-seed"] = str(run["random_shortlist_seed"])
    return options


def _validate_preflight(
    *,
    path: Path,
    repo_root: Path,
    protocol_path: Path,
    suite_dir: Path,
    runs_root: Path,
    task_sha256: str,
    manifest_sha256: str,
    task_id: str,
    arm: str,
    run: Mapping[str, Any],
    provider: Mapping[str, Any],
    protocol_sha256: str,
) -> str:
    preflight = _read_object(path)
    for field, expected in {
        "schema": "developmental-smc-benchmark-preflight-v1",
        "status": "validated-before-provider-call",
        "developmental": True,
        "task_id": task_id,
        "arm": arm,
        "run_id": f"{task_id}--{arm}",
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": manifest_sha256,
        "task_sha256": task_sha256,
        "preflight_path": str(path.resolve()),
    }.items():
        _expect(preflight.get(field), expected, name=f"{path} {field}")
    python_executable, options = _option_map(
        preflight.get("command"),
        module="research.evidence_shortlist_smc",
        name=str(path),
    )
    _expect(
        options,
        _expected_harness_options(
            repo_root=repo_root,
            protocol_path=protocol_path,
            suite_dir=suite_dir,
            runs_root=runs_root,
            arm=arm,
            task_id=task_id,
            run=run,
            provider=provider,
            protocol_sha256=protocol_sha256,
        ),
        name=f"{path} harness options",
    )
    return python_executable


def _validate_launch(
    *,
    root: Path,
    protocol_sha256: str,
    scheduler_sha256: str,
) -> dict[str, Any]:
    launch = _read_object(root / "launch-seal.json")
    for field, expected in {
        "schema": "developmental-smc-matrix-launch-v1",
        "classification": "developmental-nonblind",
        "protocol_sha256": protocol_sha256,
        "scheduler_sha256": scheduler_sha256,
        "execution_order": list(EXECUTION_ORDER),
        "task_order": list(TASK_IDS),
        "inter_task_concurrency": INTER_TASK_CONCURRENCY,
        "retry_policy": "none; every failed child is retained",
        "child_count": 36,
    }.items():
        _expect(launch.get(field), expected, name=f"launch {field}")
    started = launch.get("started_unix")
    if (
        not isinstance(started, (int, float))
        or isinstance(started, bool)
        or not math.isfinite(started)
    ):
        raise ValueError("launch started_unix must be finite")
    return launch


def _validate_driver_child(
    *,
    record: Mapping[str, Any],
    repo_root: Path,
    protocol_path: Path,
    suite_dir: Path,
    matrix_root: Path,
    task_id: str,
    arm: str,
    protocol_sha256: str,
    expected_returncode: int,
) -> None:
    _expect(record.get("task_id"), task_id, name="driver child task_id")
    _expect(record.get("arm"), arm, name="driver child arm")
    _expect(record.get("returncode"), expected_returncode, name="driver child returncode")
    started = record.get("started_unix")
    completed = record.get("completed_unix")
    if (
        not isinstance(started, (int, float))
        or isinstance(started, bool)
        or not math.isfinite(started)
        or not isinstance(completed, (int, float))
        or isinstance(completed, bool)
        or not math.isfinite(completed)
        or completed < started
    ):
        raise ValueError("driver child timestamps are invalid")
    expected_log = matrix_root / "driver-logs" / arm / f"{task_id}.log"
    _expect(record.get("log_path"), str(expected_log), name="driver child log_path")
    _regular_file(expected_log, name="driver child log")
    python_executable, options = _option_map(
        record.get("command"),
        module="research.run_developmental_smc_benchmark",
        name="driver child",
    )
    expected = {
        "--repo-root": str(repo_root),
        "--protocol": str(protocol_path),
        "--expected-protocol-sha256": protocol_sha256,
        "--suite-dir": str(suite_dir),
        "--runs-root": str(matrix_root / "runs"),
        "--task-id": task_id,
        "--arm": arm,
        "--python-executable": python_executable,
    }
    _expect(options, expected, name="driver child options")


def _normalized(log_weights: Sequence[float]) -> list[float]:
    if not log_weights or any(not math.isfinite(value) for value in log_weights):
        raise ValueError("log weights must be a nonempty finite sequence")
    maximum = max(log_weights)
    unnormalized = [math.exp(value - maximum) for value in log_weights]
    total = sum(unnormalized)
    return [value / total for value in unnormalized]


def _close_sequence(actual: object, expected: Sequence[float], *, name: str) -> None:
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise ValueError(f"{name} has the wrong shape")
    for index, (observed, reference) in enumerate(zip(actual, expected, strict=True)):
        if (
            not isinstance(observed, (int, float))
            or isinstance(observed, bool)
            or not math.isfinite(observed)
            or not math.isclose(float(observed), reference, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError(f"{name}[{index}] differs from recomputation")


def _provider_inventory(run_dir: Path) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for path in sorted((run_dir / "provider").glob("**/*.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"provider inventory rejects non-regular path: {path}")
        payload = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "records": records,
        "inventory_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
    }


def _validate_result_sidecars(
    *,
    run_dir: Path,
    result: Mapping[str, Any],
    run: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    task_id: str,
    arm: str,
) -> int:
    embedded = _object(result.get("protocol"), name="result.protocol")
    _expect(
        _read_object(run_dir / "protocol.json"),
        embedded,
        name=f"{task_id}/{arm} protocol sidecar",
    )
    _expect(
        embedded.get("harness_sha256"),
        _object(protocol["source_bindings"], name="source_bindings")["harness"]["sha256"],
        name="embedded harness SHA-256",
    )
    for field, expected in {
        "epsilon": run["epsilon"],
        "evidence_scale": run["evidence_scale"],
        "max_concurrency": run["max_concurrency"],
        "exact_reference_limit": run["exact_reference_limit"],
        "offspring_per_parent": run["offspring_per_parent"],
        "first_round_offspring": run["first_round_offspring"],
        "resampled_parent_count": run["parent_count"],
        "proposal_source": run["proposal_source"],
        "random_shortlist_seed": run["random_shortlist_seed"],
        "provider_seed": run["provider_seed"],
        "sample_seed": run["sample_seed"],
        "resample_seed": run["resample_seed"],
        "start_seed": run["start_seed"],
    }.items():
        _expect(embedded.get(field), expected, name=f"embedded {field}")
    frozen = _object(embedded.get("frozen_invocation"), name="frozen invocation")
    _expect(frozen.get("study_protocol_sha256"), protocol_sha256, name="frozen protocol")
    _expect(frozen.get("run_id"), run["id"], name="frozen run ID")
    provider = _object(protocol.get("provider"), name="provider")
    expected_arguments = {
        "base_url": provider["base_url"],
        "model": provider["model"],
        "reasoning_effort": provider["reasoning_effort"],
        "temperature": provider["temperature"],
        "max_tokens": provider["max_output_tokens"],
        "timeout_seconds": provider["timeout_seconds"],
        "epsilon": run["epsilon"],
        "evidence_scale": run["evidence_scale"],
        "start_seed": run["start_seed"],
        "parent_count": run["parent_count"],
        "offspring_per_parent": run["offspring_per_parent"],
        "first_round_offspring": run["first_round_offspring"],
        "provider_seed": run["provider_seed"],
        "sample_seed": run["sample_seed"],
        "resample_seed": run["resample_seed"],
        "max_concurrency": run["max_concurrency"],
        "exact_reference_limit": run["exact_reference_limit"],
        "proposal_source": run["proposal_source"],
        "random_shortlist_seed": run["random_shortlist_seed"],
    }
    _expect(frozen.get("validated_arguments"), expected_arguments, name="frozen arguments")
    _expect(
        frozen.get("validated_derived_fields"),
        {
            "logical_execution_cap": 29,
            "provider_call_cap": run["provider_call_cap"],
            "terminal_weighted_particles": 8,
        },
        name="frozen derived fields",
    )
    served = embedded.get("served_model")
    if arm == "llm-smc":
        served_record = _object(served, name="served model")
        _expect(served_record.get("id"), provider["model"], name="served model ID")
        _expect(
            served_record.get("revision"), provider["model_revision"], name="served revision"
        )
        root = served_record.get("root")
        if not isinstance(root, str) or Path(root).name != provider["model_revision"]:
            raise ValueError("served model root does not bind the frozen revision")
    elif served is not None:
        raise ValueError("provider-free control unexpectedly has served-model metadata")

    rounds = cast(list[dict[str, Any]], result["rounds"])
    for index, round_record in enumerate(rounds, 1):
        _expect(
            _read_object(run_dir / f"round-{index:02d}.json"),
            round_record,
            name=f"round {index} sidecar",
        )
        proposals = round_record.get("proposals")
        if not isinstance(proposals, list):
            raise ValueError("round proposals must be an array")
        log_weights: list[float] = []
        for proposal in proposals:
            record = _object(proposal, name="proposal")
            for field in ("q", "log_gamma", "log_weight"):
                value = record.get(field)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"proposal {field} must be finite")
            if not 0.0 < cast(float, record["q"]) <= 1.0:
                raise ValueError("proposal q must lie in (0,1]")
            log_weights.append(float(record["log_weight"]))
        normalized = _normalized(log_weights)
        _close_sequence(
            round_record.get("normalized_weights"),
            normalized,
            name=f"round {index} normalized weights",
        )
        ess = 1.0 / sum(weight * weight for weight in normalized)
        recorded_ess = round_record.get("ess")
        if not isinstance(recorded_ess, (int, float)) or not math.isclose(
            float(recorded_ess), ess, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"round {index} ESS differs from recomputation")

    executions = json.loads((run_dir / "executions.json").read_text(encoding="utf-8"))
    if not isinstance(executions, list) or len(executions) != 29:
        raise ValueError("executions sidecar must contain exactly 29 slots")
    _expect([record.get("slot") for record in executions], list(range(1, 30)), name="slots")
    flattened = [proposal for round_record in rounds for proposal in round_record["proposals"]]
    _expect(executions[1:], flattened, name="execution/round proposal records")
    search = _object(result.get("search"), name="search")
    _expect(
        search.get("grammar_restart_draws"),
        sum(record.get("sampled_branch") == "full-grammar-restart" for record in executions),
        name="grammar restart count",
    )
    _expect(
        search.get("shortlist_draws"),
        sum(record.get("sampled_branch") == "shortlist-slot" for record in executions),
        name="shortlist draw count",
    )
    if arm == "grammar-only":
        _expect(search.get("grammar_restart_draws"), 28, name="grammar-only restarts")
        _expect(search.get("shortlist_draws"), 0, name="grammar-only shortlist draws")

    provider_seal = _read_object(run_dir / "provider-seal.json")
    recomputed_inventory = _provider_inventory(run_dir)
    _expect(provider_seal, recomputed_inventory, name="provider byte inventory")
    _expect(
        result.get("provider_inventory_sha256"),
        recomputed_inventory["inventory_sha256"],
        name="result provider inventory",
    )
    provider_results = sorted((run_dir / "provider").glob("**/result.json"))
    provider_calls = search.get("provider_calls")
    _expect(len(provider_results), provider_calls, name="provider result count")
    _expect(
        sum(cast(int, round_record["provider_calls"]) for round_record in rounds),
        provider_calls,
        name="round provider call total",
    )
    if arm != "llm-smc":
        _expect(recomputed_inventory["records"], [], name="control provider inventory")

    statuses: Counter[str] = Counter()
    slot_statuses: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    for path in provider_results:
        record = _read_object(path)
        statuses[str(record.get("status", "missing-status"))] += 1
        slots = record.get("slots")
        if isinstance(slots, list):
            for slot in slots:
                if isinstance(slot, dict):
                    slot_statuses[str(slot.get("status", "missing-status"))] += 1
        raw_usage = record.get("usage")
        if isinstance(raw_usage, dict):
            for key, value in raw_usage.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    usage[key] += value
    _expect(search.get("provider_status_counts"), dict(sorted(statuses.items())), name="statuses")
    _expect(
        search.get("provider_slot_status_counts"),
        dict(sorted(slot_statuses.items())),
        name="slot statuses",
    )
    _expect(search.get("provider_usage_totals"), dict(sorted(usage.items())), name="usage")

    terminal_weights = cast(list[float], rounds[-1]["normalized_weights"])
    particles = _object(result.get("inference"), name="inference").get("final_particles")
    if not isinstance(particles, list):
        raise ValueError("final particles must be an array")
    _close_sequence(
        [record.get("normalized_weight") for record in particles if isinstance(record, dict)],
        terminal_weights,
        name="terminal particle weights",
    )
    final_ess = 1.0 / sum(weight * weight for weight in terminal_weights)
    recorded_final_ess = cast(dict[str, Any], result["inference"]).get("final_ess")
    if not isinstance(recorded_final_ess, (int, float)) or not math.isclose(
        float(recorded_final_ess), final_ess, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("final ESS differs from terminal weights")
    return cast(int, provider_calls)


def _validate_aborted_v1(
    *,
    repo_root: Path,
    protocol_path: Path,
    suite_dir: Path,
    v1_root: Path,
    v2_root: Path,
    protocol: Mapping[str, Any],
    task_paths: Mapping[str, Path],
    runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    launch = _validate_launch(
        root=v1_root,
        protocol_sha256=ABORTED_V1_PROTOCOL_SHA256,
        scheduler_sha256=ABORTED_V1_SCHEDULER_SHA256,
    )
    if (v1_root / "driver-result.json").exists():
        raise ValueError("aborted v1 must not have a completed driver result")
    provider_paths = [path for path in v1_root.rglob("*") if "provider" in path.parts]
    if provider_paths:
        raise ValueError("aborted v1 contains provider material")
    if list(v1_root.rglob("result.json")) or list(v1_root.rglob("protocol.json")):
        raise ValueError("aborted v1 progressed past frozen invocation validation")

    manifest_sha256 = sha256_file(suite_dir / "manifest.json")
    provider = _object(protocol.get("provider"), name="provider")
    python_executables: set[str] = set()
    expected_preflights = {
        v1_root / "runs" / arm / f"{task_id}.preflight.json"
        for arm, task_id in V1_PREFLIGHTS
    }
    actual_preflights = set(v1_root.glob("runs/*/*.preflight.json"))
    _expect(actual_preflights, expected_preflights, name="aborted v1 preflight set")
    for arm, task_id in V1_PREFLIGHTS:
        run = runs[f"{task_id}--{arm}"]
        preflight = v1_root / "runs" / arm / f"{task_id}.preflight.json"
        python_executables.add(
            _validate_preflight(
                path=preflight,
                repo_root=repo_root,
                protocol_path=protocol_path,
                suite_dir=suite_dir,
                runs_root=v1_root / "runs",
                task_sha256=sha256_file(task_paths[task_id]),
                manifest_sha256=manifest_sha256,
                task_id=task_id,
                arm=arm,
                run=run,
                provider=provider,
                protocol_sha256=ABORTED_V1_PROTOCOL_SHA256,
            )
        )
        log_path = v1_root / "driver-logs" / arm / f"{task_id}.log"
        _regular_file(log_path, name="aborted v1 child log")
        text = log_path.read_text(encoding="utf-8")
        if arm == "evidence-only":
            required = (
                "frozen_invocation = validate_frozen_invocation(args)",
                'study["proposal"])["epsilon"]',
                "KeyError: 'epsilon'",
            )
            if any(fragment not in text for fragment in required):
                raise ValueError(f"v1 failure log is not the frozen-validation failure: {log_path}")
            if "provider_metadata(" in text:
                raise ValueError("v1 failure stack entered provider metadata")
        elif text:
            raise ValueError("aborted, never-completed v1 grammar child has a nonempty log")
        if (v1_root / "runs" / arm / task_id).exists():
            raise ValueError("aborted v1 unexpectedly created a run output directory")
    if len(python_executables) != 1:
        raise ValueError("aborted v1 used multiple Python executables")

    driver_log = _regular_file(
        Path(str(v1_root) + ".driver.log"), name="aborted v1 driver log"
    )
    driver_records = _read_json_lines(driver_log, name="aborted v1 driver log")
    expected_failed = [("evidence-only", task_id) for task_id in TASK_IDS]
    if len(driver_records) != len(expected_failed):
        raise ValueError("aborted v1 driver log must contain twelve failed children")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in driver_records:
        key = (str(record.get("arm")), str(record.get("task_id")))
        if key in by_key:
            raise ValueError("aborted v1 driver log contains a duplicate child")
        by_key[key] = record
    _expect(set(by_key), set(expected_failed), name="aborted v1 failed child set")
    for arm, task_id in expected_failed:
        _validate_driver_child(
            record=by_key[(arm, task_id)],
            repo_root=repo_root,
            protocol_path=protocol_path,
            suite_dir=suite_dir,
            matrix_root=v1_root,
            task_id=task_id,
            arm=arm,
            protocol_sha256=ABORTED_V1_PROTOCOL_SHA256,
            expected_returncode=1,
        )

    # The 18 v1 preflights declared the same task-arm semantics as v2 except
    # for the corrected protocol digest and isolated output root.
    for arm, task_id in V1_PREFLIGHTS:
        old = _read_object(v1_root / "runs" / arm / f"{task_id}.preflight.json")
        new = _read_object(v2_root / "runs" / arm / f"{task_id}.preflight.json")
        _, old_options = _option_map(
            old["command"], module="research.evidence_shortlist_smc", name="v1 preflight"
        )
        _, new_options = _option_map(
            new["command"], module="research.evidence_shortlist_smc", name="v2 preflight"
        )
        for options in (old_options, new_options):
            options.pop("--output")
            options.pop("--expected-study-protocol-sha256")
        _expect(old_options, new_options, name=f"v1/v2 semantic command {task_id}/{arm}")

    inventory = _inventory(v1_root)
    _expect(inventory["file_count"], 37, name="aborted v1 file count")
    return {
        "status": "invalid-aborted-pre-provider-not-analyzed",
        "reason": (
            "twelve evidence-only children failed inside frozen invocation validation on "
            "the obsolete protocol shape; six grammar-only children were opened but never "
            "completed; no run directory, provider artifact, result, or driver result exists"
        ),
        "protocol_sha256": ABORTED_V1_PROTOCOL_SHA256,
        "launch_seal_sha256": sha256_file(v1_root / "launch-seal.json"),
        "completed_failed_children": 12,
        "opened_never_completed_children": 6,
        "provider_calls": 0,
        "inventory": inventory,
        "driver_log": {
            "path": _repo_path(repo_root, driver_log, name="aborted v1 driver log"),
            **_file_binding(driver_log),
        },
        "started_unix": launch["started_unix"],
    }


def _validate_corrected_v2(
    *,
    repo_root: Path,
    protocol_path: Path,
    suite_dir: Path,
    v2_root: Path,
    analysis_path: Path,
    protocol: Mapping[str, Any],
    task_paths: Mapping[str, Path],
    runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    scheduler_path = repo_root / "research/run_developmental_smc_matrix.py"
    _expect(
        sha256_file(scheduler_path),
        CORRECTED_V2_SCHEDULER_SHA256,
        name="corrected v2 scheduler SHA-256",
    )
    launch = _validate_launch(
        root=v2_root,
        protocol_sha256=FROZEN_PROTOCOL_SHA256,
        scheduler_sha256=CORRECTED_V2_SCHEDULER_SHA256,
    )
    driver = _read_object(v2_root / "driver-result.json")
    for field, expected in {
        "schema": "developmental-smc-matrix-driver-result-v1",
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "successful_children": 36,
        "failed_children": 0,
    }.items():
        _expect(driver.get(field), expected, name=f"driver result {field}")
    completed = driver.get("completed_unix")
    if (
        not isinstance(completed, (int, float))
        or isinstance(completed, bool)
        or not math.isfinite(completed)
        or completed < cast(float, launch["started_unix"])
    ):
        raise ValueError("driver completion timestamp is invalid")
    children = driver.get("children")
    if not isinstance(children, list) or len(children) != 36:
        raise ValueError("driver result must contain exactly 36 children")
    expected_keys = [(arm, task_id) for arm in EXECUTION_ORDER for task_id in TASK_IDS]
    _expect(
        [
            (record.get("arm"), record.get("task_id"))
            for record in children
            if isinstance(record, dict)
        ],
        expected_keys,
        name="driver child order",
    )
    for raw, (arm, task_id) in zip(children, expected_keys, strict=True):
        record = _object(raw, name="driver child")
        _validate_driver_child(
            record=record,
            repo_root=repo_root,
            protocol_path=protocol_path,
            suite_dir=suite_dir,
            matrix_root=v2_root,
            task_id=task_id,
            arm=arm,
            protocol_sha256=FROZEN_PROTOCOL_SHA256,
            expected_returncode=0,
        )

    driver_log = _regular_file(
        Path(str(v2_root) + ".driver.log"), name="corrected v2 driver log"
    )
    logged_children = _read_json_lines(driver_log, name="corrected v2 driver log")
    if len(logged_children) != 36:
        raise ValueError("corrected v2 driver log must contain exactly 36 children")
    _expect(
        Counter(canonical_bytes(record) for record in logged_children),
        Counter(canonical_bytes(cast(dict[str, Any], record)) for record in children),
        name="driver log/result child records",
    )

    manifest_sha256 = sha256_file(suite_dir / "manifest.json")
    provider = _object(protocol.get("provider"), name="provider")
    runs_root = v2_root / "runs"
    total_provider_calls = 0
    result_bindings: list[dict[str, object]] = []
    for arm in ARMS:
        for task_id in TASK_IDS:
            run = runs[f"{task_id}--{arm}"]
            preflight_path = runs_root / arm / f"{task_id}.preflight.json"
            _validate_preflight(
                path=preflight_path,
                repo_root=repo_root,
                protocol_path=protocol_path,
                suite_dir=suite_dir,
                runs_root=runs_root,
                task_sha256=sha256_file(task_paths[task_id]),
                manifest_sha256=manifest_sha256,
                task_id=task_id,
                arm=arm,
                run=run,
                provider=provider,
                protocol_sha256=FROZEN_PROTOCOL_SHA256,
            )
            run_dir = runs_root / arm / task_id
            validated = _validate_result(
                run_dir=run_dir,
                task_id=task_id,
                arm=arm,
                run=run,
                protocol_sha256=FROZEN_PROTOCOL_SHA256,
                task_sha256=sha256_file(task_paths[task_id]),
            )
            result = _read_object(run_dir / "result.json")
            total_provider_calls += _validate_result_sidecars(
                run_dir=run_dir,
                result=result,
                run=run,
                protocol=protocol,
                protocol_sha256=FROZEN_PROTOCOL_SHA256,
                task_id=task_id,
                arm=arm,
            )
            result_bindings.append(
                {
                    "task_id": task_id,
                    "arm": arm,
                    "found_public_example_exact": validated["found_public_example_exact"],
                    "first_exact_slot": validated["first_exact_slot"],
                    "result_sha256": validated["result_sha256"],
                    "preflight_sha256": validated["preflight_sha256"],
                }
            )

    expected_analysis = analyze(
        repo_root=repo_root,
        protocol_path=protocol_path,
        expected_protocol_sha256=FROZEN_PROTOCOL_SHA256,
        suite_dir=suite_dir,
        runs_root=runs_root,
    )
    _regular_file(analysis_path, name="Stage-1 analysis")
    _expect(
        analysis_path.read_bytes(),
        canonical_bytes(expected_analysis) + b"\n",
        name="analysis bytes versus fail-closed recomputation",
    )
    analysis = _read_object(analysis_path)
    _expect(analysis.get("schema"), ANALYSIS_SCHEMA, name="analysis schema")
    _expect(
        analysis.get("integrity"),
        {"passed": True, "complete_task_arm_runs": 36},
        name="analysis integrity",
    )
    inventory = _inventory(v2_root)
    return {
        "status": "complete-analyzer-validated-developmental-matrix",
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "scheduler_sha256": CORRECTED_V2_SCHEDULER_SHA256,
        "launch_seal_sha256": sha256_file(v2_root / "launch-seal.json"),
        "driver_result_sha256": sha256_file(v2_root / "driver-result.json"),
        "analysis_sha256": sha256_file(analysis_path),
        "complete_task_arm_runs": 36,
        "provider_calls": total_provider_calls,
        "results": result_bindings,
        "inventory": inventory,
        "driver_log": {
            "path": _repo_path(repo_root, driver_log, name="corrected v2 driver log"),
            **_file_binding(driver_log),
        },
        "started_unix": launch["started_unix"],
        "completed_unix": completed,
        "analysis_summary": {
            "classification": analysis["classification"],
            "endpoint": analysis["endpoint"],
            "arm_summaries": analysis["arm_summaries"],
            "inference_note": analysis["inference_note"],
        },
    }


def seal(
    *,
    repo_root: Path,
    protocol_path: Path,
    suite_dir: Path,
    aborted_v1_root: Path,
    corrected_v2_root: Path,
    analysis_path: Path,
) -> dict[str, object]:
    root = _regular_directory(repo_root, name="repository root")
    protocol_file = _regular_file(protocol_path, name="frozen protocol")
    suite = _regular_directory(suite_dir, name="public suite")
    v1 = _regular_directory(aborted_v1_root, name="aborted v1 root")
    v2 = _regular_directory(corrected_v2_root, name="corrected v2 root")
    analysis_file = _within(v2, analysis_path, name="analysis path")
    _expect(
        analysis_file.relative_to(v2).as_posix(),
        "analysis/developmental-analysis.json",
        name="analysis relative path",
    )
    for path, name in (
        (protocol_file, "protocol"),
        (suite, "suite"),
        (v1, "aborted v1"),
        (v2, "corrected v2"),
        (Path(__file__).resolve(), "sealer source"),
    ):
        _repo_path(root, path, name=name)
    protocol = validate_protocol(root, protocol_file, FROZEN_PROTOCOL_SHA256)
    task_paths = validate_public_suite(
        suite, _object(protocol.get("task_suite"), name="task_suite")
    )
    runs = cast(dict[str, Mapping[str, Any]], _index_runs(protocol))
    invalid = _validate_aborted_v1(
        repo_root=root,
        protocol_path=protocol_file,
        suite_dir=suite,
        v1_root=v1,
        v2_root=v2,
        protocol=protocol,
        task_paths=task_paths,
        runs=runs,
    )
    valid = _validate_corrected_v2(
        repo_root=root,
        protocol_path=protocol_file,
        suite_dir=suite,
        v2_root=v2,
        analysis_path=analysis_file,
        protocol=protocol,
        task_paths=task_paths,
        runs=runs,
    )
    return {
        "schema": SCHEMA,
        "classification": {
            "developmental": True,
            "blind": False,
            "confirmatory": False,
            "public_examples_only": True,
        },
        "sealer": {
            "path": _repo_path(root, Path(__file__), name="sealer source"),
            "sha256": sha256_file(Path(__file__)),
        },
        "bindings": {
            "protocol_path": _repo_path(root, protocol_file, name="protocol"),
            "protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "suite_path": _repo_path(root, suite, name="suite"),
            "manifest_sha256": sha256_file(suite / "manifest.json"),
            "aborted_v1_path": _repo_path(root, v1, name="aborted v1"),
            "corrected_v2_path": _repo_path(root, v2, name="corrected v2"),
            "analysis_path": _repo_path(root, analysis_file, name="analysis"),
        },
        "aborted_v1": invalid,
        "corrected_v2": valid,
        "claim_scope": {
            "allowed": (
                "descriptive comparison of frozen mechanisms on twelve reused public tasks "
                "at the endpoint zero public-example loss by logical slot 29"
            ),
            "forbidden": [
                "blind or confirmatory efficacy",
                "fresh hidden-test semantic success",
                "posterior calibration",
                "universal or wall-clock search speedup",
            ],
        },
    }


def _write_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value) + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--aborted-v1-root", type=Path, required=True)
    parser.add_argument("--corrected-v2-root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite public bundle seal: {output}")
    for artifact_root in (args.aborted_v1_root.resolve(), args.corrected_v2_root.resolve()):
        if output == artifact_root or artifact_root in output.parents:
            raise ValueError("seal output must be outside both inventoried matrix roots")
    result = seal(
        repo_root=args.repo_root,
        protocol_path=args.protocol,
        suite_dir=args.suite_dir,
        aborted_v1_root=args.aborted_v1_root,
        corrected_v2_root=args.corrected_v2_root,
        analysis_path=args.analysis,
    )
    _write_exclusive(output, result)
    summary = cast(dict[str, Any], result["corrected_v2"])["analysis_summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"seal_sha256={sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

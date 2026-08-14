"""Seal the complete reveal-free blind-v2 run and analysis artifact bundle.

This post-run helper accepts only the public ``runs`` and reveal-free
``analysis`` directories.  It never accepts or reads a task secret, private
reveal, target manifest, or repository root.  After strict completeness and
binding checks, it writes a sorted exact-byte ``SHA256SUMS`` inventory and one
``BUNDLE_SHA256`` commitment into a new output directory.

Run this only after the public analyzer has successfully written analysis.json
and SUMMARY.md, and before unblinding::

    python -m research.seal_blind_evidence_frontier_v2_public \
      --runs-root artifacts/.../runs \
      --analysis-root artifacts/.../analysis \
      --output artifacts/.../public-bundle-seal

Use ``--preflight-only`` to validate and print the would-be commitment without
writing anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from research.analyze_blind_evidence_frontier_v2 import (
    render_markdown as render_analysis_markdown,
)

ANALYSIS_SCHEMA = "blind-evidence-frontier-confirmation-analysis-v2"
RESULT_SCHEMA = "iterative-typed-llm-beam-experiment-v2"
PREFLIGHT_SCHEMA = "blind-v2-preflight-invocation-v1"
MODEL_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
TASK_IDS = tuple(f"blind-v2-{index:02d}" for index in range(1, 13))
TRIALS = 10_000
CHECKPOINTS = ("21", "29")
SEAL_FILENAMES = frozenset({"SHA256SUMS", "BUNDLE_SHA256"})
PROVIDER_FILENAMES = frozenset(
    {"beam-validation.json", "request.json", "response.json", "result.json"}
)
FORBIDDEN_PRIVATE_KEYS = frozenset(
    {
        "commitment_nonce",
        "commitment_preimage",
        "mapper_family",
        "nonce",
        "predicate_shape",
        "private_seed",
        "seed_hex",
        "structural_rejections_before_acceptance",
        "target",
        "target_mapper",
        "target_mapper_dsl",
        "target_predicate",
        "target_predicate_dsl",
    }
)
FORBIDDEN_PATH_PARTS = frozenset(
    {
        "private",
        "reveal",
        "reveal.json",
        "secret",
        "secret.bin",
        "seed.bin",
    }
)


def canonical_bytes(value: object) -> bytes:
    """Return the stable JSON encoding used for provider inventory hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact {path}: {error}") from error


def _read_object(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def _array(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return cast(list[Any], value)


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _expect(actual: object, expected: object, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _close(actual: object, expected: float, *, name: str) -> None:
    value = _number(actual, name=name)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {value!r}")


def _reject_private_keys(value: object, *, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_PRIVATE_KEYS:
                raise ValueError(f"private reveal key {key!r} appears at {path}")
            _reject_private_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_private_keys(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            embedded = json.loads(value)
        except json.JSONDecodeError:
            return
        _reject_private_keys(embedded, path=f"{path}<embedded-json>")


def _regular_files(root: Path, *, label: str) -> tuple[Path, ...]:
    if not root.is_dir():
        raise ValueError(f"{label} directory does not exist: {root}")
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        lowered = {part.lower() for part in relative.parts}
        forbidden = lowered.intersection(FORBIDDEN_PATH_PARTS)
        if forbidden:
            raise ValueError(f"private/reveal path appears under {label}: {relative}")
        if path.is_symlink():
            raise ValueError(f"symbolic links are forbidden in {label}: {relative}")
        if path.is_file():
            if path.name in SEAL_FILENAMES:
                raise ValueError(f"pre-existing bundle seal file under {label}: {relative}")
            files.append(path)
        elif not path.is_dir():
            raise ValueError(f"non-regular artifact under {label}: {relative}")
    return tuple(sorted(files, key=lambda path: path.relative_to(root).as_posix()))


def _validate_json_public(files: Sequence[Path], root: Path, *, label: str) -> None:
    for path in files:
        if path.suffix.lower() != ".json":
            continue
        value = _read_json(path)
        _reject_private_keys(value, path=f"{label}/{path.relative_to(root).as_posix()}")


def _validate_provider_inventory(
    run_dir: Path,
    result: Mapping[str, Any],
    *,
    task_id: str,
) -> str:
    seal = _read_object(run_dir / "provider-seal.json")
    records = _array(seal.get("records"), name=f"{task_id}.provider-seal.records")
    normalized: list[dict[str, object]] = []
    listed: set[str] = set()
    previous = ""
    for index, raw in enumerate(records):
        record = _object(raw, name=f"{task_id}.provider-seal.records[{index}]")
        relative_value = record.get("path")
        if not isinstance(relative_value, str):
            raise ValueError(f"{task_id} provider inventory path must be a string")
        relative = Path(relative_value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "provider"
        ):
            raise ValueError(f"{task_id} unsafe provider inventory path: {relative_value}")
        if (
            len(relative.parts) != 4
            or not relative.parts[1].startswith("round-")
            or not relative.parts[1][6:].isdigit()
            or not relative.parts[2].startswith("beam-")
            or relative.parts[3] not in PROVIDER_FILENAMES
        ):
            raise ValueError(
                f"{task_id} unexpected provider artifact shape: {relative_value}"
            )
        if relative_value <= previous:
            raise ValueError(f"{task_id} provider inventory is not strictly sorted")
        previous = relative_value
        artifact = run_dir / relative
        expected_bytes = _integer(record.get("bytes"), name=f"{task_id}.{relative_value}.bytes")
        expected_hash = _digest(
            record.get("sha256"), name=f"{task_id}.{relative_value}.sha256"
        )
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError(f"{task_id} provider artifact is missing/nonregular: {relative_value}")
        _expect(artifact.stat().st_size, expected_bytes, name=f"{task_id}.{relative_value}.bytes")
        _expect(sha256_file(artifact), expected_hash, name=f"{task_id}.{relative_value}.sha256")
        listed.add(relative.as_posix())
        normalized.append(
            {"path": relative_value, "bytes": expected_bytes, "sha256": expected_hash}
        )
    provider_root = run_dir / "provider"
    if not provider_root.is_dir() or provider_root.is_symlink():
        raise ValueError(f"{task_id} provider directory is missing/nonregular")
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in provider_root.rglob("*")
        if path.is_file()
    }
    _expect(listed, actual, name=f"{task_id} provider inventory completeness")
    inventory_sha256 = sha256_bytes(canonical_bytes(normalized))
    _expect(
        seal.get("inventory_sha256"), inventory_sha256, name=f"{task_id} provider seal digest"
    )
    _expect(
        result.get("provider_inventory_sha256"),
        inventory_sha256,
        name=f"{task_id} result provider digest",
    )
    return inventory_sha256


def _command_options(command: object, *, task_id: str) -> dict[str, str | bool]:
    values = _array(command, name=f"{task_id}.preflight.command")
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"{task_id}.preflight.command must contain only strings")
    argv = cast(list[str], values)
    if len(argv) < 4 or not argv[0] or argv[1:3] != [
        "-m",
        "research.iterative_beam_experiment",
    ]:
        raise ValueError(f"{task_id}.preflight.command has the wrong executable/module")
    options: dict[str, str | bool] = {}
    index = 3
    while index < len(argv):
        flag = argv[index]
        if not flag.startswith("--") or flag in options:
            raise ValueError(f"{task_id}.preflight.command has an invalid/repeated flag")
        if flag == "--singleton-evidence":
            options[flag] = True
            index += 1
            continue
        if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            raise ValueError(f"{task_id}.preflight.command lacks a value for {flag}")
        options[flag] = argv[index + 1]
        index += 2
    return options


def _validate_task_run(
    runs_root: Path,
    task_id: str,
    analysis_task: Mapping[str, Any],
    analysis_bindings: Mapping[str, Any],
) -> dict[str, object]:
    run_dir = (runs_root / task_id).resolve()
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ValueError(f"missing/nonregular run directory for {task_id}")
    _expect(
        Path(cast(str, analysis_task.get("run_directory"))).resolve(),
        run_dir,
        name=f"{task_id}.analysis run_directory",
    )
    if (run_dir / "failure.json").exists():
        raise ValueError(f"{task_id} has failure.json; public confirmation is incomplete")
    required = {
        "protocol.json",
        "result.json",
        "provider-seal.json",
        "matched-random-trials.json",
    }
    for filename in required:
        path = run_dir / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{task_id} is missing required artifact {filename}")
    result = _read_object(run_dir / "result.json")
    protocol = _read_object(run_dir / "protocol.json")
    _expect(result.get("schema"), RESULT_SCHEMA, name=f"{task_id}.result schema")
    _expect(result.get("protocol"), protocol, name=f"{task_id}.result protocol binding")
    task_sha256 = _digest(
        analysis_task.get("public_task_sha256"), name=f"{task_id}.public task SHA-256"
    )
    binding_fields = {
        "study_protocol_sha256": "study_protocol_sha256",
        "method_seal_sha256": "method_seal_sha256",
        "provider_call_seal_sha256": "provider_call_seal_sha256",
        "public_manifest_sha256": "public_manifest_sha256",
    }
    for binding_field in binding_fields.values():
        _digest(
            analysis_bindings.get(binding_field),
            name=f"analysis.bindings.{binding_field}",
        )
    _expect(
        protocol.get("study_protocol_sha256"),
        analysis_bindings["study_protocol_sha256"],
        name=f"{task_id}.protocol study binding",
    )
    _expect(
        protocol.get("blind_manifest_sha256"),
        analysis_bindings["public_manifest_sha256"],
        name=f"{task_id}.protocol manifest binding",
    )
    _expect(protocol.get("task_sha256"), task_sha256, name=f"{task_id}.protocol task hash")
    task_index = TASK_IDS.index(task_id)
    provider_seed = 301_000 + task_index * 1_000
    expected_protocol = {
        "protocol_mode": None,
        "blind_task_id": None,
        "model": "gpt-oss-120b",
        "model_revision": MODEL_REVISION,
        "reasoning_effort": "low",
        "temperature": 0.0,
        "rounds": 4,
        "beam_width": 2,
        "branching_factor": 4,
        "maximum_proposal_slot_budget": 29,
        "selection_policy": "evidence-frontier",
        "singleton_evidence": True,
        "stall_policy": "alternate-hole",
        "start_seed": 17,
        "random_baseline_trials": TRIALS,
        "max_provider_concurrency": 2,
        "maximum_provider_calls": 7,
        "primary_checkpoint_round": 4,
        "primary_checkpoint_slot": 29,
        "provider_seed": provider_seed,
        "tie_seed": provider_seed + 100,
        "random_baseline_seed": provider_seed + 200,
    }
    for field, expected in expected_protocol.items():
        _expect(protocol.get(field), expected, name=f"{task_id}.protocol.{field}")
    _expect(
        protocol.get("proposal_slot_checkpoints"),
        [1, 5, 13, 21, 29],
        name=f"{task_id}.protocol checkpoints",
    )
    metrics = _object(result.get("llm_beam_metrics"), name=f"{task_id}.llm metrics")
    rounds_completed = _integer(
        metrics.get("rounds_completed"), name=f"{task_id}.rounds_completed"
    )
    if not 1 <= rounds_completed <= 4:
        raise ValueError(f"{task_id}.rounds_completed must be in [1,4]")
    expected_round_names = {f"round-{number:02d}.json" for number in range(1, rounds_completed + 1)}
    actual_round_names = {path.name for path in run_dir.glob("round-*.json") if path.is_file()}
    _expect(actual_round_names, expected_round_names, name=f"{task_id} round-file set")
    allowed_top_files = required | expected_round_names
    actual_top_files = {path.name for path in run_dir.iterdir() if path.is_file()}
    _expect(actual_top_files, allowed_top_files, name=f"{task_id} top-level file set")
    actual_top_dirs = {path.name for path in run_dir.iterdir() if path.is_dir()}
    _expect(actual_top_dirs, {"provider"}, name=f"{task_id} top-level directory set")

    trials = _array(
        _read_json(run_dir / "matched-random-trials.json"),
        name=f"{task_id}.matched-random-trials",
    )
    if len(trials) != TRIALS:
        raise ValueError(f"{task_id} must contain exactly {TRIALS} matched-random trials")
    preflight = runs_root / f"{task_id}.preflight.json"
    preflight_record = _read_object(preflight)
    _expect(preflight_record.get("schema"), PREFLIGHT_SCHEMA, name=f"{task_id} preflight schema")
    _expect(
        preflight_record.get("status"),
        "validated-before-provider-call",
        name=f"{task_id} preflight status",
    )
    _expect(preflight_record.get("task_id"), task_id, name=f"{task_id} preflight task")
    _expect(
        Path(cast(str, preflight_record.get("preflight_record_path"))).resolve(),
        preflight.resolve(),
        name=f"{task_id} preflight self path",
    )
    for preflight_field, binding_field in binding_fields.items():
        _expect(
            preflight_record.get(preflight_field),
            analysis_bindings[binding_field],
            name=f"{task_id} preflight {preflight_field}",
        )
    _expect(
        preflight_record.get("task_file_sha256"),
        task_sha256,
        name=f"{task_id} preflight task hash",
    )
    harness_sha256 = _digest(
        preflight_record.get("harness_sha256"), name=f"{task_id} preflight harness hash"
    )
    _expect(
        protocol.get("harness_sha256"),
        harness_sha256,
        name=f"{task_id} protocol/preflight harness hash",
    )
    options = _command_options(preflight_record.get("command"), task_id=task_id)
    expected_command_options: dict[str, str | bool] = {
        "--model": "gpt-oss-120b",
        "--reasoning-effort": "low",
        "--temperature": "0",
        "--max-tokens": "1600",
        "--timeout-seconds": "420",
        "--max-concurrency": "2",
        "--rounds": "4",
        "--beam-width": "2",
        "--branching-factor": "4",
        "--start-seed": "17",
        "--provider-seed": str(provider_seed),
        "--tie-seed": str(provider_seed + 100),
        "--random-baseline-seed": str(provider_seed + 200),
        "--random-baseline-trials": "10000",
        "--selection-policy": "evidence-frontier",
        "--stall-policy": "alternate-hole",
        "--singleton-evidence": True,
        "--primary-checkpoint-round": "4",
    }
    expected_flags = {
        *expected_command_options,
        "--task",
        "--output",
        "--blind-manifest",
        "--study-protocol",
        "--base-url",
    }
    _expect(set(options), expected_flags, name=f"{task_id}.preflight command flags")
    for flag, expected in expected_command_options.items():
        _expect(options.get(flag), expected, name=f"{task_id}.preflight {flag}")
    _expect(
        Path(cast(str, options["--output"])).resolve(),
        run_dir,
        name=f"{task_id}.preflight output path",
    )
    task_command_path = Path(cast(str, options["--task"]))
    if not task_command_path.is_absolute() or task_command_path.name != f"{task_id}.json":
        raise ValueError(f"{task_id}.preflight task path is invalid")
    for flag in ("--blind-manifest", "--study-protocol"):
        if not Path(cast(str, options[flag])).is_absolute():
            raise ValueError(f"{task_id}.preflight {flag} path must be absolute")
    base_url = options["--base-url"]
    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        raise ValueError(f"{task_id}.preflight base URL is invalid")

    _expect(
        sha256_file(run_dir / "protocol.json"),
        _digest(analysis_task.get("run_protocol_sha256"), name=f"{task_id}.run protocol hash"),
        name=f"{task_id}.analysis protocol hash",
    )
    _expect(
        sha256_file(run_dir / "result.json"),
        _digest(analysis_task.get("result_sha256"), name=f"{task_id}.result hash"),
        name=f"{task_id}.analysis result hash",
    )
    _expect(
        sha256_file(run_dir / "matched-random-trials.json"),
        _digest(
            analysis_task.get("matched_random_trials_sha256"),
            name=f"{task_id}.random-trials hash",
        ),
        name=f"{task_id}.analysis random-trials hash",
    )
    _expect(
        sha256_file(preflight),
        _digest(
            analysis_task.get("preflight_invocation_sha256"),
            name=f"{task_id}.preflight hash",
        ),
        name=f"{task_id}.analysis preflight hash",
    )
    round_hashes = _object(
        analysis_task.get("round_file_sha256"), name=f"{task_id}.round_file_sha256"
    )
    _expect(
        set(round_hashes),
        {str(number) for number in range(1, rounds_completed + 1)},
        name=f"{task_id}.analysis round hash keys",
    )
    for number in range(1, rounds_completed + 1):
        _expect(
            sha256_file(run_dir / f"round-{number:02d}.json"),
            _digest(round_hashes[str(number)], name=f"{task_id}.round-{number} hash"),
            name=f"{task_id}.analysis round-{number} hash",
        )
    provider_inventory = _validate_provider_inventory(run_dir, result, task_id=task_id)
    _expect(
        analysis_task.get("provider_inventory_sha256"),
        provider_inventory,
        name=f"{task_id}.analysis provider inventory",
    )

    first_exact_value = metrics.get("first_exact_proposal_slot")
    first_exact = (
        None
        if first_exact_value is None
        else _integer(first_exact_value, name=f"{task_id}.first exact slot")
    )
    if first_exact is not None and not 1 <= first_exact <= 29:
        raise ValueError(f"{task_id}.first exact slot must be in [1,29]")
    _expect(
        analysis_task.get("first_exact_proposal_slot"),
        first_exact,
        name=f"{task_id}.analysis first exact slot",
    )
    success_map = _object(
        metrics.get("success_by_proposal_slot_checkpoint"),
        name=f"{task_id}.llm success checkpoints",
    )
    loss_map = _object(
        metrics.get("best_loss_by_proposal_slot_checkpoint"),
        name=f"{task_id}.llm loss checkpoints",
    )
    analysis_success = _object(
        analysis_task.get("exact_by_slot"), name=f"{task_id}.analysis exact checkpoints"
    )
    analysis_losses = _object(
        analysis_task.get("best_loss_by_slot"), name=f"{task_id}.analysis loss checkpoints"
    )
    llm_indicators: dict[str, bool] = {}
    for checkpoint in CHECKPOINTS:
        expected_success = first_exact is not None and first_exact <= int(checkpoint)
        _expect(
            success_map.get(checkpoint),
            expected_success,
            name=f"{task_id}.llm success[{checkpoint}]",
        )
        _expect(
            analysis_success.get(checkpoint),
            expected_success,
            name=f"{task_id}.analysis success[{checkpoint}]",
        )
        loss = _number(loss_map.get(checkpoint), name=f"{task_id}.llm loss[{checkpoint}]")
        if loss < 0.0 or (loss == 0.0) != expected_success:
            raise ValueError(f"{task_id}.llm loss/success is inconsistent at {checkpoint}")
        _close(
            analysis_losses.get(checkpoint),
            loss,
            name=f"{task_id}.analysis loss[{checkpoint}]",
        )
        llm_indicators[checkpoint] = expected_success

    random_indicators = {checkpoint: [] for checkpoint in CHECKPOINTS}
    for index, raw_trial in enumerate(trials):
        trial = _object(raw_trial, name=f"{task_id}.random trial[{index}]")
        exact_value = trial.get("first_exact_execution")
        exact_slot = (
            None
            if exact_value is None
            else _integer(exact_value, name=f"{task_id}.random trial[{index}].first exact")
        )
        if exact_slot is not None and not 1 <= exact_slot <= 29:
            raise ValueError(f"{task_id}.random trial[{index}] exact slot is outside [1,29]")
        _expect(
            trial.get("success"),
            exact_slot is not None,
            name=f"{task_id}.random trial[{index}].success",
        )
        for checkpoint in CHECKPOINTS:
            random_indicators[checkpoint].append(
                exact_slot is not None and exact_slot <= int(checkpoint)
            )
    random_metrics = _object(
        result.get("matched_random_beam"), name=f"{task_id}.random metrics"
    )
    _expect(random_metrics.get("trials"), TRIALS, name=f"{task_id}.random trial count")
    random_successes = _object(
        random_metrics.get("successes_by_proposal_slot_checkpoint"),
        name=f"{task_id}.random checkpoint successes",
    )
    analysis_random = _object(
        analysis_task.get("matched_random"), name=f"{task_id}.analysis random checkpoints"
    )
    for checkpoint in CHECKPOINTS:
        successes = sum(random_indicators[checkpoint])
        _expect(
            random_successes.get(checkpoint),
            successes,
            name=f"{task_id}.random successes[{checkpoint}]",
        )
        analysis_checkpoint = _object(
            analysis_random.get(checkpoint),
            name=f"{task_id}.analysis random[{checkpoint}]",
        )
        _expect(
            analysis_checkpoint.get("successes"),
            successes,
            name=f"{task_id}.analysis random successes[{checkpoint}]",
        )
        _expect(
            analysis_checkpoint.get("trials"),
            TRIALS,
            name=f"{task_id}.analysis random trials[{checkpoint}]",
        )
        _close(
            analysis_checkpoint.get("success_rate"),
            successes / TRIALS,
            name=f"{task_id}.analysis random rate[{checkpoint}]",
        )
    return {
        "harness_sha256": harness_sha256,
        "llm": llm_indicators,
        "random": random_indicators,
    }


def _validate_endpoints(
    analysis: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, object]],
) -> None:
    endpoints = _object(analysis.get("endpoints"), name="analysis.endpoints")
    for checkpoint in CHECKPOINTS:
        endpoint = _object(
            endpoints.get(checkpoint), name=f"analysis.endpoints[{checkpoint}]"
        )
        _expect(
            endpoint.get("role"),
            "primary" if checkpoint == "29" else "secondary-descriptive",
            name=f"endpoint {checkpoint} role",
        )
        _expect(endpoint.get("task_count"), 12, name=f"endpoint {checkpoint} task count")
        llm_successes = sum(
            cast(bool, cast(Mapping[str, bool], outcome["llm"])[checkpoint])
            for outcome in outcomes
        )
        _expect(
            endpoint.get("llm_exact_successes"),
            llm_successes,
            name=f"endpoint {checkpoint} LLM successes",
        )
        _close(
            endpoint.get("llm_exact_rate"),
            llm_successes / 12,
            name=f"endpoint {checkpoint} LLM rate",
        )
        aggregate = [
            sum(
                cast(list[bool], cast(Mapping[str, object], outcome["random"])[checkpoint])[
                    trial_index
                ]
                for outcome in outcomes
            )
            for trial_index in range(TRIALS)
        ]
        stored_aggregate = _array(
            endpoint.get("matched_random_aggregate_success_count_by_trial_index"),
            name=f"endpoint {checkpoint} aggregate random trials",
        )
        if stored_aggregate != aggregate:
            raise ValueError(f"endpoint {checkpoint} aggregate random trials mismatch")
        random_mean_task_rate = sum(aggregate) / (12 * TRIALS)
        _close(
            endpoint.get("matched_random_mean_task_success_rate"),
            random_mean_task_rate,
            name=f"endpoint {checkpoint} random mean task rate",
        )
        _close(
            endpoint.get("mean_paired_success_advantage"),
            llm_successes / 12 - random_mean_task_rate,
            name=f"endpoint {checkpoint} paired advantage",
        )
        exceedances = sum(value >= llm_successes for value in aggregate)
        _expect(
            endpoint.get("randomization_exceedances"),
            exceedances,
            name=f"endpoint {checkpoint} randomization exceedances",
        )
        p_value = (1 + exceedances) / (TRIALS + 1)
        _close(
            endpoint.get("randomization_p_value"),
            p_value,
            name=f"endpoint {checkpoint} randomization p-value",
        )
        _expect(
            endpoint.get("randomization_p_value_formula"),
            "(1 + # aggregate random trial-j >= X_LLM) / 10001",
            name=f"endpoint {checkpoint} p-value formula",
        )

    primary = _object(endpoints.get("29"), name="analysis primary endpoint")
    primary_decision = _object(
        analysis.get("primary_decision"), name="analysis.primary_decision"
    )
    primary_successes = _integer(
        primary.get("llm_exact_successes"), name="primary LLM successes"
    )
    p_value = _number(primary.get("randomization_p_value"), name="primary p-value")
    practical_pass = primary_successes >= 6
    statistical_pass = p_value <= 0.05
    _expect(
        primary_decision.get("practical_threshold"),
        6,
        name="primary practical threshold",
    )
    _expect(
        primary_decision.get("practical_threshold_passed"),
        practical_pass,
        name="primary practical decision",
    )
    _close(primary_decision.get("alpha"), 0.05, name="primary alpha")
    _expect(
        primary_decision.get("randomization_test_passed"),
        statistical_pass,
        name="primary randomization decision",
    )
    _expect(
        primary_decision.get("study_success"),
        practical_pass and statistical_pass,
        name="primary study decision",
    )


def validate_public_artifacts(runs_root: Path, analysis_root: Path) -> None:
    """Reject any incomplete, unbound, or private/reveal-bearing artifact set."""

    runs = runs_root.resolve()
    analysis_dir = analysis_root.resolve()
    if runs == analysis_dir or runs in analysis_dir.parents or analysis_dir in runs.parents:
        raise ValueError("runs and analysis roots must be separate, non-nested directories")
    run_files = _regular_files(runs, label="runs")
    analysis_files = _regular_files(analysis_dir, label="analysis")
    expected_run_entries = {*TASK_IDS, *(f"{task_id}.preflight.json" for task_id in TASK_IDS)}
    _expect(
        {path.name for path in runs.iterdir()}, expected_run_entries, name="runs-root entries"
    )
    _expect(
        {path.relative_to(analysis_dir).as_posix() for path in analysis_files},
        {"SUMMARY.md", "analysis.json"},
        name="analysis file set",
    )
    analysis = _read_object(analysis_dir / "analysis.json")
    _expect(analysis.get("schema"), ANALYSIS_SCHEMA, name="analysis schema")
    _expect(
        analysis.get("label"),
        "CONFIRMATORY BLIND-V2 — INTEGRITY VALIDATED",
        name="analysis label",
    )
    _expect(analysis.get("blind"), True, name="analysis blind flag")
    _expect(analysis.get("confirmatory"), True, name="analysis confirmatory flag")
    _expect(analysis.get("private_reveal_used"), False, name="analysis private_reveal_used")
    integrity = _object(analysis.get("integrity"), name="analysis.integrity")
    _expect(integrity.get("passed"), True, name="analysis integrity")
    bindings = _object(analysis.get("bindings"), name="analysis.bindings")
    _expect(
        set(bindings),
        {
            "study_protocol_sha256",
            "method_seal_sha256",
            "provider_call_seal_sha256",
            "public_manifest_sha256",
            "hidden_target_manifest_sha256",
        },
        name="analysis binding fields",
    )
    for field in bindings:
        _digest(bindings[field], name=f"analysis.bindings.{field}")
    raw_tasks = _array(analysis.get("tasks"), name="analysis.tasks")
    if len(raw_tasks) != 12:
        raise ValueError("analysis must contain exactly 12 task records")
    task_records: dict[str, dict[str, Any]] = {}
    for index, task_id in enumerate(TASK_IDS):
        record = _object(raw_tasks[index], name=f"analysis.tasks[{index}]")
        _expect(record.get("task_id"), task_id, name=f"analysis.tasks[{index}].task_id")
        task_records[task_id] = record
    outcomes = []
    for task_id in TASK_IDS:
        outcomes.append(
            _validate_task_run(runs, task_id, task_records[task_id], bindings)
        )
    if len({cast(str, outcome["harness_sha256"]) for outcome in outcomes}) != 1:
        raise ValueError("the 12 public runs do not share one frozen harness hash")
    _validate_endpoints(analysis, outcomes)
    expected_summary = render_analysis_markdown(analysis)
    actual_summary = (analysis_dir / "SUMMARY.md").read_text(encoding="utf-8")
    _expect(actual_summary, expected_summary, name="analysis SUMMARY.md rendering")
    _validate_json_public(run_files, runs, label="runs")
    _validate_json_public(analysis_files, analysis_dir, label="analysis")


def _inventory_bytes(runs_root: Path, analysis_root: Path) -> bytes:
    records: list[tuple[str, Path]] = []
    records.extend(
        (f"runs/{path.relative_to(runs_root).as_posix()}", path)
        for path in _regular_files(runs_root, label="runs")
    )
    records.extend(
        (f"analysis/{path.relative_to(analysis_root).as_posix()}", path)
        for path in _regular_files(analysis_root, label="analysis")
    )
    lines = [f"{sha256_file(path)}  {relative}\n" for relative, path in sorted(records)]
    return "".join(lines).encode()


def seal_public_artifacts(
    runs_root: Path,
    analysis_root: Path,
    output: Path,
    *,
    preflight_only: bool = False,
) -> dict[str, object]:
    """Validate a stable public snapshot, then optionally seal it exclusively."""

    runs = runs_root.resolve()
    analysis_dir = analysis_root.resolve()
    destination = output.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite bundle-seal output: {destination}")
    if runs == destination or runs in destination.parents or destination in runs.parents:
        raise ValueError("bundle-seal output must be outside the runs tree")
    if (
        analysis_dir == destination
        or analysis_dir in destination.parents
        or destination in analysis_dir.parents
    ):
        raise ValueError("bundle-seal output must be outside the analysis tree")
    before = _inventory_bytes(runs, analysis_dir)
    validate_public_artifacts(runs, analysis_dir)
    after = _inventory_bytes(runs, analysis_dir)
    if before != after:
        raise ValueError("public artifacts changed while they were being validated")
    bundle_sha256 = sha256_bytes(after)
    result: dict[str, object] = {
        "schema": "blind-v2-reveal-free-public-artifact-bundle-v1",
        "status": "validated-not-written" if preflight_only else "sealed-before-unblinding",
        "private_reveal_used": False,
        "task_count": 12,
        "file_count": after.count(b"\n"),
        "sha256sums_sha256": bundle_sha256,
        "bundle_sha256": bundle_sha256,
        "output": str(destination),
    }
    if preflight_only:
        return result
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "SHA256SUMS").open("xb") as stream:
        stream.write(after)
    with (destination / "BUNDLE_SHA256").open("x", encoding="utf-8") as stream:
        stream.write(bundle_sha256 + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    result = seal_public_artifacts(
        args.runs_root,
        args.analysis_root,
        args.output,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fail-closed launcher for the frozen blind-v2 confirmation.

The launcher has no task-generation capability.  It verifies the independently
supplied method- and provider-call-seal digests, all frozen source and public
task bytes, and the exact per-task settings before it can invoke the generic
beam harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

METHOD_SCHEMA = "blind-evidence-frontier-confirmation-v2"
METHOD_STATUS = "method-frozen-before-task-generation"
METHOD_SEAL_SCHEMA = "blind-evidence-frontier-method-seal-v2"
PROVIDER_SEAL_SCHEMA = "blind-v2-provider-call-seal-v1"
MANIFEST_SCHEMA = "blinded-filter-map-suite-v2"
TASK_IDS = tuple(f"blind-v2-{index:02d}" for index in range(1, 13))
MODEL_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
BUNDLE_SCHEME = "SHA256(domain || NUL || repeated(path || NUL || exact_file_bytes || NUL))"
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "commitment_nonce",
        "mapper_family",
        "nonce",
        "private_seed",
        "rejection_log",
        "seed_hex",
        "target",
        "target_mapper",
        "target_mapper_dsl",
        "target_predicate",
        "target_predicate_dsl",
    }
)


def canonical_bytes(value: object) -> bytes:
    """Return the stable JSON encoding used by task commitments."""

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
    return sha256_bytes(path.read_bytes())


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}: {error}") from error
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


def _safe_repo_path(repo_root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"{name} must be a safe repository-relative path")
    root = repo_root.resolve()
    resolved = (root / relative).resolve()
    if root not in resolved.parents:
        raise ValueError(f"{name} escaped the repository")
    return resolved


def _reject_private_keys(value: object, *, path: str = "public") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"private key {key!r} appears at {path}")
            _reject_private_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_private_keys(child, path=f"{path}[{index}]")


def _reject_placeholders(value: object, *, path: str = "protocol") -> None:
    if isinstance(value, str) and ("TO_BE_BOUND" in value or value.startswith("SEALED_")):
        raise ValueError(f"unresolved placeholder at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_placeholders(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_placeholders(child, path=f"{path}[{index}]")


def _validate_source(
    repo_root: Path,
    value: object,
    *,
    name: str,
    expected_path: str | None = None,
) -> Path:
    record = _object(value, name=name)
    if expected_path is not None:
        _expect(record.get("path"), expected_path, name=f"{name}.path")
    path = _safe_repo_path(repo_root, record.get("path"), name=f"{name}.path")
    expected = _digest(record.get("sha256"), name=f"{name}.sha256")
    _expect(sha256_file(path), expected, name=f"{name}.sha256")
    return path


def _validate_bundle(repo_root: Path, value: object, *, name: str) -> str:
    record = _object(value, name=name)
    _expect(record.get("scheme"), BUNDLE_SCHEME, name=f"{name}.scheme")
    domain = record.get("domain")
    if not isinstance(domain, str) or not domain:
        raise ValueError(f"{name}.domain must be a nonempty string")
    paths = _array(record.get("paths_in_order"), name=f"{name}.paths_in_order")
    if not paths or len(paths) != len(set(map(str, paths))):
        raise ValueError(f"{name}.paths_in_order must be nonempty and unique")
    source_hashes = _object(record.get("source_sha256"), name=f"{name}.source_sha256")
    if set(source_hashes) != set(paths):
        raise ValueError(f"{name}.source_sha256 keys must exactly match paths_in_order")
    digest = hashlib.sha256()
    digest.update(domain.encode())
    digest.update(b"\0")
    for index, relative in enumerate(paths):
        path = _safe_repo_path(repo_root, relative, name=f"{name}.paths_in_order[{index}]")
        payload = path.read_bytes()
        expected = _digest(source_hashes.get(relative), name=f"{name}.source_sha256[{relative!r}]")
        _expect(sha256_bytes(payload), expected, name=f"{name}.source_sha256[{relative!r}]")
        digest.update(cast(str, relative).encode())
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    actual = digest.hexdigest()
    _expect(
        actual,
        _digest(record.get("bundle_sha256"), name=f"{name}.bundle_sha256"),
        name=f"{name}.bundle_sha256",
    )
    return actual


def _validate_frozen_method(
    repo_root: Path,
    protocol: Mapping[str, Any],
) -> tuple[Path, tuple[dict[str, int | str], ...]]:
    _expect(protocol.get("schema"), METHOD_SCHEMA, name="study protocol schema")
    _expect(protocol.get("protocol_status"), METHOD_STATUS, name="study protocol status")
    _reject_placeholders(protocol)
    freeze = _object(protocol.get("freeze_requirements"), name="freeze_requirements")
    _expect(
        freeze.get("protocol_sha256_binding"),
        "external-provider-seal-and-run-protocol-records",
        name="freeze_requirements.protocol_sha256_binding",
    )
    _validate_source(
        repo_root,
        freeze.get("generator"),
        name="freeze_requirements.generator",
        expected_path="research/generate_blinded_filter_map_tasks.py",
    )
    harness = _validate_source(
        repo_root,
        freeze.get("harness"),
        name="freeze_requirements.harness",
        expected_path="research/iterative_beam_experiment.py",
    )
    _validate_source(repo_root, freeze.get("analysis"), name="freeze_requirements.analysis")
    _validate_source(
        repo_root,
        freeze.get("runner"),
        name="freeze_requirements.runner",
        expected_path="research/run_blind_evidence_frontier_v2.py",
    )
    prompt_digest = _validate_bundle(
        repo_root,
        freeze.get("prompt_template_binding"),
        name="freeze_requirements.prompt_template_binding",
    )
    _expect(
        freeze.get("prompt_template_sha256"),
        prompt_digest,
        name="freeze_requirements.prompt_template_sha256",
    )
    _validate_bundle(repo_root, freeze.get("test_bundle"), name="freeze_requirements.test_bundle")
    _digest(
        freeze.get("task_secret_commitment_sha256"),
        name="freeze_requirements.task_secret_commitment_sha256",
    )

    model = _object(protocol.get("model"), name="model")
    expected_model = {
        "served_name": "gpt-oss-120b",
        "revision": MODEL_REVISION,
        "reasoning_effort": "low",
        "temperature": 0,
        "max_tokens": 1600,
        "timeout_seconds": 420,
        "max_concurrency": 2,
        "provider_retry_policy": "none",
    }
    for field, expected in expected_model.items():
        _expect(model.get(field), expected, name=f"model.{field}")
    search = _object(protocol.get("search"), name="search")
    expected_search = {
        "selection_policy": "evidence-frontier",
        "branching_factor": 4,
        "beam_width": 2,
        "maximum_rounds": 4,
        "start_seed": 17,
        "singleton_evidence": True,
        "stall_policy": "alternate-hole",
        "maximum_proposal_slots": 29,
        "maximum_provider_calls": 7,
    }
    for field, expected in expected_search.items():
        _expect(search.get(field), expected, name=f"search.{field}")
    baseline = _object(protocol.get("matched_random_baseline"), name="matched_random_baseline")
    _expect(baseline.get("trials_per_task"), 10000, name="matched_random_baseline.trials_per_task")
    generator = _object(protocol.get("fresh_task_generator"), name="fresh_task_generator")
    _expect(generator.get("task_count"), 12, name="fresh_task_generator.task_count")
    schedule = _array(generator.get("fixed_balanced_schedule"), name="fixed_balanced_schedule")
    scheduled_task_ids = tuple(
        row[0] for row in schedule if isinstance(row, list) and row
    )
    _expect(scheduled_task_ids, TASK_IDS, name="task schedule")

    seeds_raw = _array(freeze.get("run_seeds"), name="freeze_requirements.run_seeds")
    if len(seeds_raw) != 12:
        raise ValueError("freeze_requirements.run_seeds must contain exactly 12 records")
    seeds: list[dict[str, int | str]] = []
    seen_values: set[int] = set()
    for index, task_id in enumerate(TASK_IDS):
        record = _object(seeds_raw[index], name=f"run_seeds[{index}]")
        _expect(record.get("task_id"), task_id, name=f"run_seeds[{index}].task_id")
        normalized: dict[str, int | str] = {"task_id": task_id}
        for field in ("provider_seed", "tie_seed", "matched_random_seed"):
            value = _integer(record.get(field), name=f"run_seeds[{index}].{field}")
            if value in seen_values:
                raise ValueError("all frozen top-level run seeds must be distinct")
            seen_values.add(value)
            normalized[field] = value
        seeds.append(normalized)
    return harness, tuple(seeds)


def _validate_method_seal(
    repo_root: Path,
    method_seal_path: Path,
    expected_seal_sha256: str,
    protocol_path: Path,
) -> dict[str, Any]:
    _expect(
        sha256_file(method_seal_path),
        _digest(expected_seal_sha256, name="expected method-seal SHA-256"),
        name="method-seal external SHA-256",
    )
    seal = _read_object(method_seal_path)
    _expect(seal.get("schema"), METHOD_SEAL_SCHEMA, name="method-seal schema")
    _expect(
        seal.get("sealed_before_task_generation"), True, name="sealed_before_task_generation"
    )
    binding = _object(seal.get("protocol"), name="method-seal.protocol")
    bound_path = _safe_repo_path(repo_root, binding.get("path"), name="method-seal.protocol.path")
    _expect(bound_path, protocol_path.resolve(), name="method-seal protocol path")
    _expect(
        sha256_file(protocol_path),
        _digest(binding.get("sha256"), name="method-seal.protocol.sha256"),
        name="method-seal protocol SHA-256",
    )
    return seal


def _validate_provider_seal(
    repo_root: Path,
    provider_seal_path: Path,
    expected_seal_sha256: str,
    *,
    protocol_sha256: str,
    method_seal_sha256: str,
    protocol: Mapping[str, Any],
    frozen_seeds: tuple[dict[str, int | str], ...],
) -> tuple[Path, dict[str, Path]]:
    _expect(
        sha256_file(provider_seal_path),
        _digest(expected_seal_sha256, name="expected provider-call-seal SHA-256"),
        name="provider-call-seal external SHA-256",
    )
    seal = _read_object(provider_seal_path)
    _expect(seal.get("schema"), PROVIDER_SEAL_SCHEMA, name="provider-call-seal schema")
    _expect(
        seal.get("sealed_before_provider_calls"), True, name="sealed_before_provider_calls"
    )
    _expect(
        seal.get("study_protocol_sha256"),
        protocol_sha256,
        name="provider-call-seal study_protocol_sha256",
    )
    _expect(
        seal.get("method_seal_sha256"),
        method_seal_sha256,
        name="provider-call-seal method_seal_sha256",
    )
    freeze = _object(protocol.get("freeze_requirements"), name="freeze_requirements")
    _expect(
        seal.get("task_secret_commitment_sha256"),
        freeze.get("task_secret_commitment_sha256"),
        name="task-secret commitment",
    )
    _digest(
        seal.get("hidden_target_manifest_sha256"),
        name="hidden_target_manifest_sha256",
    )

    manifest_binding = _object(seal.get("public_manifest"), name="public_manifest")
    manifest_path = _safe_repo_path(
        repo_root, manifest_binding.get("path"), name="public_manifest.path"
    )
    _expect(
        sha256_file(manifest_path),
        _digest(manifest_binding.get("sha256"), name="public_manifest.sha256"),
        name="public-manifest file SHA-256",
    )
    manifest = _read_object(manifest_path)
    _reject_private_keys(manifest)
    _expect(manifest.get("schema"), MANIFEST_SCHEMA, name="public-manifest schema")
    _expect(manifest.get("task_count"), 12, name="public-manifest task_count")
    _expect(
        manifest.get("seed_commitment_sha256"),
        seal.get("task_secret_commitment_sha256"),
        name="public-manifest seed commitment",
    )
    generation = _object(manifest.get("task_generation"), name="manifest.task_generation")
    generator_binding = _object(freeze.get("generator"), name="freeze_requirements.generator")
    _expect(
        generation.get("generator_sha256"),
        generator_binding.get("sha256"),
        name="manifest generator SHA-256",
    )
    public_tasks = _array(manifest.get("tasks"), name="manifest.tasks")
    sealed_tasks = _array(seal.get("tasks"), name="provider-call-seal.tasks")
    if len(public_tasks) != 12 or len(sealed_tasks) != 12:
        raise ValueError("manifest and provider-call seal must each contain exactly 12 tasks")
    task_paths: dict[str, Path] = {}
    for index, task_id in enumerate(TASK_IDS):
        public = _object(public_tasks[index], name=f"manifest.tasks[{index}]")
        sealed = _object(sealed_tasks[index], name=f"provider-call-seal.tasks[{index}]")
        frozen_seed = frozen_seeds[index]
        _expect(public.get("task_id"), task_id, name=f"manifest.tasks[{index}].task_id")
        _expect(sealed.get("task_id"), task_id, name=f"sealed.tasks[{index}].task_id")
        for field in (
            "task_file_sha256",
            "task_canonical_sha256",
            "target_commitment_sha256",
        ):
            expected = _digest(public.get(field), name=f"manifest.tasks[{index}].{field}")
            _expect(sealed.get(field), expected, name=f"sealed.tasks[{index}].{field}")
        for field in ("provider_seed", "tie_seed", "matched_random_seed"):
            _expect(sealed.get(field), frozen_seed[field], name=f"sealed.tasks[{index}].{field}")
        relative_name = public.get("path")
        _expect(relative_name, f"{task_id}.json", name=f"manifest.tasks[{index}].path")
        task_path = _safe_repo_path(
            repo_root, sealed.get("path"), name=f"sealed.tasks[{index}].path"
        )
        _expect(task_path.name, relative_name, name=f"sealed.tasks[{index}].path basename")
        task = _read_object(task_path)
        _reject_private_keys(task, path=task_id)
        _expect(
            sha256_file(task_path),
            public.get("task_file_sha256"),
            name=f"{task_id} file SHA-256",
        )
        _expect(
            sha256_bytes(canonical_bytes(task)),
            public.get("task_canonical_sha256"),
            name=f"{task_id} canonical SHA-256",
        )
        task_paths[task_id] = task_path

    endpoint_record = _object(
        seal.get("endpoint_health_record"), name="endpoint_health_record"
    )
    endpoint_path = _safe_repo_path(
        repo_root, endpoint_record.get("path"), name="endpoint_health_record.path"
    )
    _expect(
        sha256_file(endpoint_path),
        _digest(endpoint_record.get("sha256"), name="endpoint_health_record.sha256"),
        name="endpoint-health-record SHA-256",
    )
    return manifest_path, task_paths


def validate_and_build_command(
    *,
    repo_root: Path,
    study_protocol_path: Path,
    method_seal_path: Path,
    provider_call_seal_path: Path,
    expected_method_seal_sha256: str,
    expected_provider_call_seal_sha256: str,
    task_id: str,
    output: Path,
    base_url: str,
    python_executable: str,
) -> tuple[list[str], dict[str, object]]:
    """Validate every frozen binding and return the only permitted harness command."""

    root = repo_root.resolve()
    protocol_path = study_protocol_path.resolve()
    protocol = _read_object(protocol_path)
    _validate_method_seal(
        root,
        method_seal_path.resolve(),
        expected_method_seal_sha256,
        protocol_path,
    )
    harness_path, frozen_seeds = _validate_frozen_method(root, protocol)
    protocol_sha256 = sha256_file(protocol_path)
    method_seal_sha256 = sha256_file(method_seal_path.resolve())
    manifest_path, task_paths = _validate_provider_seal(
        root,
        provider_call_seal_path.resolve(),
        expected_provider_call_seal_sha256,
        protocol_sha256=protocol_sha256,
        method_seal_sha256=method_seal_sha256,
        protocol=protocol,
        frozen_seeds=frozen_seeds,
    )
    if task_id not in TASK_IDS:
        raise ValueError(f"task_id must be one of {TASK_IDS}")
    output_path = output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output path: {output_path}")
    preflight_record_path = output_path.with_name(output_path.name + ".preflight.json")
    if preflight_record_path.exists():
        raise FileExistsError(f"refusing to overwrite preflight record: {preflight_record_path}")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base_url must use http:// or https://")
    index = TASK_IDS.index(task_id)
    seeds = frozen_seeds[index]
    command = [
        python_executable,
        "-m",
        "research.iterative_beam_experiment",
        "--task",
        str(task_paths[task_id]),
        "--output",
        str(output_path),
        "--blind-manifest",
        str(manifest_path),
        "--study-protocol",
        str(protocol_path),
        "--base-url",
        base_url,
        "--model",
        "gpt-oss-120b",
        "--reasoning-effort",
        "low",
        "--temperature",
        "0",
        "--max-tokens",
        "1600",
        "--timeout-seconds",
        "420",
        "--max-concurrency",
        "2",
        "--rounds",
        "4",
        "--beam-width",
        "2",
        "--branching-factor",
        "4",
        "--start-seed",
        "17",
        "--provider-seed",
        str(seeds["provider_seed"]),
        "--tie-seed",
        str(seeds["tie_seed"]),
        "--random-baseline-seed",
        str(seeds["matched_random_seed"]),
        "--random-baseline-trials",
        "10000",
        "--selection-policy",
        "evidence-frontier",
        "--stall-policy",
        "alternate-hole",
        "--singleton-evidence",
        "--primary-checkpoint-round",
        "4",
    ]
    if "--protocol-mode" in command:
        raise AssertionError("blind-v2 must use the audited generic harness path")
    record: dict[str, object] = {
        "schema": "blind-v2-preflight-invocation-v1",
        "status": "validated-before-provider-call",
        "task_id": task_id,
        "study_protocol_sha256": protocol_sha256,
        "method_seal_sha256": method_seal_sha256,
        "provider_call_seal_sha256": sha256_file(provider_call_seal_path.resolve()),
        "public_manifest_sha256": sha256_file(manifest_path),
        "task_file_sha256": sha256_file(task_paths[task_id]),
        "harness_sha256": sha256_file(harness_path),
        "command": command,
        "preflight_record_path": str(preflight_record_path),
    }
    return command, record


def _write_exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value) + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--study-protocol", type=Path, required=True)
    parser.add_argument("--method-seal", type=Path, required=True)
    parser.add_argument("--provider-call-seal", type=Path, required=True)
    parser.add_argument("--expected-method-seal-sha256", required=True)
    parser.add_argument("--expected-provider-call-seal-sha256", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    command, record = validate_and_build_command(
        repo_root=args.repo_root,
        study_protocol_path=args.study_protocol,
        method_seal_path=args.method_seal,
        provider_call_seal_path=args.provider_call_seal,
        expected_method_seal_sha256=args.expected_method_seal_sha256,
        expected_provider_call_seal_sha256=args.expected_provider_call_seal_sha256,
        task_id=args.task_id,
        output=args.output,
        base_url=args.base_url,
        python_executable=args.python_executable,
    )
    if args.preflight_only:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    preflight_path = Path(cast(str, record["preflight_record_path"]))
    _write_exclusive_json(preflight_path, record)
    completed = subprocess.run(command, cwd=args.repo_root.resolve(), check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

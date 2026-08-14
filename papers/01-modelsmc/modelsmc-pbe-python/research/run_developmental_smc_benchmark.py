"""Fail-closed launcher for the 12-task developmental SMC benchmark.

This launcher never generates tasks and has a side-effect-free preflight mode.
The public suite was previously used and unblinded, so this benchmark is a
developmental regression study rather than a blind or confirmatory experiment.
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

SCHEMA = "developmental-evidence-shortlist-smc-benchmark-v1"
STATUS = "frozen-before-provider-calls"
MANIFEST_SCHEMA = "blinded-filter-map-suite-v2"
TASK_IDS = tuple(f"blind-v2-{index:02d}" for index in range(1, 13))
ARMS = ("llm-smc", "evidence-only", "grammar-only")
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "commitment_preimage",
        "nonce",
        "private_seed",
        "seed_hex",
        "target",
        "target_mapper_dsl",
        "target_predicate_dsl",
    }
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be a safe repository-relative path")
    root = repo_root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"{name} escaped the repository")
    return path


def _reject_private_keys(value: object, *, path: str = "public") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"private key {key!r} appears at {path}")
            _reject_private_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_private_keys(child, path=f"{path}[{index}]")


def _validate_python_tree(repo_root: Path, raw: object) -> None:
    binding = _object(raw, name="source_bindings.modelsmc_python_tree")
    _expect(binding.get("schema"), "sha256-python-tree-v1", name="tree schema")
    _expect(binding.get("include"), "**/*.py", name="tree include")
    tree_root = _safe_repo_path(repo_root, binding.get("root"), name="tree root")
    if not tree_root.is_dir():
        raise ValueError("bound Python tree root is not a directory")
    for path in tree_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Python tree contains a symlink: {path}")
    paths = sorted(
        (path for path in tree_root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(tree_root).as_posix().encode(),
    )
    entries = [
        {
            "path": path.relative_to(tree_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    manifest = {
        "schema": "sha256-python-tree-v1",
        "root": binding.get("root"),
        "include": "**/*.py",
        "entries": entries,
    }
    _expect(binding.get("file_count"), len(entries), name="tree file_count")
    _expect(
        binding.get("manifest_sha256"),
        hashlib.sha256(canonical_bytes(manifest)).hexdigest(),
        name="tree manifest_sha256",
    )


def validate_protocol(repo_root: Path, protocol_path: Path, expected_sha256: str) -> dict[str, Any]:
    """Validate the external freeze and every repository source binding."""

    _expect(
        sha256_file(protocol_path),
        _digest(expected_sha256, name="expected protocol SHA-256"),
        name="external protocol SHA-256",
    )
    protocol = _read_object(protocol_path)
    _expect(protocol.get("schema"), SCHEMA, name="protocol schema")
    _expect(protocol.get("status"), STATUS, name="protocol status")
    classification = _object(protocol.get("classification"), name="classification")
    _expect(classification.get("developmental"), True, name="developmental flag")
    _expect(classification.get("blind"), False, name="blind flag")
    _expect(classification.get("confirmatory"), False, name="confirmatory flag")
    bindings = _object(protocol.get("source_bindings"), name="source_bindings")
    for label in (
        "harness",
        "runner",
        "analyzer",
        "focused_tests",
        "benchmark_tests",
    ):
        record = _object(bindings.get(label), name=f"source_bindings.{label}")
        path = _safe_repo_path(repo_root, record.get("path"), name=f"{label}.path")
        _expect(
            sha256_file(path),
            _digest(record.get("sha256"), name=f"{label}.sha256"),
            name=f"{label} SHA-256",
        )
    dependencies = _array(bindings.get("dependency_bundle"), name="dependency_bundle")
    if not dependencies:
        raise ValueError("dependency_bundle must not be empty")
    for index, raw in enumerate(dependencies):
        record = _object(raw, name=f"dependency_bundle[{index}]")
        path = _safe_repo_path(repo_root, record.get("path"), name=f"dependency[{index}].path")
        _expect(
            sha256_file(path),
            _digest(record.get("sha256"), name=f"dependency[{index}].sha256"),
            name=f"dependency[{index}] SHA-256",
        )
    _validate_python_tree(repo_root, bindings.get("modelsmc_python_tree"))
    design = _object(protocol.get("design"), name="design")
    for field, expected in {
        "rounds": 4,
        "parent_count": 2,
        "offspring_per_parent": 4,
        "first_round_offspring": 4,
        "logical_execution_cap": 29,
        "llm_provider_call_cap": 7,
        "early_stop": False,
    }.items():
        _expect(design.get(field), expected, name=f"design.{field}")
    return protocol


def validate_public_suite(
    suite_dir: Path,
    suite_binding: Mapping[str, Any],
) -> dict[str, Path]:
    """Validate the reused public manifest and all twelve public task bytes."""

    root = suite_dir.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("public suite must be a regular directory")
    manifest_path = root / "manifest.json"
    _expect(
        sha256_file(manifest_path),
        _digest(suite_binding.get("manifest_sha256"), name="manifest_sha256"),
        name="public manifest SHA-256",
    )
    manifest = _read_object(manifest_path)
    _reject_private_keys(manifest)
    _expect(manifest.get("schema"), MANIFEST_SCHEMA, name="manifest schema")
    _expect(manifest.get("task_count"), 12, name="manifest task_count")
    public_records = _array(manifest.get("tasks"), name="manifest.tasks")
    frozen_records = _array(suite_binding.get("tasks"), name="task_suite.tasks")
    if len(public_records) != 12 or len(frozen_records) != 12:
        raise ValueError("manifest and protocol must bind exactly twelve tasks")
    paths: dict[str, Path] = {}
    for index, task_id in enumerate(TASK_IDS):
        public = _object(public_records[index], name=f"manifest.tasks[{index}]")
        frozen = _object(frozen_records[index], name=f"task_suite.tasks[{index}]")
        _expect(public, frozen, name=f"frozen manifest record {task_id}")
        _expect(public.get("task_id"), task_id, name=f"{task_id}.task_id")
        _expect(public.get("path"), f"{task_id}.json", name=f"{task_id}.path")
        path = root / f"{task_id}.json"
        if path.resolve().parent != root:
            raise ValueError(f"unsafe task path for {task_id}")
        task = _read_object(path)
        _reject_private_keys(task, path=task_id)
        _expect(
            sha256_file(path),
            _digest(public.get("task_file_sha256"), name=f"{task_id}.task_file_sha256"),
            name=f"{task_id} file SHA-256",
        )
        _expect(
            hashlib.sha256(canonical_bytes(task)).hexdigest(),
            _digest(
                public.get("task_canonical_sha256"),
                name=f"{task_id}.task_canonical_sha256",
            ),
            name=f"{task_id} canonical SHA-256",
        )
        paths[task_id] = path
    return paths


def _index_runs(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_runs = _array(protocol.get("runs"), name="runs")
    if len(raw_runs) != len(TASK_IDS) * len(ARMS):
        raise ValueError("protocol must contain exactly 36 task-arm runs")
    runs: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_runs):
        run = _object(raw, name=f"runs[{index}]")
        run_id = run.get("id")
        if not isinstance(run_id, str) or run_id in runs:
            raise ValueError("run IDs must be unique strings")
        runs[run_id] = run
    expected = tuple(f"{task_id}--{arm}" for arm in ARMS for task_id in TASK_IDS)
    _expect(tuple(runs), expected, name="run order")
    return runs


def validate_and_build_command(
    *,
    repo_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    suite_dir: Path,
    runs_root: Path,
    task_id: str,
    arm: str,
    python_executable: str,
) -> tuple[list[str], dict[str, object]]:
    root = repo_root.resolve()
    protocol_file = protocol_path.resolve()
    protocol = validate_protocol(root, protocol_file, expected_protocol_sha256)
    task_paths = validate_public_suite(
        suite_dir,
        _object(protocol.get("task_suite"), name="task_suite"),
    )
    if task_id not in TASK_IDS:
        raise ValueError(f"task_id must be one of {TASK_IDS}")
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    run_id = f"{task_id}--{arm}"
    run = _index_runs(protocol)[run_id]
    _expect(run.get("task"), task_id, name="run task")
    _expect(run.get("arm"), arm, name="run arm")
    _expect(run.get("task_sha256"), sha256_file(task_paths[task_id]), name="run task SHA-256")
    output = runs_root.resolve() / arm / task_id
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    preflight = output.with_name(output.name + ".preflight.json")
    if preflight.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {preflight}")
    provider = _object(protocol.get("provider"), name="provider")
    command = [
        python_executable,
        "-m",
        "research.evidence_shortlist_smc",
        "--task",
        str(task_paths[task_id]),
        "--output",
        str(output),
        "--study-protocol",
        str(protocol_file),
        "--run-id",
        run_id,
        "--expected-study-protocol-sha256",
        expected_protocol_sha256,
        "--base-url",
        cast(str, provider["base_url"]),
        "--model",
        cast(str, provider["model"]),
        "--reasoning-effort",
        cast(str, provider["reasoning_effort"]),
        "--temperature",
        str(provider["temperature"]),
        "--max-tokens",
        str(provider["max_output_tokens"]),
        "--timeout-seconds",
        str(provider["timeout_seconds"]),
        "--epsilon",
        str(run["epsilon"]),
        "--evidence-scale",
        str(run["evidence_scale"]),
        "--start-seed",
        str(run["start_seed"]),
        "--provider-seed",
        str(run["provider_seed"]),
        "--sample-seed",
        str(run["sample_seed"]),
        "--resample-seed",
        str(run["resample_seed"]),
        "--parent-count",
        str(run["parent_count"]),
        "--offspring-per-parent",
        str(run["offspring_per_parent"]),
        "--first-round-offspring",
        str(run["first_round_offspring"]),
        "--max-concurrency",
        str(run["max_concurrency"]),
        "--exact-reference-limit",
        str(run["exact_reference_limit"]),
        "--proposal-source",
        cast(str, run["proposal_source"]),
    ]
    random_seed = run.get("random_shortlist_seed")
    if random_seed is not None:
        command.extend(("--random-shortlist-seed", str(random_seed)))
    record: dict[str, object] = {
        "schema": "developmental-smc-benchmark-preflight-v1",
        "status": "validated-before-provider-call",
        "developmental": True,
        "task_id": task_id,
        "arm": arm,
        "run_id": run_id,
        "protocol_sha256": sha256_file(protocol_file),
        "manifest_sha256": sha256_file(suite_dir.resolve() / "manifest.json"),
        "task_sha256": sha256_file(task_paths[task_id]),
        "command": command,
        "preflight_path": str(preflight),
    }
    return command, record


def _write_exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value) + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    command, record = validate_and_build_command(
        repo_root=args.repo_root,
        protocol_path=args.protocol,
        expected_protocol_sha256=args.expected_protocol_sha256,
        suite_dir=args.suite_dir,
        runs_root=args.runs_root,
        task_id=args.task_id,
        arm=args.arm,
        python_executable=args.python_executable,
    )
    if args.preflight_only:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    _write_exclusive_json(Path(cast(str, record["preflight_path"])), record)
    return subprocess.run(command, cwd=args.repo_root.resolve(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

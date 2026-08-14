"""Fail-closed runner for the paired ExeDec DeepCoder-HO debug benchmark.

The four adapted tasks come from public released ExeDec data.  This runner only
passes a bound public task to ``evidence_shortlist_smc``.  It deliberately does
not open or evaluate the provider-private debug oracles; those are consumed by
the separate post-run analyzer after every paired run is complete.

No run occurs without an external protocol SHA-256.  LLM execution additionally
requires an explicit debug-provider authorization flag.  ``--preflight-only``
is side-effect free and makes no network or provider calls.
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

from research.evidence_shortlist_smc import population_schedule, python_tree_binding

SCHEMA = "exedec-deepcoder-ho-paired-smc-debug-benchmark-v1"
STATUS = "frozen-before-provider-calls"
PREFLIGHT_SCHEMA = "exedec-deepcoder-ho-paired-smc-debug-preflight-v1"
BUNDLE_SCHEMA = "exedec-deepcoder-filter-map-ho-debug-bundle-v1"
TASK_IDS = (
    "exedec-debug-0131",
    "exedec-debug-0245",
    "exedec-debug-0258",
    "exedec-debug-0608",
)
ARMS = ("llm-smc", "grammar-random")
SEED_IDS = tuple(range(8))
ROUNDS = 4
PARENT_COUNT = 2
OFFSPRING_PER_PARENT = 4
FIRST_ROUND_OFFSPRING = 4
LOGICAL_EXECUTION_CAP = 29
TERMINAL_WEIGHTED_PARTICLES = 8
LLM_PROVIDER_CALL_CAP = 7
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "oracle",
        "private_oracle",
        "semantic_fingerprint_sha256",
        "source_line_sha256",
        "syntax_fingerprint_sha256",
        "target",
        "target_ast",
    }
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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


def _workspace_root(repo_root: Path) -> Path:
    root = repo_root.resolve()
    try:
        workspace = root.parents[2]
    except IndexError as error:
        raise ValueError("repository path is too shallow to resolve the workspace") from error
    if (workspace / "papers/01-modelsmc/modelsmc-pbe-python").resolve() != root:
        raise ValueError("repository is not at the frozen workspace-relative path")
    return workspace


def _bound_path(repo_root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty project-relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{name} must not be absolute")
    path = (repo_root.resolve() / relative).resolve()
    workspace = _workspace_root(repo_root)
    if path != workspace and workspace not in path.parents:
        raise ValueError(f"{name} escaped the workspace")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must resolve to a regular non-symlink file")
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


def _validate_bound_files(repo_root: Path, bindings: Mapping[str, Any]) -> None:
    for label in (
        "harness",
        "runner",
        "analyzer",
        "focused_tests",
        "adapter",
        "adapter_protocol",
        "bundle_manifest",
        "pyproject",
        "uv_lock",
    ):
        record = _object(bindings.get(label), name=f"source_bindings.{label}")
        path = _bound_path(repo_root, record.get("path"), name=f"{label}.path")
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
        if set(record) != {"path", "sha256"}:
            raise ValueError("dependency binding has the wrong fields")
        path = _bound_path(
            repo_root,
            record.get("path"),
            name=f"dependency_bundle[{index}].path",
        )
        _expect(
            sha256_file(path),
            _digest(record.get("sha256"), name=f"dependency[{index}].sha256"),
            name=f"dependency[{index}] SHA-256",
        )
    tree = _object(bindings.get("modelsmc_python_tree"), name="modelsmc_python_tree")
    try:
        python_tree_binding(repo_root, tree)
    except ValueError as error:
        raise ValueError(f"ModelSMC Python tree binding failed: {error}") from error


def validate_protocol(
    repo_root: Path,
    protocol_path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    """Validate the external freeze, source bindings, design, and all run records."""

    expected = _digest(expected_sha256, name="expected protocol SHA-256")
    _expect(sha256_file(protocol_path), expected, name="external protocol SHA-256")
    protocol = _read_object(protocol_path)
    _expect(protocol.get("schema"), SCHEMA, name="protocol schema")
    _expect(protocol.get("status"), STATUS, name="protocol status")
    classification = _object(protocol.get("classification"), name="classification")
    for field, value in {
        "debug_only": True,
        "public_released_tasks": True,
        "blind": False,
        "confirmatory": False,
        "external_benchmark_generalization_claim": False,
    }.items():
        _expect(classification.get(field), value, name=f"classification.{field}")
    authorization = _object(protocol.get("authorization"), name="authorization")
    _expect(authorization.get("preflight_without_provider"), True, name="preflight auth")
    _expect(authorization.get("provider_calls_require_explicit_cli_flag"), True, name="call auth")
    _expect(authorization.get("provider_calls_started_at_freeze"), False, name="call status")
    _validate_bound_files(
        repo_root,
        _object(protocol.get("source_bindings"), name="source_bindings"),
    )
    design = _object(protocol.get("design"), name="design")
    expected_design = {
        "tasks": 4,
        "paired_seeds_per_task": 8,
        "paired_blocks": 32,
        "arms": list(ARMS),
        "rounds": ROUNDS,
        "parent_count": PARENT_COUNT,
        "offspring_per_parent": OFFSPRING_PER_PARENT,
        "first_round_offspring": FIRST_ROUND_OFFSPRING,
        "logical_execution_cap": LOGICAL_EXECUTION_CAP,
        "terminal_weighted_particles": TERMINAL_WEIGHTED_PARTICLES,
        "llm_provider_call_cap": LLM_PROVIDER_CALL_CAP,
        "grammar_random_provider_call_cap": 0,
        "early_stop": False,
    }
    for field, value in expected_design.items():
        _expect(design.get(field), value, name=f"design.{field}")
    checkpoints, call_cap = population_schedule(
        rounds=ROUNDS,
        parent_count=PARENT_COUNT,
        offspring_per_parent=OFFSPRING_PER_PARENT,
        first_round_offspring=FIRST_ROUND_OFFSPRING,
    )
    _expect(checkpoints, (1, 5, 13, 21, 29), name="derived checkpoints")
    _expect(call_cap, LLM_PROVIDER_CALL_CAP, name="derived provider cap")
    _index_runs(protocol)
    return protocol


def validate_public_bundle(
    repo_root: Path,
    bundle_dir: Path,
    bundle_binding: Mapping[str, Any],
) -> dict[str, Path]:
    """Validate only the manifest and public task side of the debug bundle."""

    expected_root = (repo_root.resolve() / cast(str, bundle_binding.get("root"))).resolve()
    root = bundle_dir.resolve()
    _expect(root, expected_root, name="bundle root")
    workspace = _workspace_root(repo_root)
    if workspace not in root.parents or not root.is_dir() or root.is_symlink():
        raise ValueError("bundle root must be a regular in-workspace directory")
    manifest_path = root / "manifest.json"
    _expect(
        sha256_file(manifest_path),
        _digest(bundle_binding.get("manifest_sha256"), name="bundle manifest SHA-256"),
        name="bundle manifest SHA-256",
    )
    manifest = _read_object(manifest_path)
    _expect(manifest.get("schema"), BUNDLE_SCHEMA, name="bundle schema")
    _expect(manifest.get("debug_seed"), 808, name="bundle debug seed")
    _expect(
        manifest.get("classification"),
        "debug-only-public-released-data-no-provider-run-authorized",
        name="bundle classification",
    )
    manifest_tasks = _array(manifest.get("tasks"), name="manifest.tasks")
    frozen_tasks = _array(bundle_binding.get("tasks"), name="debug_bundle.tasks")
    if manifest_tasks != frozen_tasks or len(frozen_tasks) != len(TASK_IDS):
        raise ValueError("manifest task records differ from the frozen bundle")
    task_paths: dict[str, Path] = {}
    for task_id, raw in zip(TASK_IDS, frozen_tasks, strict=True):
        record = _object(raw, name=f"debug_bundle task {task_id}")
        _expect(record.get("task_id"), task_id, name="task ID")
        relative = record.get("public_path")
        if not isinstance(relative, str) or relative != f"public/{task_id}.json":
            raise ValueError(f"invalid public path for {task_id}")
        path = (root / relative).resolve()
        if path.parent != (root / "public").resolve() or path.is_symlink():
            raise ValueError(f"unsafe public task path for {task_id}")
        _expect(
            sha256_file(path),
            _digest(record.get("public_sha256"), name=f"{task_id}.public_sha256"),
            name=f"{task_id} public task SHA-256",
        )
        task = _read_object(path)
        _reject_private_keys(task, path=task_id)
        task_paths[task_id] = path
    return task_paths


def _expected_run_ids() -> tuple[str, ...]:
    return tuple(
        f"{task_id}--seed-{seed_id:02d}--{arm}"
        for arm in ARMS
        for task_id in TASK_IDS
        for seed_id in SEED_IDS
    )


def _index_runs(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_runs = _array(protocol.get("runs"), name="runs")
    expected_ids = _expected_run_ids()
    if len(raw_runs) != len(expected_ids):
        raise ValueError(f"protocol must contain exactly {len(expected_ids)} runs")
    runs: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_runs):
        run = _object(raw, name=f"runs[{index}]")
        run_id = run.get("id")
        if not isinstance(run_id, str) or run_id in runs:
            raise ValueError("run IDs must be unique strings")
        runs[run_id] = run
    _expect(tuple(runs), expected_ids, name="run order")
    for run_id, run in runs.items():
        task_id = run.get("task_id")
        arm = run.get("arm")
        seed_id = run.get("seed_id")
        if task_id not in TASK_IDS or arm not in ARMS or seed_id not in SEED_IDS:
            raise ValueError(f"invalid task, arm, or seed in {run_id}")
        _expect(
            run_id,
            f"{task_id}--seed-{cast(int, seed_id):02d}--{arm}",
            name="derived run ID",
        )
        expected_source = "llm" if arm == "llm-smc" else "grammar-random"
        _expect(run.get("proposal_source"), expected_source, name="proposal source")
        expected_calls = LLM_PROVIDER_CALL_CAP if arm == "llm-smc" else 0
        _expect(run.get("provider_call_cap"), expected_calls, name="provider cap")
        _expect(run.get("logical_execution_cap"), LOGICAL_EXECUTION_CAP, name="slot cap")
        _expect(
            run.get("terminal_weighted_particles"),
            TERMINAL_WEIGHTED_PARTICLES,
            name="terminal particles",
        )
        if (arm == "grammar-random") != (run.get("random_shortlist_seed") is not None):
            raise ValueError("random shortlist seed must occur iff arm is grammar-random")
    for task_id in TASK_IDS:
        for seed_id in SEED_IDS:
            llm = runs[f"{task_id}--seed-{seed_id:02d}--llm-smc"]
            random_run = runs[f"{task_id}--seed-{seed_id:02d}--grammar-random"]
            for field in (
                "task_sha256",
                "start_seed",
                "provider_seed",
                "sample_seed",
                "resample_seed",
                "epsilon",
                "evidence_scale",
                "parent_count",
                "offspring_per_parent",
                "first_round_offspring",
                "exact_reference_limit",
            ):
                _expect(random_run.get(field), llm.get(field), name=f"paired {field}")
    return runs


def validate_and_build_command(
    *,
    repo_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    bundle_dir: Path,
    runs_root: Path,
    task_id: str,
    seed_id: int,
    arm: str,
    python_executable: str,
) -> tuple[list[str], dict[str, object]]:
    root = repo_root.resolve()
    protocol_file = protocol_path.resolve()
    protocol = validate_protocol(root, protocol_file, expected_protocol_sha256)
    task_paths = validate_public_bundle(
        root,
        bundle_dir,
        _object(protocol.get("debug_bundle"), name="debug_bundle"),
    )
    if task_id not in TASK_IDS or seed_id not in SEED_IDS or arm not in ARMS:
        raise ValueError("task_id, seed_id, or arm is outside the frozen design")
    run_id = f"{task_id}--seed-{seed_id:02d}--{arm}"
    run = _index_runs(protocol)[run_id]
    _expect(run.get("task_sha256"), sha256_file(task_paths[task_id]), name="task hash")
    output = runs_root.resolve() / arm / task_id / f"seed-{seed_id:02d}"
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
    if run.get("random_shortlist_seed") is not None:
        command.extend(("--random-shortlist-seed", str(run["random_shortlist_seed"])))
    record: dict[str, object] = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "validated-before-debug-run",
        "classification": "debug-only-public-released-task-not-confirmatory",
        "task_id": task_id,
        "seed_id": seed_id,
        "arm": arm,
        "run_id": run_id,
        "protocol_sha256": sha256_file(protocol_file),
        "bundle_manifest_sha256": sha256_file(bundle_dir.resolve() / "manifest.json"),
        "public_task_sha256": sha256_file(task_paths[task_id]),
        "private_oracle_opened": False,
        "provider_call_cap": run["provider_call_cap"],
        "logical_execution_cap": LOGICAL_EXECUTION_CAP,
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
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--task-id", choices=TASK_IDS, required=True)
    parser.add_argument("--seed-id", type=int, choices=SEED_IDS, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-debug-provider-call", action="store_true")
    args = parser.parse_args(argv)
    command, record = validate_and_build_command(
        repo_root=args.repo_root,
        protocol_path=args.protocol,
        expected_protocol_sha256=args.expected_protocol_sha256,
        bundle_dir=args.bundle_dir,
        runs_root=args.runs_root,
        task_id=args.task_id,
        seed_id=args.seed_id,
        arm=args.arm,
        python_executable=args.python_executable,
    )
    if args.preflight_only:
        if args.authorize_debug_provider_call:
            parser.error("do not authorize a provider call during preflight")
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    if args.arm == "llm-smc" and not args.authorize_debug_provider_call:
        parser.error("LLM debug execution requires --authorize-debug-provider-call")
    if args.arm != "llm-smc" and args.authorize_debug_provider_call:
        parser.error("grammar-random execution must not authorize a provider call")
    _write_exclusive_json(Path(cast(str, record["preflight_path"])), record)
    return subprocess.run(command, cwd=args.repo_root.resolve(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

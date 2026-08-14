"""Fail-closed launcher for one arm of the fresh blind-v3 matched study.

The launcher accepts only a task ID, arm, sealed paths, and runs root.  It
derives all semantic harness arguments and a one-run frozen invocation record
from Stage 1 plus the pre-completion provider seal.  It has no task-generation
or unblinding interface.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from research.blind_filter_map_confirmation_v3 import (
    BUNDLE_SCHEME,
    CUSTODY_SEAL_SCHEMA,
    EPSILON,
    EVIDENCE_SCALE,
    EXACT_REFERENCE_LIMIT,
    FIRST_ROUND_OFFSPRING,
    MANIFEST_SCHEMA,
    MAX_CONCURRENCY,
    MAX_TOKENS,
    METHOD_SCHEMA,
    METHOD_SEAL_SCHEMA,
    METHOD_STATUS,
    MODEL_NAME,
    OFFSPRING_PER_PARENT,
    PARENT_COUNT,
    PROVIDER_SEAL_SCHEMA,
    REASONING_EFFORT,
    RESULT_SCHEMA,
    SLOT_CHECKPOINTS,
    TASK_IDS,
    TEMPERATURE,
    TIMEOUT_SECONDS,
    canonical_bytes,
    expect,
    frozen_run_seeds,
    read_object,
    reject_private_keys,
    require_array,
    require_digest,
    require_object,
    safe_repo_path,
    sha256_bytes,
    sha256_file,
    write_json_exclusive,
)

ARMS = ("grammar-random", "llm")


def _validate_bundle(repo_root: Path, raw: object, *, name: str) -> str:
    bundle = require_object(raw, name=name)
    expect(bundle.get("scheme"), BUNDLE_SCHEME, name=f"{name}.scheme")
    domain = bundle.get("domain")
    if not isinstance(domain, str) or not domain:
        raise ValueError(f"{name}.domain must be nonempty")
    paths = require_array(bundle.get("paths_in_order"), name=f"{name}.paths")
    hashes = require_object(bundle.get("source_sha256"), name=f"{name}.source_sha256")
    expect(set(hashes), set(paths), name=f"{name} source paths")
    material = bytearray(domain.encode() + b"\0")
    for index, relative in enumerate(paths):
        path = safe_repo_path(repo_root, relative, name=f"{name}.paths[{index}]")
        payload = path.read_bytes()
        expect(hashes.get(relative), sha256_bytes(payload), name=f"{name}.{relative}")
        material.extend(cast(str, relative).encode() + b"\0" + payload + b"\0")
    digest = sha256_bytes(bytes(material))
    expect(bundle.get("bundle_sha256"), digest, name=f"{name}.bundle_sha256")
    return digest


def _python_tree_binding(repo_root: Path, raw: object) -> dict[str, object]:
    binding = require_object(raw, name="modelsmc_python_tree")
    expect(binding.get("schema"), "sha256-python-tree-v1", name="tree schema")
    expect(binding.get("include"), "**/*.py", name="tree include")
    relative_root = binding.get("root")
    if not isinstance(relative_root, str):
        raise ValueError("tree root must be a string")
    source_root = safe_repo_path(repo_root, relative_root, name="tree root")
    paths = sorted(
        (path for path in source_root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(source_root).as_posix().encode(),
    )
    if any(path.is_symlink() for path in source_root.rglob("*")):
        raise ValueError("source tree contains a symlink")
    entries = [
        {
            "path": path.relative_to(source_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    manifest = {
        "schema": "sha256-python-tree-v1",
        "root": relative_root,
        "include": "**/*.py",
        "entries": entries,
    }
    actual: dict[str, object] = {
        "schema": "sha256-python-tree-v1",
        "root": relative_root,
        "include": "**/*.py",
        "file_count": len(entries),
        "manifest_sha256": sha256_bytes(canonical_bytes(manifest)),
    }
    expect(binding, actual, name="modelsmc Python-tree binding")
    return actual


def _validate_method(
    *,
    repo_root: Path,
    protocol_path: Path,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str, Path]:
    root = repo_root.resolve()
    protocol_file = protocol_path.resolve()
    method_file = method_seal_path.resolve()
    method_sha256 = sha256_file(method_file)
    expect(
        method_sha256,
        require_digest(expected_method_seal_sha256, name="expected method-seal SHA-256"),
        name="external method-seal SHA-256",
    )
    seal = read_object(method_file)
    expect(seal.get("schema"), METHOD_SEAL_SCHEMA, name="method-seal schema")
    expect(seal.get("sealed_before_secret_preparation"), True, name="method seal timing")
    binding = require_object(seal.get("protocol"), name="method-seal.protocol")
    expect(
        safe_repo_path(root, binding.get("path"), name="method protocol path"),
        protocol_file,
        name="method protocol path",
    )
    protocol_sha256 = sha256_file(protocol_file)
    expect(binding.get("sha256"), protocol_sha256, name="method protocol SHA-256")
    protocol = read_object(protocol_file)
    expect(protocol.get("schema"), METHOD_SCHEMA, name="method schema")
    expect(protocol.get("protocol_status"), METHOD_STATUS, name="method status")
    freeze = require_object(protocol.get("freeze_requirements"), name="freeze_requirements")
    for label in (
        "common",
        "generator",
        "harness",
        "runner",
        "analysis",
        "public_sealer",
    ):
        source = require_object(freeze.get(label), name=f"freeze.{label}")
        path = safe_repo_path(root, source.get("path"), name=f"freeze.{label}.path")
        expect(source.get("sha256"), sha256_file(path), name=f"frozen {label} SHA-256")
        if label == "runner":
            expect(path, Path(__file__).resolve(), name="executing runner path")
    dependencies = require_array(freeze.get("dependency_bundle"), name="dependency_bundle")
    for index, raw in enumerate(dependencies):
        source = require_object(raw, name=f"dependency_bundle[{index}]")
        path = safe_repo_path(root, source.get("path"), name=f"dependency[{index}].path")
        expect(source.get("sha256"), sha256_file(path), name=f"dependency[{index}] hash")
    prompt_sha256 = _validate_bundle(
        root,
        freeze.get("prompt_template_binding"),
        name="prompt_template_binding",
    )
    expect(freeze.get("prompt_template_sha256"), prompt_sha256, name="prompt SHA-256")
    _validate_bundle(root, freeze.get("test_bundle"), name="test_bundle")
    _python_tree_binding(root, freeze.get("modelsmc_python_tree"))
    expect(
        freeze.get("run_seeds"),
        [record.to_dict() for record in frozen_run_seeds()],
        name="frozen run seeds",
    )
    harness = safe_repo_path(
        root,
        require_object(freeze.get("harness"), name="harness").get("path"),
        name="harness path",
    )
    return protocol, freeze, protocol_sha256, method_sha256, harness


def _validate_provider_boundary(
    *,
    repo_root: Path,
    protocol_sha256: str,
    method_sha256: str,
    custody_seal_path: Path,
    expected_custody_seal_sha256: str,
    provider_seal_path: Path,
    expected_provider_seal_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path], str, str]:
    root = repo_root.resolve()
    custody_file = custody_seal_path.resolve()
    custody_sha256 = sha256_file(custody_file)
    expect(
        custody_sha256,
        require_digest(expected_custody_seal_sha256, name="expected custody SHA-256"),
        name="external custody SHA-256",
    )
    custody = read_object(custody_file)
    expect(custody.get("schema"), CUSTODY_SEAL_SCHEMA, name="custody schema")
    expect(custody.get("sealed_before_task_generation"), True, name="custody timing")
    expect(custody.get("study_protocol_sha256"), protocol_sha256, name="custody protocol")
    expect(custody.get("method_seal_sha256"), method_sha256, name="custody method seal")
    provider_file = provider_seal_path.resolve()
    provider_sha256 = sha256_file(provider_file)
    expect(
        provider_sha256,
        require_digest(expected_provider_seal_sha256, name="expected provider-seal SHA-256"),
        name="external provider-seal SHA-256",
    )
    provider = read_object(provider_file)
    expect(provider.get("schema"), PROVIDER_SEAL_SCHEMA, name="provider-seal schema")
    expect(provider.get("sealed_before_completion_calls"), True, name="provider seal timing")
    expect(provider.get("endpoint_health_query_precedes_seal"), True, name="health timing")
    expect(provider.get("study_protocol_sha256"), protocol_sha256, name="provider protocol")
    expect(provider.get("method_seal_sha256"), method_sha256, name="provider method seal")
    expect(provider.get("custody_seal_sha256"), custody_sha256, name="provider custody seal")
    expect(
        provider.get("task_secret_commitment_sha256"),
        custody.get("task_secret_commitment_sha256"),
        name="provider secret commitment",
    )
    require_digest(provider.get("hidden_target_manifest_sha256"), name="hidden target hash")
    manifest_binding = require_object(provider.get("public_manifest"), name="public_manifest")
    manifest_path = safe_repo_path(root, manifest_binding.get("path"), name="manifest path")
    expect(manifest_binding.get("sha256"), sha256_file(manifest_path), name="manifest hash")
    manifest = read_object(manifest_path)
    reject_private_keys(manifest)
    expect(manifest.get("schema"), MANIFEST_SCHEMA, name="manifest schema")
    expect(manifest.get("task_count"), len(TASK_IDS), name="manifest task count")
    expect(
        manifest.get("seed_commitment_sha256"),
        custody.get("task_secret_commitment_sha256"),
        name="manifest secret commitment",
    )
    public_records = require_array(manifest.get("tasks"), name="manifest.tasks")
    sealed_records = require_array(provider.get("tasks"), name="provider.tasks")
    expect(len(public_records), len(TASK_IDS), name="manifest record count")
    expect(len(sealed_records), len(TASK_IDS), name="sealed record count")
    task_paths: dict[str, Path] = {}
    for index, (task_id, seed_record) in enumerate(
        zip(TASK_IDS, frozen_run_seeds(), strict=True)
    ):
        public = require_object(public_records[index], name=f"manifest.tasks[{index}]")
        sealed = require_object(sealed_records[index], name=f"provider.tasks[{index}]")
        expect(public.get("task_id"), task_id, name=f"public task {index} ID")
        expect(sealed.get("task_id"), task_id, name=f"sealed task {index} ID")
        for field in (
            "task_file_sha256",
            "task_canonical_sha256",
            "target_commitment_sha256",
        ):
            expect(sealed.get(field), public.get(field), name=f"{task_id}.{field}")
        for field, value in seed_record.to_dict().items():
            expect(sealed.get(field), value, name=f"{task_id}.{field}")
        task_path = safe_repo_path(root, sealed.get("path"), name=f"{task_id}.path")
        expect(task_path.name, public.get("path"), name=f"{task_id} basename")
        expect(sha256_file(task_path), public.get("task_file_sha256"), name=f"{task_id} hash")
        task = read_object(task_path)
        reject_private_keys(task, path=task_id)
        expect(
            sha256_bytes(canonical_bytes(task)),
            public.get("task_canonical_sha256"),
            name=f"{task_id} canonical hash",
        )
        task_paths[task_id] = task_path
    endpoint = require_object(provider.get("endpoint_health_record"), name="endpoint record")
    endpoint_path = safe_repo_path(root, endpoint.get("path"), name="endpoint path")
    expect(endpoint.get("sha256"), sha256_file(endpoint_path), name="endpoint hash")
    return manifest, task_paths, custody_sha256, provider_sha256


def _assert_complete_result(path: Path, *, arm: str) -> None:
    result = read_object(path)
    expect(result.get("schema"), RESULT_SCHEMA, name=f"{path} schema")
    protocol = require_object(result.get("protocol"), name=f"{path} protocol")
    expect(protocol.get("proposal_source"), arm, name=f"{path} arm")
    search = require_object(result.get("search"), name=f"{path} search")
    expect(
        search.get("logical_complete_program_executions"),
        SLOT_CHECKPOINTS[-1],
        name=f"{path} logical slots",
    )
    if arm == "grammar-random":
        expect(search.get("provider_calls"), 0, name=f"{path} provider calls")


def _validate_order(runs_root: Path, task_id: str, arm: str) -> Path:
    root = runs_root.resolve()
    current_index = TASK_IDS.index(task_id)
    for prior_id in TASK_IDS[:current_index]:
        for prior_arm in ARMS:
            _assert_complete_result(root / prior_id / prior_arm / "result.json", arm=prior_arm)
    output = root / task_id / arm
    if output.exists():
        raise FileExistsError(f"refusing to overwrite run output: {output}")
    if arm == "llm":
        _assert_complete_result(
            root / task_id / "grammar-random" / "result.json",
            arm="grammar-random",
        )
    elif (root / task_id / "llm").exists():
        raise ValueError("grammar-random must precede llm for each task")
    for later_id in TASK_IDS[current_index + 1 :]:
        if (root / later_id).exists():
            raise ValueError("later task output exists before the frozen run order")
    return output


def validate_and_build(
    *,
    repo_root: Path,
    protocol_path: Path,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
    custody_seal_path: Path,
    expected_custody_seal_sha256: str,
    provider_seal_path: Path,
    expected_provider_seal_sha256: str,
    task_id: str,
    arm: str,
    runs_root: Path,
    base_url: str,
    python_executable: str,
) -> tuple[list[str], dict[str, object], dict[str, object]]:
    if task_id not in TASK_IDS:
        raise ValueError(f"unknown task ID: {task_id}")
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base_url must use HTTP or HTTPS")
    root = repo_root.resolve()
    protocol, freeze, protocol_sha256, method_sha256, harness = _validate_method(
        repo_root=root,
        protocol_path=protocol_path,
        method_seal_path=method_seal_path,
        expected_method_seal_sha256=expected_method_seal_sha256,
    )
    _, task_paths, custody_sha256, provider_sha256 = _validate_provider_boundary(
        repo_root=root,
        protocol_sha256=protocol_sha256,
        method_sha256=method_sha256,
        custody_seal_path=custody_seal_path,
        expected_custody_seal_sha256=expected_custody_seal_sha256,
        provider_seal_path=provider_seal_path,
        expected_provider_seal_sha256=expected_provider_seal_sha256,
    )
    output = _validate_order(runs_root, task_id, arm)
    invocation_path = output.with_name(output.name + ".invocation.json")
    preflight_path = output.with_name(output.name + ".preflight.json")
    for artifact in (invocation_path, preflight_path):
        if artifact.exists():
            raise FileExistsError(f"refusing to overwrite {artifact}")
    seeds = frozen_run_seeds()[TASK_IDS.index(task_id)]
    run_id = f"{task_id}-{arm}"
    source_bindings = {
        label: freeze[label]
        for label in (
            "common",
            "generator",
            "harness",
            "runner",
            "analysis",
            "public_sealer",
        )
    } | {
        "dependency_bundle": freeze["dependency_bundle"],
        "modelsmc_python_tree": freeze["modelsmc_python_tree"],
    }
    run_record: dict[str, object] = {
        "id": run_id,
        "task": task_paths[task_id].relative_to(root).as_posix(),
        "task_sha256": sha256_file(task_paths[task_id]),
        "role": f"fresh blind matched {arm} arm",
        "proposal_source": arm,
        "parent_count": PARENT_COUNT,
        "offspring_per_parent": OFFSPRING_PER_PARENT,
        "first_round_offspring": FIRST_ROUND_OFFSPRING,
        "terminal_weighted_particles": PARENT_COUNT * OFFSPRING_PER_PARENT,
        "logical_execution_cap": SLOT_CHECKPOINTS[-1],
        "provider_call_cap": 0 if arm == "grammar-random" else 7,
        "provider_seed": seeds.provider_seed,
        "random_shortlist_seed": (
            seeds.random_shortlist_seed if arm == "grammar-random" else None
        ),
        "epsilon": EPSILON,
        "evidence_scale": EVIDENCE_SCALE,
        "start_seed": seeds.start_seed,
        "sample_seed": seeds.sample_seed,
        "resample_seed": seeds.resample_seed,
        "max_concurrency": MAX_CONCURRENCY,
        "exact_reference_limit": EXACT_REFERENCE_LIMIT,
    }
    invocation: dict[str, object] = {
        "schema": "blind-filter-map-confirmation-v3-derived-harness-invocation-v1",
        "status": "frozen-before-large-space-provider-calls",
        "derived_from_study_protocol_sha256": protocol_sha256,
        "derived_from_provider_seal_sha256": provider_sha256,
        "source_bindings": source_bindings,
        "provider": require_object(protocol.get("provider"), name="provider"),
        "proposal": require_object(protocol.get("proposal"), name="proposal"),
        "target_schedule": {"evidence_scale": EVIDENCE_SCALE},
        "runs": [run_record],
    }
    invocation_sha256 = sha256_bytes(canonical_bytes(invocation) + b"\n")
    command = [
        python_executable,
        "-m",
        "research.evidence_shortlist_smc",
        "--task",
        str(task_paths[task_id]),
        "--output",
        str(output),
        "--base-url",
        base_url,
        "--model",
        MODEL_NAME,
        "--reasoning-effort",
        REASONING_EFFORT,
        "--temperature",
        str(TEMPERATURE),
        "--max-tokens",
        str(MAX_TOKENS),
        "--timeout-seconds",
        str(TIMEOUT_SECONDS),
        "--epsilon",
        str(EPSILON),
        "--evidence-scale",
        str(EVIDENCE_SCALE),
        "--start-seed",
        str(seeds.start_seed),
        "--provider-seed",
        str(seeds.provider_seed),
        "--sample-seed",
        str(seeds.sample_seed),
        "--resample-seed",
        str(seeds.resample_seed),
        "--parent-count",
        str(PARENT_COUNT),
        "--offspring-per-parent",
        str(OFFSPRING_PER_PARENT),
        "--first-round-offspring",
        str(FIRST_ROUND_OFFSPRING),
        "--max-concurrency",
        str(MAX_CONCURRENCY),
        "--exact-reference-limit",
        str(EXACT_REFERENCE_LIMIT),
        "--proposal-source",
        arm,
        "--study-protocol",
        str(invocation_path),
        "--run-id",
        run_id,
        "--expected-study-protocol-sha256",
        invocation_sha256,
    ]
    if arm == "grammar-random":
        command.extend(["--random-shortlist-seed", str(seeds.random_shortlist_seed)])
    record: dict[str, object] = {
        "schema": "blind-filter-map-confirmation-v3-preflight-v1",
        "status": "validated-before-arm-execution",
        "task_id": task_id,
        "arm": arm,
        "study_protocol_sha256": protocol_sha256,
        "method_seal_sha256": method_sha256,
        "custody_seal_sha256": custody_sha256,
        "provider_seal_sha256": provider_sha256,
        "task_file_sha256": sha256_file(task_paths[task_id]),
        "harness_sha256": sha256_file(harness),
        "derived_invocation_path": str(invocation_path),
        "derived_invocation_sha256": invocation_sha256,
        "command": command,
        "output": str(output),
    }
    return command, record, invocation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--study-protocol", type=Path, required=True)
    parser.add_argument("--method-seal", type=Path, required=True)
    parser.add_argument("--expected-method-seal-sha256", required=True)
    parser.add_argument("--custody-seal", type=Path, required=True)
    parser.add_argument("--expected-custody-seal-sha256", required=True)
    parser.add_argument("--provider-seal", type=Path, required=True)
    parser.add_argument("--expected-provider-seal-sha256", required=True)
    parser.add_argument("--task-id", choices=TASK_IDS, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    command, record, invocation = validate_and_build(
        repo_root=args.repo_root,
        protocol_path=args.study_protocol,
        method_seal_path=args.method_seal,
        expected_method_seal_sha256=args.expected_method_seal_sha256,
        custody_seal_path=args.custody_seal,
        expected_custody_seal_sha256=args.expected_custody_seal_sha256,
        provider_seal_path=args.provider_seal,
        expected_provider_seal_sha256=args.expected_provider_seal_sha256,
        task_id=args.task_id,
        arm=args.arm,
        runs_root=args.runs_root,
        base_url=args.base_url,
        python_executable=args.python_executable,
    )
    if args.preflight_only:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    invocation_path = Path(cast(str, record["derived_invocation_path"]))
    preflight_path = Path(cast(str, record["output"])).with_name(
        Path(cast(str, record["output"])).name + ".preflight.json"
    )
    write_json_exclusive(invocation_path, invocation)
    write_json_exclusive(preflight_path, record)
    completed = subprocess.run(command, cwd=args.repo_root.resolve(), check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

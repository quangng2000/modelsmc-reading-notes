from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.run_blind_evidence_frontier_v2 import (
    BUNDLE_SCHEME,
    TASK_IDS,
    canonical_bytes,
    sha256_bytes,
    sha256_file,
    validate_and_build_command,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def _source(root: Path, relative: str, payload: bytes) -> dict[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": relative, "sha256": sha256_bytes(payload)}


def _bundle(root: Path, domain: str, paths: list[str]) -> dict[str, object]:
    digest = __import__("hashlib").sha256()
    digest.update(domain.encode())
    digest.update(b"\0")
    source_sha256 = {}
    for relative in paths:
        payload = (root / relative).read_bytes()
        source_sha256[relative] = sha256_bytes(payload)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return {
        "scheme": BUNDLE_SCHEME,
        "domain": domain,
        "paths_in_order": paths,
        "source_sha256": source_sha256,
        "bundle_sha256": digest.hexdigest(),
    }


def _fixture(tmp_path: Path, *, leak_private_key: bool = False) -> dict[str, object]:
    root = tmp_path.resolve()
    generator = _source(root, "research/generate_blinded_filter_map_tasks.py", b"generator-v2\n")
    harness = _source(root, "research/iterative_beam_experiment.py", b"harness-v2\n")
    analysis = _source(root, "research/analyze_blind_evidence_frontier_v2.py", b"analysis-v2\n")
    runner = _source(root, "research/run_blind_evidence_frontier_v2.py", b"runner-v2\n")
    _source(root, "research/prompt.py", b"prompt-v2\n")
    _source(root, "research/tests/test_confirmation.py", b"tests-v2\n")
    prompt_bundle = _bundle(root, "prompt-bundle-v2", ["research/prompt.py"])
    test_bundle = _bundle(
        root,
        "test-bundle-v2",
        ["research/tests/test_confirmation.py"],
    )
    run_seeds = [
        {
            "task_id": task_id,
            "provider_seed": 301000 + index * 1000,
            "tie_seed": 301100 + index * 1000,
            "matched_random_seed": 301200 + index * 1000,
        }
        for index, task_id in enumerate(TASK_IDS)
    ]
    protocol = {
        "schema": "blind-evidence-frontier-confirmation-v2",
        "protocol_status": "method-frozen-before-task-generation",
        "freeze_requirements": {
            "protocol_sha256_binding": "external-provider-seal-and-run-protocol-records",
            "generator": generator,
            "harness": harness,
            "analysis": analysis,
            "runner": runner,
            "prompt_template_sha256": prompt_bundle["bundle_sha256"],
            "prompt_template_binding": prompt_bundle,
            "test_bundle": test_bundle,
            "task_secret_commitment_sha256": "1" * 64,
            "run_seeds": run_seeds,
        },
        "model": {
            "served_name": "gpt-oss-120b",
            "revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
            "reasoning_effort": "low",
            "temperature": 0,
            "max_tokens": 1600,
            "timeout_seconds": 420,
            "max_concurrency": 2,
            "provider_retry_policy": "none",
        },
        "fresh_task_generator": {
            "task_count": 12,
            "fixed_balanced_schedule": [[task_id, "shape", "mapper"] for task_id in TASK_IDS],
        },
        "search": {
            "selection_policy": "evidence-frontier",
            "branching_factor": 4,
            "beam_width": 2,
            "maximum_rounds": 4,
            "start_seed": 17,
            "singleton_evidence": True,
            "stall_policy": "alternate-hole",
            "maximum_proposal_slots": 29,
            "maximum_provider_calls": 7,
        },
        "matched_random_baseline": {"trials_per_task": 10000},
    }
    protocol_path = root / "research/protocol-blind-evidence-frontier-v2.json"
    _write_json(protocol_path, protocol)
    method_seal = {
        "schema": "blind-evidence-frontier-method-seal-v2",
        "sealed_before_task_generation": True,
        "protocol": {
            "path": str(protocol_path.relative_to(root)),
            "sha256": sha256_file(protocol_path),
        },
    }
    method_seal_path = root / "artifacts/method-seal.json"
    _write_json(method_seal_path, method_seal)
    endpoint_path = root / "artifacts/endpoint-health.json"
    _write_json(
        endpoint_path,
        {
            "served_model_name": "gpt-oss-120b",
            "model_snapshot_revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
            "health_checks": {"models_endpoint": "ok"},
        },
    )

    public_records = []
    sealed_records = []
    for index, task_id in enumerate(TASK_IDS):
        task: dict[str, object] = {"name": task_id, "examples": []}
        if leak_private_key and index == 0:
            task["target"] = {"predicate": "secret"}
        task_path = root / "suite/public" / f"{task_id}.json"
        _write_json(task_path, task)
        public = {
            "task_id": task_id,
            "path": task_path.name,
            "task_file_sha256": sha256_file(task_path),
            "task_canonical_sha256": sha256_bytes(canonical_bytes(task)),
            "target_commitment_sha256": f"{index + 2:x}" * 64,
        }
        public["target_commitment_sha256"] = cast_digest(public["target_commitment_sha256"])
        public_records.append(public)
        sealed_records.append(
            public
            | {
                "path": str(task_path.relative_to(root)),
                "provider_seed": run_seeds[index]["provider_seed"],
                "tie_seed": run_seeds[index]["tie_seed"],
                "matched_random_seed": run_seeds[index]["matched_random_seed"],
            }
        )
    manifest = {
        "schema": "blinded-filter-map-suite-v2",
        "task_count": 12,
        "seed_commitment_sha256": "1" * 64,
        "task_generation": {
            "constants": [str(value) for value in range(-3, 5)],
            "provider_boundary": "public only",
            "generator_sha256": generator["sha256"],
        },
        "tasks": public_records,
    }
    manifest_path = root / "suite/public/manifest.json"
    _write_json(manifest_path, manifest)
    provider_seal = {
        "schema": "blind-v2-provider-call-seal-v1",
        "sealed_before_provider_calls": True,
        "study_protocol_sha256": sha256_file(protocol_path),
        "method_seal_sha256": sha256_file(method_seal_path),
        "endpoint_health_record": {
            "path": str(endpoint_path.relative_to(root)),
            "sha256": sha256_file(endpoint_path),
        },
        "public_manifest": {
            "path": str(manifest_path.relative_to(root)),
            "sha256": sha256_file(manifest_path),
        },
        "hidden_target_manifest_sha256": "e" * 64,
        "task_secret_commitment_sha256": "1" * 64,
        "tasks": sealed_records,
    }
    provider_seal_path = root / "artifacts/provider-call-seal.json"
    _write_json(provider_seal_path, provider_seal)
    return {
        "repo_root": root,
        "study_protocol_path": protocol_path,
        "method_seal_path": method_seal_path,
        "provider_call_seal_path": provider_seal_path,
        "expected_method_seal_sha256": sha256_file(method_seal_path),
        "expected_provider_call_seal_sha256": sha256_file(provider_seal_path),
        "task_id": TASK_IDS[2],
        "output": root / "runs/blind-v2-03",
        "base_url": "http://127.0.0.1:18000/v1",
        "python_executable": "/frozen/python",
    }


def cast_digest(value: object) -> str:
    text = str(value)
    return (text + "0" * 64)[:64]


def test_valid_preflight_builds_only_the_frozen_generic_command(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    command, record = validate_and_build_command(**arguments)  # type: ignore[arg-type]

    assert command[:3] == ["/frozen/python", "-m", "research.iterative_beam_experiment"]
    assert "--protocol-mode" not in command
    assert command[command.index("--provider-seed") + 1] == "303000"
    assert command[command.index("--tie-seed") + 1] == "303100"
    assert command[command.index("--random-baseline-seed") + 1] == "303200"
    assert "--singleton-evidence" in command
    assert record["status"] == "validated-before-provider-call"
    assert not cast_path(arguments["output"]).exists()
    assert not cast_path(arguments["output"]).with_name("blind-v2-03.preflight.json").exists()


def test_preflight_rejects_wrong_external_seal_digest(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    arguments["expected_provider_call_seal_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="external SHA-256"):
        validate_and_build_command(**arguments)  # type: ignore[arg-type]


def test_preflight_rejects_frozen_source_drift(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    (cast_path(arguments["repo_root"]) / "research/iterative_beam_experiment.py").write_text(
        "changed\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"harness\.sha256"):
        validate_and_build_command(**arguments)  # type: ignore[arg-type]


def test_preflight_rejects_provider_seed_drift(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    seal_path = cast_path(arguments["provider_call_seal_path"])
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["tasks"][2]["provider_seed"] += 1
    _write_json(seal_path, seal)
    arguments["expected_provider_call_seal_sha256"] = sha256_file(seal_path)

    with pytest.raises(ValueError, match="provider_seed"):
        validate_and_build_command(**arguments)  # type: ignore[arg-type]


def test_preflight_rejects_frozen_setting_drift(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    protocol_path = cast_path(arguments["study_protocol_path"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["search"]["maximum_rounds"] = 5
    _write_json(protocol_path, protocol)
    method_path = cast_path(arguments["method_seal_path"])
    method = json.loads(method_path.read_text(encoding="utf-8"))
    method["protocol"]["sha256"] = sha256_file(protocol_path)
    _write_json(method_path, method)
    arguments["expected_method_seal_sha256"] = sha256_file(method_path)
    provider_path = cast_path(arguments["provider_call_seal_path"])
    provider = json.loads(provider_path.read_text(encoding="utf-8"))
    provider["study_protocol_sha256"] = sha256_file(protocol_path)
    provider["method_seal_sha256"] = sha256_file(method_path)
    _write_json(provider_path, provider)
    arguments["expected_provider_call_seal_sha256"] = sha256_file(provider_path)

    with pytest.raises(ValueError, match="maximum_rounds"):
        validate_and_build_command(**arguments)  # type: ignore[arg-type]


def test_preflight_rejects_missing_or_drifted_endpoint_health_record(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    provider_path = cast_path(arguments["provider_call_seal_path"])
    provider = json.loads(provider_path.read_text(encoding="utf-8"))
    endpoint_path = cast_path(arguments["repo_root"]) / provider["endpoint_health_record"]["path"]
    endpoint_path.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="endpoint-health-record SHA-256"):
        validate_and_build_command(**arguments)  # type: ignore[arg-type]

    endpoint_path.unlink()
    provider.pop("endpoint_health_record")
    _write_json(provider_path, provider)
    arguments["expected_provider_call_seal_sha256"] = sha256_file(provider_path)
    with pytest.raises(ValueError, match="endpoint_health_record must be an object"):
        validate_and_build_command(**arguments)  # type: ignore[arg-type]


def test_preflight_rejects_private_target_key_in_public_task(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path, leak_private_key=True)

    with pytest.raises(ValueError, match="private key 'target'"):
        validate_and_build_command(**arguments)  # type: ignore[arg-type]


def test_preflight_rejects_existing_output_or_invocation_record(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    output = cast_path(arguments["output"])
    output.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="output path"):
        validate_and_build_command(**arguments)  # type: ignore[arg-type]


def cast_path(value: object) -> Path:
    assert isinstance(value, Path)
    return value

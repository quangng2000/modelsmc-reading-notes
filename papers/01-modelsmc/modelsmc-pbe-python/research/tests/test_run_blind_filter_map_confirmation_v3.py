from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import research.evidence_shortlist_smc as harness
import research.run_blind_filter_map_confirmation_v3 as runner
from research.blind_filter_map_confirmation_v3 import (
    CUSTODY_SEAL_SCHEMA,
    MANIFEST_SCHEMA,
    METHOD_SCHEMA,
    METHOD_SEAL_SCHEMA,
    METHOD_STATUS,
    PROVIDER_SEAL_SCHEMA,
    TASK_IDS,
    canonical_bytes,
    frozen_run_seeds,
    sha256_bytes,
    sha256_file,
)
from research.freeze_blind_filter_map_confirmation_v3 import (
    _bundle,
    _python_tree_binding,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path | str]:
    repo = tmp_path / "repo"
    research = repo / "research"
    research.mkdir(parents=True)
    source_paths = {
        "common": "research/blind_filter_map_confirmation_v3.py",
        "generator": "research/generate_blind_filter_map_confirmation_v3.py",
        "harness": "research/evidence_shortlist_smc.py",
        "runner": "research/run_blind_filter_map_confirmation_v3.py",
        "analysis": "research/analyze_blind_filter_map_confirmation_v3.py",
        "public_sealer": "research/seal_blind_filter_map_confirmation_v3_public.py",
    }
    for label, relative in source_paths.items():
        path = repo / relative
        path.write_text(f"# {label} fixture\n", encoding="utf-8")
    monkeypatch.setattr(runner, "__file__", str(repo / source_paths["runner"]))
    dependency = research / "dependency.py"
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    prompt = research / "prompt.py"
    prompt.write_text("PROMPT = 'fixed'\n", encoding="utf-8")
    test_source = research / "test_fixture.py"
    test_source.write_text("def test_fixed(): pass\n", encoding="utf-8")
    tree_file = repo / "src/modelsmc_pbe/core.py"
    tree_file.parent.mkdir(parents=True)
    tree_file.write_text("VALUE = 1\n", encoding="utf-8")
    freeze = {
        **{
            label: {"path": relative, "sha256": sha256_file(repo / relative)}
            for label, relative in source_paths.items()
        },
        "dependency_bundle": [
            {
                "path": dependency.relative_to(repo).as_posix(),
                "sha256": sha256_file(dependency),
            }
        ],
        "modelsmc_python_tree": _python_tree_binding(repo),
        "prompt_template_binding": _bundle(
            repo,
            domain="fixture-prompt",
            paths=(prompt.relative_to(repo).as_posix(),),
        ),
        "test_bundle": _bundle(
            repo,
            domain="fixture-tests",
            paths=(test_source.relative_to(repo).as_posix(),),
        ),
        "run_seeds": [record.to_dict() for record in frozen_run_seeds()],
    }
    freeze["prompt_template_sha256"] = freeze["prompt_template_binding"][
        "bundle_sha256"
    ]
    protocol = research / "protocol-v3.json"
    _write_json(
        protocol,
        {
            "schema": METHOD_SCHEMA,
            "protocol_status": METHOD_STATUS,
            "provider": {
                "base_url": "http://127.0.0.1:18000/v1",
                "model": "gpt-oss-120b",
                "model_revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
                "reasoning_effort": "low",
                "temperature": 0,
                "max_output_tokens": 1200,
                "timeout_seconds": 420,
            },
            "proposal": {"epsilon": 0.05},
            "freeze_requirements": freeze,
        },
    )
    method = research / "method-seal-v3.json"
    _write_json(
        method,
        {
            "schema": METHOD_SEAL_SCHEMA,
            "sealed_before_secret_preparation": True,
            "protocol": {
                "path": protocol.relative_to(repo).as_posix(),
                "sha256": sha256_file(protocol),
            },
        },
    )
    secret_commitment = "1" * 64
    custody = research / "custody-seal-v3.json"
    _write_json(
        custody,
        {
            "schema": CUSTODY_SEAL_SCHEMA,
            "sealed_before_task_generation": True,
            "study_protocol_sha256": sha256_file(protocol),
            "method_seal_sha256": sha256_file(method),
            "task_secret_commitment_sha256": secret_commitment,
        },
    )
    public = repo / "artifacts/public"
    public.mkdir(parents=True)
    task_records = []
    sealed_records = []
    for task_id, seeds in zip(TASK_IDS, frozen_run_seeds(), strict=True):
        task = public / f"{task_id}.json"
        _write_json(task, {"name": task_id, "examples": []})
        record = {
            "task_id": task_id,
            "path": task.name,
            "task_file_sha256": sha256_file(task),
            "task_canonical_sha256": sha256_bytes(
                canonical_bytes(json.loads(task.read_text(encoding="utf-8")))
            ),
            "target_commitment_sha256": "2" * 64,
        }
        task_records.append(record)
        sealed_records.append(
            {
                **record,
                "path": task.relative_to(repo).as_posix(),
                **seeds.to_dict(),
            }
        )
    manifest = public / "manifest.json"
    _write_json(
        manifest,
        {
            "schema": MANIFEST_SCHEMA,
            "task_count": len(TASK_IDS),
            "seed_commitment_sha256": secret_commitment,
            "tasks": task_records,
        },
    )
    endpoint = repo / "evidence/endpoint.json"
    _write_json(endpoint, {"status": "healthy"})
    provider = repo / "evidence/provider-seal-v3.json"
    _write_json(
        provider,
        {
            "schema": PROVIDER_SEAL_SCHEMA,
            "endpoint_health_query_precedes_seal": True,
            "sealed_before_completion_calls": True,
            "study_protocol_sha256": sha256_file(protocol),
            "method_seal_sha256": sha256_file(method),
            "custody_seal_sha256": sha256_file(custody),
            "task_secret_commitment_sha256": secret_commitment,
            "hidden_target_manifest_sha256": "3" * 64,
            "public_manifest": {
                "path": manifest.relative_to(repo).as_posix(),
                "sha256": sha256_file(manifest),
            },
            "endpoint_health_record": {
                "path": endpoint.relative_to(repo).as_posix(),
                "sha256": sha256_file(endpoint),
            },
            "tasks": sealed_records,
        },
    )
    return {
        "repo": repo,
        "protocol": protocol,
        "method": method,
        "method_sha": sha256_file(method),
        "custody": custody,
        "custody_sha": sha256_file(custody),
        "provider": provider,
        "provider_sha": sha256_file(provider),
        "runs": repo / "runs",
        "public": public,
        "first_task": public / "blind-v3-01.json",
    }


def _validate(
    fixture: dict[str, Path | str],
    *,
    task_id: str = "blind-v3-01",
    arm: str = "grammar-random",
):
    return runner.validate_and_build(
        repo_root=Path(fixture["repo"]),
        protocol_path=Path(fixture["protocol"]),
        method_seal_path=Path(fixture["method"]),
        expected_method_seal_sha256=str(fixture["method_sha"]),
        custody_seal_path=Path(fixture["custody"]),
        expected_custody_seal_sha256=str(fixture["custody_sha"]),
        provider_seal_path=Path(fixture["provider"]),
        expected_provider_seal_sha256=str(fixture["provider_sha"]),
        task_id=task_id,
        arm=arm,
        runs_root=Path(fixture["runs"]),
        base_url="http://127.0.0.1:18000/v1",
        python_executable="python",
    )


def test_random_preflight_derives_exact_29_slot_zero_provider_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    command, record, invocation = _validate(fixture)
    assert command[command.index("--proposal-source") + 1] == "grammar-random"
    assert "--random-shortlist-seed" in command
    frozen = invocation["runs"][0]
    assert frozen["logical_execution_cap"] == 29
    assert frozen["provider_call_cap"] == 0
    assert record["arm"] == "grammar-random"
    assert not Path(record["output"]).exists()


def _command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_every_task_and_arm_derived_invocation_passes_underlying_harness_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regress the exact r3 failure across the complete 12-by-2 launch matrix."""

    fixture = _fixture(tmp_path, monkeypatch)
    repo = Path(fixture["repo"])
    monkeypatch.setattr(
        harness,
        "__file__",
        str(repo / "research/evidence_shortlist_smc.py"),
    )
    monkeypatch.setattr(
        runner,
        "_validate_order",
        lambda runs_root, task_id, arm: runs_root.resolve() / task_id / arm,
    )

    required_run_fields = {
        "epsilon",
        "evidence_scale",
        "start_seed",
        "parent_count",
        "offspring_per_parent",
        "first_round_offspring",
        "provider_seed",
        "sample_seed",
        "resample_seed",
        "max_concurrency",
        "exact_reference_limit",
        "proposal_source",
        "random_shortlist_seed",
        "logical_execution_cap",
        "provider_call_cap",
        "terminal_weighted_particles",
    }
    for task_id in TASK_IDS:
        for arm in runner.ARMS:
            command, record, invocation = _validate(
                fixture,
                task_id=task_id,
                arm=arm,
            )
            frozen = invocation["runs"][0]
            assert required_run_fields <= frozen.keys()
            assert frozen["epsilon"] == 0.05
            assert frozen["evidence_scale"] == 2.0

            invocation_path = Path(record["derived_invocation_path"])
            _write_json(invocation_path, invocation)
            args = SimpleNamespace(
                study_protocol=invocation_path,
                run_id=_command_value(command, "--run-id"),
                expected_study_protocol_sha256=_command_value(
                    command, "--expected-study-protocol-sha256"
                ),
                task=Path(_command_value(command, "--task")),
                base_url=_command_value(command, "--base-url"),
                model=_command_value(command, "--model"),
                reasoning_effort=_command_value(command, "--reasoning-effort"),
                temperature=float(_command_value(command, "--temperature")),
                max_tokens=int(_command_value(command, "--max-tokens")),
                timeout_seconds=float(
                    _command_value(command, "--timeout-seconds")
                ),
                epsilon=float(_command_value(command, "--epsilon")),
                evidence_scale=float(
                    _command_value(command, "--evidence-scale")
                ),
                start_seed=int(_command_value(command, "--start-seed")),
                parent_count=int(_command_value(command, "--parent-count")),
                offspring_per_parent=int(
                    _command_value(command, "--offspring-per-parent")
                ),
                first_round_offspring=int(
                    _command_value(command, "--first-round-offspring")
                ),
                provider_seed=int(_command_value(command, "--provider-seed")),
                sample_seed=int(_command_value(command, "--sample-seed")),
                resample_seed=int(_command_value(command, "--resample-seed")),
                max_concurrency=int(
                    _command_value(command, "--max-concurrency")
                ),
                exact_reference_limit=int(
                    _command_value(command, "--exact-reference-limit")
                ),
                proposal_source=_command_value(command, "--proposal-source"),
                random_shortlist_seed=(
                    int(_command_value(command, "--random-shortlist-seed"))
                    if "--random-shortlist-seed" in command
                    else None
                ),
                allow_unfrozen_developmental=False,
            )
            validated = harness.validate_frozen_invocation(args)
            assert validated is not None
            assert validated["run_id"] == f"{task_id}-{arm}"
            assert validated["validated_arguments"]["epsilon"] == 0.05
            assert validated["validated_arguments"]["evidence_scale"] == 2.0


def test_preflight_fails_on_public_task_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    task = Path(fixture["first_task"])
    task.write_text(task.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        _validate(fixture)


def test_preflight_refuses_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = Path(fixture["runs"]) / "blind-v3-01/grammar-random"
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        _validate(fixture)

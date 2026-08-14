from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from research.analyze_exedec_deepcoder_ho_debug_benchmark_v1 import (
    _semantic_evaluation,
    _validate_private_oracles,
    _validate_result,
    exact_two_sided_sign_p,
)
from research.run_exedec_deepcoder_ho_debug_benchmark_v1 import (
    ARMS,
    LOGICAL_EXECUTION_CAP,
    PREFLIGHT_SCHEMA,
    SCHEMA,
    SEED_IDS,
    TASK_IDS,
    _index_runs,
    canonical_bytes,
    sha256_file,
    validate_and_build_command,
    validate_protocol,
    validate_public_bundle,
)

PROJECT_ROOT = Path(__file__).parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[2]
PROTOCOL = PROJECT_ROOT / "research/protocol-exedec-deepcoder-ho-debug-benchmark-v1.json"
BUNDLE = WORKSPACE_ROOT / "artifacts/exedec-deepcoder-ho-debug-v1"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def _protocol() -> tuple[dict[str, Any], str]:
    digest = sha256_file(PROTOCOL)
    protocol = validate_protocol(PROJECT_ROOT, PROTOCOL, digest)
    return protocol, digest


def test_frozen_protocol_bundle_and_all_paired_runs_validate() -> None:
    protocol, digest = _protocol()
    assert protocol["schema"] == SCHEMA
    task_paths = validate_public_bundle(
        PROJECT_ROOT,
        BUNDLE,
        cast(dict[str, Any], protocol["debug_bundle"]),
    )
    assert tuple(task_paths) == TASK_IDS
    assert len(_index_runs(protocol)) == len(TASK_IDS) * len(SEED_IDS) * len(ARMS) == 64
    with pytest.raises(ValueError, match="external protocol"):
        validate_protocol(PROJECT_ROOT, PROTOCOL, "0" * 64)
    assert len(digest) == 64


def test_preflight_commands_are_paired_and_never_expose_oracle(tmp_path: Path) -> None:
    _, digest = _protocol()
    common = {
        "repo_root": PROJECT_ROOT,
        "protocol_path": PROTOCOL,
        "expected_protocol_sha256": digest,
        "bundle_dir": BUNDLE,
        "runs_root": tmp_path / "runs",
        "task_id": TASK_IDS[0],
        "seed_id": 0,
        "python_executable": "/bound/python",
    }
    llm_command, llm_record = validate_and_build_command(arm="llm-smc", **common)
    random_command, random_record = validate_and_build_command(
        arm="grammar-random",
        **common,
    )

    assert llm_record["private_oracle_opened"] is False
    assert random_record["private_oracle_opened"] is False
    assert llm_record["logical_execution_cap"] == LOGICAL_EXECUTION_CAP
    assert llm_record["provider_call_cap"] == 7
    assert random_record["provider_call_cap"] == 0
    assert "--random-shortlist-seed" not in llm_command
    assert "--random-shortlist-seed" in random_command
    assert not any("oracle" in value or "/private/" in value for value in llm_command)
    for flag in ("--start-seed", "--sample-seed", "--resample-seed"):
        assert llm_command[llm_command.index(flag) + 1] == random_command[
            random_command.index(flag) + 1
        ]


@pytest.mark.parametrize(
    ("llm_wins", "grammar_wins", "expected"),
    [
        (0, 0, 1.0),
        (4, 0, 0.125),
        (3, 1, 0.625),
        (5, 5, 1.0),
    ],
)
def test_exact_two_sided_sign_test(
    llm_wins: int,
    grammar_wins: int,
    expected: float,
) -> None:
    assert exact_two_sided_sign_p(llm_wins, grammar_wins) == pytest.approx(expected)


def test_hidden_semantic_evaluation_checks_every_executed_slot() -> None:
    public_task = BUNDLE / "public/exedec-debug-0131.json"
    oracle = json.loads(
        (BUNDLE / "private/exedec-debug-0131.oracle.json").read_text(encoding="utf-8")
    )
    wrong = ("lt(item,0)", "item")
    exact = ("lt(item,0)", "sub(0,item)")
    keys = [wrong] * LOGICAL_EXECUTION_CAP
    keys[16] = exact
    result = _semantic_evaluation(
        public_task=public_task,
        oracle=oracle,
        program_keys=keys,
    )
    assert result["found_hidden_semantic_exact"] is True
    assert result["first_hidden_semantic_exact_slot"] == 17
    assert result["best_hidden_semantic_loss"] == 0
    assert result["unique_executed_programs"] == 2
    assert cast(int, result["hidden_probe_count"]) > 100


def _fake_run_artifact(
    root: Path,
    *,
    run: dict[str, Any],
    protocol_sha256: str,
    manifest_sha256: str,
    task_sha256: str,
) -> None:
    preflight = root.with_name(root.name + ".preflight.json")
    _write_json(
        preflight,
        {
            "schema": PREFLIGHT_SCHEMA,
            "status": "validated-before-debug-run",
            "classification": "debug-only-public-released-task-not-confirmatory",
            "protocol_sha256": protocol_sha256,
            "bundle_manifest_sha256": manifest_sha256,
            "public_task_sha256": task_sha256,
            "private_oracle_opened": False,
            "task_id": run["task_id"],
            "seed_id": run["seed_id"],
            "arm": run["arm"],
            "command": ["python", "public/task.json"],
        },
    )
    key = {"predicate": "lt(item,0)", "mapper": "sub(0,item)"}
    executions = [
        {
            "slot": slot,
            **({"program": key} if slot == 1 else {"sampled_program": key}),
        }
        for slot in range(1, LOGICAL_EXECUTION_CAP + 1)
    ]
    _write_json(root / "executions.json", executions)
    provider_seal = {"inventory_sha256": "empty", "records": []}
    _write_json(root / "provider-seal.json", provider_seal)
    rounds = []
    for index, proposals in enumerate((4, 8, 8, 8)):
        record: dict[str, object] = {
            "round": index + 1,
            "proposal_count": proposals,
            "parent_count": 1 if index == 0 else 2,
        }
        if index < 3:
            record["systematic_resample_ancestors"] = [0, 0]
        rounds.append(record)
    _write_json(
        root / "result.json",
        {
            "schema": "evidence-shortlist-product-path-smc-result-v1",
            "protocol": {
                "task_sha256": task_sha256,
                "proposal_source": run["proposal_source"],
                "epsilon": run["epsilon"],
                "evidence_scale": run["evidence_scale"],
                "early_stop": False,
                "proposal_slot_checkpoints": [1, 5, 13, 21, 29],
                "logical_complete_program_execution_cap": 29,
                "maximum_provider_calls": run["provider_call_cap"],
                "frozen_invocation": {
                    "run_id": run["id"],
                    "study_protocol_sha256": protocol_sha256,
                },
            },
            "search": {
                "logical_complete_program_executions": 29,
                "provider_calls": 0,
                "found_exact": True,
                "first_exact_slot": 1,
                "best_score": {
                    "exact_program": True,
                    "total_loss": 0,
                },
                "physical_scorer_calls_before_reference": 1,
            },
            "rounds": rounds,
            "inference": {
                "final_particles": [{} for _ in range(8)],
                "final_ess": 8.0,
                "final_relative_ess": 1.0,
                "self_normalized_exact_mass": 1.0,
                "self_normalized_target_mean_loss": 0.0,
            },
            "exact_reference": {
                "post_provider_exhaustive_programs": 15732,
                "exact_target_mass": 0.5,
                "target_mean_loss": 1.0,
            },
            "provider_inventory_sha256": "empty",
        },
    )


def test_result_validator_enforces_public_before_private_contract(tmp_path: Path) -> None:
    protocol, protocol_sha256 = _protocol()
    run = _index_runs(protocol)["exedec-debug-0131--seed-00--grammar-random"]
    run_dir = tmp_path / "grammar-random/exedec-debug-0131/seed-00"
    manifest_sha256 = sha256_file(BUNDLE / "manifest.json")
    task_sha256 = sha256_file(BUNDLE / "public/exedec-debug-0131.json")
    _fake_run_artifact(
        run_dir,
        run=run,
        protocol_sha256=protocol_sha256,
        manifest_sha256=manifest_sha256,
        task_sha256=task_sha256,
    )
    record, keys = _validate_result(
        run_dir=run_dir,
        task_id="exedec-debug-0131",
        seed_id=0,
        arm="grammar-random",
        run=run,
        protocol_sha256=protocol_sha256,
        bundle_manifest_sha256=manifest_sha256,
        task_sha256=task_sha256,
    )
    assert record["provider_calls"] == 0
    assert len(keys) == 29

    preflight = run_dir.with_name(run_dir.name + ".preflight.json")
    bad = json.loads(preflight.read_text(encoding="utf-8"))
    bad["command"].append(str(BUNDLE / "private/exedec-debug-0131.oracle.json"))
    _write_json(preflight, bad)
    with pytest.raises(ValueError, match="exposes a private"):
        _validate_result(
            run_dir=run_dir,
            task_id="exedec-debug-0131",
            seed_id=0,
            arm="grammar-random",
            run=run,
            protocol_sha256=protocol_sha256,
            bundle_manifest_sha256=manifest_sha256,
            task_sha256=task_sha256,
        )


def test_private_oracles_are_bound_and_debug_only() -> None:
    protocol, _ = _protocol()
    oracles = _validate_private_oracles(
        repo_root=PROJECT_ROOT,
        bundle_dir=BUNDLE,
        bundle_binding=cast(dict[str, Any], protocol["debug_bundle"]),
    )
    assert tuple(oracles) == TASK_IDS
    for oracle, digest in oracles.values():
        assert oracle["classification"] == "debug-only-public-released-target-not-confirmatory"
        assert len(digest) == 64

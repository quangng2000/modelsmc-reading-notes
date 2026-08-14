from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest

from research.analyze_blind_evidence_frontier_v2 import (
    TASK_IDS,
    TRIALS,
    analyze_blind_evidence_frontier_v2,
    canonical_bytes,
    render_markdown,
    sha256_bytes,
    sha256_file,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _bundle(repo: Path, domain: str, relative_paths: list[str]) -> dict[str, object]:
    material = bytearray(domain.encode() + b"\0")
    source_sha256 = {}
    for relative in relative_paths:
        payload = (repo / relative).read_bytes()
        source_sha256[relative] = sha256_bytes(payload)
        material.extend(relative.encode() + b"\0" + payload + b"\0")
    return {
        "scheme": "SHA256(domain || NUL || repeated(path || NUL || exact_file_bytes || NUL))",
        "domain": domain,
        "paths_in_order": relative_paths,
        "source_sha256": source_sha256,
        "bundle_sha256": sha256_bytes(bytes(material)),
    }


def _state(round_number: int, *, exact: bool) -> dict[str, object]:
    mapper = list(range(-3, 5))
    if not exact:
        mapper[0] = 99
    return {
        "state_id": f"state-{round_number}",
        "predicate": "predicate",
        "mapper": "mapper",
        "score": {"total_loss": 0.0 if exact else 1.0, "exact_program": exact},
        "history": [],
        "singleton_constraint_violations": {
            "predicate": 0,
            "mapper": 0 if exact else 1,
        },
        "finite_component_semantics": {
            "predicate_keep_mask": [True] * 8,
            "mapper_values": mapper,
        },
    }


def _make_bundle(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, dict[str, Path]]:
    repo = tmp_path / "repo"
    research = repo / "research"
    research.mkdir(parents=True)
    analyzer_source = Path(analyze_blind_evidence_frontier_v2.__code__.co_filename)
    files = {
        "research/generate_blinded_filter_map_tasks.py": b"# generator\n",
        "research/iterative_beam_experiment.py": b"# harness\n",
        "research/analyze_blind_evidence_frontier_v2.py": analyzer_source.read_bytes(),
        "research/run_blind_evidence_frontier_v2.py": b"# runner\n",
        "research/prompt.py": b"# prompt\n",
        "research/test_source.py": b"# tests\n",
    }
    for relative, payload in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    prompt = _bundle(repo, "prompt-v2", ["research/prompt.py"])
    tests = _bundle(repo, "tests-v2", ["research/test_source.py"])
    secret = "a" * 64
    seed_records = []
    for index, task_id in enumerate(TASK_IDS):
        base = 301_000 + index * 1_000
        seed_records.append(
            {
                "task_id": task_id,
                "provider_seed": base,
                "tie_seed": base + 100,
                "matched_random_seed": base + 200,
            }
        )
    study = {
        "schema": "blind-evidence-frontier-confirmation-v2",
        "protocol_status": "method-frozen-before-task-generation",
        "freeze_requirements": {
            "protocol_sha256_binding": "external-provider-seal-and-run-protocol-records",
            "generator": {
                "path": "research/generate_blinded_filter_map_tasks.py",
                "sha256": sha256_file(repo / "research/generate_blinded_filter_map_tasks.py"),
            },
            "harness": {
                "path": "research/iterative_beam_experiment.py",
                "sha256": sha256_file(repo / "research/iterative_beam_experiment.py"),
            },
            "analysis": {
                "path": "research/analyze_blind_evidence_frontier_v2.py",
                "sha256": sha256_file(repo / "research/analyze_blind_evidence_frontier_v2.py"),
            },
            "runner": {
                "path": "research/run_blind_evidence_frontier_v2.py",
                "sha256": sha256_file(repo / "research/run_blind_evidence_frontier_v2.py"),
            },
            "prompt_template_sha256": prompt["bundle_sha256"],
            "prompt_template_binding": prompt,
            "test_bundle": tests,
            "task_secret_commitment_sha256": secret,
            "run_seeds": seed_records,
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
    }
    study_path = research / "protocol-blind-evidence-frontier-v2.json"
    _write_json(study_path, study)
    study_sha256 = sha256_file(study_path)
    method_path = repo / "method-seal.json"
    _write_json(
        method_path,
        {
            "schema": "blind-evidence-frontier-method-seal-v2",
            "sealed_before_task_generation": True,
            "protocol": {
                "path": study_path.relative_to(repo).as_posix(),
                "sha256": study_sha256,
            },
        },
    )
    method_sha256 = sha256_file(method_path)

    public = repo / "suite" / "public"
    public.mkdir(parents=True)
    manifest_tasks = []
    task_documents = {}
    for task_id in TASK_IDS:
        task = {
            "name": task_id,
            "signature": {"input": "List<Int>", "output": "List<Int>"},
            "examples": [
                {"input": [str(item)], "output": [str(item)]}
                for item in range(-3, 5)
            ],
            "integerConstants": [str(item) for item in range(-3, 5)],
        }
        task_path = public / f"{task_id}.json"
        _write_json(task_path, task)
        record = {
            "task_id": task_id,
            "path": task_path.name,
            "task_file_sha256": sha256_file(task_path),
            "task_canonical_sha256": sha256_bytes(canonical_bytes(task)),
            "target_commitment_sha256": sha256_bytes(f"target-{task_id}".encode()),
        }
        manifest_tasks.append(record)
        task_documents[task_id] = record
    manifest = {
        "schema": "blinded-filter-map-suite-v2",
        "task_count": 12,
        "seed_commitment_sha256": secret,
        "task_generation": {},
        "tasks": manifest_tasks,
    }
    manifest_path = public / "manifest.json"
    _write_json(manifest_path, manifest)

    seal_tasks = []
    for record, seeds in zip(manifest_tasks, seed_records, strict=True):
        seal_tasks.append(
            record
            | {
                "path": (public / cast(str, record["path"])).relative_to(repo).as_posix(),
                **{key: value for key, value in seeds.items() if key != "task_id"},
            }
        )
    endpoint_path = repo / "endpoint-health.json"
    _write_json(endpoint_path, {"status": "healthy"})
    seal = {
        "schema": "blind-v2-provider-call-seal-v1",
        "sealed_before_provider_calls": True,
        "study_protocol_sha256": study_sha256,
        "method_seal_sha256": method_sha256,
        "public_manifest": {
            "path": manifest_path.relative_to(repo).as_posix(),
            "sha256": sha256_file(manifest_path),
        },
        "hidden_target_manifest_sha256": "b" * 64,
        "task_secret_commitment_sha256": secret,
        "endpoint_health_record": {
            "path": endpoint_path.relative_to(repo).as_posix(),
            "sha256": sha256_file(endpoint_path),
        },
        "tasks": seal_tasks,
    }
    seal_path = repo / "provider-call-seal.json"
    _write_json(seal_path, seal)
    seal_sha256 = sha256_file(seal_path)

    run_dirs = {}
    for task_index, (task_id, task_record, seeds) in enumerate(
        zip(TASK_IDS, manifest_tasks, seed_records, strict=True)
    ):
        run_dir = repo / "runs" / task_id
        run_dirs[task_id] = run_dir
        provider = run_dir / "provider" / "round-01" / "response.json"
        response = canonical_bytes(
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                }
            }
        )
        provider.parent.mkdir(parents=True)
        provider.write_bytes(response)
        inventory_records = [
            {
                "path": provider.relative_to(run_dir).as_posix(),
                "bytes": len(response),
                "sha256": sha256_bytes(response),
            }
        ]
        inventory_sha256 = sha256_bytes(canonical_bytes(inventory_records))
        _write_json(
            run_dir / "provider-seal.json",
            {"records": inventory_records, "inventory_sha256": inventory_sha256},
        )
        exact = task_index < 6
        first_exact = 18 if exact else None
        completed = 3 if exact else 4
        protocol = {
            "protocol_mode": None,
            "blind_task_id": None,
            "task_sha256": task_record["task_file_sha256"],
            "study_protocol_sha256": study_sha256,
            "blind_manifest_sha256": sha256_file(manifest_path),
            "harness_sha256": sha256_file(repo / "research/iterative_beam_experiment.py"),
            "model": "gpt-oss-120b",
            "model_revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
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
            "proposal_slot_checkpoints": [1, 5, 13, 21, 29],
            "provider_seed": seeds["provider_seed"],
            "tie_seed": seeds["tie_seed"],
            "random_baseline_seed": seeds["matched_random_seed"],
        }
        _write_json(run_dir / "protocol.json", protocol)
        preflight_path = run_dir.with_name(run_dir.name + ".preflight.json")
        command = [
            "/python",
            "-m",
            "research.iterative_beam_experiment",
            "--task",
            str((public / f"{task_id}.json").resolve()),
            "--output",
            str(run_dir.resolve()),
            "--blind-manifest",
            str(manifest_path.resolve()),
            "--study-protocol",
            str(study_path.resolve()),
            "--base-url",
            "http://127.0.0.1:18000/v1",
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
        _write_json(
            preflight_path,
            {
                "schema": "blind-v2-preflight-invocation-v1",
                "status": "validated-before-provider-call",
                "task_id": task_id,
                "study_protocol_sha256": study_sha256,
                "method_seal_sha256": method_sha256,
                "provider_call_seal_sha256": seal_sha256,
                "public_manifest_sha256": sha256_file(manifest_path),
                "task_file_sha256": task_record["task_file_sha256"],
                "harness_sha256": sha256_file(
                    repo / "research/iterative_beam_experiment.py"
                ),
                "command": command,
                "preflight_record_path": str(preflight_path.resolve()),
            },
        )
        for round_number in range(1, completed + 1):
            _write_json(
                run_dir / f"round-{round_number:02d}.json",
                {
                    "round": round_number,
                    "selected_beam": [
                        _state(round_number, exact=exact and round_number == completed)
                    ],
                },
            )
        random_success = task_index == 0
        trials = [
            {
                "success": random_success and trial_index == 0,
                "first_exact_execution": 18 if random_success and trial_index == 0 else None,
            }
            for trial_index in range(TRIALS)
        ]
        _write_json(run_dir / "matched-random-trials.json", trials)
        successes = 1 if random_success else 0
        metrics = {
            "success": exact,
            "first_exact_proposal_slot": first_exact,
            "success_by_proposal_slot_checkpoint": {
                "21": exact,
                "29": exact,
            },
            "best_loss_by_proposal_slot_checkpoint": {
                "21": 0.0 if exact else 1.0,
                "29": 0.0 if exact else 1.0,
            },
            "rounds_completed": completed,
            "proposal_slots_consumed": 21 if exact else 29,
            "logical_candidate_evaluations": 21 if exact else 29,
            "physical_scorer_calls": 20 if exact else 28,
            "score_cache_hits": 1,
            "unique_joint_semantic_cells_evaluated": 10,
            "semantic_redundancy_rate": 0.5,
            "provider_calls_attempted": 5 if exact else 7,
            "provider_calls_valid": 5 if exact else 7,
            "provider_calls_invalid": 0,
        }
        result = {
            "schema": "iterative-typed-llm-beam-experiment-v2",
            "protocol": protocol,
            "provider_inventory_sha256": inventory_sha256,
            "llm_beam_metrics": metrics,
            "matched_random_beam": {
                "trials": TRIALS,
                "successes_by_proposal_slot_checkpoint": {
                    "21": successes,
                    "29": successes,
                },
                "best_loss_summary_by_proposal_slot_checkpoint": {
                    "21": {"observed_trials": TRIALS},
                    "29": {"observed_trials": TRIALS},
                },
            },
            "timing": {"total_seconds": 1.0},
        }
        _write_json(run_dir / "result.json", result)
    return repo, study_path, method_path, seal_path, manifest_path, run_dirs


def test_valid_bundle_computes_exact_trial_index_randomization_test(tmp_path: Path) -> None:
    repo, study, method, seal, manifest, runs = _make_bundle(tmp_path)

    analysis = analyze_blind_evidence_frontier_v2(
        repo,
        study,
        method,
        seal,
        manifest,
        runs,
        expected_method_seal_sha256=sha256_file(method),
        expected_provider_call_seal_sha256=sha256_file(seal),
    )

    assert analysis["confirmatory"] is True
    assert analysis["integrity"]["passed"] is True
    assert len(analysis["tasks"]) == 12
    primary = analysis["endpoints"]["29"]
    assert primary["llm_exact_successes"] == 6
    assert primary["matched_random_aggregate_success_count_by_trial_index"][0] == 1
    assert primary["randomization_exceedances"] == 0
    assert primary["randomization_p_value"] == pytest.approx(1 / 10001)
    assert analysis["primary_decision"]["study_success"] is True
    assert "Every preregistered task" in render_markdown(analysis)


def test_rejects_provider_tampering_before_confirmatory_label(tmp_path: Path) -> None:
    repo, study, method, seal, manifest, runs = _make_bundle(tmp_path)
    response = runs[TASK_IDS[4]] / "provider" / "round-01" / "response.json"
    response.write_bytes(response.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="bytes"):
        analyze_blind_evidence_frontier_v2(
            repo,
            study,
            method,
            seal,
            manifest,
            runs,
            expected_method_seal_sha256=sha256_file(method),
            expected_provider_call_seal_sha256=sha256_file(seal),
        )


def test_rejects_wrong_task_seed_and_singleton_violation_count(tmp_path: Path) -> None:
    repo, study, method, seal, manifest, runs = _make_bundle(tmp_path)
    protocol_path = runs[TASK_IDS[3]] / "protocol.json"
    result_path = runs[TASK_IDS[3]] / "result.json"
    protocol = _read(protocol_path)
    result = _read(result_path)
    protocol["tie_seed"] = 1
    result["protocol"] = protocol
    _write_json(protocol_path, protocol)
    _write_json(result_path, result)
    with pytest.raises(ValueError, match="tie_seed"):
        analyze_blind_evidence_frontier_v2(
            repo,
            study,
            method,
            seal,
            manifest,
            runs,
            expected_method_seal_sha256=sha256_file(method),
            expected_provider_call_seal_sha256=sha256_file(seal),
        )

    repo, study, method, seal, manifest, runs = _make_bundle(tmp_path / "second")
    round_path = runs[TASK_IDS[8]] / "round-01.json"
    round_record = copy.deepcopy(_read(round_path))
    round_record["selected_beam"][0]["singleton_constraint_violations"]["mapper"] = 0
    _write_json(round_path, round_record)
    with pytest.raises(ValueError, match="mapper violations"):
        analyze_blind_evidence_frontier_v2(
            repo,
            study,
            method,
            seal,
            manifest,
            runs,
            expected_method_seal_sha256=sha256_file(method),
            expected_provider_call_seal_sha256=sha256_file(seal),
        )


def test_rejects_trial_index_or_stage_two_binding_drift(tmp_path: Path) -> None:
    repo, study, method, seal, manifest, runs = _make_bundle(tmp_path)
    trials_path = runs[TASK_IDS[0]] / "matched-random-trials.json"
    trials = json.loads(trials_path.read_text(encoding="utf-8"))
    trials.pop()
    _write_json(trials_path, trials)
    with pytest.raises(ValueError, match="exactly 10000"):
        analyze_blind_evidence_frontier_v2(
            repo,
            study,
            method,
            seal,
            manifest,
            runs,
            expected_method_seal_sha256=sha256_file(method),
            expected_provider_call_seal_sha256=sha256_file(seal),
        )

    repo, study, method, seal, manifest, runs = _make_bundle(tmp_path / "second")
    seal_record = _read(seal)
    seal_record["tasks"][0]["task_file_sha256"] = "f" * 64
    _write_json(seal, seal_record)
    with pytest.raises(ValueError, match="task_file_sha256"):
        analyze_blind_evidence_frontier_v2(
            repo,
            study,
            method,
            seal,
            manifest,
            runs,
            expected_method_seal_sha256=sha256_file(method),
            expected_provider_call_seal_sha256=sha256_file(seal),
        )

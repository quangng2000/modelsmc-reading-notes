from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.analyze_posthoc_evidence_frontier import (
    LABEL,
    TASK_IDS,
    analyze_posthoc_evidence_frontier,
    canonical_bytes,
    render_markdown,
    sha256_bytes,
    sha256_file,
    write_analysis,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def _state(
    round_number: int,
    index: int,
    *,
    predicate_violations: int,
    mapper_violations: int,
    loss: float,
    exact: bool = False,
) -> dict[str, object]:
    keep_mask = [False] * predicate_violations + [True] * (4 - predicate_violations)
    mapper_values = [99 if item < mapper_violations else item + 10 for item in range(4)]
    return {
        "state_id": f"state-{round_number}-{index}",
        "predicate": f"p-{round_number}-{index}",
        "mapper": f"m-{round_number}-{index}",
        "score": {"total_loss": loss, "exact_program": exact},
        "history": [
            {
                "round": round_number,
                "hole": "mapper" if exact else "predicate",
                "repair": "repair",
                "loss_after": loss,
            }
        ],
        "singleton_constraint_violations": {
            "predicate": predicate_violations,
            "mapper": mapper_violations,
        },
        "finite_component_semantics": {
            "predicate_keep_mask": keep_mask,
            "mapper_values": mapper_values,
        },
    }


def _make_bundle(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    repo = tmp_path / "repo"
    harness = repo / "research" / "iterative_beam_experiment.py"
    harness.parent.mkdir(parents=True)
    harness.write_text("# frozen harness\n", encoding="utf-8")
    analyzer_path = Path(analyze_posthoc_evidence_frontier.__code__.co_filename)

    records = []
    seeds = {
        "blind-02": (202000, 202100, 202200),
        "blind-03": (203000, 203100, 203200),
    }
    for task_id in TASK_IDS:
        task_path = repo / "artifacts" / "suite" / f"{task_id}.json"
        _write_json(
            task_path,
            {
                "name": task_id,
                "signature": {"input": "List<Int>", "output": "List<Int>"},
                "examples": [
                    {"input": [item], "output": [item + 10]} for item in range(4)
                ],
                "integerConstants": [-1, 0, 1],
            },
        )
        provider_seed, tie_seed, random_seed = seeds[task_id]
        records.append(
            {
                "task_id": task_id,
                "path": task_path.relative_to(repo).as_posix(),
                "sha256": sha256_file(task_path),
                "provider_seed": provider_seed,
                "tie_seed": tie_seed,
                "matched_random_seed": random_seed,
            }
        )

    study_path = repo / "research" / "developmental-protocol.json"
    study = {
        "schema": "post-hoc-developmental-evidence-frontier-v1",
        "protocol_status": "developmental-post-hoc-frozen",
        "study_classification": {"blind": False, "confirmatory": False},
        "bindings_to_original_study": {"tasks_in_fixed_run_order": records},
        "freeze_before_followup_calls": {
            "harness_path": harness.relative_to(repo).as_posix(),
            "harness_sha256": sha256_file(harness),
            "analysis_code_sha256": sha256_file(analyzer_path),
            "prompt_template_sha256": "a" * 64,
        },
    }
    _write_json(study_path, study)
    study_sha256 = sha256_file(study_path)

    run_dirs: dict[str, Path] = {}
    first_exact = {"blind-02": 18, "blind-03": 25}
    completed = {"blind-02": 3, "blind-03": 4}
    for record in records:
        task_id = str(record["task_id"])
        run_dir = repo / "runs" / task_id
        run_dirs[task_id] = run_dir
        run_dir.mkdir(parents=True)
        provider_path = run_dir / "provider" / "round-01" / "request.json"
        provider_payload = canonical_bytes({"public_task": task_id})
        provider_path.parent.mkdir(parents=True)
        provider_path.write_bytes(provider_payload)
        inventory_records = [
            {
                "path": provider_path.relative_to(run_dir).as_posix(),
                "bytes": len(provider_payload),
                "sha256": sha256_bytes(provider_payload),
            }
        ]
        inventory_sha256 = sha256_bytes(canonical_bytes(inventory_records))
        _write_json(
            run_dir / "provider-seal.json",
            {"records": inventory_records, "inventory_sha256": inventory_sha256},
        )
        protocol = {
            "task_sha256": record["sha256"],
            "study_protocol_sha256": study_sha256,
            "harness_sha256": sha256_file(harness),
            "selection_policy": "evidence-frontier",
            "branching_factor": 4,
            "beam_width": 2,
            "rounds": 4,
            "maximum_proposal_slot_budget": 29,
            "maximum_provider_calls": 7,
            "proposal_slot_checkpoints": [1, 5, 13, 21, 29],
            "start_seed": 17,
            "random_baseline_trials": 10000,
            "singleton_evidence": True,
            "stall_policy": "alternate-hole",
            "primary_checkpoint_round": 4,
            "provider_seed": record["provider_seed"],
            "tie_seed": record["tie_seed"],
            "random_baseline_seed": record["matched_random_seed"],
        }
        _write_json(run_dir / "protocol.json", protocol)
        exact_at = first_exact[task_id]
        metrics = {
            "success": True,
            "first_exact_proposal_slot": exact_at,
            "success_by_proposal_slot_checkpoint": {
                "21": exact_at <= 21,
                "29": exact_at <= 29,
            },
            "best_loss_by_proposal_slot_checkpoint": {
                "21": 0.0 if exact_at <= 21 else 2.0,
                "29": 0.0,
            },
            "rounds_completed": completed[task_id],
        }
        result = {
            "schema": "iterative-typed-llm-beam-experiment-v2",
            "protocol": protocol,
            "provider_inventory_sha256": inventory_sha256,
            "llm_beam_metrics": metrics,
        }
        _write_json(run_dir / "result.json", result)

        for round_number in range(1, completed[task_id] + 1):
            is_final = round_number == completed[task_id]
            selected = [
                _state(
                    round_number,
                    0,
                    predicate_violations=max(0, 4 - round_number),
                    mapper_violations=4,
                    loss=float(10 - round_number),
                ),
                _state(
                    round_number,
                    1,
                    predicate_violations=0,
                    mapper_violations=0 if is_final else max(1, 4 - round_number),
                    loss=0.0 if is_final else float(20 - round_number),
                    exact=is_final,
                ),
            ]
            _write_json(
                run_dir / f"round-{round_number:02d}.json",
                {"round": round_number, "selected_beam": selected},
            )
    return repo, study_path, run_dirs


def test_analyzer_validates_bindings_endpoints_and_violation_trajectories(
    tmp_path: Path,
) -> None:
    repo, study_path, run_dirs = _make_bundle(tmp_path)

    analysis = analyze_posthoc_evidence_frontier(repo, study_path, run_dirs)

    assert analysis["label"] == LABEL
    assert analysis["blind"] is False
    assert analysis["confirmatory"] is False
    assert analysis["developmental_gate"]["passed"] is True
    tasks = {task["task_id"]: task for task in analysis["tasks"]}
    assert tasks["blind-02"]["exact_by_slot"] == {"21": True, "29": True}
    assert tasks["blind-03"]["exact_by_slot"] == {"21": False, "29": True}
    assert tasks["blind-02"]["selected_state_violation_trajectory"][-1]["exact"] is True
    assert all(task["label"] == LABEL for task in analysis["tasks"])

    markdown = render_markdown(analysis)
    assert markdown.startswith(f"# {LABEL}")
    assert markdown.count(LABEL) >= 3
    json_path, markdown_path = write_analysis(tmp_path / "report", analysis)
    assert json_path.name == "POST_HOC_NON_BLIND_ANALYSIS.json"
    assert markdown_path.name == "POST_HOC_NON_BLIND_SUMMARY.md"
    assert LABEL in json_path.read_text(encoding="utf-8")
    assert LABEL in markdown_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("field", "wrong", "match"),
    [
        ("selection_policy", "semantic-diverse", "selection_policy"),
        ("branching_factor", 5, "branching_factor"),
        ("beam_width", 3, "beam_width"),
        ("rounds", 5, "rounds"),
        ("provider_seed", 999, "provider_seed"),
    ],
)
def test_analyzer_rejects_search_or_seed_drift(
    tmp_path: Path,
    field: str,
    wrong: object,
    match: str,
) -> None:
    repo, study_path, run_dirs = _make_bundle(tmp_path)
    run_dir = run_dirs["blind-02"]
    protocol = _read_json(run_dir / "protocol.json")
    result = _read_json(run_dir / "result.json")
    protocol[field] = wrong
    result["protocol"] = protocol
    _write_json(run_dir / "protocol.json", protocol)
    _write_json(run_dir / "result.json", result)

    with pytest.raises(ValueError, match=match):
        analyze_posthoc_evidence_frontier(repo, study_path, run_dirs)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_analyzer_rejects_task_harness_and_study_hash_drift(tmp_path: Path) -> None:
    repo, study_path, run_dirs = _make_bundle(tmp_path)
    task = repo / "artifacts" / "suite" / "blind-03.json"
    task.write_bytes(task.read_bytes() + b" ")
    with pytest.raises(ValueError, match=r"blind-03\.task SHA-256"):
        analyze_posthoc_evidence_frontier(repo, study_path, run_dirs)

    repo, study_path, run_dirs = _make_bundle(tmp_path / "second")
    (repo / "research" / "iterative_beam_experiment.py").write_text(
        "# changed harness\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="frozen harness SHA-256"):
        analyze_posthoc_evidence_frontier(repo, study_path, run_dirs)

    repo, study_path, run_dirs = _make_bundle(tmp_path / "third")
    study = _read_json(study_path)
    study["note"] = "changed after the run"
    _write_json(study_path, study)
    with pytest.raises(ValueError, match="study protocol SHA-256"):
        analyze_posthoc_evidence_frontier(repo, study_path, run_dirs)


def test_analyzer_rejects_provider_tampering_and_inconsistent_exact_checkpoint(
    tmp_path: Path,
) -> None:
    repo, study_path, run_dirs = _make_bundle(tmp_path)
    provider = run_dirs["blind-02"] / "provider" / "round-01" / "request.json"
    provider.write_bytes(provider.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="bytes"):
        analyze_posthoc_evidence_frontier(repo, study_path, run_dirs)

    repo, study_path, run_dirs = _make_bundle(tmp_path / "second")
    result_path = run_dirs["blind-03"] / "result.json"
    result = copy.deepcopy(_read_json(result_path))
    result["llm_beam_metrics"]["success_by_proposal_slot_checkpoint"]["21"] = True
    _write_json(result_path, result)
    with pytest.raises(ValueError, match=r"success\[21\]"):
        analyze_posthoc_evidence_frontier(repo, study_path, run_dirs)

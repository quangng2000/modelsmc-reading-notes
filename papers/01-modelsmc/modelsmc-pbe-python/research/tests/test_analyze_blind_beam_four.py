from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.analyze_blind_beam_four import (
    RUN_SEEDS,
    TASK_IDS,
    analyze_blind_beam_four,
    canonical_bytes,
    poisson_binomial_distribution,
    poisson_binomial_tail,
    render_markdown,
    sha256_bytes,
    wilson_interval,
    write_analysis,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def _random_checkpoint(successes: int, trials: int = 10000) -> dict[str, object]:
    rate = successes / trials
    return {
        "successes": successes,
        "rate": rate,
        "interval": wilson_interval(successes, trials),
        "summary": {
            "observed_trials": trials,
            "mean": 4.0,
            "median": 3.0,
            "p05": 1.0,
            "p95": 8.0,
            "fraction_at_most_llm": 0.25,
        },
    }


def _make_bundle(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    public = tmp_path / "public"
    runs = tmp_path / "runs"
    public.mkdir()
    task_records = []
    task_hashes: dict[str, str] = {}
    for index, task_id in enumerate(TASK_IDS, start=1):
        task = {
            "name": task_id,
            "signature": {"input": "List<Int>", "output": "List<Int>"},
            "examples": [{"input": [], "output": []}],
            "integerConstants": [-3, -2, -1, 0, 1, 2, 3, 4],
        }
        task_path = public / f"{task_id}.json"
        _write_json(task_path, task)
        file_digest = sha256_bytes(task_path.read_bytes())
        canonical_digest = sha256_bytes(canonical_bytes(task))
        task_hashes[task_id] = file_digest
        task_records.append(
            {
                "task_id": task_id,
                "path": task_path.name,
                "task_file_sha256": file_digest,
                "task_canonical_sha256": canonical_digest,
                "target_commitment_sha256": f"{index}" * 64,
            }
        )
    manifest = {
        "schema": "blinded-filter-map-suite-v1",
        "task_count": 4,
        "seed_commitment_sha256": "e" * 64,
        "task_generation": {
            "constants": ["-3", "-2", "-1", "0", "1", "2", "3", "4"],
            "provider_boundary": "task files only",
            "generator_sha256": "f" * 64,
        },
        "tasks": task_records,
    }
    manifest_path = public / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha256 = sha256_bytes(manifest_path.read_bytes())

    first_exact = {
        "blind-01": 4,
        "blind-02": 20,
        "blind-03": 31,
        "blind-04": None,
    }
    losses = {
        "blind-01": {"29": 0.0, "37": 0.0},
        "blind-02": {"29": 0.0, "37": 0.0},
        "blind-03": {"29": 2.0, "37": 0.0},
        "blind-04": {"29": 3.0, "37": 1.0},
    }
    random_29 = (100, 200, 300, 400)
    random_37 = (200, 300, 400, 500)
    run_dirs: dict[str, Path] = {}
    for index, task_id in enumerate(TASK_IDS):
        run_dir = runs / task_id
        run_dir.mkdir(parents=True)
        run_dirs[task_id] = run_dir
        provider_payload = canonical_bytes({"task_id": task_id})
        provider_path = run_dir / "provider" / "round-01" / "request.json"
        provider_path.parent.mkdir(parents=True)
        provider_path.write_bytes(provider_payload)
        inventory_records = [
            {
                "path": "provider/round-01/request.json",
                "bytes": len(provider_payload),
                "sha256": sha256_bytes(provider_payload),
            }
        ]
        inventory_sha256 = sha256_bytes(canonical_bytes(inventory_records))
        _write_json(
            run_dir / "provider-seal.json",
            {"records": inventory_records, "inventory_sha256": inventory_sha256},
        )
        provider_seed, tie_seed, random_seed = RUN_SEEDS[task_id]
        protocol = {
            "protocol_mode": "blind-four-v1",
            "blind_task_id": task_id,
            "task_sha256": task_hashes[task_id],
            "blind_manifest_sha256": manifest_sha256,
            "study_protocol_sha256": "a" * 64,
            "harness_sha256": "b" * 64,
            "model": "gpt-oss-120b",
            "reasoning_effort": "low",
            "temperature": 0.0,
            "rounds": 5,
            "beam_width": 2,
            "branching_factor": 4,
            "maximum_proposal_slot_budget": 37,
            "maximum_provider_calls": 9,
            "selection_policy": "semantic-diverse",
            "singleton_evidence": True,
            "stall_policy": "alternate-hole",
            "start_seed": 17,
            "random_baseline_trials": 10000,
            "primary_checkpoint_round": 4,
            "primary_checkpoint_slot": 29,
            "proposal_slot_checkpoints": [1, 5, 13, 21, 29, 37],
            "provider_seed": provider_seed,
            "tie_seed": tie_seed,
            "random_baseline_seed": random_seed,
        }
        _write_json(run_dir / "protocol.json", protocol)
        exact_at = first_exact[task_id]
        success_29 = exact_at is not None and exact_at <= 29
        success_37 = exact_at is not None and exact_at <= 37
        consumed = (
            5
            if task_id == "blind-01"
            else 21
            if task_id == "blind-02"
            else 37
        )
        checkpoint_29 = _random_checkpoint(random_29[index])
        checkpoint_37 = _random_checkpoint(random_37[index])
        result = {
            "schema": "iterative-typed-llm-beam-experiment-v2",
            "protocol": protocol,
            "provider_inventory_sha256": inventory_sha256,
            "llm_beam_metrics": {
                "success": success_37,
                "first_exact_proposal_slot": exact_at,
                "success_by_proposal_slot_checkpoint": {
                    "29": success_29,
                    "37": success_37,
                },
                "primary_checkpoint_slot": 29,
                "primary_checkpoint_success": success_29,
                "best_loss_by_proposal_slot_checkpoint": losses[task_id],
                "proposal_slots_consumed": consumed,
                "best_loss": losses[task_id]["37"],
            },
            "matched_random_beam": {
                "trials": 10000,
                "successes": checkpoint_37["successes"],
                "success_rate": checkpoint_37["rate"],
                "success_rate_wilson_95": checkpoint_37["interval"],
                "successes_by_proposal_slot_checkpoint": {
                    "29": checkpoint_29["successes"],
                    "37": checkpoint_37["successes"],
                },
                "success_rates_by_proposal_slot_checkpoint": {
                    "29": checkpoint_29["rate"],
                    "37": checkpoint_37["rate"],
                },
                "success_rate_wilson_95_by_proposal_slot_checkpoint": {
                    "29": checkpoint_29["interval"],
                    "37": checkpoint_37["interval"],
                },
                "best_loss_summary_by_proposal_slot_checkpoint": {
                    "29": checkpoint_29["summary"],
                    "37": checkpoint_37["summary"],
                },
            },
        }
        _write_json(run_dir / "result.json", result)
    return manifest_path, run_dirs


def test_poisson_binomial_math_is_exact_for_two_trials() -> None:
    distribution = poisson_binomial_distribution((0.1, 0.2))
    assert distribution == pytest.approx((0.72, 0.26, 0.02))
    assert poisson_binomial_tail((0.1, 0.2), 1) == pytest.approx(0.28)
    assert poisson_binomial_tail((0.1, 0.2), 2) == pytest.approx(0.02)
    assert poisson_binomial_tail((0.1, 0.2), 0) == pytest.approx(1.0)


def test_analyzer_validates_and_aggregates_four_public_runs(tmp_path: Path) -> None:
    manifest_path, run_dirs = _make_bundle(tmp_path)
    private_reveal = tmp_path / "private" / "reveal.json"
    _write_json(private_reveal, {"this": "must not be read", "target": "private"})

    analysis = analyze_blind_beam_four(manifest_path, run_dirs)

    assert analysis["public_only_analysis"] is True
    primary = analysis["checkpoints"]["29"]
    secondary = analysis["checkpoints"]["37"]
    assert primary["llm_exact_successes"] == 2
    assert secondary["llm_exact_successes"] == 3
    assert primary["mean_paired_advantage"] == pytest.approx(0.475)
    assert secondary["mean_paired_advantage"] == pytest.approx(0.715)
    assert primary["poisson_binomial"][
        "tail_probability_random_at_least_observed"
    ] == pytest.approx(poisson_binomial_tail((0.01, 0.02, 0.03, 0.04), 2))
    assert [task["task_id"] for task in analysis["tasks"]] == list(TASK_IDS)
    assert "private/reveal.json" not in canonical_bytes(analysis).decode()

    markdown = render_markdown(analysis)
    assert "LLM exact 2/4" in markdown
    assert "LLM exact 3/4" in markdown
    json_path, markdown_path = write_analysis(tmp_path / "analysis", analysis)
    assert json_path.is_file()
    assert markdown_path.read_text(encoding="utf-8") == markdown


def test_analyzer_rejects_public_task_hash_mismatch(tmp_path: Path) -> None:
    manifest_path, run_dirs = _make_bundle(tmp_path)
    task_path = manifest_path.parent / "blind-03.json"
    task_path.write_bytes(task_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match=r"blind-03\.task file hash"):
        analyze_blind_beam_four(manifest_path, run_dirs)


def test_analyzer_rejects_wrong_frozen_checkpoints(tmp_path: Path) -> None:
    manifest_path, run_dirs = _make_bundle(tmp_path)
    run_dir = run_dirs["blind-02"]
    result = copy.deepcopy(
        json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    )
    result["protocol"]["proposal_slot_checkpoints"] = [1, 5, 13, 21, 30, 38]
    _write_json(run_dir / "protocol.json", result["protocol"])
    _write_json(run_dir / "result.json", result)

    with pytest.raises(ValueError, match="proposal_slot_checkpoints"):
        analyze_blind_beam_four(manifest_path, run_dirs)

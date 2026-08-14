from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.aggregate import aggregate_matrix

GOLDEN = Path(__file__).with_name("golden_aggregation.json")


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        if row["core_artifact"] is not None:
            row["core_artifact"] = "<CORE>"
    return rows


def test_golden_intention_to_treat_aggregation(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    plans = [
        {
            "cell_id": "signed--QD--seed-101",
            "task_id": "signed",
            "arm": "QD",
            "seed": 101,
            "analysis_label": "confirmatory",
            "model_scope": "model-specific",
            "model_id": "qwen25-coder-3b",
            "model_alias": "qwen25-coder-3b",
            "model_hf_repository": "Qwen/Qwen2.5-Coder-3B-Instruct",
            "model_architecture": "Qwen2ForCausalLM",
            "model_parameterization": "dense",
            "model_total_parameters_billion": 3.0,
            "model_active_parameters_billion": 3.0,
            "model_dtype": "bfloat16",
            "model_quantization": "none",
            "model_revision": "revision-3b",
            "tokenizer_revision": "tokenizer-3b",
        },
        {
            "cell_id": "signed--U--seed-101",
            "task_id": "signed",
            "arm": "U",
            "seed": 101,
            "analysis_label": "confirmatory",
            "model_scope": "size-invariant",
            "model_id": None,
            "model_alias": None,
            "model_hf_repository": None,
            "model_architecture": None,
            "model_parameterization": None,
            "model_total_parameters_billion": None,
            "model_active_parameters_billion": None,
            "model_dtype": None,
            "model_quantization": None,
            "model_revision": None,
            "tokenizer_revision": None,
        },
    ]
    _write(
        matrix / "matrix_manifest.json",
        {
            "protocol_id": "golden-v1",
            "protocol_sha256": "abc123",
            "planned_cells": plans,
        },
    )
    cell = matrix / "cells" / "signed--QD--seed-101"
    core = cell / "artifacts" / "core-run"
    _write(
        cell / "cell.json",
        {
            "status": "completed",
            "wall_time_seconds": 12.5,
            "exit_code": 0,
            "exception": None,
            "core_artifact": "artifacts/core-run",
        },
    )
    _write(
        cell / "heldout.json",
        {
            "status": "completed",
            "exact": True,
            "accuracy": 1,
            "cases": 96,
            "program_source": "best_visited",
        },
    )
    _write(
        core / "manifest.json",
        {
            "status": "completed",
            "seed": 101,
            "metrics": {
                "candidate_score_cache": {
                    "provider_score_requests": 4,
                    "provider_candidates": 80,
                    "provider_scored_tokens": 640,
                    "provider_await_wall_seconds": 2.5,
                    "hit_candidates": 20,
                    "miss_candidates": 80,
                    "cache_served_scored_tokens": 120,
                }
            },
        },
    )
    _write(
        core / "result.json",
        {
            "status": "completed",
            "run_id": "core-1",
            "result": {
                "best_visited": {"exact_program": True, "total_loss": 0, "cost": 25},
                "sampled_best": {"exact_program": False, "total_loss": 1, "cost": 4},
                "search": {"exact_found": True},
                "stages": [{"ess_after": 3.5}, {"ess_after": 2.25}],
                "reference": {
                    "total_variation_distance": 0.2,
                    "log_path_z_error": -0.1,
                    "absolute_log_path_z_error": 0.1,
                },
                "scored_candidates": 1688,
                "support_states": 132198,
                "exact_programs": 4,
            },
        },
    )
    (core / "events.jsonl").write_text(
        json.dumps({"elapsed_seconds": 11.0, "event": "run.completed", "data": {}}) + "\n",
        encoding="utf-8",
    )

    actual = _normalize(aggregate_matrix(matrix))
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))

    assert actual == expected

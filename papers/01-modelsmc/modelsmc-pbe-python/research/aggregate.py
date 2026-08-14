"""Aggregate matrix manifests, core artifacts, and events into tidy records."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from research.heldout import write_json_atomic

FIELDNAMES = (
    "protocol_id",
    "protocol_sha256",
    "cell_id",
    "analysis_label",
    "task_id",
    "arm",
    "seed",
    "model_scope",
    "model_id",
    "model_alias",
    "model_hf_repository",
    "model_architecture",
    "model_parameterization",
    "model_total_parameters_billion",
    "model_active_parameters_billion",
    "model_dtype",
    "model_quantization",
    "model_revision",
    "tokenizer_revision",
    "status",
    "run_completed",
    "success",
    "training_exact",
    "heldout_evaluable",
    "heldout_correct",
    "heldout_accuracy",
    "heldout_cases",
    "heldout_program_source",
    "loss",
    "cost",
    "training_program_source",
    "ess",
    "ess_min",
    "tv_distance",
    "logz_error",
    "absolute_logz_error",
    "scored_candidates",
    "provider_score_requests",
    "provider_candidates",
    "provider_scored_tokens",
    "provider_await_wall_seconds",
    "score_cache_hit_candidates",
    "score_cache_miss_candidates",
    "cache_served_scored_tokens",
    "wall_time_seconds",
    "support_states",
    "support_exact_programs",
    "exit_code",
    "failure_reason",
    "core_run_id",
    "core_artifact",
)


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    return cast(dict[str, Any], value)


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _events(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(cast(dict[str, Any], value))
    return records


def _event_data(event: Mapping[str, Any]) -> Mapping[str, Any]:
    value = event.get("data")
    return cast(Mapping[str, Any], value) if isinstance(value, dict) else {}


def _stage_ess(result: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    stages = result.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            value = _number(stage.get("ess_after"))
            if value is None:
                value = _number(stage.get("ess"))
            if value is not None:
                values.append(value)
    if values:
        return values
    final_ess = _number(result.get("final_ess"))
    if final_ess is not None:
        return [final_ess]
    for event in events:
        if event.get("event") not in {
            "importance_smc.stage.completed",
            "grammar_smc.stage.completed",
        }:
            continue
        data = _event_data(event)
        value = _number(data.get("ess_after"))
        if value is None:
            value = _number(data.get("ess"))
        if value is not None:
            values.append(value)
    return values


def _candidate_count(result: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> int | None:
    direct = _integer(result.get("scored_candidates"))
    if direct is not None:
        return direct
    direct = _integer(result.get("proposal_calls"))
    if direct is not None:
        return direct
    total = 0
    found = False
    for event in events:
        if event.get("event") != "importance.proposal.scoring.completed":
            continue
        candidates = _integer(_event_data(event).get("candidates"))
        if candidates is not None:
            found = True
            total += candidates
    return total if found else None


def _cache_metrics(manifest: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if manifest is None:
        return {}
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    cache = metrics.get("candidate_score_cache")
    return cast(Mapping[str, Any], cache) if isinstance(cache, dict) else {}


def _champion_metrics(
    result: Mapping[str, Any],
) -> tuple[bool, float | None, int | None, str | None]:
    search = result.get("search")
    search_exact: bool | None = None
    if isinstance(search, dict) and isinstance(search.get("exact_found"), bool):
        search_exact = cast(bool, search["exact_found"])
    if "best_visited" in result:
        best = result.get("best_visited")
        if not isinstance(best, dict):
            return False, None, None, "best_visited_invalid"
        exact = search_exact if search_exact is not None else best.get("exact_program") is True
        return (
            exact,
            _number(best.get("total_loss")),
            _integer(best.get("cost")),
            "best_visited",
        )
    sampled = result.get("sampled_best")
    if isinstance(sampled, dict):
        exact = search_exact if search_exact is not None else sampled.get("exact_program") is True
        return (
            exact,
            _number(sampled.get("total_loss")),
            _integer(sampled.get("cost")),
            "sampled_best_legacy_fallback",
        )
    champion = result.get("champion")
    if isinstance(champion, dict):
        score = champion.get("score")
        if isinstance(score, dict):
            exact = search_exact if search_exact is not None else score.get("exact_program") is True
            return (
                exact,
                _number(score.get("total_loss")),
                _integer(score.get("cost")),
                "champion_legacy_fallback",
            )
    exact = search_exact if search_exact is not None else result.get("exact") is True
    return exact, None, None, None


def _failure_reason(
    cell: Mapping[str, Any] | None, core_envelope: Mapping[str, Any] | None
) -> str | None:
    if cell is None:
        return "planned cell was not started"
    exception = cell.get("exception")
    if isinstance(exception, str) and exception:
        return exception
    if core_envelope is not None:
        error = core_envelope.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return cast(str, error["message"]) or cast(str, error.get("type", "core run failed"))
    if cell.get("status") != "completed":
        return f"cell status is {cell.get('status', 'unknown')}"
    return None


def aggregate_cell(
    matrix_dir: Path,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one planned cell; absent artifacts remain negative ITT rows."""

    cell_id = str(plan["cell_id"])
    cell_dir = matrix_dir / "cells" / cell_id
    cell = _json(cell_dir / "cell.json")
    core_dir: Path | None = None
    if cell is not None and isinstance(cell.get("core_artifact"), str):
        core_dir = (cell_dir / cast(str, cell["core_artifact"])).resolve()
    core_envelope = _json(core_dir / "result.json") if core_dir else None
    core_manifest = _json(core_dir / "manifest.json") if core_dir else None
    events = _events(core_dir / "events.jsonl") if core_dir else []
    heldout = _json(cell_dir / "heldout.json")

    completed = (
        cell is not None
        and cell.get("status") == "completed"
        and core_envelope is not None
        and core_envelope.get("status") == "completed"
    )
    result_value = core_envelope.get("result") if core_envelope else None
    result: Mapping[str, Any] = (
        cast(Mapping[str, Any], result_value) if isinstance(result_value, dict) else {}
    )
    training_exact, loss, cost, training_source = _champion_metrics(result)
    training_exact = completed and training_exact
    heldout_evaluable = heldout is not None and heldout.get("status") == "completed"
    heldout_correct = heldout_evaluable and heldout is not None and heldout.get("exact") is True
    ess_values = _stage_ess(result, events)
    reference_value = result.get("reference")
    reference: Mapping[str, Any] = (
        cast(Mapping[str, Any], reference_value) if isinstance(reference_value, dict) else {}
    )
    tv = _number(reference.get("total_variation_distance"))
    logz_error = _number(reference.get("log_path_z_error"))
    if logz_error is None:
        logz_error = _number(reference.get("log_z_error"))
    abs_logz = _number(reference.get("absolute_log_path_z_error"))
    if abs_logz is None:
        abs_logz = _number(reference.get("absolute_log_z_error"))
    if abs_logz is None and logz_error is not None:
        abs_logz = abs(logz_error)
    wall = _number(cell.get("wall_time_seconds")) if cell else None
    if wall is None and events:
        wall = _number(events[-1].get("elapsed_seconds"))
    support_states = _integer(result.get("support_states"))
    if support_states is None:
        support_states = _integer(result.get("grammar_states"))
    status = str(cell.get("status")) if cell is not None else "not_started"
    cache = _cache_metrics(core_manifest)
    return {
        "protocol_id": manifest.get("protocol_id"),
        "protocol_sha256": manifest.get("protocol_sha256"),
        "cell_id": cell_id,
        "analysis_label": plan.get("analysis_label"),
        "task_id": plan.get("task_id"),
        "arm": plan.get("arm"),
        "seed": plan.get("seed"),
        "model_scope": plan.get("model_scope"),
        "model_id": plan.get("model_id"),
        "model_alias": plan.get("model_alias"),
        "model_hf_repository": plan.get("model_hf_repository"),
        "model_architecture": plan.get("model_architecture"),
        "model_parameterization": plan.get("model_parameterization"),
        "model_total_parameters_billion": _number(
            plan.get("model_total_parameters_billion")
        ),
        "model_active_parameters_billion": _number(
            plan.get("model_active_parameters_billion")
        ),
        "model_dtype": plan.get("model_dtype"),
        "model_quantization": plan.get("model_quantization"),
        "model_revision": plan.get("model_revision"),
        "tokenizer_revision": plan.get("tokenizer_revision"),
        "status": status,
        "run_completed": completed,
        "success": training_exact,
        "training_exact": training_exact,
        "heldout_evaluable": heldout_evaluable,
        "heldout_correct": heldout_correct,
        "heldout_accuracy": _number(heldout.get("accuracy")) if heldout else 0.0,
        "heldout_cases": _integer(heldout.get("cases")) if heldout else None,
        "heldout_program_source": heldout.get("program_source") if heldout else None,
        "loss": loss,
        "cost": cost,
        "training_program_source": training_source,
        "ess": ess_values[-1] if ess_values else None,
        "ess_min": min(ess_values) if ess_values else None,
        "tv_distance": tv,
        "logz_error": logz_error,
        "absolute_logz_error": abs_logz,
        "scored_candidates": _candidate_count(result, events),
        "provider_score_requests": _integer(cache.get("provider_score_requests")),
        "provider_candidates": _integer(cache.get("provider_candidates")),
        "provider_scored_tokens": _integer(cache.get("provider_scored_tokens")),
        "provider_await_wall_seconds": _number(cache.get("provider_await_wall_seconds")),
        "score_cache_hit_candidates": _integer(cache.get("hit_candidates")),
        "score_cache_miss_candidates": _integer(cache.get("miss_candidates")),
        "cache_served_scored_tokens": _integer(cache.get("cache_served_scored_tokens")),
        "wall_time_seconds": wall,
        "support_states": support_states,
        "support_exact_programs": _integer(result.get("exact_programs")),
        "exit_code": _integer(cell.get("exit_code")) if cell else None,
        "failure_reason": _failure_reason(cell, core_envelope),
        "core_run_id": (
            core_envelope.get("run_id")
            if core_envelope and core_envelope.get("run_id") is not None
            else core_manifest.get("run_id")
            if core_manifest
            else None
        ),
        "core_artifact": str(core_dir) if core_dir else None,
    }


def aggregate_matrix(matrix_dir: Path) -> list[dict[str, Any]]:
    """Aggregate every planned cell, including those that never started."""

    manifest = _json(matrix_dir / "matrix_manifest.json")
    if manifest is None:
        raise ValueError(f"matrix manifest is unavailable: {matrix_dir}")
    planned = manifest.get("planned_cells")
    if not isinstance(planned, list):
        raise ValueError("matrix manifest has no planned_cells array")
    rows: list[dict[str, Any]] = []
    for value in planned:
        if not isinstance(value, dict) or "cell_id" not in value:
            raise ValueError("matrix manifest contains an invalid cell plan")
        rows.append(aggregate_cell(matrix_dir, cast(Mapping[str, Any], value), manifest))
    return rows


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    matrix_dir = args.matrix.expanduser().resolve()
    output = (args.output or matrix_dir / "analysis").expanduser().resolve()
    rows = aggregate_matrix(matrix_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "metrics.json", rows)
    write_csv(output / "metrics.csv", rows)
    successes = sum(row["success"] is True for row in rows)
    heldout = sum(row["heldout_correct"] is True for row in rows)
    print(
        f"[aggregate] rows={len(rows)} training_success={successes} "
        f"heldout_exact={heldout} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

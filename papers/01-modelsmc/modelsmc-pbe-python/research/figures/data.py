"""Load, validate, and normalize tidy aggregate rows for plotting."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

Row = dict[str, Any]

_BOOLEAN_FIELDS = {
    "run_completed",
    "success",
    "training_exact",
    "heldout_evaluable",
    "heldout_correct",
}
_INTEGER_FIELDS = {
    "seed",
    "heldout_cases",
    "cost",
    "scored_candidates",
    "provider_score_requests",
    "provider_candidates",
    "provider_scored_tokens",
    "score_cache_hit_candidates",
    "score_cache_miss_candidates",
    "cache_served_scored_tokens",
    "support_states",
    "support_exact_programs",
    "exit_code",
}
_FLOAT_FIELDS = {
    "model_total_parameters_billion",
    "model_active_parameters_billion",
    "heldout_accuracy",
    "loss",
    "ess",
    "ess_min",
    "tv_distance",
    "logz_error",
    "absolute_logz_error",
    "provider_await_wall_seconds",
    "wall_time_seconds",
}
_REQUIRED_FIELDS = {
    "protocol_sha256",
    "cell_id",
    "task_id",
    "arm",
    "seed",
    "model_id",
    "status",
    "run_completed",
    "training_exact",
    "heldout_evaluable",
    "heldout_correct",
}
_FIGURE_FIELDS = (
    "protocol_id",
    "protocol_sha256",
    "cell_id",
    "analysis_label",
    "task_id",
    "arm",
    "seed",
    "model_id",
    "model_total_parameters_billion",
    "status",
    "run_completed",
    "training_exact",
    "heldout_evaluable",
    "heldout_correct",
    "heldout_accuracy",
    "provider_candidates",
    "provider_scored_tokens",
    "provider_await_wall_seconds",
    "score_cache_hit_candidates",
    "score_cache_miss_candidates",
    "cache_served_scored_tokens",
    "failure_reason",
)


def _parse_csv_value(name: str, value: str) -> object:
    if value == "":
        return None
    if name in _BOOLEAN_FIELDS:
        lowered = value.lower()
        if lowered not in {"true", "false"}:
            raise ValueError(f"CSV field {name} must be true, false, or empty")
        return lowered == "true"
    if name in _INTEGER_FIELDS:
        return int(value)
    if name in _FLOAT_FIELDS:
        return float(value)
    return value


def _rows_from_json(path: Path) -> list[Row]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid aggregate JSON {path}: {error}") from error
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"aggregate JSON must be an array of objects: {path}")
    return [cast(Row, row) for row in value]


def _rows_from_csv(path: Path) -> list[Row]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return [
            {name: _parse_csv_value(name, value) for name, value in row.items()}
            for row in reader
        ]


def load_rows(paths: Sequence[Path]) -> tuple[list[Row], list[dict[str, str]]]:
    """Load JSON/CSV aggregates, reject conflicts, and hash every input."""

    if not paths:
        raise ValueError("at least one aggregate input is required")
    inputs: list[dict[str, str]] = []
    by_cell: dict[str, Row] = {}
    for index, raw_path in enumerate(paths, start=1):
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"aggregate input does not exist: {path}")
        content = path.read_bytes()
        inputs.append(
            {
                "index": str(index),
                "name": path.name,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
        if path.suffix.lower() == ".json":
            rows = _rows_from_json(path)
        elif path.suffix.lower() == ".csv":
            rows = _rows_from_csv(path)
        else:
            raise ValueError(f"aggregate input must end in .json or .csv: {path}")
        for row in rows:
            missing = _REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"aggregate row is missing fields: {sorted(missing)}")
            cell_id = row.get("cell_id")
            if not isinstance(cell_id, str) or not cell_id:
                raise ValueError("aggregate cell_id must be a nonempty string")
            prior = by_cell.get(cell_id)
            if prior is not None and prior != row:
                raise ValueError(f"conflicting duplicate aggregate cell: {cell_id}")
            by_cell[cell_id] = row
    rows = list(by_cell.values())
    protocols = {
        row.get("protocol_sha256")
        for row in rows
        if isinstance(row.get("protocol_sha256"), str)
    }
    if len(protocols) != 1:
        raise ValueError(
            "figure inputs must contain exactly one protocol hash; preflight and "
            "amended outcomes cannot be pooled"
        )
    return rows, inputs


def select_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    models: Sequence[str],
    tasks: Sequence[str],
    arms: Sequence[str],
    seeds: Sequence[int],
) -> list[Row]:
    """Select the declared grid and sort it in protocol-facing order."""

    model_order = {value: index for index, value in enumerate(models)}
    task_order = {value: index for index, value in enumerate(tasks)}
    arm_order = {value: index for index, value in enumerate(arms)}
    seed_order = {value: index for index, value in enumerate(seeds)}
    selected = [
        dict(row)
        for row in rows
        if row.get("model_id") in model_order
        and row.get("task_id") in task_order
        and row.get("arm") in arm_order
        and row.get("seed") in seed_order
    ]
    selected.sort(
        key=lambda row: (
            model_order[cast(str, row["model_id"])],
            task_order[cast(str, row["task_id"])],
            arm_order[cast(str, row["arm"])],
            seed_order[cast(int, row["seed"])],
        )
    )
    return selected


def figure_rows(rows: Iterable[Mapping[str, Any]]) -> list[Row]:
    """Remove absolute artifact paths and retain only plotted/audited fields."""

    return [{name: row.get(name) for name in _FIGURE_FIELDS} for row in rows]


def cache_status(row: Mapping[str, Any]) -> str:
    hit = row.get("score_cache_hit_candidates")
    miss = row.get("score_cache_miss_candidates")
    if not isinstance(hit, int) or isinstance(hit, bool):
        return "unavailable"
    if not isinstance(miss, int) or isinstance(miss, bool):
        return "unavailable"
    if hit > 0 and miss == 0:
        return "warm"
    if miss > 0 and hit == 0:
        return "cold"
    if hit > 0 and miss > 0:
        return "mixed"
    return "unavailable"

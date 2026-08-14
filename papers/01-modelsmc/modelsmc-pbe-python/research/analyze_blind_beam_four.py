"""Validate and summarize the public four-task blind-beam pilot artifacts.

This analyzer deliberately has no private-reveal input. It binds each result to
the public task manifest, checks the frozen beam protocol and checkpoint
statistics, and emits a descriptive JSON/Markdown report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any, cast

SCHEMA = "blind-beam-four-task-analysis-v1"
PUBLIC_MANIFEST_SCHEMA = "blinded-filter-map-suite-v1"
RESULT_SCHEMA = "iterative-typed-llm-beam-experiment-v2"
TASK_IDS = ("blind-01", "blind-02", "blind-03", "blind-04")
ALL_CHECKPOINTS = (1, 5, 13, 21, 29, 37)
REPORT_CHECKPOINTS = (29, 37)
RUN_SEEDS = {
    "blind-01": (101000, 101100, 101200),
    "blind-02": (102000, 102100, 102200),
    "blind-03": (103000, 103100, 103200),
    "blind-04": (104000, 104100, 104200),
}
FORBIDDEN_PRIVATE_KEYS = frozenset(
    {
        "commitment_nonce",
        "mapper_family",
        "nonce",
        "private_seed",
        "seed_hex",
        "target",
        "target_mapper",
        "target_predicate",
        "target_mapper_dsl",
        "target_predicate_dsl",
    }
)


def canonical_bytes(value: object) -> bytes:
    """Return the repository's stable JSON encoding without a trailing newline."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def _array(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return cast(list[Any], value)


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be Boolean")
    return value


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _expect(actual: object, expected: object, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _close(actual: float, expected: float, *, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _reject_private_keys(value: object, *, path: str = "public") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_PRIVATE_KEYS:
                raise ValueError(f"private key {key!r} appears at {path}")
            _reject_private_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_private_keys(child, path=f"{path}[{index}]")


def _safe_public_task_path(manifest_path: Path, value: object, *, task_id: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"manifest task path for {task_id} must be a string")
    relative = Path(value)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name in {"", ".", ".."}:
        raise ValueError(f"manifest task path for {task_id} must be one file name")
    return manifest_path.resolve().parent / relative


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval by default."""

    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("successes and trials are inconsistent")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def poisson_binomial_distribution(probabilities: Sequence[float]) -> tuple[float, ...]:
    """Return P(sum Bernoulli(p_i) == k) for every k via stable dynamic programming."""

    distribution = [1.0]
    for index, probability in enumerate(probabilities):
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability {index} is outside [0, 1]")
        updated = [0.0] * (len(distribution) + 1)
        for successes, mass in enumerate(distribution):
            updated[successes] += mass * (1.0 - probability)
            updated[successes + 1] += mass * probability
        distribution = updated
    return tuple(distribution)


def poisson_binomial_tail(probabilities: Sequence[float], observed: int) -> float:
    """Return P(sum Bernoulli(p_i) >= observed)."""

    if not 0 <= observed <= len(probabilities):
        raise ValueError("observed successes are inconsistent with probabilities")
    tail = sum(poisson_binomial_distribution(probabilities)[observed:])
    return min(1.0, max(0.0, tail))


def _validate_provider_seal(run_dir: Path, result: Mapping[str, Any], *, task_id: str) -> str:
    seal_path = run_dir / "provider-seal.json"
    seal = _read_object(seal_path)
    records = _array(seal.get("records"), name=f"{task_id}.provider-seal.records")
    normalized: list[dict[str, object]] = []
    previous_path = ""
    for index, raw_record in enumerate(records):
        record = _object(raw_record, name=f"{task_id}.provider-seal.records[{index}]")
        relative_value = record.get("path")
        if not isinstance(relative_value, str):
            raise ValueError(f"{task_id} provider inventory path must be a string")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"{task_id} provider inventory path is unsafe: {relative_value}")
        if relative.parts[0] != "provider":
            raise ValueError(f"{task_id} provider inventory escaped provider/: {relative_value}")
        if relative_value <= previous_path:
            raise ValueError(f"{task_id} provider inventory paths are not strictly sorted")
        previous_path = relative_value
        artifact = run_dir / relative
        payload = artifact.read_bytes()
        size = _integer(record.get("bytes"), name=f"{task_id}.{relative_value}.bytes")
        digest = _digest(record.get("sha256"), name=f"{task_id}.{relative_value}.sha256")
        _expect(len(payload), size, name=f"{task_id}.{relative_value}.bytes")
        _expect(sha256_bytes(payload), digest, name=f"{task_id}.{relative_value}.sha256")
        normalized.append({"path": relative_value, "bytes": size, "sha256": digest})
    inventory_digest = sha256_bytes(canonical_bytes(normalized))
    _expect(
        seal.get("inventory_sha256"),
        inventory_digest,
        name=f"{task_id}.provider-seal.inventory_sha256",
    )
    _expect(
        result.get("provider_inventory_sha256"),
        inventory_digest,
        name=f"{task_id}.result.provider_inventory_sha256",
    )
    return inventory_digest


def _validate_stored_interval(
    value: object,
    expected: tuple[float, float],
    *,
    name: str,
) -> tuple[float, float]:
    interval = _array(value, name=name)
    if len(interval) != 2:
        raise ValueError(f"{name} must contain two endpoints")
    lower = _number(interval[0], name=f"{name}[0]")
    upper = _number(interval[1], name=f"{name}[1]")
    _close(lower, expected[0], name=f"{name}[0]")
    _close(upper, expected[1], name=f"{name}[1]")
    return lower, upper


def _validate_random_checkpoint(
    random_metrics: Mapping[str, Any],
    *,
    checkpoint: int,
    trials: int,
    task_id: str,
) -> dict[str, object]:
    key = str(checkpoint)
    successes_map = _object(
        random_metrics.get("successes_by_proposal_slot_checkpoint"),
        name=f"{task_id}.random.successes_by_checkpoint",
    )
    rates_map = _object(
        random_metrics.get("success_rates_by_proposal_slot_checkpoint"),
        name=f"{task_id}.random.success_rates_by_checkpoint",
    )
    intervals_map = _object(
        random_metrics.get("success_rate_wilson_95_by_proposal_slot_checkpoint"),
        name=f"{task_id}.random.wilson_by_checkpoint",
    )
    successes = _integer(successes_map.get(key), name=f"{task_id}.random.successes[{key}]")
    if not 0 <= successes <= trials:
        raise ValueError(f"{task_id}.random.successes[{key}] is outside [0, trials]")
    rate = _number(rates_map.get(key), name=f"{task_id}.random.rate[{key}]")
    _close(rate, successes / trials, name=f"{task_id}.random.rate[{key}]")
    expected_interval = wilson_interval(successes, trials)
    interval = _validate_stored_interval(
        intervals_map.get(key),
        expected_interval,
        name=f"{task_id}.random.wilson[{key}]",
    )
    summaries = _object(
        random_metrics.get("best_loss_summary_by_proposal_slot_checkpoint"),
        name=f"{task_id}.random.best_loss_summary_by_checkpoint",
    )
    summary = _object(summaries.get(key), name=f"{task_id}.random.best_loss[{key}]")
    _expect(
        _integer(summary.get("observed_trials"), name=f"{task_id}.observed_trials[{key}]"),
        trials,
        name=f"{task_id}.observed_trials[{key}]",
    )
    normalized_summary: dict[str, float | int | None] = {"observed_trials": trials}
    for field in ("mean", "median", "p05", "p95", "fraction_at_most_llm"):
        raw_value = summary.get(field)
        normalized_summary[field] = (
            None if raw_value is None else _number(raw_value, name=f"{task_id}.{field}[{key}]")
        )
    return {
        "successes": successes,
        "success_rate": rate,
        "success_rate_wilson_95": interval,
        "best_loss_summary": normalized_summary,
    }


def _validate_task_run(
    *,
    task_id: str,
    task_record: Mapping[str, Any],
    task_path: Path,
    task_file_sha256: str,
    public_manifest_sha256: str,
    run_dir: Path,
) -> dict[str, object]:
    if (run_dir / "failure.json").exists():
        raise ValueError(f"{task_id} contains failure.json instead of a complete result")
    result_path = run_dir / "result.json"
    result = _read_object(result_path)
    _expect(result.get("schema"), RESULT_SCHEMA, name=f"{task_id}.result.schema")
    protocol = _object(result.get("protocol"), name=f"{task_id}.result.protocol")
    stored_protocol = _read_object(run_dir / "protocol.json")
    _expect(stored_protocol, protocol, name=f"{task_id}.stored protocol")

    expected_protocol = {
        "protocol_mode": "blind-four-v1",
        "blind_task_id": task_id,
        "task_sha256": task_file_sha256,
        "blind_manifest_sha256": public_manifest_sha256,
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
    }
    for field, expected in expected_protocol.items():
        _expect(protocol.get(field), expected, name=f"{task_id}.protocol.{field}")
    _expect(
        protocol.get("proposal_slot_checkpoints"),
        list(ALL_CHECKPOINTS),
        name=f"{task_id}.protocol.proposal_slot_checkpoints",
    )
    expected_seeds = RUN_SEEDS[task_id]
    for field, expected in zip(
        ("provider_seed", "tie_seed", "random_baseline_seed"), expected_seeds, strict=True
    ):
        _expect(protocol.get(field), expected, name=f"{task_id}.protocol.{field}")
    harness_sha256 = _digest(protocol.get("harness_sha256"), name=f"{task_id}.harness_sha256")
    study_protocol_sha256 = _digest(
        protocol.get("study_protocol_sha256"), name=f"{task_id}.study_protocol_sha256"
    )

    llm_metrics = _object(result.get("llm_beam_metrics"), name=f"{task_id}.llm")
    llm_successes = _object(
        llm_metrics.get("success_by_proposal_slot_checkpoint"),
        name=f"{task_id}.llm.success_by_checkpoint",
    )
    llm_losses = _object(
        llm_metrics.get("best_loss_by_proposal_slot_checkpoint"),
        name=f"{task_id}.llm.best_loss_by_checkpoint",
    )
    first_exact_raw = llm_metrics.get("first_exact_proposal_slot")
    first_exact = (
        None
        if first_exact_raw is None
        else _integer(first_exact_raw, name=f"{task_id}.llm.first_exact_proposal_slot")
    )
    if first_exact is not None and not 1 <= first_exact <= 37:
        raise ValueError(f"{task_id}.llm.first_exact_proposal_slot is outside [1, 37]")
    checkpoint_success: dict[str, bool] = {}
    checkpoint_loss: dict[str, float] = {}
    for checkpoint in REPORT_CHECKPOINTS:
        key = str(checkpoint)
        success = _boolean(llm_successes.get(key), name=f"{task_id}.llm.success[{key}]")
        expected_success = first_exact is not None and first_exact <= checkpoint
        _expect(success, expected_success, name=f"{task_id}.llm.success[{key}]")
        loss = _number(llm_losses.get(key), name=f"{task_id}.llm.best_loss[{key}]")
        if loss < 0.0:
            raise ValueError(f"{task_id}.llm.best_loss[{key}] must be nonnegative")
        _expect(loss == 0.0, success, name=f"{task_id}.llm zero loss at {key}")
        checkpoint_success[key] = success
        checkpoint_loss[key] = loss
    _expect(
        llm_metrics.get("primary_checkpoint_success"),
        checkpoint_success["29"],
        name=f"{task_id}.llm.primary_checkpoint_success",
    )
    _expect(
        llm_metrics.get("primary_checkpoint_slot"),
        29,
        name=f"{task_id}.llm.primary_checkpoint_slot",
    )
    _expect(
        llm_metrics.get("success"),
        checkpoint_success["37"],
        name=f"{task_id}.llm.success",
    )
    final_best_loss = _number(llm_metrics.get("best_loss"), name=f"{task_id}.llm.best_loss")
    _close(final_best_loss, checkpoint_loss["37"], name=f"{task_id}.llm.final best loss")
    if checkpoint_loss["37"] > checkpoint_loss["29"]:
        raise ValueError(f"{task_id}.llm best loss increased between checkpoints")
    consumed = _integer(
        llm_metrics.get("proposal_slots_consumed"),
        name=f"{task_id}.llm.proposal_slots_consumed",
    )
    expected_consumed = (
        37
        if first_exact is None
        else min(checkpoint for checkpoint in ALL_CHECKPOINTS if checkpoint >= first_exact)
    )
    _expect(consumed, expected_consumed, name=f"{task_id}.llm.proposal_slots_consumed")

    random_metrics = _object(result.get("matched_random_beam"), name=f"{task_id}.random")
    trials = _integer(random_metrics.get("trials"), name=f"{task_id}.random.trials")
    _expect(trials, 10000, name=f"{task_id}.random.trials")
    random_checkpoints = {
        str(checkpoint): _validate_random_checkpoint(
            random_metrics,
            checkpoint=checkpoint,
            trials=trials,
            task_id=task_id,
        )
        for checkpoint in REPORT_CHECKPOINTS
    }
    if cast(int, random_checkpoints["37"]["successes"]) < cast(
        int, random_checkpoints["29"]["successes"]
    ):
        raise ValueError(f"{task_id}.random checkpoint successes are not monotone")
    final_random = cast(dict[str, object], random_checkpoints["37"])
    _expect(
        random_metrics.get("successes"),
        final_random["successes"],
        name=f"{task_id}.random.successes",
    )
    _close(
        _number(random_metrics.get("success_rate"), name=f"{task_id}.random.success_rate"),
        cast(float, final_random["success_rate"]),
        name=f"{task_id}.random.success_rate",
    )
    _validate_stored_interval(
        random_metrics.get("success_rate_wilson_95"),
        cast(tuple[float, float], final_random["success_rate_wilson_95"]),
        name=f"{task_id}.random.success_rate_wilson_95",
    )

    inventory_sha256 = _validate_provider_seal(run_dir, result, task_id=task_id)
    return {
        "task_id": task_id,
        "public_task": {
            "path": task_path.name,
            "task_file_sha256": task_file_sha256,
            "task_canonical_sha256": task_record["task_canonical_sha256"],
            "target_commitment_sha256": task_record["target_commitment_sha256"],
        },
        "run": {
            "directory": str(run_dir),
            "result_sha256": sha256_file(result_path),
            "stored_protocol_sha256": sha256_file(run_dir / "protocol.json"),
            "provider_inventory_sha256": inventory_sha256,
            "harness_sha256": harness_sha256,
            "study_protocol_sha256": study_protocol_sha256,
        },
        "llm": {
            "success_by_proposal_slot_checkpoint": checkpoint_success,
            "best_loss_by_proposal_slot_checkpoint": checkpoint_loss,
            "first_exact_proposal_slot": first_exact,
            "proposal_slots_consumed": consumed,
            "final_best_loss": final_best_loss,
        },
        "matched_random": {
            "trials": trials,
            "checkpoints": random_checkpoints,
        },
    }


def analyze_blind_beam_four(
    public_manifest_path: Path,
    run_dirs: Mapping[str, Path],
) -> dict[str, object]:
    """Validate four public run bundles and return their descriptive aggregate."""

    manifest_path = public_manifest_path.expanduser().resolve()
    manifest = _read_object(manifest_path)
    _reject_private_keys(manifest)
    _expect(manifest.get("schema"), PUBLIC_MANIFEST_SCHEMA, name="public manifest schema")
    _expect(manifest.get("task_count"), 4, name="public manifest task_count")
    records = _array(manifest.get("tasks"), name="public manifest tasks")
    if len(records) != 4:
        raise ValueError("public manifest must contain exactly four task records")
    if set(run_dirs) != set(TASK_IDS):
        raise ValueError(f"run directories must be keyed by exactly {TASK_IDS}")

    indexed_records: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(records):
        record = _object(raw_record, name=f"public manifest tasks[{index}]")
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or task_id not in TASK_IDS:
            raise ValueError(f"public manifest has unexpected task ID: {task_id!r}")
        if task_id in indexed_records:
            raise ValueError(f"public manifest repeats task ID: {task_id}")
        indexed_records[task_id] = record
    _expect(tuple(indexed_records), TASK_IDS, name="public manifest task order")

    public_manifest_sha256 = sha256_file(manifest_path)
    public_manifest_canonical_sha256 = sha256_bytes(canonical_bytes(manifest))
    task_rows: list[dict[str, object]] = []
    for task_id in TASK_IDS:
        record = indexed_records[task_id]
        task_path = _safe_public_task_path(manifest_path, record.get("path"), task_id=task_id)
        task_document = _read_object(task_path)
        _reject_private_keys(task_document, path=f"public task {task_id}")
        task_file_sha256 = _digest(
            record.get("task_file_sha256"), name=f"{task_id}.task_file_sha256"
        )
        task_canonical_sha256 = _digest(
            record.get("task_canonical_sha256"), name=f"{task_id}.task_canonical_sha256"
        )
        _digest(
            record.get("target_commitment_sha256"),
            name=f"{task_id}.target_commitment_sha256",
        )
        _expect(sha256_file(task_path), task_file_sha256, name=f"{task_id}.task file hash")
        _expect(
            sha256_bytes(canonical_bytes(task_document)),
            task_canonical_sha256,
            name=f"{task_id}.task canonical hash",
        )
        run_dir = run_dirs[task_id].expanduser().resolve()
        if not run_dir.is_dir():
            raise ValueError(f"run directory does not exist for {task_id}: {run_dir}")
        task_rows.append(
            _validate_task_run(
                task_id=task_id,
                task_record=record,
                task_path=task_path,
                task_file_sha256=task_file_sha256,
                public_manifest_sha256=public_manifest_sha256,
                run_dir=run_dir,
            )
        )

    harness_hashes = {
        cast(str, cast(dict[str, object], row["run"])["harness_sha256"])
        for row in task_rows
    }
    study_protocol_hashes = {
        cast(str, cast(dict[str, object], row["run"])["study_protocol_sha256"])
        for row in task_rows
    }
    if len(harness_hashes) != 1:
        raise ValueError("the four runs used different harness hashes")
    if len(study_protocol_hashes) != 1:
        raise ValueError("the four runs used different study-protocol hashes")

    checkpoint_aggregates: dict[str, dict[str, object]] = {}
    for checkpoint in REPORT_CHECKPOINTS:
        key = str(checkpoint)
        llm_indicators = [
            cast(bool, cast(dict[str, object], row["llm"])[
                "success_by_proposal_slot_checkpoint"
            ][key])
            for row in task_rows
        ]
        random_probabilities = [
            cast(
                float,
                cast(dict[str, object], cast(dict[str, object], row["matched_random"])[
                    "checkpoints"
                ])[key]["success_rate"],
            )
            for row in task_rows
        ]
        observed = sum(llm_indicators)
        distribution = poisson_binomial_distribution(random_probabilities)
        checkpoint_aggregates[key] = {
            "endpoint": "primary" if checkpoint == 29 else "secondary",
            "llm_exact_successes": observed,
            "task_count": 4,
            "llm_exact_rate": observed / 4,
            "llm_exact_rate_wilson_95": wilson_interval(observed, 4),
            "mean_matched_random_success_rate": mean(random_probabilities),
            "expected_matched_random_successes": sum(random_probabilities),
            "mean_paired_advantage": mean(
                float(indicator) - probability
                for indicator, probability in zip(
                    llm_indicators, random_probabilities, strict=True
                )
            ),
            "paired_advantage_formula": "mean_task(LLM_exact_indicator - random_exact_rate)",
            "poisson_binomial": {
                "plug_in_probabilities": random_probabilities,
                "success_count_mass": distribution,
                "tail_probability_random_at_least_observed": sum(distribution[observed:]),
                "interpretation": (
                    "descriptive plug-in comparison; matched-random probability estimation "
                    "uncertainty is not propagated"
                ),
            },
        }

    return {
        "schema": SCHEMA,
        "analysis_scope": (
            "small exploratory four-task blind pilot; not a powered or confirmatory "
            "general-speedup study"
        ),
        "public_only_analysis": True,
        "inputs": {
            "public_manifest": str(manifest_path),
            "public_manifest_sha256": public_manifest_sha256,
            "public_manifest_canonical_sha256": public_manifest_canonical_sha256,
            "task_ids": TASK_IDS,
            "harness_sha256": next(iter(harness_hashes)),
            "study_protocol_sha256": next(iter(study_protocol_hashes)),
        },
        "checkpoints": checkpoint_aggregates,
        "tasks": task_rows,
        "limitations": [
            "Only four generated tasks and one temperature-zero LLM trajectory per task.",
            "Poisson-binomial tails use estimated random rates as fixed plug-in probabilities.",
            "Proposal-slot comparisons are not wall-clock speedup measurements.",
            "Exact means zero loss on frozen public examples; it does not identify a unique AST.",
            "No claim is made about million- or billion-program spaces or asymptotic scaling.",
        ],
    }


def _format_rate(value: float) -> str:
    if 0.0 < value < 0.0001:
        return f"{value:.3e}"
    return f"{value:.4f}"


def render_markdown(analysis: Mapping[str, object]) -> str:
    """Render a concise human-readable summary from validated aggregate JSON."""

    checkpoints = cast(dict[str, dict[str, object]], analysis["checkpoints"])
    tasks = cast(list[dict[str, object]], analysis["tasks"])
    inputs = cast(dict[str, object], analysis["inputs"])
    lines = [
        "# Blind Beam Four-Task Exploratory Pilot",
        "",
        (
            "This report uses only the public task manifest and sealed run artifacts. "
            "It does not read the private target reveal."
        ),
        "",
        "## Outcome",
        "",
    ]
    for checkpoint in REPORT_CHECKPOINTS:
        key = str(checkpoint)
        record = checkpoints[key]
        endpoint = cast(str, record["endpoint"])
        tail = cast(
            float,
            cast(dict[str, object], record["poisson_binomial"])[
                "tail_probability_random_at_least_observed"
            ],
        )
        lines.append(
            f"- {endpoint.capitalize()} checkpoint, slot {key}: "
            f"LLM exact {record['llm_exact_successes']}/4; mean matched-random rate "
            f"{_format_rate(cast(float, record['mean_matched_random_success_rate']))}; "
            f"mean paired advantage "
            f"{cast(float, record['mean_paired_advantage']):+.4f}; "
            f"plug-in P(random successes >= observed) = {_format_rate(tail)}."
        )
    lines.extend(["", "## Task results", ""])
    for row in tasks:
        task_id = cast(str, row["task_id"])
        llm = cast(dict[str, object], row["llm"])
        random_record = cast(dict[str, object], row["matched_random"])
        random_checkpoints = cast(dict[str, dict[str, object]], random_record["checkpoints"])
        llm_success = cast(dict[str, bool], llm["success_by_proposal_slot_checkpoint"])
        llm_loss = cast(dict[str, float], llm["best_loss_by_proposal_slot_checkpoint"])
        details = []
        for checkpoint in REPORT_CHECKPOINTS:
            key = str(checkpoint)
            random_at_checkpoint = random_checkpoints[key]
            lower, upper = cast(tuple[float, float], random_at_checkpoint[
                "success_rate_wilson_95"
            ])
            details.append(
                f"slot {key}: LLM {'exact' if llm_success[key] else 'not exact'} "
                f"(best loss {llm_loss[key]:g}), random "
                f"{random_at_checkpoint['successes']}/10000 "
                f"[{_format_rate(lower)}, {_format_rate(upper)}]"
            )
        lines.append(f"- {task_id}: " + "; ".join(details) + ".")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The paired advantages and Poisson-binomial tails are descriptive mechanism "
                "checks. Four tasks cannot establish a general search speedup, and proposal "
                "slots are not wall-clock measurements."
            ),
            "",
            "## Public bindings",
            "",
            f"- Manifest SHA-256: `{inputs['public_manifest_sha256']}`",
            f"- Harness SHA-256: `{inputs['harness_sha256']}`",
            f"- Study protocol SHA-256: `{inputs['study_protocol_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_analysis(output_dir: Path, analysis: Mapping[str, object]) -> tuple[Path, Path]:
    """Write canonical JSON and Markdown without overwriting an existing report."""

    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "analysis.json"
    markdown_path = output / "SUMMARY.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(f"refusing to overwrite analysis in {output}")
    json_path.write_bytes(canonical_bytes(analysis) + b"\n")
    markdown_path.write_text(render_markdown(analysis), encoding="utf-8")
    return json_path, markdown_path


def _parse_run_specs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        task_id, separator, raw_path = value.partition("=")
        if not separator or not task_id or not raw_path:
            raise ValueError("--run must have the form TASK_ID=PATH")
        if task_id in result:
            raise ValueError(f"duplicate --run task ID: {task_id}")
        result[task_id] = Path(raw_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-manifest", type=Path, required=True)
    runs = parser.add_mutually_exclusive_group(required=True)
    runs.add_argument(
        "--runs-root",
        type=Path,
        help="Directory containing blind-01 through blind-04 run directories.",
    )
    runs.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="TASK_ID=PATH",
        help="Explicit task-to-run binding; repeat exactly four times.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.runs_root is not None:
            run_dirs = {task_id: args.runs_root / task_id for task_id in TASK_IDS}
        else:
            run_dirs = _parse_run_specs(args.run)
        analysis = analyze_blind_beam_four(args.public_manifest, run_dirs)
        json_path, markdown_path = write_analysis(args.output, analysis)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "analysis_json": str(json_path),
                "summary_markdown": str(markdown_path),
                "checkpoints": analysis["checkpoints"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

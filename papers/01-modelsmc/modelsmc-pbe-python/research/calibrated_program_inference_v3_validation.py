"""Native validation and analysis for the V3 calibration confirmation.

This module deliberately owns the V3 ledger, schema, bootstrap, gate, and
artifact-inventory invariants.  It does not import the V2 fresh analyzer: a V3
artifact can therefore be checked without inheriting any V2 task, seed, arm,
or schema globals.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

STUDY_SCHEMA = "provider-free-calibrated-program-inference-v3-fresh"
RUN_SCHEMA = f"{STUDY_SCHEMA}-run-v1"
REFERENCE_SCHEMA = f"{STUDY_SCHEMA}-reference-v1"
ANALYSIS_SCHEMA = f"{STUDY_SCHEMA}-analysis-v1"
INVENTORY_SCHEMA = f"{STUDY_SCHEMA}-inventory-v1"
METADATA_SCHEMA = f"{STUDY_SCHEMA}-metadata-v1"

TASK_COUNT = 32
TASK_IDS = tuple(f"fresh-cal-v3-{index:03d}" for index in range(1, TASK_COUNT + 1))
PARTICLE_COUNTS = (256, 512, 1024)
PRIMARY_PARTICLE_COUNT = 256
REPETITIONS = 64
REPETITION_SEEDS = tuple(range(936_001, 936_001 + REPETITIONS))
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 937_001
IDENTITY_TOLERANCE = 1e-12
PRIOR_V2_SECRET_COMMITMENT = (
    "702606b13152d78067d2f230a64914de191038db77651d36ae1d7d916b36365b"
)

PRIMARY_ARM_ID = "factorized-k8-a025-e075-proposal-bridge"
FACTORIZED_IS_ARM_ID = "factorized-k8-a025-e075-terminal-is"
STICKY_IS_ARM_ID = "sticky-epsilon-005-terminal-is"
ARM_PARTICLES: dict[str, tuple[int, ...]] = {
    PRIMARY_ARM_ID: PARTICLE_COUNTS,
    FACTORIZED_IS_ARM_ID: (PRIMARY_PARTICLE_COUNT,),
    STICKY_IS_ARM_ID: (PRIMARY_PARTICLE_COUNT,),
}
ARM_ALGORITHMS = {
    PRIMARY_ARM_ID: "proposal-bridge-smc",
    FACTORIZED_IS_ARM_ID: "terminal-is",
    STICKY_IS_ARM_ID: "terminal-is",
}
ARM_ROLES = {
    PRIMARY_ARM_ID: "confirmatory-primary",
    FACTORIZED_IS_ARM_ID: "matched-proposal-terminal-is-ablation",
    STICKY_IS_ARM_ID: "historical-sticky-terminal-is-baseline",
}
EXPECTED_CELLS = tuple(
    (arm_id, particles)
    for arm_id, particle_counts in ARM_PARTICLES.items()
    for particles in particle_counts
)
EXPECTED_RUN_COUNT = TASK_COUNT * REPETITIONS * len(EXPECTED_CELLS)
EXPECTED_INITIAL_PROPOSAL_DRAWS = TASK_COUNT * REPETITIONS * sum(
    sum(particle_counts) for particle_counts in ARM_PARTICLES.values()
)

EXPECTED_ARTIFACT_PATHS = frozenset(
    {
        "SUMMARY.md",
        "analysis.json",
        "protocol.json",
        "public-suite-manifest.json",
        "runs.json",
        "study-metadata.json",
        *(f"references/{task_id}.json" for task_id in TASK_IDS),
    }
)


class V3ValidationError(ValueError):
    """A native V3 ledger, analysis, or artifact invariant was violated."""


def canonical_bytes(value: object) -> bytes:
    """Return strict, deterministic JSON bytes without a trailing newline."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _expected_sample_seed(
    task_sha256: str,
    arm_id: str,
    particles: int,
    repetition: int,
    base_seed: int,
) -> int:
    payload = "\0".join(
        str(part)
        for part in (
            STUDY_SCHEMA,
            "smc",
            task_sha256,
            arm_id,
            particles,
            repetition,
            base_seed,
        )
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V3ValidationError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise V3ValidationError(f"{name} is not finite")
    return result


def _require_int(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not _is_int(value):
        raise V3ValidationError(f"{name} is not an integer")
    result = cast(int, value)
    if minimum is not None and result < minimum:
        raise V3ValidationError(f"{name} is below {minimum}")
    if maximum is not None and result > maximum:
        raise V3ValidationError(f"{name} is above {maximum}")
    return result


def _require_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V3ValidationError(f"{name} is not a lowercase SHA-256 digest")
    return value


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise V3ValidationError(f"{name} is not a string-keyed object")
    return cast(Mapping[str, object], value)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V3ValidationError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise V3ValidationError(f"JSON contains non-finite constant: {value}")


def _parse_json_bytes(raw: bytes, *, name: str) -> object:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise V3ValidationError(f"{name} is not UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise V3ValidationError(f"{name} is not strict JSON") from error


def _read_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise V3ValidationError(f"artifact member is not a regular file: {path}")
    return _parse_json_bytes(path.read_bytes(), name=path.as_posix())


def _validate_reference(
    task_id: str,
    value: object,
) -> Mapping[str, object]:
    reference = _require_mapping(value, name=f"reference {task_id}")
    if set(reference) != {
        "schema",
        "task_id",
        "task_sha256",
        "provider_calls",
        "program_syntaxes",
        "exact_program_syntaxes",
        "exact_target_mass",
        "target_mean_loss",
        "program_population_sha256",
        "mode_bank",
        "proposal_census",
    }:
        raise V3ValidationError(f"reference fields differ for {task_id}")
    if reference.get("schema") != REFERENCE_SCHEMA:
        raise V3ValidationError(f"reference schema differs for {task_id}")
    if reference.get("task_id") != task_id:
        raise V3ValidationError(f"reference task ID differs for {task_id}")
    if reference.get("provider_calls") != 0 or isinstance(
        reference.get("provider_calls"), bool
    ):
        raise V3ValidationError(f"reference is not provider-free for {task_id}")
    _require_digest(reference.get("task_sha256"), name=f"{task_id} task SHA-256")
    _require_digest(
        reference.get("program_population_sha256"),
        name=f"{task_id} program-population SHA-256",
    )
    exact_mass = _finite(
        reference.get("exact_target_mass"), name=f"{task_id} exact target mass"
    )
    mean_loss = _finite(
        reference.get("target_mean_loss"), name=f"{task_id} target mean loss"
    )
    if not 0.0 <= exact_mass <= 1.0 or mean_loss < 0.0:
        raise V3ValidationError(f"reference endpoint is invalid for {task_id}")
    if reference.get("program_syntaxes") != 36_000:
        raise V3ValidationError(f"reference support size differs for {task_id}")
    _require_int(
        reference.get("exact_program_syntaxes"),
        name=f"{task_id} exact syntax count",
        minimum=1,
        maximum=36_000,
    )
    mode_bank = _require_mapping(reference.get("mode_bank"), name=f"{task_id} mode bank")
    if (
        mode_bank.get("acquisition_completed_before_reference_materialization")
        is not True
        or mode_bank.get("reference_materialized_after_acquisition") is not True
        or mode_bank.get("hidden_target_used") is not False
        or mode_bank.get("complete_program_catalog_ranked") is not False
        or mode_bank.get("predicate_syntaxes_evaluated") != 600
        or mode_bank.get("mapper_syntaxes_evaluated") != 60
        or _require_int(
            mode_bank.get("complete_representatives_scored"),
            name=f"{task_id} complete representatives scored",
            minimum=1,
            maximum=64,
        )
        > 64
    ):
        raise V3ValidationError(f"reference acquisition boundary differs for {task_id}")
    census = _require_mapping(
        reference.get("proposal_census"), name=f"{task_id} proposal census"
    )
    if set(census) != set(ARM_PARTICLES):
        raise V3ValidationError(f"proposal-census arm set differs for {task_id}")
    for arm_id in ARM_PARTICLES:
        arm_census = _require_mapping(
            census[arm_id], name=f"{task_id}/{arm_id} proposal census"
        )
        identity_error = _finite(
            arm_census.get("importance_identity_maximum_absolute_error"),
            name=f"{task_id}/{arm_id} identity error",
        )
        if identity_error < 0.0:
            raise V3ValidationError(f"negative identity error for {task_id}/{arm_id}")
        if arm_id in {PRIMARY_ARM_ID, FACTORIZED_IS_ARM_ID}:
            ledger = _require_mapping(
                arm_census.get("factorized_mass_ledger"),
                name=f"{task_id}/{arm_id} factorized mass ledger",
            )
            if (
                not math.isclose(
                    _finite(
                        ledger.get("grammar_component_mass"),
                        name=f"{task_id}/{arm_id} grammar mass",
                    ),
                    0.25,
                    rel_tol=0.0,
                    abs_tol=IDENTITY_TOLERANCE,
                )
                or not math.isclose(
                    _finite(
                        ledger.get("local_component_mass"),
                        name=f"{task_id}/{arm_id} local mass",
                    ),
                    0.75,
                    rel_tol=0.0,
                    abs_tol=IDENTITY_TOLERANCE,
                )
                or abs(
                    _finite(
                        ledger.get("pre_normalization_sum_error"),
                        name=f"{task_id}/{arm_id} proposal sum error",
                    )
                )
                > IDENTITY_TOLERANCE
                or _finite(
                    arm_census.get(
                        "factorized_reconstruction_maximum_absolute_error"
                    ),
                    name=f"{task_id}/{arm_id} proposal reconstruction error",
                )
                > IDENTITY_TOLERANCE
            ):
                raise V3ValidationError(
                    f"factorized proposal accounting differs for {task_id}/{arm_id}"
                )
    return reference


def validate_references(
    references: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    """Validate the complete, native V3 reference set."""

    if set(references) != set(TASK_IDS):
        raise V3ValidationError("reference task set differs from the frozen V3 suite")
    return {
        task_id: _validate_reference(task_id, references[task_id])
        for task_id in TASK_IDS
    }


def _validate_run(
    value: object,
    references: Mapping[str, Mapping[str, object]],
) -> tuple[str, str, int, int]:
    run = _require_mapping(value, name="run")
    if set(run) != {
        "schema",
        "task_id",
        "task_sha256",
        "arm_id",
        "algorithm",
        "particles",
        "repetition",
        "base_seed",
        "sample_seed",
        "provider_calls",
        "logical_proposal_draws",
        "estimate",
        "reference",
        "error",
        "diagnostics",
        "stages",
    }:
        raise V3ValidationError("run fields differ")
    if run.get("schema") != RUN_SCHEMA:
        raise V3ValidationError("run schema differs")
    if run.get("provider_calls") != 0 or isinstance(run.get("provider_calls"), bool):
        raise V3ValidationError("run is not provider-free")

    task_id = run.get("task_id")
    if not isinstance(task_id, str) or task_id not in references:
        raise V3ValidationError("run task ID is outside the frozen V3 suite")
    arm_id = run.get("arm_id")
    if not isinstance(arm_id, str) or arm_id not in ARM_PARTICLES:
        raise V3ValidationError("run arm ID is outside the frozen V3 design")
    particles = _require_int(run.get("particles"), name="run particles", minimum=2)
    if particles not in ARM_PARTICLES[arm_id]:
        raise V3ValidationError(f"undeclared run cell: {arm_id}/N={particles}")
    repetition = _require_int(
        run.get("repetition"),
        name="run repetition",
        minimum=0,
        maximum=REPETITIONS - 1,
    )
    if run.get("base_seed") != REPETITION_SEEDS[repetition] or isinstance(
        run.get("base_seed"), bool
    ):
        raise V3ValidationError("run repetition/base-seed binding differs")
    task_digest = _require_digest(run.get("task_sha256"), name="run task SHA-256")
    if task_digest != references[task_id]["task_sha256"]:
        raise V3ValidationError("run task digest differs from its reference")
    if run.get("algorithm") != ARM_ALGORITHMS[arm_id]:
        raise V3ValidationError("run algorithm differs from its arm")
    sample_seed = _require_int(run.get("sample_seed"), name="run sample seed", minimum=0)
    if sample_seed != _expected_sample_seed(
        task_digest,
        arm_id,
        particles,
        repetition,
        cast(int, run["base_seed"]),
    ):
        raise V3ValidationError("run sample seed does not match V3 derivation")

    estimate = _require_mapping(run.get("estimate"), name="run estimate")
    run_reference = _require_mapping(run.get("reference"), name="run reference")
    error = _require_mapping(run.get("error"), name="run error")
    diagnostics = _require_mapping(run.get("diagnostics"), name="run diagnostics")
    stages = run.get("stages")
    if not isinstance(stages, list):
        raise V3ValidationError("run stages are malformed")
    if set(estimate) != {"exact_target_mass", "target_mean_loss"}:
        raise V3ValidationError("run estimate fields differ")
    if set(run_reference) != {"exact_target_mass", "target_mean_loss"}:
        raise V3ValidationError("run reference fields differ")
    if set(error) != {"exact_mass_signed", "target_mean_loss_signed"}:
        raise V3ValidationError("run error fields differ")
    if set(diagnostics) != {
        "importance_ess",
        "relative_importance_ess",
        "maximum_normalized_weight",
        "unique_terminal_programs",
        "exact_terminal_particles",
        "resampling_events",
        "mh_attempts",
        "mh_accepted",
        "mh_acceptance_rate",
        "annealing_stages",
    }:
        raise V3ValidationError("run diagnostic fields differ")

    estimate_mass = _finite(
        estimate.get("exact_target_mass"), name="estimated exact target mass"
    )
    estimate_loss = _finite(
        estimate.get("target_mean_loss"), name="estimated target mean loss"
    )
    reference_mass = _finite(
        run_reference.get("exact_target_mass"), name="run reference exact mass"
    )
    reference_loss = _finite(
        run_reference.get("target_mean_loss"), name="run reference mean loss"
    )
    exact_error = _finite(
        error.get("exact_mass_signed"), name="signed exact-mass error"
    )
    loss_error = _finite(
        error.get("target_mean_loss_signed"), name="signed mean-loss error"
    )
    if not 0.0 <= estimate_mass <= 1.0 + 1e-12 or estimate_loss < 0.0:
        raise V3ValidationError("estimated endpoint is outside its valid range")
    task_reference = references[task_id]
    if (
        reference_mass != task_reference["exact_target_mass"]
        or reference_loss != task_reference["target_mean_loss"]
    ):
        raise V3ValidationError("run endpoint reference differs from task reference")
    if not math.isclose(
        exact_error,
        estimate_mass - reference_mass,
        rel_tol=1e-15,
        abs_tol=1e-15,
    ) or not math.isclose(
        loss_error,
        estimate_loss - reference_loss,
        rel_tol=1e-15,
        abs_tol=1e-15,
    ):
        raise V3ValidationError("stored run error does not match its endpoints")

    importance_ess = _finite(
        diagnostics.get("importance_ess"), name="run importance ESS"
    )
    relative_ess = _finite(
        diagnostics.get("relative_importance_ess"), name="run relative ESS"
    )
    maximum_weight = _finite(
        diagnostics.get("maximum_normalized_weight"),
        name="run maximum normalized weight",
    )
    if not 1.0 - 1e-9 <= importance_ess <= particles + 1e-9:
        raise V3ValidationError("run importance ESS is outside its valid range")
    if not math.isclose(
        relative_ess,
        importance_ess / particles,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise V3ValidationError("run relative ESS does not match importance ESS")
    if not 1.0 / particles - 1e-12 <= maximum_weight <= 1.0 + 1e-12:
        raise V3ValidationError("run maximum normalized weight is invalid")
    logical_draws = _require_int(
        run.get("logical_proposal_draws"),
        name="logical proposal draws",
        minimum=particles,
    )
    if logical_draws != particles:
        raise V3ValidationError("V3 no-MH logical draw count differs")
    _require_int(
        diagnostics.get("unique_terminal_programs"),
        name="unique terminal programs",
        minimum=1,
        maximum=particles,
    )
    _require_int(
        diagnostics.get("exact_terminal_particles"),
        name="exact terminal particles",
        minimum=0,
        maximum=particles,
    )
    resampling_events = _require_int(
        diagnostics.get("resampling_events"),
        name="resampling events",
        minimum=0,
        maximum=64,
    )
    if (
        diagnostics.get("mh_attempts") != 0
        or diagnostics.get("mh_accepted") != 0
        or diagnostics.get("mh_acceptance_rate") != 0.0
    ):
        raise V3ValidationError("V3 no-MH diagnostics differ")
    annealing_stages = _require_int(
        diagnostics.get("annealing_stages"),
        name="annealing stages",
        minimum=0,
        maximum=64,
    )
    if arm_id == PRIMARY_ARM_ID:
        if not stages or annealing_stages != len(stages):
            raise V3ValidationError("V3 bridge stage count differs")
        previous_beta = 0.0
        counted_resamples = 0
        for expected_stage, value in enumerate(stages, 1):
            stage = _require_mapping(value, name="bridge stage")
            if set(stage) != {
                "stage",
                "beta_previous",
                "beta_current",
                "relative_ess_before_optional_resampling",
                "resampled",
            }:
                raise V3ValidationError("V3 bridge stage fields differ")
            if stage.get("stage") != expected_stage:
                raise V3ValidationError("V3 bridge stage order differs")
            beta_previous = _finite(stage.get("beta_previous"), name="previous beta")
            beta_current = _finite(stage.get("beta_current"), name="current beta")
            relative_stage_ess = _finite(
                stage.get("relative_ess_before_optional_resampling"),
                name="stage relative ESS",
            )
            if (
                not math.isclose(beta_previous, previous_beta, abs_tol=1e-15)
                or not beta_previous < beta_current <= 1.0
                or not 0.0 < relative_stage_ess <= 1.0 + 1e-12
                or not isinstance(stage.get("resampled"), bool)
            ):
                raise V3ValidationError("V3 bridge stage values differ")
            counted_resamples += int(cast(bool, stage["resampled"]))
            previous_beta = beta_current
        if previous_beta != 1.0 or counted_resamples != resampling_events:
            raise V3ValidationError("V3 bridge endpoint or resample count differs")
    elif stages or annealing_stages != 0 or resampling_events != 0:
        raise V3ValidationError("terminal-IS arm recorded bridge stages")
    return task_id, arm_id, particles, repetition


def validate_ledger(
    runs: Sequence[Mapping[str, object]],
    references: Mapping[str, Mapping[str, object]],
) -> tuple[
    dict[str, Mapping[str, object]],
    dict[tuple[str, str, int, int], Mapping[str, object]],
]:
    """Validate and index the exact 10,240-row frozen V3 ledger."""

    validated_references = validate_references(references)
    if isinstance(runs, (str, bytes)) or len(runs) != EXPECTED_RUN_COUNT:
        raise V3ValidationError(
            f"run count {len(runs)} differs from {EXPECTED_RUN_COUNT}"
        )
    ledger: dict[tuple[str, str, int, int], Mapping[str, object]] = {}
    for value in runs:
        key = _validate_run(value, validated_references)
        if key in ledger:
            raise V3ValidationError(f"duplicate run row: {key}")
        ledger[key] = value
    expected = {
        (task_id, arm_id, particles, repetition)
        for task_id in TASK_IDS
        for arm_id, particles in EXPECTED_CELLS
        for repetition in range(REPETITIONS)
    }
    if set(ledger) != expected:
        missing = len(expected - set(ledger))
        extra = len(set(ledger) - expected)
        raise V3ValidationError(
            f"run grid differs (missing={missing}, extra={extra})"
        )
    return validated_references, ledger


def _quantile(values: Sequence[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not records:
        raise V3ValidationError("cannot summarize an empty run group")
    exact_errors = np.asarray(
        [
            float(cast(Mapping[str, object], record["error"])["exact_mass_signed"])
            for record in records
        ],
        dtype=np.float64,
    )
    loss_errors = np.asarray(
        [
            float(
                cast(Mapping[str, object], record["error"])[
                    "target_mean_loss_signed"
                ]
            )
            for record in records
        ],
        dtype=np.float64,
    )
    relative_ess = [
        float(
            cast(Mapping[str, object], record["diagnostics"])[
                "relative_importance_ess"
            ]
        )
        for record in records
    ]
    max_weights = [
        float(
            cast(Mapping[str, object], record["diagnostics"])[
                "maximum_normalized_weight"
            ]
        )
        for record in records
    ]
    draws = [float(record["logical_proposal_draws"]) for record in records]
    task_count = len({cast(str, record["task_id"]) for record in records})
    return {
        "observations": len(records),
        "task_count": task_count,
        "task_weighting": "equal-task",
        "exact_mass": {
            "bias": float(exact_errors.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(exact_errors)))),
            "mae": float(np.mean(np.abs(exact_errors))),
        },
        "target_mean_loss": {
            "bias": float(loss_errors.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(loss_errors)))),
            "mae": float(np.mean(np.abs(loss_errors))),
        },
        "relative_importance_ess": {
            "mean": statistics.fmean(relative_ess),
            "q05": _quantile(relative_ess, 0.05),
        },
        "maximum_normalized_weight": {
            "mean": statistics.fmean(max_weights),
            "q95": _quantile(max_weights, 0.95),
            "maximum": max(max_weights),
            "fraction_above_0_25": sum(value > 0.25 for value in max_weights)
            / len(max_weights),
        },
        "logical_proposal_draws": {
            "mean": statistics.fmean(draws),
            "maximum": max(draws),
        },
    }


def task_first_bootstrap(
    task_errors: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Run the frozen task-first, then within-task, nonparametric bootstrap."""

    if task_errors.shape != (TASK_COUNT, REPETITIONS):
        raise V3ValidationError(
            "bootstrap error matrix must have shape "
            f"({TASK_COUNT}, {REPETITIONS})"
        )
    if not _is_int(replicates) or replicates < 1:
        raise V3ValidationError("bootstrap replicate count is invalid")
    if not _is_int(seed) or seed < 0:
        raise V3ValidationError("bootstrap seed is invalid")
    if not np.isfinite(task_errors).all():
        raise V3ValidationError("bootstrap errors are not finite")
    rng = np.random.Generator(np.random.PCG64(seed))
    biases = np.empty(replicates, dtype=np.float64)
    rmses = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        task_indices = rng.integers(0, TASK_COUNT, size=TASK_COUNT)
        repetition_indices = rng.integers(
            0,
            REPETITIONS,
            size=(TASK_COUNT, REPETITIONS),
        )
        sampled = task_errors[task_indices[:, None], repetition_indices]
        biases[index] = float(sampled.mean())
        rmses[index] = float(np.sqrt(np.mean(np.square(sampled))))
    return {
        "method": "task-first-then-within-task-nonparametric-bootstrap",
        "replicates": replicates,
        "seed": seed,
        "rmse_upper_95": float(np.quantile(rmses, 0.95)),
        "rmse_interval_95": [
            float(np.quantile(rmses, 0.025)),
            float(np.quantile(rmses, 0.975)),
        ],
        "bias_interval_90": [
            float(np.quantile(biases, 0.05)),
            float(np.quantile(biases, 0.95)),
        ],
        "bias_interval_95": [
            float(np.quantile(biases, 0.025)),
            float(np.quantile(biases, 0.975)),
        ],
    }


def analyze(
    runs: Sequence[Mapping[str, object]],
    references: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Recompute all equal-task summaries and the frozen V3 gate."""

    validated_references, ledger = validate_ledger(runs, references)

    pooled: dict[str, object] = {}
    per_task: dict[str, object] = {}
    for arm_id, particle_counts in ARM_PARTICLES.items():
        pooled_by_n: dict[str, object] = {}
        for particles in particle_counts:
            group = [
                ledger[(task_id, arm_id, particles, repetition)]
                for task_id in TASK_IDS
                for repetition in range(REPETITIONS)
            ]
            pooled_by_n[str(particles)] = _summary(group)
        pooled[arm_id] = {
            "role": ARM_ROLES[arm_id],
            "by_particle_count": pooled_by_n,
        }
    for task_id in TASK_IDS:
        task_arms: dict[str, object] = {}
        for arm_id, particle_counts in ARM_PARTICLES.items():
            by_n: dict[str, object] = {}
            for particles in particle_counts:
                group = [
                    ledger[(task_id, arm_id, particles, repetition)]
                    for repetition in range(REPETITIONS)
                ]
                by_n[str(particles)] = _summary(group)
            task_arms[arm_id] = {
                "role": ARM_ROLES[arm_id],
                "by_particle_count": by_n,
            }
        per_task[task_id] = {"arms": task_arms}

    primary_errors = np.empty((TASK_COUNT, REPETITIONS), dtype=np.float64)
    for task_index, task_id in enumerate(TASK_IDS):
        primary_errors[task_index] = [
            float(
                cast(
                    Mapping[str, object],
                    ledger[(task_id, PRIMARY_ARM_ID, PRIMARY_PARTICLE_COUNT, repetition)][
                        "error"
                    ],
                )["exact_mass_signed"]
            )
            for repetition in range(REPETITIONS)
        ]
    bootstrap = task_first_bootstrap(
        primary_errors,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )

    primary_by_n = cast(
        Mapping[str, object], pooled[PRIMARY_ARM_ID]
    )["by_particle_count"]
    rmse_by_n = {
        particles: float(
            cast(
                Mapping[str, object],
                cast(Mapping[str, object], primary_by_n[str(particles)])["exact_mass"],
            )["rmse"]
        )
        for particles in PARTICLE_COUNTS
    }
    task_rmses = {
        task_id: float(
            cast(
                Mapping[str, object],
                cast(
                    Mapping[str, object],
                    cast(
                        Mapping[str, object],
                        cast(Mapping[str, object], per_task[task_id])["arms"],
                    )[PRIMARY_ARM_ID],
                )["by_particle_count"],
            )[str(PRIMARY_PARTICLE_COUNT)]["exact_mass"]["rmse"]
        )
        for task_id in TASK_IDS
    }
    primary_256 = cast(Mapping[str, object], primary_by_n[str(PRIMARY_PARTICLE_COUNT)])
    max_weight = cast(
        Mapping[str, object], primary_256["maximum_normalized_weight"]
    )
    identity_error = max(
        float(
            cast(Mapping[str, object], reference["proposal_census"])[arm_id][
                "importance_identity_maximum_absolute_error"
            ]
        )
        for reference in validated_references.values()
        for arm_id in ARM_PARTICLES
    )
    bias_interval_90 = cast(list[float], bootstrap["bias_interval_90"])
    checks = {
        "identity_maximum_at_most_1e_12": identity_error <= IDENTITY_TOLERANCE,
        "rmse_upper_95_strictly_below_0_10": (
            float(bootstrap["rmse_upper_95"]) < 0.10
        ),
        "bias_interval_90_strictly_inside_equivalence": (
            bias_interval_90[0] > -0.03 and bias_interval_90[1] < 0.03
        ),
        "every_task_rmse_strictly_below_0_20": max(task_rmses.values()) < 0.20,
        "point_rmse_nonincreasing_256_512_1024": (
            rmse_by_n[512] <= rmse_by_n[256]
            and rmse_by_n[1024] <= rmse_by_n[512]
        ),
        "primary_weight_tail": (
            float(max_weight["q95"]) < 0.05
            and float(max_weight["fraction_above_0_25"]) <= 0.01
        ),
    }
    return {
        "schema": ANALYSIS_SCHEMA,
        "status": "completed-fresh-v3-calibration-analysis",
        "run_count": EXPECTED_RUN_COUNT,
        "task_count": TASK_COUNT,
        "cell_count": len(EXPECTED_CELLS),
        "repetitions_per_task_cell": REPETITIONS,
        "task_weighting": "equal-task",
        "provider_calls": 0,
        "pooled": pooled,
        "per_task": per_task,
        "bootstrap": bootstrap,
        "primary_gate": {
            "all_required": True,
            "checks": checks,
            "pass": all(checks.values()),
            "primary_arm_id": PRIMARY_ARM_ID,
            "primary_particle_count": PRIMARY_PARTICLE_COUNT,
            "maximum_identity_error": identity_error,
            "task_rmse": task_rmses,
            "rmse_by_particle_count": {
                str(particles): rmse_by_n[particles] for particles in PARTICLE_COUNTS
            },
            "weight_tail": {
                "q95_maximum_normalized_weight": float(max_weight["q95"]),
                "fraction_above_0_25": float(max_weight["fraction_above_0_25"]),
            },
        },
        "claim_boundary": (
            "fresh provider-free terminal finite-support calibration on the declared "
            "singleton-complete synthetic generator; primary factorized acquisition is "
            "completed before exact reference enumeration, while the descriptive sticky "
            "baseline is reference-derived; no LLM, full-pipeline, or large-DSL claim"
        ),
    }


def _safe_inventory_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise V3ValidationError("inventory path is not a safe POSIX relative path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
        or value == "inventory.json"
    ):
        raise V3ValidationError(f"unsafe inventory path: {value!r}")
    return value


def _artifact_files(output: Path) -> list[Path]:
    if output.is_symlink() or not output.is_dir():
        raise V3ValidationError("artifact root is not a regular directory")
    members = list(output.rglob("*"))
    if any(member.is_symlink() for member in members):
        raise V3ValidationError("artifact tree contains a symlink")
    return sorted(
        (
            member
            for member in members
            if member.is_file() and member.relative_to(output).as_posix() != "inventory.json"
        ),
        key=lambda member: member.relative_to(output).as_posix().encode("utf-8"),
    )


def _write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")


def seal_inventory(output: Path) -> dict[str, object]:
    """Exclusively write the deterministic native V3 artifact inventory."""

    files = _artifact_files(output)
    entries = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "file_count_excluding_inventory": len(entries),
        "entries": entries,
        "entries_sha256": _sha256_bytes(canonical_bytes(entries)),
    }
    _write_json_exclusive(output / "inventory.json", inventory)
    return inventory


def validate_inventory(output: Path) -> dict[str, object]:
    """Validate canonical inventory bytes, safe paths, and every artifact hash."""

    inventory_path = output / "inventory.json"
    inventory_value = _read_json(inventory_path)
    inventory = dict(_require_mapping(inventory_value, name="artifact inventory"))
    if set(inventory) != {
        "schema",
        "file_count_excluding_inventory",
        "entries",
        "entries_sha256",
    }:
        raise V3ValidationError("inventory fields differ from the V3 schema")
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise V3ValidationError("inventory schema differs")
    if inventory_path.read_bytes() != canonical_bytes(inventory) + b"\n":
        raise V3ValidationError("inventory is not in canonical byte form")
    entries = inventory.get("entries")
    if not isinstance(entries, list):
        raise V3ValidationError("inventory entries are malformed")
    count = _require_int(
        inventory.get("file_count_excluding_inventory"),
        name="inventory file count",
        minimum=0,
    )
    if count != len(entries):
        raise V3ValidationError("inventory file count differs from its entries")
    expected_digest = _require_digest(
        inventory.get("entries_sha256"), name="inventory entries SHA-256"
    )
    if expected_digest != _sha256_bytes(canonical_bytes(entries)):
        raise V3ValidationError("inventory entries digest differs")

    normalized: list[dict[str, object]] = []
    paths: list[str] = []
    for value in entries:
        record = dict(_require_mapping(value, name="inventory entry"))
        if set(record) != {"path", "bytes", "sha256"}:
            raise V3ValidationError("inventory entry fields differ")
        relative = _safe_inventory_path(record.get("path"))
        byte_count = _require_int(
            record.get("bytes"), name=f"inventory byte count for {relative}", minimum=0
        )
        digest = _require_digest(
            record.get("sha256"), name=f"inventory SHA-256 for {relative}"
        )
        paths.append(relative)
        normalized.append({"path": relative, "bytes": byte_count, "sha256": digest})
    if paths != sorted(paths, key=lambda value: value.encode("utf-8")):
        raise V3ValidationError("inventory paths are not deterministically sorted")
    if len(paths) != len(set(paths)):
        raise V3ValidationError("inventory contains duplicate paths")

    actual_files = _artifact_files(output)
    actual_paths = [path.relative_to(output).as_posix() for path in actual_files]
    if paths != actual_paths:
        raise V3ValidationError("artifact path set differs from its inventory")
    for record, path in zip(normalized, actual_files, strict=True):
        if path.stat().st_size != record["bytes"] or _sha256_file(path) != record["sha256"]:
            raise V3ValidationError(f"artifact file differs: {record['path']}")
    return inventory


def _load_artifact_references(output: Path) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for task_id in TASK_IDS:
        value = _read_json(output / "references" / f"{task_id}.json")
        result[task_id] = _require_mapping(value, name=f"reference file {task_id}")
    return result


def validate_and_recompute_artifact(output: Path) -> dict[str, object]:
    """Validate a finalized artifact and byte-recompute its complete analysis."""

    inventory = validate_inventory(output)
    entries = cast(list[Mapping[str, object]], inventory["entries"])
    paths = {cast(str, entry["path"]) for entry in entries}
    if paths != EXPECTED_ARTIFACT_PATHS:
        missing = sorted(EXPECTED_ARTIFACT_PATHS - paths)
        extra = sorted(paths - EXPECTED_ARTIFACT_PATHS)
        raise V3ValidationError(
            f"finalized artifact path set differs (missing={missing}, extra={extra})"
        )

    protocol = _require_mapping(_read_json(output / "protocol.json"), name="protocol")
    if protocol.get("schema") != STUDY_SCHEMA:
        raise V3ValidationError("artifact protocol schema differs")
    authorization = _require_mapping(
        protocol.get("authorization"), name="protocol authorization"
    )
    if authorization.get("provider_calls") is not False:
        raise V3ValidationError("artifact protocol authorizes provider calls")
    task_distribution = _require_mapping(
        protocol.get("task_distribution"), name="protocol task distribution"
    )
    if (
        task_distribution.get("task_count") != TASK_COUNT
        or task_distribution.get("task_ids") != list(TASK_IDS)
    ):
        raise V3ValidationError("artifact protocol task design differs")
    design = _require_mapping(protocol.get("design"), name="protocol design")
    expected_arm_particles = {
        arm_id: list(particle_counts)
        for arm_id, particle_counts in ARM_PARTICLES.items()
    }
    if (
        design.get("arms") != list(ARM_PARTICLES)
        or design.get("arm_particles") != expected_arm_particles
        or design.get("repetitions") != REPETITIONS
        or design.get("repetition_seeds") != list(REPETITION_SEEDS)
        or design.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
        or design.get("bootstrap_seed") != BOOTSTRAP_SEED
        or design.get("expected_run_count") != EXPECTED_RUN_COUNT
        or design.get("expected_initial_proposal_draws")
        != EXPECTED_INITIAL_PROPOSAL_DRAWS
    ):
        raise V3ValidationError("artifact protocol Monte Carlo design differs")
    manifest = _require_mapping(
        _read_json(output / "public-suite-manifest.json"), name="public manifest"
    )
    if set(manifest) != {
        "schema",
        "protocol_sha256",
        "method_seal_sha256",
        "custody_seal_sha256",
        "secret_commitment_sha256",
        "secret_commitment_under_v2_domain_sha256",
        "prior_v2_secret_commitment_sha256",
        "same_domain_secret_nonreuse_verified",
        "task_count",
        "task_ids",
        "tasks",
    }:
        raise V3ValidationError("artifact public-manifest fields differ")
    if manifest.get("schema") != f"{STUDY_SCHEMA}-public-suite-v1":
        raise V3ValidationError("artifact public-manifest schema differs")
    if (
        manifest.get("task_count") != TASK_COUNT
        or manifest.get("task_ids") != list(TASK_IDS)
    ):
        raise V3ValidationError("artifact public-manifest task design differs")
    protocol_sha256 = _sha256_file(output / "protocol.json")
    method_seal_sha256 = _require_digest(
        manifest.get("method_seal_sha256"), name="manifest method-seal SHA-256"
    )
    custody_seal_sha256 = _require_digest(
        manifest.get("custody_seal_sha256"), name="manifest custody-seal SHA-256"
    )
    _require_digest(
        manifest.get("secret_commitment_sha256"), name="manifest V3 commitment"
    )
    v2_domain_commitment = _require_digest(
        manifest.get("secret_commitment_under_v2_domain_sha256"),
        name="manifest V2-domain commitment",
    )
    if (
        manifest.get("protocol_sha256") != protocol_sha256
        or manifest.get("prior_v2_secret_commitment_sha256")
        != PRIOR_V2_SECRET_COMMITMENT
        or manifest.get("same_domain_secret_nonreuse_verified") is not True
        or v2_domain_commitment == PRIOR_V2_SECRET_COMMITMENT
    ):
        raise V3ValidationError("artifact public-manifest binding differs")
    manifest_tasks = manifest.get("tasks")
    if not isinstance(manifest_tasks, list) or len(manifest_tasks) != TASK_COUNT:
        raise V3ValidationError("artifact public-manifest task records differ")
    metadata = _require_mapping(
        _read_json(output / "study-metadata.json"), name="study metadata"
    )
    if set(metadata) != {
        "schema",
        "protocol_sha256",
        "method_seal_sha256",
        "custody_seal_sha256",
        "public_manifest_sha256",
        "provider_calls",
        "private_reveal_read",
        "exact_support_materialized_after_mode_bank",
        "runtime",
    }:
        raise V3ValidationError("study metadata fields differ")
    if metadata.get("schema") != METADATA_SCHEMA:
        raise V3ValidationError("study metadata schema differs")
    if metadata.get("provider_calls") != 0 or isinstance(
        metadata.get("provider_calls"), bool
    ):
        raise V3ValidationError("study metadata is not provider-free")
    if metadata.get("private_reveal_read") is not False:
        raise V3ValidationError("study metadata does not preserve the blind boundary")
    if metadata.get("protocol_sha256") != protocol_sha256:
        raise V3ValidationError("study metadata protocol hash differs")
    if (
        metadata.get("method_seal_sha256") != method_seal_sha256
        or metadata.get("custody_seal_sha256") != custody_seal_sha256
    ):
        raise V3ValidationError("study metadata custody binding differs")
    if metadata.get("exact_support_materialized_after_mode_bank") is not True:
        raise V3ValidationError("study metadata acquisition/reference order differs")
    if metadata.get("public_manifest_sha256") != _sha256_file(
        output / "public-suite-manifest.json"
    ):
        raise V3ValidationError("study metadata public-manifest hash differs")

    runs_value = _read_json(output / "runs.json")
    if not isinstance(runs_value, list):
        raise V3ValidationError("runs.json is not a JSON array")
    runs = cast(list[Mapping[str, object]], runs_value)
    references = _load_artifact_references(output)
    for task_id, value in zip(TASK_IDS, manifest_tasks, strict=True):
        record = _require_mapping(value, name=f"public-manifest task {task_id}")
        if set(record) != {
            "task_id",
            "path",
            "task_sha256",
            "target_commitment_sha256",
        }:
            raise V3ValidationError("artifact public-manifest task fields differ")
        if record.get("task_id") != task_id:
            raise V3ValidationError("artifact public-manifest task order differs")
        if record.get("path") != f"{task_id}.json":
            raise V3ValidationError("artifact public-manifest task path differs")
        _require_digest(
            record.get("target_commitment_sha256"),
            name=f"public-manifest {task_id} target commitment",
        )
        manifest_digest = _require_digest(
            record.get("task_sha256"), name=f"public-manifest {task_id} task SHA-256"
        )
        if manifest_digest != references[task_id].get("task_sha256"):
            raise V3ValidationError(
                "artifact reference digest differs from the public manifest"
            )
    recomputed = analyze(runs, references)
    analysis_path = output / "analysis.json"
    stored_value = _read_json(analysis_path)
    _require_mapping(stored_value, name="stored analysis")
    expected_bytes = canonical_bytes(recomputed) + b"\n"
    if analysis_path.read_bytes() != expected_bytes:
        raise V3ValidationError(
            "analysis.json is not byte-identical to native V3 recomputation"
        )
    return recomputed


def validate_artifact(output: Path) -> dict[str, object]:
    """Alias for full inventory, ledger, and byte-recomputation validation."""

    return validate_and_recompute_artifact(output)

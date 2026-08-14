"""Terminal-only diagnosis of provider-free particle-calibration V1.

This is a separate post-result diagnostic.  It never edits or supersedes the
frozen V1 protocol, harness, gate, or artifact.  For each of the four enumerable
tasks it constructs the exact 36,000-program terminal target, mechanically picks
one canonical zero-loss parent, and compares six exactly evaluable proposals.

The global top-64 arm is an exhaustive, posthoc developmental positive control.
It is not a search algorithm and supports no search-efficiency claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np

from modelsmc_pbe.config import ExperimentConfig, load_experiment_config
from modelsmc_pbe.core.evaluate import evaluate_expression
from modelsmc_pbe.core.results import ScoredProgram
from modelsmc_pbe.core.scorer import ProgramScorer
from modelsmc_pbe.core.types import Node
from modelsmc_pbe.grammar import arithmetic_expressions, filter_predicates
from research.evidence_shortlist_smc import (
    ProgramKey,
    program_grammar_probability,
    python_tree_binding,
    recursive_grammar_probabilities,
)
from research.execution_guided_repair import _assemble_program, _dsl_catalog
from research.iterative_beam_experiment import derive_singleton_constraints

STUDY_SCHEMA = "provider-free-particle-calibration-terminal-diagnostic-v2"
RUN_SCHEMA = f"{STUDY_SCHEMA}-run-v1"
REFERENCE_SCHEMA = f"{STUDY_SCHEMA}-reference-v1"
ANALYSIS_SCHEMA = f"{STUDY_SCHEMA}-analysis-v1"
STATUS = "frozen-provider-free-posthoc-terminal-diagnostic-before-v2-draws"

HARNESS_PATH = "research/particle_calibration_terminal_diagnostic_v2.py"
TEST_PATH = "research/tests/test_particle_calibration_terminal_diagnostic_v2.py"
DEFAULT_PROTOCOL = "research/protocol-particle-calibration-terminal-diagnostic-v2.json"
DEFAULT_OUTPUT = "artifacts/particle-calibration-terminal-diagnostic-v2"

PARTICLE_COUNTS = (256, 512)
REPETITIONS = 128
REPETITION_SEEDS = tuple(range(842001, 842001 + REPETITIONS))
GLOBAL_SLATE_SIZE = 64
IDENTITY_TOLERANCE = 1e-12

DEPENDENCY_PATHS = (
    "research/adaptive_shortlist_experiment.py",
    "research/automatic_repair_feedback.py",
    "research/automatic_shortlist_experiment.py",
    "research/direct_json_repair_choice.py",
    "research/evidence_shortlist_smc.py",
    "research/execution_guided_repair.py",
    "research/iterative_beam_experiment.py",
    "research/local_repair_importance.py",
)
LOCKFILE_PATHS = ("pyproject.toml", "uv.lock")
TASK_SPECS = (
    ("foldr-bounded-square", "examples/foldr-bounded-square.json"),
    (
        "calibration-lower-shift",
        "examples/calibration-filter-map-lower-shift-v1.json",
    ),
    (
        "calibration-upper-negate",
        "examples/calibration-filter-map-upper-negate-v1.json",
    ),
    (
        "calibration-equality-scale",
        "examples/calibration-filter-map-equality-scale-v1.json",
    ),
)


class DiagnosticError(ValueError):
    """A frozen diagnostic invariant was violated."""


@dataclass(frozen=True, slots=True)
class ArmSpec:
    arm_id: str
    proposal_kind: str
    epsilon: float
    local_slots: int
    role: str


ARMS = (
    ArmSpec(
        "sticky-epsilon-005",
        "sticky-exact-parent",
        0.05,
        4,
        "v1-terminal-mechanism",
    ),
    ArmSpec(
        "sticky-epsilon-025",
        "sticky-exact-parent",
        0.25,
        4,
        "defensive-restart-ladder",
    ),
    ArmSpec(
        "sticky-epsilon-050",
        "sticky-exact-parent",
        0.50,
        4,
        "defensive-restart-ladder",
    ),
    ArmSpec(
        "grammar-only-epsilon-100",
        "sticky-exact-parent",
        1.00,
        4,
        "defensive-restart-ladder",
    ),
    ArmSpec(
        "nonsticky-mapper-epsilon-005",
        "nonsticky-mapper-top4",
        0.05,
        4,
        "remove-exact-absorption-control",
    ),
    ArmSpec(
        "global-top64-epsilon-050-positive-control",
        "global-public-score-top64",
        0.50,
        GLOBAL_SLATE_SIZE,
        "posthoc-developmental-positive-control",
    ),
)


@dataclass(frozen=True, slots=True)
class TaskBinding:
    task_id: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class StudyPlan:
    tasks: tuple[TaskBinding, ...]
    arms: tuple[ArmSpec, ...]
    particle_counts: tuple[int, ...]
    repetition_seeds: tuple[int, ...]
    protocol_sha256: str
    harness_sha256: str


@dataclass(frozen=True, slots=True)
class ArmDistribution:
    arm: ArmSpec
    probabilities: np.ndarray
    normalized_importance_ratios: np.ndarray
    local_indices: tuple[int, ...]
    census: dict[str, object]


def canonical_bytes(value: object) -> bytes:
    """Return stable strict-JSON bytes."""

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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiagnosticError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DiagnosticError(f"{name} must be finite")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DiagnosticError(f"{name} must be a positive integer")
    return value


def _runtime_record() -> dict[str, object]:
    return {
        "python_major_minor": ".".join(platform.python_version_tuple()[:2]),
        "python_version": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pydantic", "torch")
        },
        "calculation_dtype": "numpy.float64",
        "device": "cpu",
    }


def _derived_seed(
    *,
    task_sha256: str,
    arm_id: str,
    particles: int,
    base_seed: int,
) -> int:
    digest = hashlib.sha256(
        (
            f"{STUDY_SCHEMA}\0terminal-draw\0{task_sha256}\0{arm_id}\0"
            f"{particles}\0{base_seed}"
        ).encode()
    ).digest()
    return int.from_bytes(digest[:16], "big")


class TerminalTask:
    """Exact terminal population and the six frozen proposal distributions."""

    def __init__(self, binding: TaskBinding) -> None:
        self.binding = binding
        self.config: ExperimentConfig = load_experiment_config(binding.path)
        signature = self.config.spec.signature
        if signature is None or signature.input_type.value != "List<Int>":
            raise DiagnosticError(f"{binding.task_id} must take one List<Int> input")
        if signature.output_type.value != "List<Int>":
            raise DiagnosticError(f"{binding.task_id} must return List<Int>")

        self.scorer = ProgramScorer(self.config)
        constants = tuple(self.config.spec.integer_constants)
        self.predicate_catalog = _dsl_catalog(filter_predicates(constants))
        self.mapper_catalog = _dsl_catalog(arithmetic_expressions("Item", constants))
        self.predicate_order = tuple(sorted(self.predicate_catalog))
        self.mapper_order = tuple(sorted(self.mapper_catalog))
        if len(self.predicate_order) != len(self.predicate_catalog):
            raise DiagnosticError("predicate catalog contains duplicate syntax")
        if len(self.mapper_order) != len(self.mapper_catalog):
            raise DiagnosticError("mapper catalog contains duplicate syntax")
        (
            self.predicate_probability,
            self.mapper_probability,
        ) = recursive_grammar_probabilities(self.predicate_order, self.mapper_order)

        observed_items = tuple(
            sorted(
                {
                    item
                    for example in self.config.spec.examples
                    for item in cast(list[int], example.input_value)
                }
            )
        )
        if not observed_items:
            raise DiagnosticError(f"{binding.task_id} has no observed scalar items")
        self.observed_items = observed_items
        self.observed_index = {item: index for index, item in enumerate(observed_items)}
        self.singleton = derive_singleton_constraints(self.config.spec)
        self.mapper_signatures = self._mapper_signatures()
        self.score_cache: dict[ProgramKey, ScoredProgram] = {}

        keys: list[ProgramKey] = []
        grammar: list[float] = []
        log_gamma: list[float] = []
        exact: list[float] = []
        losses: list[float] = []
        costs: list[int] = []
        inventory_hash = hashlib.sha256()
        for predicate in self.predicate_order:
            for mapper in self.mapper_order:
                key = ProgramKey(predicate, mapper)
                score = self.score_key(key)
                prior = program_grammar_probability(
                    key,
                    self.predicate_probability,
                    self.mapper_probability,
                )
                keys.append(key)
                grammar.append(float(prior))
                log_gamma.append(math.log(float(prior)) + score.log_target)
                exact.append(float(score.exact_program))
                losses.append(score.total_loss)
                costs.append(score.cost)
                inventory_hash.update(
                    canonical_bytes(
                        {
                            "program": asdict(key),
                            "grammar_probability": {
                                "numerator": prior.numerator,
                                "denominator": prior.denominator,
                            },
                            "total_loss": score.total_loss,
                            "cost": score.cost,
                            "log_target": score.log_target,
                            "exact_program": score.exact_program,
                        }
                    )
                    + b"\n"
                )
        expected_programs = len(self.predicate_order) * len(self.mapper_order)
        if len(keys) != expected_programs or expected_programs != 36_000:
            raise DiagnosticError(
                f"{binding.task_id} terminal support must contain exactly 36,000 programs"
            )

        self.keys = tuple(keys)
        self.key_index = {key: index for index, key in enumerate(self.keys)}
        self.grammar = np.asarray(grammar, dtype=np.float64)
        self.log_gamma = np.asarray(log_gamma, dtype=np.float64)
        self.exact = np.asarray(exact, dtype=np.float64)
        self.losses = np.asarray(losses, dtype=np.float64)
        self.costs = np.asarray(costs, dtype=np.int64)
        if abs(float(self.grammar.sum()) - 1.0) > 1e-12:
            raise DiagnosticError("recursive grammar probabilities do not sum to one")
        maximum = float(self.log_gamma.max())
        scaled = np.exp(self.log_gamma - maximum)
        scaled_sum = float(scaled.sum())
        self.target = scaled / scaled_sum
        self.log_normalizer = maximum + math.log(scaled_sum)
        self.exact_mass = float(self.target @ self.exact)
        self.target_mean_loss = float(self.target @ self.losses)
        self.program_inventory_sha256 = inventory_hash.hexdigest()

        exact_indices = np.flatnonzero(self.exact == 1.0).tolist()
        if not exact_indices:
            raise DiagnosticError(f"{binding.task_id} has no exact program")
        self.fixed_parent_index = min(
            exact_indices,
            key=lambda index: (
                int(self.costs[index]),
                self.keys[index].predicate,
                self.keys[index].mapper,
            ),
        )
        self.fixed_parent = self.keys[self.fixed_parent_index]
        self.nonsticky_mapper_indices = self._nonsticky_mapper_indices()
        self.global_indices = tuple(
            sorted(
                range(len(self.keys)),
                key=lambda index: (
                    float(self.losses[index]),
                    int(self.costs[index]),
                    self.keys[index].predicate,
                    self.keys[index].mapper,
                ),
            )[:GLOBAL_SLATE_SIZE]
        )
        if len(set(self.global_indices)) != GLOBAL_SLATE_SIZE:
            raise DiagnosticError("global positive-control slate is not unique")
        self._arm_cache: dict[str, ArmDistribution] = {}

    def _mapper_signatures(self) -> dict[str, tuple[int, ...]]:
        signatures: dict[str, tuple[int, ...]] = {}
        for dsl, expression in self.mapper_catalog.items():
            values = tuple(
                evaluate_expression(
                    cast(Node, expression),
                    list(self.observed_items),
                    item=item,
                )
                for item in self.observed_items
            )
            if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
                raise DiagnosticError("mapper catalog returned a non-integer value")
            signatures[dsl] = cast(tuple[int, ...], values)
        return signatures

    def score_key(self, key: ProgramKey) -> ScoredProgram:
        cached = self.score_cache.get(key)
        if cached is not None:
            return cached
        score = self.scorer.score(
            _assemble_program(
                self.predicate_catalog[key.predicate],
                self.mapper_catalog[key.mapper],
            )
        )
        if not isinstance(score, ScoredProgram):
            raise DiagnosticError(f"grammar-valid program was rejected: {score.reason}")
        self.score_cache[key] = score
        return score

    def _mapper_violation_count(self, mapper: str) -> int:
        signature = self.mapper_signatures[mapper]
        return sum(
            signature[self.observed_index[cast(int, fact["item"])]]
            != cast(int, fact["expected_value"])
            for fact in self.singleton["mapper"]
        )

    def _nonsticky_mapper_indices(self) -> tuple[int, ...]:
        ranked: list[tuple[tuple[int | float | str, ...], int]] = []
        for mapper in self.mapper_order:
            key = ProgramKey(self.fixed_parent.predicate, mapper)
            if key == self.fixed_parent:
                continue
            score = self.score_key(key)
            ranked.append(
                (
                    (
                        self._mapper_violation_count(mapper),
                        score.total_loss,
                        score.cost,
                        mapper,
                    ),
                    self.key_index[key],
                )
            )
        ranked.sort(key=lambda item: item[0])
        result = tuple(index for _, index in ranked[:4])
        if len(result) != 4 or len(set(result)) != 4:
            raise DiagnosticError("nonsticky mapper proposal did not produce four slots")
        if self.fixed_parent_index in result:
            raise DiagnosticError("nonsticky mapper proposal retained the exact parent")
        return result

    def _local_indices(self, arm: ArmSpec) -> tuple[int, ...]:
        if arm.proposal_kind == "sticky-exact-parent":
            return (self.fixed_parent_index,) * arm.local_slots
        if arm.proposal_kind == "nonsticky-mapper-top4":
            return self.nonsticky_mapper_indices
        if arm.proposal_kind == "global-public-score-top64":
            return self.global_indices
        raise DiagnosticError(f"unsupported proposal kind {arm.proposal_kind}")

    def arm_distribution(self, arm: ArmSpec) -> ArmDistribution:
        cached = self._arm_cache.get(arm.arm_id)
        if cached is not None:
            return cached
        local_indices = self._local_indices(arm)
        if len(local_indices) != arm.local_slots:
            raise DiagnosticError(f"{arm.arm_id} local-slot count differs")
        probabilities = arm.epsilon * self.grammar.copy()
        local_mass = (1.0 - arm.epsilon) / len(local_indices)
        for index in local_indices:
            probabilities[index] += local_mass
        theoretical_sum_error = float(probabilities.sum()) - 1.0
        probabilities /= float(probabilities.sum())
        if np.any(probabilities <= 0.0):
            raise DiagnosticError(f"{arm.arm_id} does not have full support")
        ratios = self.target / probabilities
        exact_weight_mean = float(probabilities @ ratios)
        exact_numerator_mean = float(probabilities @ (ratios * self.exact))
        loss_numerator_mean = float(probabilities @ (ratios * self.losses))
        second_moment = float(probabilities @ np.square(ratios))
        ess_fraction = 1.0 / second_moment
        exact_asymptotic_variance = float(
            np.sum(
                np.square(self.target)
                * np.square(self.exact - self.exact_mass)
                / probabilities
            )
        )
        loss_asymptotic_variance = float(
            np.sum(
                np.square(self.target)
                * np.square(self.losses - self.target_mean_loss)
                / probabilities
            )
        )
        unique_local = tuple(sorted(set(local_indices)))
        local_mask = np.zeros(len(self.keys), dtype=bool)
        local_mask[list(unique_local)] = True
        census: dict[str, object] = {
            "arm_id": arm.arm_id,
            "proposal_kind": arm.proposal_kind,
            "epsilon": arm.epsilon,
            "role": arm.role,
            "local_slot_count_with_multiplicity": len(local_indices),
            "unique_local_programs": len(unique_local),
            "local_programs": [asdict(self.keys[index]) for index in local_indices],
            "proposal_probability_sum": float(probabilities.sum()),
            "proposal_sum_error_before_float_renormalization": theoretical_sum_error,
            "minimum_proposal_probability": float(probabilities.min()),
            "maximum_proposal_probability": float(probabilities.max()),
            "proposal_exact_probability": float(probabilities @ self.exact),
            "proposal_probability_on_local_support": float(
                probabilities[local_mask].sum()
            ),
            "target_mass_on_local_support": float(self.target[local_mask].sum()),
            "target_exact_mass_on_local_support": float(
                self.target[local_mask] @ self.exact[local_mask]
            ),
            "importance_identity_weight_mean": exact_weight_mean,
            "importance_identity_exact_numerator": exact_numerator_mean,
            "importance_identity_loss_numerator": loss_numerator_mean,
            "importance_identity_maximum_absolute_error": max(
                abs(exact_weight_mean - 1.0),
                abs(exact_numerator_mean - self.exact_mass),
                abs(loss_numerator_mean - self.target_mean_loss),
            ),
            "normalized_importance_weight_second_moment": second_moment,
            "population_importance_ess_fraction": ess_fraction,
            "chi_square_target_to_proposal": second_moment - 1.0,
            "maximum_normalized_importance_ratio": float(ratios.max()),
            "exact_mass_snis_asymptotic_variance": exact_asymptotic_variance,
            "target_mean_loss_snis_asymptotic_variance": loss_asymptotic_variance,
            "exact_mass_asymptotic_sd_by_particle_count": {
                str(n): math.sqrt(exact_asymptotic_variance / n)
                for n in PARTICLE_COUNTS
            },
            "target_mean_loss_asymptotic_sd_by_particle_count": {
                str(n): math.sqrt(loss_asymptotic_variance / n)
                for n in PARTICLE_COUNTS
            },
            "particles_for_exact_mass_asymptotic_sd_0_03": math.ceil(
                exact_asymptotic_variance / 0.03**2
            ),
            "particles_for_exact_mass_asymptotic_rmse_0_10": math.ceil(
                exact_asymptotic_variance / 0.10**2
            ),
        }
        result = ArmDistribution(
            arm=arm,
            probabilities=probabilities,
            normalized_importance_ratios=ratios,
            local_indices=local_indices,
            census=census,
        )
        self._arm_cache[arm.arm_id] = result
        return result

    def reference_record(self, arms: Sequence[ArmSpec]) -> dict[str, object]:
        distributions = [self.arm_distribution(arm) for arm in arms]
        result = {
            "schema": REFERENCE_SCHEMA,
            "task_id": self.binding.task_id,
            "task_sha256": self.binding.sha256,
            "program_syntaxes": len(self.keys),
            "predicate_syntaxes": len(self.predicate_order),
            "mapper_syntaxes": len(self.mapper_order),
            "exact_program_syntaxes": int(self.exact.sum()),
            "log_terminal_normalizer": self.log_normalizer,
            "exact_target_mass": self.exact_mass,
            "target_mean_loss": self.target_mean_loss,
            "terminal_target_ess": 1.0 / float(np.square(self.target).sum()),
            "maximum_terminal_target_probability": float(self.target.max()),
            "program_population_sha256": self.program_inventory_sha256,
            "fixed_parent_selection": (
                "minimum (program cost, predicate syntax, mapper syntax) among all "
                "public-example zero-loss programs"
            ),
            "fixed_exact_parent": asdict(self.fixed_parent),
            "fixed_exact_parent_target_probability": float(
                self.target[self.fixed_parent_index]
            ),
            "global_positive_control_selection": (
                "first 64 complete programs under exhaustive public-example "
                "(loss, cost, predicate syntax, mapper syntax) ranking"
            ),
            "arm_census": {
                distribution.arm.arm_id: distribution.census
                for distribution in distributions
            },
        }
        _validate_reference(result, arms=arms)
        return result


def _validate_reference(
    reference: Mapping[str, object],
    *,
    arms: Sequence[ArmSpec],
) -> None:
    if reference.get("schema") != REFERENCE_SCHEMA:
        raise DiagnosticError("reference schema differs")
    if reference.get("program_syntaxes") != 36_000:
        raise DiagnosticError("reference support size differs")
    mass = _finite_number(reference.get("exact_target_mass"), name="exact target mass")
    loss = _finite_number(reference.get("target_mean_loss"), name="target mean loss")
    if not 0.0 <= mass <= 1.0 or loss < 0.0:
        raise DiagnosticError("reference endpoint is outside its valid range")
    census = reference.get("arm_census")
    if not isinstance(census, Mapping) or set(census) != {arm.arm_id for arm in arms}:
        raise DiagnosticError("reference arm census differs")
    for arm in arms:
        record = cast(Mapping[str, object], census[arm.arm_id])
        error = _finite_number(
            record.get("importance_identity_maximum_absolute_error"),
            name=f"{arm.arm_id} identity error",
        )
        if error > IDENTITY_TOLERANCE:
            raise DiagnosticError(f"{arm.arm_id} exact importance identity failed")


def run_repetition(
    task: TerminalTask,
    distribution: ArmDistribution,
    *,
    particles: int,
    repetition: int,
    base_seed: int,
) -> dict[str, object]:
    """Run one terminal-only self-normalized importance population."""

    particles = _positive_int(particles, name="particles")
    seed = _derived_seed(
        task_sha256=task.binding.sha256,
        arm_id=distribution.arm.arm_id,
        particles=particles,
        base_seed=base_seed,
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled = rng.choice(
        len(task.keys),
        size=particles,
        replace=True,
        p=distribution.probabilities,
    )
    raw_weights = distribution.normalized_importance_ratios[sampled]
    weight_sum = float(raw_weights.sum())
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        raise DiagnosticError("sampled importance weights have an invalid sum")
    normalized = raw_weights / weight_sum
    exact_estimate = float(normalized @ task.exact[sampled])
    loss_estimate = float(normalized @ task.losses[sampled])
    ess = weight_sum**2 / float(np.square(raw_weights).sum())
    local_mask = np.zeros(len(task.keys), dtype=bool)
    local_mask[list(set(distribution.local_indices))] = True
    result = {
        "schema": RUN_SCHEMA,
        "task_id": task.binding.task_id,
        "task_sha256": task.binding.sha256,
        "arm_id": distribution.arm.arm_id,
        "proposal_kind": distribution.arm.proposal_kind,
        "arm_role": distribution.arm.role,
        "epsilon": distribution.arm.epsilon,
        "particles": particles,
        "repetition": repetition,
        "base_seed": base_seed,
        "sample_seed": seed,
        "provider_calls": 0,
        "terminal_proposal_draws": particles,
        "estimate": {
            "exact_target_mass": exact_estimate,
            "target_mean_loss": loss_estimate,
        },
        "reference": {
            "exact_target_mass": task.exact_mass,
            "target_mean_loss": task.target_mean_loss,
        },
        "error": {
            "exact_mass_signed": exact_estimate - task.exact_mass,
            "target_mean_loss_signed": loss_estimate - task.target_mean_loss,
        },
        "diagnostics": {
            "importance_ess": ess,
            "relative_importance_ess": ess / particles,
            "maximum_normalized_weight": float(normalized.max()),
            "maximum_raw_normalized_target_to_q_ratio": float(raw_weights.max()),
            "minimum_raw_normalized_target_to_q_ratio": float(raw_weights.min()),
            "unique_terminal_programs": int(np.unique(sampled).size),
            "exact_draws": int(task.exact[sampled].sum()),
            "draws_in_local_support": int(local_mask[sampled].sum()),
            "exact_draws_outside_local_support": int(
                (np.logical_and(task.exact[sampled] == 1.0, ~local_mask[sampled])).sum()
            ),
        },
    }
    _validate_run(result)
    return result


def _validate_run(run: Mapping[str, object]) -> None:
    if run.get("schema") != RUN_SCHEMA or run.get("provider_calls") != 0:
        raise DiagnosticError("run schema or provider-free invariant failed")
    particles = _positive_int(run.get("particles"), name="run particles")
    if run.get("terminal_proposal_draws") != particles:
        raise DiagnosticError("terminal draw accounting differs")
    estimate = run.get("estimate")
    diagnostics = run.get("diagnostics")
    if not isinstance(estimate, Mapping) or not isinstance(diagnostics, Mapping):
        raise DiagnosticError("run estimate or diagnostics is malformed")
    mass = _finite_number(estimate.get("exact_target_mass"), name="estimated mass")
    loss = _finite_number(estimate.get("target_mean_loss"), name="estimated loss")
    ess = _finite_number(diagnostics.get("importance_ess"), name="importance ESS")
    if not 0.0 <= mass <= 1.0 + 1e-12 or loss < 0.0:
        raise DiagnosticError("estimated endpoint is outside its valid range")
    if not 1.0 <= ess <= particles + 1e-9:
        raise DiagnosticError("sample importance ESS is outside its valid range")


def _sample_sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise DiagnosticError("cannot take a quantile of no values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _error_summary(
    estimates: Sequence[float],
    references: Sequence[float],
) -> dict[str, float | int]:
    if len(estimates) != len(references) or not estimates:
        raise DiagnosticError("estimates and references must be nonempty and aligned")
    errors = [
        estimate - reference
        for estimate, reference in zip(estimates, references, strict=True)
    ]
    absolute = [abs(error) for error in errors]
    squared = [error * error for error in errors]
    error_sd = _sample_sd(errors)
    return {
        "observations": len(errors),
        "estimate_mean": statistics.fmean(estimates),
        "reference_mean": statistics.fmean(references),
        "bias": statistics.fmean(errors),
        "rmse": math.sqrt(statistics.fmean(squared)),
        "mae": statistics.fmean(absolute),
        "error_sample_sd": error_sd,
        "bias_monte_carlo_se": error_sd / math.sqrt(len(errors)),
        "estimate_q05": _quantile(estimates, 0.05),
        "estimate_median": _quantile(estimates, 0.50),
        "estimate_q95": _quantile(estimates, 0.95),
    }


def _diagnostic_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": _sample_sd(values),
        "minimum": min(values),
        "q05": _quantile(values, 0.05),
        "median": _quantile(values, 0.50),
        "q95": _quantile(values, 0.95),
        "maximum": max(values),
    }


def _group_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    exact_estimates: list[float] = []
    exact_references: list[float] = []
    loss_estimates: list[float] = []
    loss_references: list[float] = []
    relative_ess: list[float] = []
    maximum_weights: list[float] = []
    outside_exact: list[float] = []
    for run in records:
        estimate = cast(Mapping[str, object], run["estimate"])
        reference = cast(Mapping[str, object], run["reference"])
        diagnostics = cast(Mapping[str, object], run["diagnostics"])
        exact_estimates.append(_finite_number(estimate["exact_target_mass"], name="mass"))
        exact_references.append(
            _finite_number(reference["exact_target_mass"], name="mass reference")
        )
        loss_estimates.append(_finite_number(estimate["target_mean_loss"], name="loss"))
        loss_references.append(
            _finite_number(reference["target_mean_loss"], name="loss reference")
        )
        relative_ess.append(
            _finite_number(diagnostics["relative_importance_ess"], name="relative ESS")
        )
        maximum_weights.append(
            _finite_number(diagnostics["maximum_normalized_weight"], name="maximum weight")
        )
        outside_exact.append(
            _finite_number(
                diagnostics["exact_draws_outside_local_support"],
                name="outside exact draws",
            )
        )
    return {
        "exact_mass": _error_summary(exact_estimates, exact_references),
        "target_mean_loss": _error_summary(loss_estimates, loss_references),
        "relative_importance_ess": _diagnostic_summary(relative_ess),
        "maximum_normalized_weight": _diagnostic_summary(maximum_weights),
        "exact_draws_outside_local_support": _diagnostic_summary(outside_exact),
        "outside_local_exact_discovery_rate": sum(value > 0 for value in outside_exact)
        / len(outside_exact),
    }


def analyze_runs(
    runs: Sequence[Mapping[str, object]],
    references: Mapping[str, Mapping[str, object]],
    *,
    task_ids: Sequence[str],
    arms: Sequence[ArmSpec],
    particle_counts: Sequence[int],
    repetitions: int,
) -> dict[str, object]:
    expected = len(task_ids) * len(arms) * len(particle_counts) * repetitions
    if len(runs) != expected:
        raise DiagnosticError(f"analysis received {len(runs)} runs, expected {expected}")
    per_task: dict[str, object] = {}
    pooled: dict[str, object] = {}
    for task_id in task_ids:
        task_arms: dict[str, object] = {}
        for arm in arms:
            by_n: dict[str, object] = {}
            for particles in particle_counts:
                group = [
                    run
                    for run in runs
                    if run["task_id"] == task_id
                    and run["arm_id"] == arm.arm_id
                    and run["particles"] == particles
                ]
                if len(group) != repetitions:
                    raise DiagnosticError(
                        f"{task_id}/{arm.arm_id}/N={particles} is incomplete"
                    )
                by_n[str(particles)] = _group_summary(group)
            task_arms[arm.arm_id] = {
                "role": arm.role,
                "by_particle_count": by_n,
            }
        per_task[task_id] = {"arms": task_arms}
    for arm in arms:
        by_n = {}
        for particles in particle_counts:
            group = [
                run
                for run in runs
                if run["arm_id"] == arm.arm_id and run["particles"] == particles
            ]
            if len(group) != repetitions * len(task_ids):
                raise DiagnosticError(f"pooled {arm.arm_id}/N={particles} is incomplete")
            by_n[str(particles)] = _group_summary(group)
        pooled[arm.arm_id] = {"role": arm.role, "by_particle_count": by_n}

    identity_errors: list[float] = []
    sticky_ess: dict[str, float] = {}
    positive_ess: dict[str, float] = {}
    for task_id in task_ids:
        census = cast(Mapping[str, object], references[task_id]["arm_census"])
        for arm in arms:
            record = cast(Mapping[str, object], census[arm.arm_id])
            identity_errors.append(
                _finite_number(
                    record["importance_identity_maximum_absolute_error"],
                    name="importance identity error",
                )
            )
        sticky_record = cast(Mapping[str, object], census["sticky-epsilon-005"])
        positive_record = cast(
            Mapping[str, object],
            census["global-top64-epsilon-050-positive-control"],
        )
        sticky_ess[task_id] = _finite_number(
            sticky_record["population_importance_ess_fraction"],
            name="sticky population ESS",
        )
        positive_ess[task_id] = _finite_number(
            positive_record["population_importance_ess_fraction"],
            name="positive-control population ESS",
        )
    sticky_rmse = _finite_number(
        cast(Mapping[str, object], cast(Mapping[str, object], pooled["sticky-epsilon-005"])[
            "by_particle_count"
        ])["512"]["exact_mass"]["rmse"],
        name="sticky pooled RMSE",
    )
    positive_rmse = _finite_number(
        cast(
            Mapping[str, object],
            cast(Mapping[str, object], pooled[
                "global-top64-epsilon-050-positive-control"
            ])["by_particle_count"],
        )["512"]["exact_mass"]["rmse"],
        name="positive-control pooled RMSE",
    )
    identities_pass = max(identity_errors) <= IDENTITY_TOLERANCE
    population_overlap_improves = all(
        positive_ess[task_id] > sticky_ess[task_id] for task_id in task_ids
    )
    observed_error_improves = positive_rmse < sticky_rmse
    diagnosis_supported = (
        identities_pass and population_overlap_improves and observed_error_improves
    )
    return {
        "schema": ANALYSIS_SCHEMA,
        "run_count": len(runs),
        "task_count": len(task_ids),
        "arm_count": len(arms),
        "particle_counts": list(particle_counts),
        "repetitions_per_task_arm_particle_count": repetitions,
        "per_task": per_task,
        "pooled": pooled,
        "diagnostic_conclusion": {
            "not_a_calibration_gate": True,
            "v1_gate_remains_failed_and_unchanged": True,
            "maximum_exact_importance_identity_error": max(identity_errors),
            "exact_importance_identities_pass": identities_pass,
            "identity_tolerance": IDENTITY_TOLERANCE,
            "positive_control_population_ess_exceeds_sticky_for_every_task": (
                population_overlap_improves
            ),
            "population_ess_fraction_by_task": {
                task_id: {
                    "sticky_epsilon_005": sticky_ess[task_id],
                    "global_positive_control": positive_ess[task_id],
                    "positive_to_sticky_ratio": positive_ess[task_id]
                    / sticky_ess[task_id],
                }
                for task_id in task_ids
            },
            "pooled_n512_exact_mass_rmse": {
                "sticky_epsilon_005": sticky_rmse,
                "global_positive_control": positive_rmse,
            },
            "positive_control_observed_rmse_below_sticky": observed_error_improves,
            "proposal_target_overlap_diagnosis_supported": diagnosis_supported,
            "interpretation": (
                "If supported, the exact finite-state identities rule out a target/q/weight "
                "accounting mismatch in this terminal diagnostic, while the population and "
                "Monte Carlo contrasts identify poor proposal-target overlap as the V1 "
                "finite-N mechanism. The positive control is posthoc and exhaustive."
            ),
        },
        "claim_boundary": (
            "provider-free posthoc terminal diagnosis only; no revision of V1, no provider "
            "claim, no search-efficiency claim, and no confirmatory calibration claim"
        ),
    }


def _resolve_bound_file(project_root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise DiagnosticError(f"{label} path must be a nonempty string")
    root = project_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise DiagnosticError(f"{label} must resolve to a regular in-project file")
    return path


def _file_record(project_root: Path, relative: str) -> dict[str, str]:
    path = _resolve_bound_file(project_root, relative, label=relative)
    return {"path": relative, "sha256": _sha256_file(path)}


def _python_tree_record(project_root: Path) -> dict[str, object]:
    relative_root = "src/modelsmc_pbe"
    tree_root = (project_root / relative_root).resolve()
    paths = sorted(
        (path for path in tree_root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(tree_root).as_posix().encode("utf-8"),
    )
    if not paths or any(path.is_symlink() for path in tree_root.rglob("*")):
        raise DiagnosticError("ModelSMC source tree is empty or contains a symlink")
    entries = [
        {
            "path": path.relative_to(tree_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]
    manifest = {
        "schema": "sha256-python-tree-v1",
        "root": relative_root,
        "include": "**/*.py",
        "entries": entries,
    }
    return {
        "schema": "sha256-python-tree-v1",
        "root": relative_root,
        "include": "**/*.py",
        "file_count": len(entries),
        "manifest_sha256": _sha256_bytes(canonical_bytes(manifest)),
    }


def build_frozen_protocol(project_root: Path) -> dict[str, object]:
    """Construct the protocol record before any V2 terminal draws."""

    tasks = [
        {"id": task_id, **_file_record(project_root, relative)}
        for task_id, relative in TASK_SPECS
    ]
    return {
        "schema": STUDY_SCHEMA,
        "status": STATUS,
        "frozen_at": "2026-08-13",
        "study_type": "posthoc provider-free terminal-only mechanism diagnostic",
        "purpose": (
            "Distinguish V1 finite-N proposal-target overlap failure from terminal target, "
            "proposal probability, importance weight, resampling, or estimator mismatch."
        ),
        "authorization": {
            "live_provider_calls": False,
            "cached_provider_replay": False,
            "terminal_proposal_draws_only": True,
            "exact_finite_state_census": True,
        },
        "v1_preservation": {
            "v1_harness_sha256": (
                "b93939fdb208ec652a92a1be5f35d85aff3851fa852eef3ede2fa9e84f9cead4"
            ),
            "v1_protocol_sha256": (
                "dd7d5ba3d5e2e8b3427b9ef0db5bc7443ae2fcfb40a642497da31590fe34d0cb"
            ),
            "v1_n256_exact_mass_rmse": 0.23952653096132373,
            "v1_n256_exact_mass_signed_bias": 0.11301839407999795,
            "v1_primary_gate_pass": False,
            "rule": "V2 does not overwrite, revise, or replace any V1 input or result.",
        },
        "fixed_parent": {
            "selection": (
                "minimum (program cost, predicate syntax, mapper syntax) among all "
                "public-example zero-loss programs"
            ),
            "selection_is_mechanical": True,
            "privileged_target_ast_used": False,
        },
        "proposal_formula": (
            "q(y)=(1-epsilon)*empirical_uniform(frozen local/global slots)+epsilon*g(y)"
        ),
        "design": {
            "arms": [asdict(arm) for arm in ARMS],
            "particle_counts": list(PARTICLE_COUNTS),
            "repetitions": REPETITIONS,
            "repetition_seeds": list(REPETITION_SEEDS),
            "task_count": len(TASK_SPECS),
            "terminal_draws_per_repetition": "N",
            "total_repetitions": len(TASK_SPECS)
            * len(ARMS)
            * len(PARTICLE_COUNTS)
            * REPETITIONS,
            "total_terminal_proposal_draws": len(TASK_SPECS)
            * len(ARMS)
            * REPETITIONS
            * sum(PARTICLE_COUNTS),
            "exact_programs_per_task": 36_000,
            "identity_tolerance": IDENTITY_TOLERANCE,
            "device": "cpu",
        },
        "arm_boundaries": {
            "sticky_epsilon_005": "reproduces the V1 exact-parent terminal mechanism",
            "epsilon_ladder": "coverage sensitivity; not expected by itself to calibrate",
            "nonsticky_mapper": "tests removal of exact absorption while retaining locality",
            "global_positive_control": {
                "classification": "posthoc developmental exhaustive positive control",
                "construction": (
                    "top 64 complete programs by public-example loss, cost, and syntax"
                ),
                "not_a_search_algorithm": True,
                "search_efficiency_claim_authorized": False,
            },
        },
        "exact_census_endpoints": [
            "proposal normalization",
            "E_q[pi/q]=1",
            "E_q[(pi/q)*I_exact]=exact target mass",
            "E_q[(pi/q)*loss]=target mean loss",
            "target-to-proposal chi-square divergence",
            "population importance ESS fraction",
            "self-normalized endpoint asymptotic variance",
        ],
        "diagnostic_rule": {
            "not_a_calibration_gate": True,
            "mechanism_supported_when": [
                "all exact importance identities pass at tolerance 1e-12",
                "global positive-control population ESS exceeds sticky-epsilon-005 for every task",
                "global positive-control pooled N=512 exact-mass RMSE is below sticky-epsilon-005",
            ],
            "interpretation": (
                "This contrast diagnoses overlap under a posthoc exhaustive positive control; "
                "it does not turn V1 into a passing study."
            ),
        },
        "runtime": {
            "python_major_minor": _runtime_record()["python_major_minor"],
            "packages": _runtime_record()["packages"],
        },
        "bindings": {
            "harness": _file_record(project_root, HARNESS_PATH),
            "test": _file_record(project_root, TEST_PATH),
            "dependency_bundle": [
                _file_record(project_root, relative) for relative in DEPENDENCY_PATHS
            ],
            "lockfiles": [
                _file_record(project_root, relative) for relative in LOCKFILE_PATHS
            ],
            "modelsmc_python_tree": _python_tree_record(project_root),
            "tasks": tasks,
        },
        "failure_policy": {
            "binding_or_runtime_mismatch": "abort before exact census or draws",
            "identity_failure": "write sealed failed-closed artifact and abort",
            "invalid_or_nonfinite_run": "write sealed failed-closed artifact and abort",
            "existing_output": "refuse to overwrite",
        },
        "claim_boundary": [
            "V1 remains failed under its frozen gate.",
            "V2 is a post-result provider-free terminal mechanism diagnostic.",
            "The global arm is exhaustive and posthoc, so it supports no efficiency claim.",
            "No LLM, provider, generalization, or full-SMC calibration claim is authorized.",
        ],
    }


def freeze_protocol(protocol_path: Path, project_root: Path) -> str:
    if protocol_path.exists():
        raise FileExistsError(f"refusing to overwrite existing protocol: {protocol_path}")
    protocol = build_frozen_protocol(project_root)
    _write_json(protocol_path, protocol)
    return _sha256_file(protocol_path)


def _validate_file_bundle(
    records: object,
    *,
    expected_paths: Sequence[str],
    project_root: Path,
    label: str,
) -> None:
    if not isinstance(records, list):
        raise DiagnosticError(f"{label} binding must be a list")
    observed: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise DiagnosticError(f"{label} binding record has the wrong fields")
        path = _resolve_bound_file(project_root, record["path"], label=label)
        if record["sha256"] != _sha256_file(path):
            raise DiagnosticError(f"{label} SHA-256 differs for {record['path']}")
        observed.append(cast(str, record["path"]))
    if tuple(observed) != tuple(expected_paths):
        raise DiagnosticError(f"{label} paths or ordering differ")


def load_frozen_plan(
    protocol_path: Path,
    project_root: Path,
    *,
    expected_protocol_sha256: str,
) -> tuple[StudyPlan, dict[str, object]]:
    protocol_sha256 = _sha256_file(protocol_path)
    if expected_protocol_sha256 != protocol_sha256:
        raise DiagnosticError("protocol differs from externally expected SHA-256")
    try:
        raw = json.loads(protocol_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DiagnosticError(f"invalid protocol JSON: {error.msg}") from error
    if not isinstance(raw, dict) or raw.get("schema") != STUDY_SCHEMA:
        raise DiagnosticError("protocol schema differs")
    if raw.get("status") != STATUS:
        raise DiagnosticError("protocol is not frozen before V2 draws")
    if raw.get("authorization") != {
        "live_provider_calls": False,
        "cached_provider_replay": False,
        "terminal_proposal_draws_only": True,
        "exact_finite_state_census": True,
    }:
        raise DiagnosticError("protocol authorization differs")
    design = raw.get("design")
    if not isinstance(design, dict):
        raise DiagnosticError("protocol design is missing")
    if design.get("arms") != [asdict(arm) for arm in ARMS]:
        raise DiagnosticError("frozen arm declarations differ")
    if design.get("particle_counts") != list(PARTICLE_COUNTS):
        raise DiagnosticError("frozen particle counts differ")
    if design.get("repetition_seeds") != list(REPETITION_SEEDS):
        raise DiagnosticError("frozen repetition seeds differ")
    if design.get("repetitions") != REPETITIONS:
        raise DiagnosticError("frozen repetition count differs")
    expected_total = len(TASK_SPECS) * len(ARMS) * REPETITIONS * sum(PARTICLE_COUNTS)
    if design.get("total_terminal_proposal_draws") != expected_total:
        raise DiagnosticError("frozen terminal draw budget differs")

    runtime = raw.get("runtime")
    actual_runtime = _runtime_record()
    if not isinstance(runtime, dict) or runtime != {
        "python_major_minor": actual_runtime["python_major_minor"],
        "packages": actual_runtime["packages"],
    }:
        raise DiagnosticError("frozen runtime binding differs")
    bindings = raw.get("bindings")
    if not isinstance(bindings, dict):
        raise DiagnosticError("protocol bindings are missing")
    harness = bindings.get("harness")
    if not isinstance(harness, dict) or harness.get("path") != HARNESS_PATH:
        raise DiagnosticError("harness binding differs")
    harness_path = _resolve_bound_file(project_root, HARNESS_PATH, label="harness")
    harness_sha256 = _sha256_file(harness_path)
    if harness.get("sha256") != harness_sha256:
        raise DiagnosticError("harness SHA-256 differs")
    test = bindings.get("test")
    test_path = _resolve_bound_file(project_root, TEST_PATH, label="test")
    if not isinstance(test, dict) or test != {
        "path": TEST_PATH,
        "sha256": _sha256_file(test_path),
    }:
        raise DiagnosticError("test SHA-256 differs")
    _validate_file_bundle(
        bindings.get("dependency_bundle"),
        expected_paths=DEPENDENCY_PATHS,
        project_root=project_root,
        label="dependency bundle",
    )
    _validate_file_bundle(
        bindings.get("lockfiles"),
        expected_paths=LOCKFILE_PATHS,
        project_root=project_root,
        label="lockfile bundle",
    )
    source_tree = bindings.get("modelsmc_python_tree")
    if not isinstance(source_tree, dict):
        raise DiagnosticError("ModelSMC source-tree binding is missing")
    try:
        python_tree_binding(project_root, source_tree)
    except ValueError as error:
        raise DiagnosticError(str(error)) from error

    raw_tasks = bindings.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != len(TASK_SPECS):
        raise DiagnosticError("task bindings differ")
    tasks: list[TaskBinding] = []
    for raw_task, (expected_id, expected_relative) in zip(
        raw_tasks,
        TASK_SPECS,
        strict=True,
    ):
        if not isinstance(raw_task, dict):
            raise DiagnosticError("task binding must be an object")
        if raw_task.get("id") != expected_id or raw_task.get("path") != expected_relative:
            raise DiagnosticError("task ID or path differs")
        task_path = _resolve_bound_file(project_root, expected_relative, label=expected_id)
        task_sha256 = _sha256_file(task_path)
        if raw_task.get("sha256") != task_sha256:
            raise DiagnosticError(f"task SHA-256 differs for {expected_id}")
        tasks.append(TaskBinding(expected_id, task_path, task_sha256))
    return (
        StudyPlan(
            tasks=tuple(tasks),
            arms=ARMS,
            particle_counts=PARTICLE_COUNTS,
            repetition_seeds=REPETITION_SEEDS,
            protocol_sha256=protocol_sha256,
            harness_sha256=harness_sha256,
        ),
        raw,
    )


def _summary_markdown(
    analysis: Mapping[str, object],
    references: Mapping[str, Mapping[str, object]],
) -> str:
    pooled = cast(Mapping[str, object], analysis["pooled"])
    conclusion = cast(Mapping[str, object], analysis["diagnostic_conclusion"])
    lines = [
        "# Provider-Free Particle Calibration Terminal Diagnostic V2",
        "",
        "This is a posthoc terminal-only mechanism diagnostic. The frozen V1 gate "
        "remains failed and unchanged.",
        "",
        "The global top-64 arm is an exhaustive developmental positive control, not a "
        "search algorithm and not evidence of search efficiency.",
        "",
        "## Exact terminal references",
        "",
        "| Task | Exact syntaxes | Exact mass | Mean loss |",
        "|---|---:|---:|---:|",
    ]
    for task_id, reference in references.items():
        lines.append(
            f"| {task_id} | {reference['exact_program_syntaxes']} | "
            f"{float(reference['exact_target_mass']):.8g} | "
            f"{float(reference['target_mean_loss']):.8g} |"
        )
    lines.extend(
        [
            "",
            "## Pooled Monte Carlo exact-mass error",
            "",
            "| Arm | N | Bias | RMSE | Mean relative ESS |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        arm_record = cast(Mapping[str, object], pooled[arm.arm_id])
        by_n = cast(Mapping[str, object], arm_record["by_particle_count"])
        for particles in PARTICLE_COUNTS:
            summary = cast(Mapping[str, object], by_n[str(particles)])
            exact = cast(Mapping[str, object], summary["exact_mass"])
            ess = cast(Mapping[str, object], summary["relative_importance_ess"])
            lines.append(
                f"| {arm.arm_id} | {particles} | {float(exact['bias']):.6g} | "
                f"{float(exact['rmse']):.6g} | {float(ess['mean']):.6g} |"
            )
    lines.extend(
        [
            "",
            "## Diagnostic conclusion",
            "",
            "Exact importance identities pass: "
            f"**{conclusion['exact_importance_identities_pass']}**.",
            "",
            "Positive-control population ESS exceeds sticky epsilon=0.05 for every "
            "task: "
            f"**{conclusion['positive_control_population_ess_exceeds_sticky_for_every_task']}**.",
            "",
            "Positive-control pooled N=512 exact-mass RMSE is below sticky "
            f"epsilon=0.05: **{conclusion['positive_control_observed_rmse_below_sticky']}**.",
            "",
            "Proposal-target overlap diagnosis supported: "
            f"**{conclusion['proposal_target_overlap_diagnosis_supported']}**.",
            "",
            "This conclusion is diagnostic only. It does not turn V1 into a passing "
            "calibration study.",
        ]
    )
    return "\n".join(lines) + "\n"


def _seal_inventory(output: Path) -> dict[str, object]:
    records = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "inventory.json"
    ]
    inventory = {
        "schema": f"{STUDY_SCHEMA}-artifact-inventory-v1",
        "file_count": len(records),
        "records": records,
        "records_sha256": _sha256_bytes(canonical_bytes(records)),
    }
    _write_json(output / "inventory.json", inventory)
    return inventory


def validate_artifact(output: Path) -> dict[str, object]:
    output = output.resolve()
    inventory_path = output / "inventory.json"
    if not inventory_path.is_file():
        raise DiagnosticError("artifact inventory is missing")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    records = inventory.get("records")
    if not isinstance(records, list):
        raise DiagnosticError("artifact inventory records are malformed")
    actual_paths = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "inventory.json"
    )
    expected_paths: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise DiagnosticError("artifact inventory record is malformed")
        relative = cast(str, record.get("path"))
        path = (output / relative).resolve()
        if not path.is_relative_to(output) or not path.is_file() or path.is_symlink():
            raise DiagnosticError(f"invalid artifact path {relative}")
        if record.get("bytes") != path.stat().st_size:
            raise DiagnosticError(f"artifact byte count differs for {relative}")
        if record.get("sha256") != _sha256_file(path):
            raise DiagnosticError(f"artifact SHA-256 differs for {relative}")
        expected_paths.append(relative)
    if sorted(expected_paths) != actual_paths:
        raise DiagnosticError("artifact files differ from inventory")
    if inventory.get("records_sha256") != _sha256_bytes(canonical_bytes(records)):
        raise DiagnosticError("artifact inventory aggregate SHA-256 differs")
    runs = json.loads((output / "runs.json").read_text(encoding="utf-8"))
    expected_runs = len(TASK_SPECS) * len(ARMS) * len(PARTICLE_COUNTS) * REPETITIONS
    if not isinstance(runs, list) or len(runs) != expected_runs:
        raise DiagnosticError("artifact run count differs")
    if any(run.get("provider_calls") != 0 for run in runs if isinstance(run, dict)):
        raise DiagnosticError("artifact contains a provider call")
    analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    if analysis.get("schema") != ANALYSIS_SCHEMA or analysis.get("run_count") != expected_runs:
        raise DiagnosticError("artifact analysis schema or run count differs")
    return {
        "schema": STUDY_SCHEMA,
        "status": "validated",
        "run_count": expected_runs,
        "file_count": len(records) + 1,
        "records_sha256": inventory["records_sha256"],
        "diagnostic_conclusion": analysis["diagnostic_conclusion"],
    }


def run_study(
    plan: StudyPlan,
    protocol_record: Mapping[str, object],
    output: Path,
) -> dict[str, object]:
    """Run the frozen terminal study and publish a sealed artifact."""

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    staging = output.with_name(f".{output.name}.incomplete")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite incomplete output: {staging}")
    staging.mkdir(parents=True)
    completed_runs = 0
    try:
        _write_json(staging / "protocol.json", protocol_record)
        _write_json(
            staging / "study-metadata.json",
            {
                "schema": f"{STUDY_SCHEMA}-metadata-v1",
                "protocol_sha256": plan.protocol_sha256,
                "harness_sha256": plan.harness_sha256,
                "runtime": _runtime_record(),
                "provider_calls_authorized": 0,
                "provider_calls_observed": 0,
                "terminal_only": True,
                "posthoc_developmental_positive_control": True,
                "v1_gate_remains_failed_and_unchanged": True,
            },
        )
        references: dict[str, dict[str, object]] = {}
        runs: list[dict[str, object]] = []
        for binding in plan.tasks:
            task = TerminalTask(binding)
            reference = task.reference_record(plan.arms)
            references[binding.task_id] = reference
            _write_json(staging / "references" / f"{binding.task_id}.json", reference)
            for arm in plan.arms:
                distribution = task.arm_distribution(arm)
                for particles in plan.particle_counts:
                    for repetition, seed in enumerate(plan.repetition_seeds):
                        run = run_repetition(
                            task,
                            distribution,
                            particles=particles,
                            repetition=repetition,
                            base_seed=seed,
                        )
                        runs.append(run)
                        completed_runs += 1
            del task
            gc.collect()
        expected_runs = (
            len(plan.tasks)
            * len(plan.arms)
            * len(plan.particle_counts)
            * len(plan.repetition_seeds)
        )
        if completed_runs != expected_runs:
            raise DiagnosticError("completed run count differs")
        if any(run["provider_calls"] != 0 for run in runs):
            raise DiagnosticError("provider-free run invariant failed")
        _write_json(staging / "runs.json", runs)
        analysis = analyze_runs(
            runs,
            references,
            task_ids=[binding.task_id for binding in plan.tasks],
            arms=plan.arms,
            particle_counts=plan.particle_counts,
            repetitions=len(plan.repetition_seeds),
        )
        _write_json(staging / "analysis.json", analysis)
        (staging / "SUMMARY.md").write_text(
            _summary_markdown(analysis, references),
            encoding="utf-8",
        )
        inventory = _seal_inventory(staging)
        staging.rename(output)
        validated = validate_artifact(output)
        return {
            "schema": STUDY_SCHEMA,
            "status": "complete",
            "output": output.as_posix(),
            "run_count": completed_runs,
            "inventory_sha256": inventory["records_sha256"],
            "diagnostic_conclusion": validated["diagnostic_conclusion"],
        }
    except Exception as error:
        failure = {
            "schema": f"{STUDY_SCHEMA}-failure-v1",
            "status": "failed-closed",
            "error_type": type(error).__name__,
            "detail": str(error),
            "completed_runs": completed_runs,
            "traceback": traceback.format_exc(),
        }
        _write_json(staging / "failure.json", failure)
        _seal_inventory(staging)
        staging.rename(output)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze a new V2 protocol")
    freeze.add_argument("--project-root", type=Path, default=Path.cwd())
    freeze.add_argument("--protocol", type=Path, default=Path(DEFAULT_PROTOCOL))
    run = subparsers.add_parser("run", help="run the frozen V2 terminal diagnostic")
    run.add_argument("--project-root", type=Path, default=Path.cwd())
    run.add_argument("--protocol", type=Path, default=Path(DEFAULT_PROTOCOL))
    run.add_argument("--expected-protocol-sha256", required=True)
    run.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    validate = subparsers.add_parser("validate", help="validate a completed artifact")
    validate.add_argument("--artifact", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        protocol_path = args.protocol.resolve()
        sha256 = freeze_protocol(protocol_path, args.project_root.resolve())
        print(
            json.dumps(
                {"status": "frozen", "protocol": protocol_path.as_posix(), "sha256": sha256},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run":
        plan, protocol = load_frozen_plan(
            args.protocol.resolve(),
            args.project_root.resolve(),
            expected_protocol_sha256=args.expected_protocol_sha256,
        )
        result = run_study(plan, protocol, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        print(json.dumps(validate_artifact(args.artifact), indent=2, sort_keys=True))
        return 0
    raise AssertionError("argparse accepted an unknown command")


if __name__ == "__main__":
    raise SystemExit(main())

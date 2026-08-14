"""Provider-free repeated particle calibration for evidence-shortlist SMC.

The harness reuses the normalized proposal and staged target mathematics from
``research.evidence_shortlist_smc`` but never calls a model provider.  A frozen
deterministic proposal oracle ranks legal one-hole repairs using only public
singleton evidence, public-example execution loss, program cost, and canonical
syntax.  Its four slots are mixed with the same epsilon recursive-grammar
restart, so every sampled proposal probability remains exactly evaluable.

This experiment calibrates the SMC estimator under that declared oracle.  It is
not evidence about LLM proposal quality, provider variability, wall-clock
speed, or arbitrary program-synthesis tasks.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch

from modelsmc_pbe.config import ExperimentConfig, load_experiment_config
from modelsmc_pbe.core.evaluate import evaluate_expression
from modelsmc_pbe.core.results import ScoredProgram
from modelsmc_pbe.core.scorer import ProgramScorer
from modelsmc_pbe.core.types import Node
from modelsmc_pbe.grammar import arithmetic_expressions, filter_predicates
from modelsmc_pbe.smc import effective_sample_size, normalize_log_weights
from research.automatic_repair_feedback import derive_automatic_feedback
from research.evidence_shortlist_smc import (
    ROUNDS,
    SHORTLIST_SIZE,
    Particle,
    ProgramKey,
    normalized_child_weights,
    program_grammar_probability,
    python_tree_binding,
    recursive_grammar_probabilities,
    sample_proposal,
    stage_log_gamma,
    systematic_resample_count,
)
from research.execution_guided_repair import _assemble_program, _dsl_catalog
from research.iterative_beam_experiment import (
    BeamState,
    derive_singleton_constraints,
    select_repair_hole,
)

STUDY_SCHEMA = "provider-free-particle-calibration-study-v1"
RUN_SCHEMA = "provider-free-particle-calibration-run-v1"
REFERENCE_SCHEMA = "provider-free-particle-calibration-reference-v1"
ANALYSIS_SCHEMA = "provider-free-particle-calibration-analysis-v1"
ORACLE_ID = "structured-singleton-execution-ranking-oracle-v1"
PARTICLE_COUNTS = (32, 64, 128, 256, 512)
PRIMARY_GATE_N = 256
PRIMARY_EXACT_MASS_RMSE_THRESHOLD = 0.10
PRIMARY_EXACT_MASS_BIAS_INTERVAL = (-0.03, 0.03)
DEFAULT_PROTOCOL = "research/protocol-particle-calibration-v1.json"
HARNESS_PATH = "research/particle_calibration_study_v1.py"
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
TEST_PATH = "research/tests/test_particle_calibration_study_v1.py"


class CalibrationError(ValueError):
    """A frozen calibration invariant was violated."""


@dataclass(frozen=True, slots=True)
class TaskBinding:
    task_id: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class StudyPlan:
    tasks: tuple[TaskBinding, ...]
    particle_counts: tuple[int, ...]
    repetition_seeds: tuple[int, ...]
    epsilon: float
    evidence_scale: float
    start_seed: int
    exact_reference_limit: int
    protocol_sha256: str
    harness_sha256: str


def canonical_bytes(value: object) -> bytes:
    """Return deterministic JSON bytes for artifacts and bindings."""

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
        raise CalibrationError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise CalibrationError(f"{name} must be finite")
    return number


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CalibrationError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CalibrationError(f"{name} must be a nonnegative integer")
    return value


def _derived_seed(label: str, base_seed: int, task_sha256: str, particles: int) -> int:
    digest = hashlib.sha256(
        f"{STUDY_SCHEMA}\0{label}\0{base_seed}\0{task_sha256}\0{particles}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def _runtime_record() -> dict[str, object]:
    return {
        "python_major_minor": ".".join(platform.python_version_tuple()[:2]),
        "python_version": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pydantic", "torch")
        },
        "torch_default_dtype": str(torch.get_default_dtype()),
        "calibration_dtype": "torch.float64",
    }


def _counter_delta(
    after: Mapping[str, int],
    before: Mapping[str, int],
) -> dict[str, int]:
    if set(after) != set(before):
        raise CalibrationError("internal work-counter schema changed during a run")
    result = {name: after[name] - before[name] for name in after}
    if any(value < 0 for value in result.values()):
        raise CalibrationError("internal work counter decreased")
    return result


class CalibrationTask:
    """One enumerable task, exact reference, and frozen proposal oracle cache."""

    def __init__(self, binding: TaskBinding, *, start_seed: int) -> None:
        self.binding = binding
        self.config: ExperimentConfig = load_experiment_config(binding.path)
        signature = self.config.spec.signature
        if signature is None or signature.input_type.value != "List<Int>":
            raise CalibrationError(f"{binding.task_id} must take one List<Int> input")
        if signature.output_type.value != "List<Int>":
            raise CalibrationError(f"{binding.task_id} must return List<Int>")
        self.scorer = ProgramScorer(self.config)
        self.constants = tuple(self.config.spec.integer_constants)
        self.predicate_catalog = _dsl_catalog(filter_predicates(self.constants))
        self.mapper_catalog = _dsl_catalog(arithmetic_expressions("Item", self.constants))
        self.predicate_order = tuple(sorted(self.predicate_catalog))
        self.mapper_order = tuple(sorted(self.mapper_catalog))
        if len(self.predicate_order) != len(self.predicate_catalog):
            raise CalibrationError("predicate DSL catalog contains duplicate syntax")
        if len(self.mapper_order) != len(self.mapper_catalog):
            raise CalibrationError("mapper DSL catalog contains duplicate syntax")
        (
            self.predicate_probability,
            self.mapper_probability,
        ) = recursive_grammar_probabilities(self.predicate_order, self.mapper_order)
        self.observed_items = tuple(
            sorted(
                {
                    item
                    for example in self.config.spec.examples
                    for item in cast(list[int], example.input_value)
                }
            )
        )
        if not self.observed_items:
            raise CalibrationError(f"{binding.task_id} has no observed scalar items")
        self.observed_index = {item: index for index, item in enumerate(self.observed_items)}
        self.singleton = derive_singleton_constraints(self.config.spec)
        self.predicate_signatures = self._predicate_signatures()
        self.mapper_signatures = self._mapper_signatures()
        self.score_cache: dict[ProgramKey, ScoredProgram] = {}
        self.violation_cache: dict[ProgramKey, tuple[int, int]] = {}
        self.feedback_cache: dict[ProgramKey, dict[str, object]] = {}
        self.hole_cache: dict[tuple[int, ProgramKey], tuple[str, dict[str, object]]] = {}
        self.oracle_cache: dict[tuple[ProgramKey, str], tuple[ProgramKey, ...]] = {}
        self.score_requests = 0
        self.score_cache_hits = 0
        self.score_cache_misses = 0
        self.oracle_slot_requests = 0
        self.oracle_slot_cache_hits = 0
        self.oracle_rankings = 0
        self.oracle_candidate_score_lookups = 0

        start_rng = random.Random(start_seed)
        initial_mapper = self.mapper_order[start_rng.randrange(len(self.mapper_order))]
        initial_predicate = self.predicate_order[start_rng.randrange(len(self.predicate_order))]
        self.initial = self.make_particle(
            ProgramKey(initial_predicate, initial_mapper),
            lineage="root",
        )
        if self.initial.score.exact_program:
            raise CalibrationError(
                f"{binding.task_id} has an exact initial state under start seed {start_seed}"
            )

    def _predicate_signatures(self) -> dict[str, tuple[bool, ...]]:
        signatures: dict[str, tuple[bool, ...]] = {}
        for dsl, expression in self.predicate_catalog.items():
            values = tuple(
                evaluate_expression(
                    cast(Node, expression),
                    list(self.observed_items),
                    item=item,
                )
                for item in self.observed_items
            )
            if any(not isinstance(value, bool) for value in values):
                raise CalibrationError("predicate catalog returned a non-Boolean value")
            signatures[dsl] = cast(tuple[bool, ...], values)
        return signatures

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
                raise CalibrationError("mapper catalog returned a non-integer value")
            signatures[dsl] = cast(tuple[int, ...], values)
        return signatures

    def score_key(self, key: ProgramKey) -> ScoredProgram:
        self.score_requests += 1
        cached = self.score_cache.get(key)
        if cached is not None:
            self.score_cache_hits += 1
            return cached
        self.score_cache_misses += 1
        score = self.scorer.score(
            _assemble_program(
                self.predicate_catalog[key.predicate],
                self.mapper_catalog[key.mapper],
            )
        )
        if not isinstance(score, ScoredProgram):
            raise CalibrationError(f"grammar-valid program was rejected: {score.reason}")
        self.score_cache[key] = score
        return score

    def work_counters(self) -> dict[str, int]:
        return {
            "score_requests": self.score_requests,
            "score_cache_hits": self.score_cache_hits,
            "score_cache_misses": self.score_cache_misses,
            "oracle_slot_requests": self.oracle_slot_requests,
            "oracle_slot_cache_hits": self.oracle_slot_cache_hits,
            "oracle_rankings": self.oracle_rankings,
            "oracle_candidate_score_lookups": self.oracle_candidate_score_lookups,
        }

    def violation_counts(self, key: ProgramKey) -> tuple[int, int]:
        cached = self.violation_cache.get(key)
        if cached is not None:
            return cached
        predicate_signature = self.predicate_signatures[key.predicate]
        mapper_signature = self.mapper_signatures[key.mapper]
        predicate_count = sum(
            predicate_signature[self.observed_index[cast(int, fact["item"])]]
            != cast(bool, fact["keep"])
            for fact in self.singleton["predicate"]
        )
        mapper_count = sum(
            mapper_signature[self.observed_index[cast(int, fact["item"])]]
            != cast(int, fact["expected_value"])
            for fact in self.singleton["mapper"]
        )
        result = (predicate_count, mapper_count)
        self.violation_cache[key] = result
        return result

    def make_particle(self, key: ProgramKey, lineage: str) -> Particle:
        predicate_count, mapper_count = self.violation_counts(key)
        return Particle(
            key=key,
            score=self.score_key(key),
            predicate_violations=predicate_count,
            mapper_violations=mapper_count,
            lineage=lineage,
        )

    def log_prior(self, key: ProgramKey) -> float:
        return math.log(
            float(
                program_grammar_probability(
                    key,
                    self.predicate_probability,
                    self.mapper_probability,
                )
            )
        )

    def feedback_for(self, particle: Particle) -> dict[str, object]:
        cached = self.feedback_cache.get(particle.key)
        if cached is not None:
            return cached
        feedback = derive_automatic_feedback(
            self.config.spec,
            cast(Node, self.predicate_catalog[particle.key.predicate]),
            cast(Node, self.mapper_catalog[particle.key.mapper]),
        ).to_dict()
        predicate_violations = tuple(
            fact
            for fact in self.singleton["predicate"]
            if self.predicate_signatures[particle.key.predicate][
                self.observed_index[cast(int, fact["item"])]
            ]
            != cast(bool, fact["keep"])
        )
        mapper_violations = tuple(
            fact
            for fact in self.singleton["mapper"]
            if self.mapper_signatures[particle.key.mapper][
                self.observed_index[cast(int, fact["item"])]
            ]
            != cast(int, fact["expected_value"])
        )
        result = feedback | {
            "singleton_predicate_constraints": self.singleton["predicate"],
            "singleton_mapper_constraints": self.singleton["mapper"],
            "singleton_predicate_violations": predicate_violations,
            "singleton_mapper_violations": mapper_violations,
        }
        self.feedback_cache[particle.key] = result
        return result

    def select_hole(
        self,
        particle: Particle,
        *,
        round_number: int,
    ) -> tuple[str, dict[str, object]]:
        cache_key = (round_number, particle.key)
        cached = self.hole_cache.get(cache_key)
        if cached is not None:
            return cached
        state = BeamState(
            predicate=particle.key.predicate,
            mapper=particle.key.mapper,
            score=particle.score,
            predicate_signature=self.predicate_signatures[particle.key.predicate],
            mapper_signature=self.mapper_signatures[particle.key.mapper],
            singleton_predicate_violations=particle.predicate_violations,
            singleton_mapper_violations=particle.mapper_violations,
        )
        result = select_repair_hole(
            self.feedback_for(particle),
            state,
            round_number=round_number,
            stall_policy="alternate-hole",
        )
        self.hole_cache[cache_key] = result
        return result

    def oracle_slots(self, parent: Particle, hole: str) -> tuple[ProgramKey, ...]:
        """Return four deterministic evidence/execution-ranked local repairs."""

        self.oracle_slot_requests += 1
        cache_key = (parent.key, hole)
        cached = self.oracle_cache.get(cache_key)
        if cached is not None:
            self.oracle_slot_cache_hits += 1
            return cached
        if hole not in {"predicate", "mapper"}:
            raise CalibrationError(f"oracle received unsupported repair hole {hole}")
        candidates = self.predicate_order if hole == "predicate" else self.mapper_order
        self.oracle_rankings += 1
        ranked: list[tuple[tuple[float | int | str, ...], ProgramKey]] = []
        for candidate in candidates:
            key = (
                ProgramKey(candidate, parent.key.mapper)
                if hole == "predicate"
                else ProgramKey(parent.key.predicate, candidate)
            )
            if key == parent.key:
                continue
            predicate_count, mapper_count = self.violation_counts(key)
            selected_violations = predicate_count if hole == "predicate" else mapper_count
            self.oracle_candidate_score_lookups += 1
            score = self.score_key(key)
            rank = (
                selected_violations,
                score.total_loss,
                score.cost,
                candidate,
            )
            ranked.append((rank, key))
        ranked.sort(key=lambda item: item[0])
        slots = tuple(key for _, key in ranked[:SHORTLIST_SIZE])
        if len(slots) != SHORTLIST_SIZE or len(set(slots)) != SHORTLIST_SIZE:
            raise CalibrationError("proposal oracle did not produce four unique local repairs")
        self.oracle_cache[cache_key] = slots
        return slots

    def exact_reference(self, *, limit: int, evidence_scale: float) -> dict[str, object]:
        """Enumerate the exact terminal target once for this task."""

        program_count = len(self.predicate_order) * len(self.mapper_order)
        if program_count > limit:
            raise CalibrationError(
                f"{self.binding.task_id} reference size {program_count} exceeds limit {limit}"
            )
        work_before = self.work_counters()
        log_gammas: list[float] = []
        losses: list[float] = []
        exact: list[bool] = []
        for predicate in self.predicate_order:
            for mapper in self.mapper_order:
                key = ProgramKey(predicate, mapper)
                particle = self.make_particle(key, "reference")
                log_gammas.append(
                    stage_log_gamma(
                        round_number=ROUNDS,
                        score=particle.score,
                        predicate_violations=particle.predicate_violations,
                        mapper_violations=particle.mapper_violations,
                        evidence_scale=evidence_scale,
                        log_prior=self.log_prior(key),
                    )
                )
                losses.append(particle.score.total_loss)
                exact.append(particle.score.exact_program)
        normalized = normalize_log_weights(torch.tensor(log_gammas, dtype=torch.float64))
        weights = normalized.weights
        exact_mass = sum(
            weight
            for weight, is_exact in zip(weights.tolist(), exact, strict=True)
            if is_exact
        )
        mean_loss = sum(
            weight * loss
            for weight, loss in zip(weights.tolist(), losses, strict=True)
        )
        result = {
            "schema": REFERENCE_SCHEMA,
            "task_id": self.binding.task_id,
            "task_sha256": self.binding.sha256,
            "program_syntaxes": program_count,
            "predicate_syntaxes": len(self.predicate_order),
            "mapper_syntaxes": len(self.mapper_order),
            "exact_program_syntaxes": sum(exact),
            "log_normalizer": normalized.log_normalizer,
            "exact_target_mass": exact_mass,
            "target_mean_loss": mean_loss,
            "target_distribution_ess": effective_sample_size(weights),
            "maximum_target_probability": max(weights.tolist()),
            "target": "recursive grammar prior times exp(scorer.log_target)",
            "reference_complete_program_evaluations": program_count,
            "reference_work": _counter_delta(self.work_counters(), work_before),
        }
        _validate_reference(result)
        return result

    def oracle_inventory(self) -> dict[str, object]:
        entries = [
            {
                "parent": asdict(parent),
                "hole": hole,
                "slots": [asdict(slot) for slot in slots],
            }
            for (parent, hole), slots in sorted(
                self.oracle_cache.items(),
                key=lambda item: (item[0][0].predicate, item[0][0].mapper, item[0][1]),
            )
        ]
        return {
            "task_id": self.binding.task_id,
            "oracle_id": ORACLE_ID,
            "queried_state_hole_pairs": len(entries),
            "entries_sha256": _sha256_bytes(canonical_bytes(entries)),
            "ranking": [
                "selected-hole singleton violation count",
                "complete-program public-example execution loss",
                "program cost",
                "canonical component syntax",
            ],
            "target_ast_used": False,
            "target_weights_or_reference_aggregates_used": False,
            "public_execution_score_cache_shared_with_reference_enumeration": True,
            "work_counters": self.work_counters(),
        }


def _validate_reference(reference: Mapping[str, object]) -> None:
    mass = _finite_number(reference.get("exact_target_mass"), name="exact target mass")
    loss = _finite_number(reference.get("target_mean_loss"), name="target mean loss")
    ess = _finite_number(reference.get("target_distribution_ess"), name="target ESS")
    programs = _positive_int(reference.get("program_syntaxes"), name="program syntaxes")
    if not 0.0 <= mass <= 1.0:
        raise CalibrationError("exact target mass is outside [0, 1]")
    if loss < 0.0:
        raise CalibrationError("target mean loss is negative")
    if not 1.0 <= ess <= programs + 1e-9:
        raise CalibrationError("target ESS is outside its finite-support bounds")


def run_repetition(
    task: CalibrationTask,
    reference: Mapping[str, object],
    *,
    particles: int,
    repetition: int,
    base_seed: int,
    epsilon: float,
    evidence_scale: float,
) -> dict[str, object]:
    """Run one four-stage, fixed-population SMC repetition."""

    particles = _positive_int(particles, name="particles")
    work_before = task.work_counters()
    proposal_seed = _derived_seed("proposal", base_seed, task.binding.sha256, particles)
    resample_seed = _derived_seed("resample", base_seed, task.binding.sha256, particles)
    proposal_rng = random.Random(proposal_seed)
    parent_particles = [task.initial]
    parent_weights = torch.tensor([1.0], dtype=torch.float64)
    logical_executions = 1
    first_exact_slot: int | None = None
    best = task.initial
    grammar_restart_draws = 0
    shortlist_draws = 0
    exact_absorbing_parents = 0
    oracle_queries = 0
    stage_records: list[dict[str, object]] = []

    for round_number in range(1, ROUNDS + 1):
        children: list[Particle] = []
        child_log_weights: list[float] = []
        sampled_probabilities: list[float] = []
        round_restart_draws = 0
        round_shortlist_draws = 0
        for parent_index, parent in enumerate(parent_particles):
            if parent.score.exact_program:
                hole = "exact-absorbing"
                slots = (parent.key,) * SHORTLIST_SIZE
                exact_absorbing_parents += 1
            else:
                hole, _ = task.select_hole(parent, round_number=round_number)
                slots = task.oracle_slots(parent, hole)
                oracle_queries += 1
            offspring_count = particles if round_number == 1 else 1
            for branch_index in range(offspring_count):
                draw = sample_proposal(
                    slots=slots,
                    predicate_order=task.predicate_order,
                    mapper_order=task.mapper_order,
                    predicate_probability=task.predicate_probability,
                    mapper_probability=task.mapper_probability,
                    epsilon=epsilon,
                    rng=proposal_rng,
                )
                if draw.branch == "full-grammar-restart":
                    grammar_restart_draws += 1
                    round_restart_draws += 1
                else:
                    shortlist_draws += 1
                    round_shortlist_draws += 1
                logical_executions += 1
                lineage = f"root.{round_number}:{parent_index}:{branch_index}"
                child = task.make_particle(draw.key, lineage)
                children.append(child)
                log_gamma = stage_log_gamma(
                    round_number=round_number,
                    score=child.score,
                    predicate_violations=child.predicate_violations,
                    mapper_violations=child.mapper_violations,
                    evidence_scale=evidence_scale,
                    log_prior=task.log_prior(child.key),
                )
                log_weight = (
                    math.log(float(parent_weights[parent_index].item()))
                    - math.log(offspring_count)
                    + log_gamma
                    - math.log(draw.probability)
                )
                child_log_weights.append(log_weight)
                sampled_probabilities.append(draw.probability)
                if child.score.exact_program and first_exact_slot is None:
                    first_exact_slot = logical_executions
                child_rank = (
                    child.score.total_loss,
                    child.score.cost,
                    child.key.predicate,
                    child.key.mapper,
                )
                best_rank = (
                    best.score.total_loss,
                    best.score.cost,
                    best.key.predicate,
                    best.key.mapper,
                )
                if child_rank < best_rank:
                    best = child
        if len(children) != particles:
            raise CalibrationError(
                f"round {round_number} produced {len(children)} children, expected {particles}"
            )
        weights, ess = normalized_child_weights(child_log_weights)
        if abs(float(weights.sum().item()) - 1.0) > 1e-12:
            raise CalibrationError("normalized particle weights do not sum to one")
        stage: dict[str, object] = {
            "round": round_number,
            "particles": len(children),
            "ess": ess,
            "relative_ess": ess / particles,
            "maximum_normalized_weight": max(weights.tolist()),
            "unique_programs": len({child.key for child in children}),
            "exact_particles": sum(child.score.exact_program for child in children),
            "grammar_restart_draws": round_restart_draws,
            "shortlist_draws": round_shortlist_draws,
            "minimum_sampled_q": min(sampled_probabilities),
            "maximum_sampled_q": max(sampled_probabilities),
        }
        if round_number < ROUNDS:
            generator = torch.Generator(device="cpu")
            generator.manual_seed((resample_seed + round_number) % (2**63))
            ancestors = systematic_resample_count(
                weights,
                count=particles,
                generator=generator,
            ).tolist()
            parent_particles = [children[index] for index in ancestors]
            parent_weights = torch.full(
                (particles,),
                1.0 / particles,
                dtype=torch.float64,
            )
            stage["unique_resampled_ancestors"] = len(set(ancestors))
        else:
            final_particles = children
            final_weights = weights
        stage_records.append(stage)

    if logical_executions != 1 + ROUNDS * particles:
        raise CalibrationError("logical execution accounting does not equal 1 + rounds*N")
    exact_estimate = sum(
        weight
        for weight, particle in zip(final_weights.tolist(), final_particles, strict=True)
        if particle.score.exact_program
    )
    mean_loss_estimate = sum(
        weight * particle.score.total_loss
        for weight, particle in zip(final_weights.tolist(), final_particles, strict=True)
    )
    reference_exact = _finite_number(
        reference.get("exact_target_mass"),
        name="reference exact mass",
    )
    reference_loss = _finite_number(
        reference.get("target_mean_loss"),
        name="reference mean loss",
    )
    final_ess = effective_sample_size(final_weights)
    result = {
        "schema": RUN_SCHEMA,
        "task_id": task.binding.task_id,
        "task_sha256": task.binding.sha256,
        "particles": particles,
        "repetition": repetition,
        "base_seed": base_seed,
        "proposal_seed": proposal_seed,
        "resample_seed": resample_seed,
        "proposal_source": ORACLE_ID,
        "provider_calls": 0,
        "logical_complete_program_executions": logical_executions,
        "sampled_complete_program_slots": logical_executions,
        "estimate": {
            "exact_target_mass": exact_estimate,
            "target_mean_loss": mean_loss_estimate,
        },
        "reference": {
            "exact_target_mass": reference_exact,
            "target_mean_loss": reference_loss,
        },
        "error": {
            "exact_mass_signed": exact_estimate - reference_exact,
            "exact_mass_absolute": abs(exact_estimate - reference_exact),
            "mean_loss_signed": mean_loss_estimate - reference_loss,
            "mean_loss_absolute": abs(mean_loss_estimate - reference_loss),
        },
        "diagnostics": {
            "final_ess": final_ess,
            "final_relative_ess": final_ess / particles,
            "maximum_normalized_weight": max(final_weights.tolist()),
            "unique_terminal_programs": len({particle.key for particle in final_particles}),
            "grammar_restart_draws": grammar_restart_draws,
            "shortlist_draws": shortlist_draws,
            "exact_absorbing_parents": exact_absorbing_parents,
            "oracle_queries": oracle_queries,
            "found_exact": first_exact_slot is not None,
            "first_exact_slot": first_exact_slot,
            "best_loss": best.score.total_loss,
            "best_program": asdict(best.key),
        },
        "proposal_work": _counter_delta(task.work_counters(), work_before),
        "budget_scope": (
            "N counts terminal particles and 1+4N sampled program slots; exact-reference "
            "enumeration and exhaustive local oracle ranking are reported separately and "
            "are not included in N"
        ),
        "stages": stage_records,
    }
    _validate_run(result)
    return result


def _validate_run(run: Mapping[str, object]) -> None:
    if run.get("schema") != RUN_SCHEMA or run.get("provider_calls") != 0:
        raise CalibrationError("run schema or provider-free invariant failed")
    particles = _positive_int(run.get("particles"), name="run particles")
    if run.get("logical_complete_program_executions") != 1 + ROUNDS * particles:
        raise CalibrationError("run logical execution total is inconsistent")
    if run.get("sampled_complete_program_slots") != 1 + ROUNDS * particles:
        raise CalibrationError("run sampled-slot total is inconsistent")
    estimate = run.get("estimate")
    diagnostics = run.get("diagnostics")
    proposal_work = run.get("proposal_work")
    if not all(
        isinstance(record, Mapping)
        for record in (estimate, diagnostics, proposal_work)
    ):
        raise CalibrationError("run estimate or diagnostics record is malformed")
    estimate = cast(Mapping[str, object], estimate)
    diagnostics = cast(Mapping[str, object], diagnostics)
    proposal_work = cast(Mapping[str, object], proposal_work)
    exact_mass = _finite_number(estimate.get("exact_target_mass"), name="estimated mass")
    mean_loss = _finite_number(estimate.get("target_mean_loss"), name="estimated mean loss")
    ess = _finite_number(diagnostics.get("final_ess"), name="final ESS")
    if not 0.0 <= exact_mass <= 1.0 + 1e-12:
        raise CalibrationError("estimated exact mass is outside [0, 1]")
    if mean_loss < 0.0 or not 1.0 <= ess <= particles + 1e-9:
        raise CalibrationError("estimated mean loss or ESS is outside valid bounds")
    for name, value in proposal_work.items():
        _nonnegative_int(value, name=f"proposal work {name}")


def _sample_sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise CalibrationError("cannot take a quantile of no values")
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
        raise CalibrationError("metric estimates and references must be nonempty and aligned")
    errors = [
        estimate - reference
        for estimate, reference in zip(estimates, references, strict=True)
    ]
    absolute = [abs(error) for error in errors]
    squared = [error * error for error in errors]
    error_sd = _sample_sd(errors)
    return {
        "repetitions": len(errors),
        "estimate_mean": statistics.fmean(estimates),
        "reference_mean": statistics.fmean(references),
        "bias": statistics.fmean(errors),
        "rmse": math.sqrt(statistics.fmean(squared)),
        "mae": statistics.fmean(absolute),
        "error_sample_sd": error_sd,
        "bias_monte_carlo_se": error_sd / math.sqrt(len(errors)),
        "estimate_q05": _quantile(estimates, 0.05),
        "estimate_median": _quantile(estimates, 0.5),
        "estimate_q95": _quantile(estimates, 0.95),
    }


def _diagnostic_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": _sample_sd(values),
        "q05": _quantile(values, 0.05),
        "median": _quantile(values, 0.5),
        "q95": _quantile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }


def _log_log_slope(points: Mapping[int, float]) -> float | None:
    usable = [(math.log(float(n)), math.log(value)) for n, value in points.items() if value > 0]
    if len(usable) < 2:
        return None
    xs = [point[0] for point in usable]
    ys = [point[1] for point in usable]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in usable) / denominator


def _group_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    exact_estimates: list[float] = []
    exact_references: list[float] = []
    loss_estimates: list[float] = []
    loss_references: list[float] = []
    ess_values: list[float] = []
    relative_ess_values: list[float] = []
    maximum_weights: list[float] = []
    oracle_candidate_lookups: list[float] = []
    oracle_rankings: list[float] = []
    discoveries = 0
    for run in records:
        estimate = cast(Mapping[str, object], run["estimate"])
        reference = cast(Mapping[str, object], run["reference"])
        diagnostics = cast(Mapping[str, object], run["diagnostics"])
        proposal_work = cast(Mapping[str, object], run["proposal_work"])
        exact_estimates.append(_finite_number(estimate["exact_target_mass"], name="mass"))
        exact_references.append(_finite_number(reference["exact_target_mass"], name="ref mass"))
        loss_estimates.append(_finite_number(estimate["target_mean_loss"], name="loss"))
        loss_references.append(_finite_number(reference["target_mean_loss"], name="ref loss"))
        ess_values.append(_finite_number(diagnostics["final_ess"], name="ESS"))
        relative_ess_values.append(
            _finite_number(diagnostics["final_relative_ess"], name="relative ESS")
        )
        maximum_weights.append(
            _finite_number(diagnostics["maximum_normalized_weight"], name="maximum weight")
        )
        oracle_candidate_lookups.append(
            _finite_number(
                proposal_work["oracle_candidate_score_lookups"],
                name="oracle candidate score lookups",
            )
        )
        oracle_rankings.append(
            _finite_number(proposal_work["oracle_rankings"], name="oracle rankings")
        )
        discoveries += int(diagnostics["found_exact"] is True)
    return {
        "exact_mass": _error_summary(exact_estimates, exact_references),
        "target_mean_loss": _error_summary(loss_estimates, loss_references),
        "final_ess": _diagnostic_summary(ess_values),
        "final_relative_ess": _diagnostic_summary(relative_ess_values),
        "maximum_normalized_weight": _diagnostic_summary(maximum_weights),
        "oracle_candidate_score_lookups": _diagnostic_summary(oracle_candidate_lookups),
        "oracle_rankings": _diagnostic_summary(oracle_rankings),
        "exact_discovery_rate": discoveries / len(records),
    }


def _scaling_summary(by_n: Mapping[int, Mapping[str, object]]) -> dict[str, object]:
    exact_rmse = {
        n: _finite_number(cast(Mapping[str, object], summary["exact_mass"])["rmse"], name="rmse")
        for n, summary in by_n.items()
    }
    loss_rmse = {
        n: _finite_number(
            cast(Mapping[str, object], summary["target_mean_loss"])["rmse"],
            name="rmse",
        )
        for n, summary in by_n.items()
    }
    mean_ess = {
        n: _finite_number(cast(Mapping[str, object], summary["final_ess"])["mean"], name="ess")
        for n, summary in by_n.items()
    }
    return {
        "exact_mass_rmse_log_log_slope": _log_log_slope(exact_rmse),
        "mean_loss_rmse_log_log_slope": _log_log_slope(loss_rmse),
        "mean_ess_log_log_slope": _log_log_slope(mean_ess),
        "exact_mass_rmse_times_sqrt_n": {
            str(n): exact_rmse[n] * math.sqrt(n) for n in sorted(exact_rmse)
        },
        "mean_loss_rmse_times_sqrt_n": {
            str(n): loss_rmse[n] * math.sqrt(n) for n in sorted(loss_rmse)
        },
        "reference_rates": {
            "monte_carlo_rmse_slope": -0.5,
            "linear_ess_slope": 1.0,
        },
        "interpretation": "descriptive finite-grid slopes, not asymptotic rate proofs",
    }


def analyze_runs(
    runs: Sequence[Mapping[str, object]],
    *,
    task_ids: Sequence[str],
    particle_counts: Sequence[int],
    repetitions: int,
) -> dict[str, object]:
    expected = len(task_ids) * len(particle_counts) * repetitions
    if len(runs) != expected:
        raise CalibrationError(f"analysis received {len(runs)} runs, expected {expected}")
    per_task: dict[str, object] = {}
    pooled_by_n: dict[int, dict[str, object]] = {}
    for task_id in task_ids:
        by_n: dict[int, dict[str, object]] = {}
        for particles in particle_counts:
            group = [
                run
                for run in runs
                if run["task_id"] == task_id and run["particles"] == particles
            ]
            if len(group) != repetitions:
                raise CalibrationError(f"{task_id}/N={particles} has incomplete repetitions")
            by_n[particles] = _group_summary(group)
        per_task[task_id] = {
            "by_particle_count": {str(n): by_n[n] for n in sorted(by_n)},
            "scaling": _scaling_summary(by_n),
        }
    for particles in particle_counts:
        group = [run for run in runs if run["particles"] == particles]
        if len(group) != repetitions * len(task_ids):
            raise CalibrationError(f"pooled N={particles} group is incomplete")
        pooled_by_n[particles] = _group_summary(group)
    if PRIMARY_GATE_N not in pooled_by_n:
        raise CalibrationError(f"primary gate N={PRIMARY_GATE_N} is missing")
    primary_summary = pooled_by_n[PRIMARY_GATE_N]
    primary_exact = cast(Mapping[str, object], primary_summary["exact_mass"])
    primary_rmse = _finite_number(primary_exact["rmse"], name="primary exact-mass RMSE")
    primary_bias = _finite_number(primary_exact["bias"], name="primary exact-mass bias")
    rmse_pass = primary_rmse < PRIMARY_EXACT_MASS_RMSE_THRESHOLD
    bias_pass = PRIMARY_EXACT_MASS_BIAS_INTERVAL[0] <= primary_bias <= (
        PRIMARY_EXACT_MASS_BIAS_INTERVAL[1]
    )
    per_task_primary = {
        task_id: cast(Mapping[str, object], per_task[task_id])["by_particle_count"][
            str(PRIMARY_GATE_N)
        ]
        for task_id in task_ids
    }
    return {
        "schema": ANALYSIS_SCHEMA,
        "run_count": len(runs),
        "task_count": len(task_ids),
        "particle_counts": list(particle_counts),
        "repetitions_per_task_particle_count": repetitions,
        "per_task": per_task,
        "pooled": {
            "by_particle_count": {
                str(n): pooled_by_n[n] for n in sorted(pooled_by_n)
            },
            "scaling": _scaling_summary(pooled_by_n),
        },
        "primary_gate": {
            "particle_count": PRIMARY_GATE_N,
            "population": (
                f"pooled over all {len(task_ids)} tasks x {repetitions} repetitions"
            ),
            "exact_mass_rmse": primary_rmse,
            "exact_mass_rmse_threshold_exclusive": PRIMARY_EXACT_MASS_RMSE_THRESHOLD,
            "exact_mass_rmse_pass": rmse_pass,
            "exact_mass_signed_bias": primary_bias,
            "exact_mass_signed_bias_interval_inclusive": list(
                PRIMARY_EXACT_MASS_BIAS_INTERVAL
            ),
            "exact_mass_signed_bias_pass": bias_pass,
            "overall_pass": rmse_pass and bias_pass,
            "per_task_descriptive": per_task_primary,
            "per_task_pass_required": False,
        },
        "claim_boundary": (
            "provider-free calibration of the declared frozen proposal oracle and finite task "
            "suite; no LLM, universal convergence-rate, or wall-clock claim"
        ),
    }


def _resolve_bound_file(project_root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise CalibrationError(f"{label} path must be a nonempty string")
    root = project_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise CalibrationError(f"{label} must resolve to a regular in-project file")
    return path


def _validate_file_bundle(
    records: object,
    *,
    expected_paths: Sequence[str],
    project_root: Path,
    label: str,
) -> None:
    if not isinstance(records, list):
        raise CalibrationError(f"{label} binding must be a list")
    observed_paths: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise CalibrationError(f"{label} binding record has the wrong fields")
        relative = record["path"]
        path = _resolve_bound_file(project_root, relative, label=label)
        if record["sha256"] != _sha256_file(path):
            raise CalibrationError(f"{label} SHA-256 differs for {relative}")
        observed_paths.append(cast(str, relative))
    if tuple(observed_paths) != tuple(expected_paths):
        raise CalibrationError(f"{label} paths or ordering differ from the harness")


def _validate_runtime(expected: object) -> None:
    if not isinstance(expected, dict) or set(expected) != {
        "python_major_minor",
        "packages",
    }:
        raise CalibrationError("runtime binding has the wrong fields")
    packages = expected["packages"]
    if not isinstance(packages, dict) or set(packages) != {"numpy", "pydantic", "torch"}:
        raise CalibrationError("runtime package binding has the wrong fields")
    actual = _runtime_record()
    if expected["python_major_minor"] != actual["python_major_minor"]:
        raise CalibrationError("Python major/minor differs from the frozen runtime")
    if packages != actual["packages"]:
        raise CalibrationError("package versions differ from the frozen runtime")


def load_frozen_plan(
    protocol_path: Path,
    project_root: Path,
    *,
    expected_protocol_sha256: str,
) -> tuple[StudyPlan, dict[str, object]]:
    """Load and fail-closed validate the frozen provider-free protocol."""

    protocol_sha256 = _sha256_file(protocol_path)
    if expected_protocol_sha256 != protocol_sha256:
        raise CalibrationError("protocol differs from the externally expected SHA-256")
    try:
        raw = json.loads(protocol_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CalibrationError(f"invalid protocol JSON: {error.msg}") from error
    if not isinstance(raw, dict) or raw.get("schema") != STUDY_SCHEMA:
        raise CalibrationError("protocol schema is not supported")
    if raw.get("status") != "frozen-provider-free-developmental-before-calibration-runs":
        raise CalibrationError("protocol is not frozen for this study")
    authorization = raw.get("authorization")
    if not isinstance(authorization, dict):
        raise CalibrationError("protocol authorization is missing")
    if authorization != {
        "live_provider_calls": False,
        "cached_provider_replay": False,
        "frozen_proposal_oracle": True,
    }:
        raise CalibrationError("protocol does not authorize exactly the frozen offline oracle")
    proposal = raw.get("proposal_oracle")
    if not isinstance(proposal, dict) or proposal.get("id") != ORACLE_ID:
        raise CalibrationError("protocol proposal oracle differs from this harness")
    design = raw.get("design")
    if not isinstance(design, dict):
        raise CalibrationError("protocol design is missing")
    particle_counts = tuple(
        _positive_int(value, name="particle count")
        for value in cast(list[object], design.get("particle_counts"))
    )
    if particle_counts != PARTICLE_COUNTS:
        raise CalibrationError(f"particle grid must be exactly {PARTICLE_COUNTS}")
    seeds = tuple(
        _nonnegative_int(value, name="repetition seed")
        for value in cast(list[object], design.get("repetition_seeds"))
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise CalibrationError("repetition seeds must be nonempty and unique")
    if design.get("repetitions") != len(seeds):
        raise CalibrationError("declared repetition count differs from the seed list")
    if design.get("rounds") != ROUNDS:
        raise CalibrationError(f"calibration requires exactly {ROUNDS} rounds")
    if design.get("resample_after_rounds") != [1, 2, 3]:
        raise CalibrationError("calibration must resample after rounds 1, 2, and 3")
    gate = raw.get("primary_gate")
    expected_gate = {
        "particle_count": PRIMARY_GATE_N,
        "population": "pooled over all 4 tasks x 32 repetitions",
        "exact_mass_rmse": {
            "operator": "<",
            "threshold": PRIMARY_EXACT_MASS_RMSE_THRESHOLD,
        },
        "exact_mass_signed_bias": {
            "operator": "inclusive_interval",
            "lower": PRIMARY_EXACT_MASS_BIAS_INTERVAL[0],
            "upper": PRIMARY_EXACT_MASS_BIAS_INTERVAL[1],
        },
        "per_task_pass_required": False,
        "per_task_reporting_required": True,
    }
    if gate != expected_gate:
        raise CalibrationError("primary N=256 gate differs from the frozen decision rule")
    epsilon = _finite_number(design.get("epsilon"), name="epsilon")
    evidence_scale = _finite_number(design.get("evidence_scale"), name="evidence scale")
    if not 0.0 < epsilon <= 1.0 or evidence_scale <= 0.0:
        raise CalibrationError("epsilon and evidence scale are outside their valid ranges")
    start_seed = _nonnegative_int(design.get("start_seed"), name="start seed")
    reference_limit = _positive_int(
        design.get("exact_reference_limit"),
        name="exact reference limit",
    )

    bindings = raw.get("bindings")
    if not isinstance(bindings, dict):
        raise CalibrationError("protocol bindings are missing")
    harness = bindings.get("harness")
    if not isinstance(harness, dict) or harness.get("path") != HARNESS_PATH:
        raise CalibrationError("harness binding is missing")
    harness_path = _resolve_bound_file(
        project_root,
        harness.get("path"),
        label="harness",
    )
    actual_harness_sha256 = _sha256_file(harness_path)
    if harness.get("sha256") != actual_harness_sha256:
        raise CalibrationError("calibration harness SHA-256 binding differs")
    _validate_file_bundle(
        bindings.get("dependency_bundle"),
        expected_paths=DEPENDENCY_PATHS,
        project_root=project_root,
        label="research dependency bundle",
    )
    _validate_file_bundle(
        bindings.get("lockfiles"),
        expected_paths=LOCKFILE_PATHS,
        project_root=project_root,
        label="lockfile bundle",
    )
    test_binding = bindings.get("test")
    if not isinstance(test_binding, dict) or test_binding.get("path") != TEST_PATH:
        raise CalibrationError("test binding is missing or has the wrong path")
    test_path = _resolve_bound_file(project_root, TEST_PATH, label="test")
    if test_binding.get("sha256") != _sha256_file(test_path):
        raise CalibrationError("calibration test SHA-256 binding differs")
    source_tree = bindings.get("modelsmc_python_tree")
    if not isinstance(source_tree, dict):
        raise CalibrationError("ModelSMC Python source-tree binding is missing")
    try:
        python_tree_binding(project_root, source_tree)
    except ValueError as error:
        raise CalibrationError(str(error)) from error
    _validate_runtime(raw.get("runtime"))
    raw_tasks = bindings.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise CalibrationError("protocol task bindings are missing")
    tasks: list[TaskBinding] = []
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            raise CalibrationError("task binding must be an object")
        task_id = raw_task.get("id")
        relative = raw_task.get("path")
        expected_sha256 = raw_task.get("sha256")
        fields = (task_id, relative, expected_sha256)
        if not all(isinstance(value, str) and value for value in fields):
            raise CalibrationError("task binding fields must be nonempty strings")
        task_path = _resolve_bound_file(project_root, relative, label=f"task {task_id}")
        actual_sha256 = _sha256_file(task_path)
        if actual_sha256 != expected_sha256:
            raise CalibrationError(f"task SHA-256 binding differs for {task_id}")
        tasks.append(TaskBinding(cast(str, task_id), task_path, actual_sha256))
    task_ids = [task.task_id for task in tasks]
    task_paths = [task.path for task in tasks]
    if len(set(task_ids)) != len(task_ids) or len(set(task_paths)) != len(task_paths):
        raise CalibrationError("task IDs and paths must be unique")
    return (
        StudyPlan(
            tasks=tuple(tasks),
            particle_counts=particle_counts,
            repetition_seeds=seeds,
            epsilon=epsilon,
            evidence_scale=evidence_scale,
            start_seed=start_seed,
            exact_reference_limit=reference_limit,
            protocol_sha256=protocol_sha256,
            harness_sha256=actual_harness_sha256,
        ),
        raw,
    )


def _summary_markdown(
    analysis: Mapping[str, object],
    references: Mapping[str, Mapping[str, object]],
) -> str:
    lines = [
        "# Provider-Free Particle Calibration V1",
        "",
        "This artifact calibrates the evidence-shortlist SMC estimator under the frozen ",
        f"`{ORACLE_ID}` proposal. It contains no live or cached provider calls.",
        "",
        "## Exact references",
        "",
        "| Task | Programs | Exact mass | Mean loss |",
        "|---|---:|---:|---:|",
    ]
    for task_id, reference in references.items():
        lines.append(
            f"| {task_id} | {reference['program_syntaxes']} | "
            f"{cast(float, reference['exact_target_mass']):.8g} | "
            f"{cast(float, reference['target_mean_loss']):.8g} |"
        )
    lines.extend(
        [
            "",
            "## Pooled calibration",
            "",
            "| N | Exact-mass bias | Exact-mass RMSE | Mean-loss bias | "
            "Mean-loss RMSE | Mean relative ESS |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    pooled = cast(Mapping[str, object], analysis["pooled"])
    by_n = cast(Mapping[str, object], pooled["by_particle_count"])
    for n in sorted(by_n, key=int):
        summary = cast(Mapping[str, object], by_n[n])
        mass = cast(Mapping[str, object], summary["exact_mass"])
        loss = cast(Mapping[str, object], summary["target_mean_loss"])
        relative_ess = cast(Mapping[str, object], summary["final_relative_ess"])
        lines.append(
            f"| {n} | {cast(float, mass['bias']):.6g} | "
            f"{cast(float, mass['rmse']):.6g} | {cast(float, loss['bias']):.6g} | "
            f"{cast(float, loss['rmse']):.6g} | {cast(float, relative_ess['mean']):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Frozen N=256 primary gate",
            "",
        ]
    )
    gate = cast(Mapping[str, object], analysis["primary_gate"])
    lines.append(
        f"Overall pass: **{gate['overall_pass']}**; pooled exact-mass RMSE "
        f"{cast(float, gate['exact_mass_rmse']):.6g} (<0.10 required), signed bias "
        f"{cast(float, gate['exact_mass_signed_bias']):.6g} ([-0.03,+0.03] required)."
    )
    lines.extend(
        [
            "",
            "Per-task N=256 estimates are recorded under `analysis.json` at "
            "`primary_gate.per_task_descriptive`; no per-task pass was frozen.",
            "",
            "## Scientific boundary",
            "",
        "The oracle is deterministic and mechanically evidence-aware. These results measure ",
        "finite-sample calibration for that proposal and these enumerable tasks. They do not ",
        "measure an LLM, provider variability, arbitrary-task generalization, wall-clock ",
        "speed, or an asymptotic convergence rate.",
        "",
        "N counts terminal particles and 1+4N sampled program slots per repetition. The exact ",
        "reference enumerates every grammar program, and the oracle exhaustively ranks each ",
        "queried one-hole neighborhood using public execution loss. Those costs are reported ",
        "separately and are not included in N, so this is not a compute-efficiency benchmark.",
        "",
        ]
    )
    return "\n".join(lines)


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


def run_study(
    plan: StudyPlan,
    protocol_record: Mapping[str, object],
    output: Path,
) -> dict[str, object]:
    """Execute the complete frozen study and atomically publish its artifact."""

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
                "task_sha256": {
                    binding.task_id: binding.sha256 for binding in plan.tasks
                },
                "provider_calls_authorized": 0,
                "provider_calls_observed": 0,
                "sampled_slot_formula": "1 + 4*N per repetition",
                "reference_and_oracle_work_excluded_from_n": True,
            },
        )
        references: dict[str, dict[str, object]] = {}
        runs: list[dict[str, object]] = []
        oracle_inventories: list[dict[str, object]] = []
        for binding in plan.tasks:
            task = CalibrationTask(binding, start_seed=plan.start_seed)
            reference = task.exact_reference(
                limit=plan.exact_reference_limit,
                evidence_scale=plan.evidence_scale,
            )
            references[binding.task_id] = reference
            _write_json(staging / "references" / f"{binding.task_id}.json", reference)
            for particles in plan.particle_counts:
                for repetition, seed in enumerate(plan.repetition_seeds):
                    run = run_repetition(
                        task,
                        reference,
                        particles=particles,
                        repetition=repetition,
                        base_seed=seed,
                        epsilon=plan.epsilon,
                        evidence_scale=plan.evidence_scale,
                    )
                    runs.append(run)
                    completed_runs += 1
                    _write_json(
                        staging
                        / "runs"
                        / binding.task_id
                        / f"n-{particles:04d}"
                        / f"rep-{repetition:03d}.json",
                        run,
                    )
            oracle_inventories.append(task.oracle_inventory())
            del task
            gc.collect()
        expected_runs = (
            len(plan.tasks) * len(plan.particle_counts) * len(plan.repetition_seeds)
        )
        if completed_runs != expected_runs or any(run["provider_calls"] != 0 for run in runs):
            raise CalibrationError("completed run count or provider-free invariant failed")
        analysis = analyze_runs(
            runs,
            task_ids=[binding.task_id for binding in plan.tasks],
            particle_counts=plan.particle_counts,
            repetitions=len(plan.repetition_seeds),
        )
        _write_json(staging / "analysis.json", analysis)
        _write_json(
            staging / "oracle-inventory.json",
            {
                "schema": f"{STUDY_SCHEMA}-oracle-inventory-v1",
                "oracle_id": ORACLE_ID,
                "tasks": oracle_inventories,
            },
        )
        (staging / "SUMMARY.md").write_text(
            _summary_markdown(analysis, references),
            encoding="utf-8",
        )
        inventory = _seal_inventory(staging)
        staging.rename(output)
        return {
            "schema": STUDY_SCHEMA,
            "status": "complete",
            "output": output.as_posix(),
            "run_count": completed_runs,
            "inventory_sha256": inventory["records_sha256"],
            "analysis": analysis,
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
    parser.add_argument("--protocol", type=Path, default=Path(DEFAULT_PROTOCOL))
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    protocol_path = args.protocol
    if not protocol_path.is_absolute():
        protocol_path = project_root / protocol_path
    plan, protocol = load_frozen_plan(
        protocol_path.resolve(),
        project_root,
        expected_protocol_sha256=args.expected_protocol_sha256,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "schema": STUDY_SCHEMA,
                    "status": "preflight-passed",
                    "protocol_sha256": plan.protocol_sha256,
                    "harness_sha256": plan.harness_sha256,
                    "tasks": [binding.task_id for binding in plan.tasks],
                    "particle_counts": plan.particle_counts,
                    "repetitions": len(plan.repetition_seeds),
                    "provider_calls": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.output is None:
        raise CalibrationError("--output is required unless --preflight-only is used")
    result = run_study(plan, protocol, args.output)
    print(json.dumps(result | {"analysis": "written to analysis.json"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

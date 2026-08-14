"""Fresh V3 confirmation of provider-free terminal finite-support inference.

The immutable V2 failure and reused-task V3 diagnostic motivate this method,
but neither is overwritten or reclassified.  V3 builds its factorized proposal
from public singleton evidence *before* materializing the exact 36,000-program
reference, freezes a native analysis and custody chain, and then evaluates one
independently generated 32-task suite without provider calls.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.metadata
import json
import os
import platform
import secrets
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np

from modelsmc_pbe.config import ExperimentConfig, load_experiment_config
from modelsmc_pbe.core.cost import program_cost
from modelsmc_pbe.core.evaluate import evaluate_expression, evaluate_program
from modelsmc_pbe.core.results import ScoredProgram
from modelsmc_pbe.core.scorer import ProgramScorer
from modelsmc_pbe.core.types import Node
from modelsmc_pbe.grammar import arithmetic_expressions, filter_predicates
from research.blind_filter_map_confirmation_v3 import assemble_program
from research.calibrated_program_inference_v2_fresh import (
    FreshCalibrationError,
    _python_tree_record,
    _sha256_bytes,
    _sha256_file,
    _write_json_exclusive,
    canonical_bytes,
)
from research.calibrated_program_inference_v2_fresh import (
    secret_commitment as v2_secret_commitment,
)
from research.calibrated_program_inference_v2_screen import (
    ArmSpec,
    ModeBank,
    Proposal,
    _ess,
    _normalize,
    _reference_record,
    _sample_categorical,
    _systematic_resample,
    build_proposal,
    next_beta,
)
from research.calibrated_program_inference_v3_validation import (
    analyze,
    seal_inventory,
    validate_artifact,
)
from research.evidence_shortlist_smc import (
    ProgramKey,
    recursive_grammar_probabilities,
)
from research.execution_guided_repair import (
    _assemble_program,
    _dsl_catalog,
)
from research.particle_calibration_terminal_diagnostic_v2 import (
    TaskBinding,
    TerminalTask,
)

STUDY_SCHEMA = "provider-free-calibrated-program-inference-v3-fresh"
RUN_SCHEMA = f"{STUDY_SCHEMA}-run-v1"
REFERENCE_SCHEMA = f"{STUDY_SCHEMA}-reference-v1"
ANALYSIS_SCHEMA = f"{STUDY_SCHEMA}-analysis-v1"
REPLAY_SCHEMA = f"{STUDY_SCHEMA}-deterministic-replay-receipt-v1"
STATUS = "factorized-method-r2-frozen-before-second-fresh-secret"
HARNESS_PATH = "research/calibrated_program_inference_v3_fresh.py"
TEST_PATH = "research/tests/test_calibrated_program_inference_v3_fresh.py"
DEFAULT_PROTOCOL = "research/protocol-calibrated-program-inference-v3-fresh-r2.json"
DEFAULT_METHOD_SEAL = (
    "research/protocol-calibrated-program-inference-v3-fresh-r2.method-seal.json"
)
DEFAULT_SUITE = "artifacts/calibrated-program-inference-v3-fresh-suite"
DEFAULT_OUTPUT = "artifacts/calibrated-program-inference-v3-fresh"

TASK_COUNT = 32
TASK_IDS = tuple(f"fresh-cal-v3-{index:03d}" for index in range(1, TASK_COUNT + 1))
CONSTANTS = tuple(range(-3, 5))
DOMAIN = CONSTANTS
PARTICLE_COUNTS = (256, 512, 1024)
REPETITIONS = 64
REPETITION_SEEDS = tuple(range(936_001, 936_001 + REPETITIONS))
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 937_001
PRIMARY_PARTICLE_COUNT = 256
MODE_COUNT = 8
ALIASES_PER_MODE = 4
COMPONENT_POOL_SIZE = 8
GRAMMAR_MASS = 0.25
CENTER_MASS = 0.75
IDENTITY_TOLERANCE = 1e-12
ANNEAL_CONDITIONAL_ESS = 0.80
RESAMPLE_RELATIVE_ESS = 0.50
MAX_ANNEAL_STAGES = 64

PRIMARY_ARM_ID = "factorized-k8-a025-e075-proposal-bridge"
FACTOR_TERMINAL_ARM_ID = "factorized-k8-a025-e075-terminal-is"
STICKY_ARM_ID = "sticky-epsilon-005-terminal-is"
PRIMARY_ARM = ArmSpec(
    PRIMARY_ARM_ID,
    "sampled-semantic-modes",
    MODE_COUNT,
    GRAMMAR_MASS,
    CENTER_MASS,
    "proposal-bridge-smc",
    0,
    "primary-fresh-v3-terminal-calibration",
)
FACTOR_TERMINAL_ARM = ArmSpec(
    FACTOR_TERMINAL_ARM_ID,
    "sampled-semantic-modes",
    MODE_COUNT,
    GRAMMAR_MASS,
    CENTER_MASS,
    "terminal-is",
    0,
    "matched-q-terminal-is-ablation",
)
STICKY_ARM = ArmSpec(
    STICKY_ARM_ID,
    "sticky-exact-parent",
    1,
    0.05,
    1.0,
    "terminal-is",
    0,
    "failed-v1-terminal-baseline",
)
FROZEN_ARMS = (PRIMARY_ARM, FACTOR_TERMINAL_ARM, STICKY_ARM)
ARM_PARTICLES = {
    PRIMARY_ARM_ID: PARTICLE_COUNTS,
    FACTOR_TERMINAL_ARM_ID: (PRIMARY_PARTICLE_COUNT,),
    STICKY_ARM_ID: (PRIMARY_PARTICLE_COUNT,),
}
EXPECTED_RUN_COUNT = TASK_COUNT * REPETITIONS * sum(
    len(ARM_PARTICLES[arm.arm_id]) for arm in FROZEN_ARMS
)
EXPECTED_INITIAL_PROPOSAL_DRAWS = TASK_COUNT * REPETITIONS * sum(
    sum(ARM_PARTICLES[arm.arm_id]) for arm in FROZEN_ARMS
)

PRIOR_SECRET_COMMITMENT = (
    "702606b13152d78067d2f230a64914de191038db77651d36ae1d7d916b36365b"
)
SECRET_DOMAIN = b"calibrated-program-inference-v3-fresh-secret\0"
DRAW_DOMAIN = b"calibrated-program-inference-v3-fresh-draw\0"
TARGET_COMMITMENT_DOMAIN = b"calibrated-program-inference-v3-target\0"

DEPENDENCY_PATHS = (
    "research/calibrated_program_inference_v2_fresh.py",
    "research/calibrated_program_inference_v2_screen.py",
    "research/calibrated_program_inference_v3_factorized_diagnostic.py",
    "research/calibrated_program_inference_v3_replay.py",
    "research/calibrated_program_inference_v3_validation.py",
    "research/tests/test_calibrated_program_inference_v3_replay.py",
    "research/tests/test_calibrated_program_inference_v3_validation.py",
    "research/protocol-calibrated-program-inference-v3-factorized-diagnostic.json",
    "research/blind_filter_map_confirmation_v3.py",
    "research/evidence_shortlist_smc.py",
    "research/execution_guided_repair.py",
    "research/iterative_beam_experiment.py",
    "research/particle_calibration_terminal_diagnostic_v2.py",
)
LOCKFILE_PATHS = ("pyproject.toml", "uv.lock")
ATTEMPT_BINDINGS = (
    (
        "v2_failed_protocol",
        "papers/01-modelsmc/modelsmc-pbe-python/research/"
        "protocol-calibrated-program-inference-v2-fresh.json",
    ),
    (
        "v2_failed_method_seal",
        "papers/01-modelsmc/modelsmc-pbe-python/research/"
        "protocol-calibrated-program-inference-v2-fresh.method-seal.json",
    ),
    (
        "v2_failed_custody_seal",
        "artifacts/calibrated-program-inference-v2-fresh-custody-seal.json",
    ),
    (
        "v2_failed_public_manifest",
        "artifacts/calibrated-program-inference-v2-fresh-suite/public/manifest.json",
    ),
    (
        "v2_failed_runs",
        "artifacts/calibrated-program-inference-v2-fresh/runs.json",
    ),
    (
        "v2_failed_analysis",
        "artifacts/calibrated-program-inference-v2-fresh/analysis.json",
    ),
    (
        "v2_failed_inventory",
        "artifacts/calibrated-program-inference-v2-fresh/inventory.json",
    ),
    (
        "v2_failed_unblind",
        "artifacts/calibrated-program-inference-v2-fresh-unblind-verification.json",
    ),
    (
        "v3_diagnostic_protocol",
        "papers/01-modelsmc/modelsmc-pbe-python/research/"
        "protocol-calibrated-program-inference-v3-factorized-diagnostic.json",
    ),
    (
        "v3_diagnostic_analysis",
        "artifacts/calibrated-program-inference-v3-factorized-diagnostic/analysis.json",
    ),
    (
        "v3_diagnostic_runs",
        "artifacts/calibrated-program-inference-v3-factorized-diagnostic/runs.json",
    ),
    (
        "v3_diagnostic_inventory",
        "artifacts/calibrated-program-inference-v3-factorized-diagnostic/inventory.json",
    ),
    (
        "v3_fresh_r1_protocol_superseded_pre_secret",
        "papers/01-modelsmc/modelsmc-pbe-python/research/"
        "protocol-calibrated-program-inference-v3-fresh.json",
    ),
    (
        "v3_fresh_r1_method_seal_superseded_pre_secret",
        "papers/01-modelsmc/modelsmc-pbe-python/research/"
        "protocol-calibrated-program-inference-v3-fresh.method-seal.json",
    ),
    (
        "v3_fresh_r1_supersession_record",
        "papers/01-modelsmc/modelsmc-pbe-python/research/"
        "CALIBRATED_PROGRAM_INFERENCE_V3_FRESH_R1_SUPERSESSION.md",
    ),
)


def _resolve_project_file(project_root: Path, relative: str) -> Path:
    root = project_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise FreshCalibrationError(f"bound V3 file is invalid: {relative}")
    return path


def _project_record(project_root: Path, relative: str) -> dict[str, str]:
    path = _resolve_project_file(project_root, relative)
    return {"path": relative, "sha256": _sha256_file(path)}


def _resolve_workspace_file(project_root: Path, logical_path: str) -> Path:
    suffix = Path(logical_path)
    candidates: list[Path] = []
    for root in (project_root.resolve(), *project_root.resolve().parents):
        candidates.append((root / suffix).resolve())
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise FreshCalibrationError(f"bound workspace file is missing: {logical_path}")


def _workspace_record(project_root: Path, logical_path: str) -> dict[str, str]:
    path = _resolve_workspace_file(project_root, logical_path)
    return {"path": logical_path, "sha256": _sha256_file(path)}


def _research_tree_record(project_root: Path) -> dict[str, object]:
    relative_root = "research"
    tree_root = (project_root / relative_root).resolve()
    if not tree_root.is_dir() or any(path.is_symlink() for path in tree_root.rglob("*")):
        raise FreshCalibrationError("V3 research source tree is invalid")
    paths = sorted(
        (path for path in tree_root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(tree_root).as_posix().encode("utf-8"),
    )
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


def _v3_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in (STUDY_SCHEMA, *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _require_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise FreshCalibrationError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise FreshCalibrationError(f"{name} must be hexadecimal") from error
    return value


@dataclass(frozen=True, slots=True)
class PublicModeBank:
    """Program-key bank selected without an exact-support object."""

    groups: tuple[tuple[ProgramKey, ...], ...]
    metadata: dict[str, object]


class PublicAcquisitionView:
    """Least-authority view used before exact reference enumeration.

    This object owns only the public task, component catalogs, singleton facts,
    and a scorer capped at 64 complete representatives.  It deliberately has no
    target, exact-mass, full-loss, or 36,000-program arrays.
    """

    def __init__(self, binding: TaskBinding) -> None:
        self.binding = binding
        self.config: ExperimentConfig = load_experiment_config(binding.path)
        signature = self.config.spec.signature
        if signature is None or signature.input_type.value != "List<Int>":
            raise FreshCalibrationError("V3 task must take List<Int>")
        if signature.output_type.value != "List<Int>":
            raise FreshCalibrationError("V3 task must return List<Int>")
        self.scorer = ProgramScorer(self.config)
        constants = tuple(self.config.spec.integer_constants)
        self.predicate_catalog = _dsl_catalog(filter_predicates(constants))
        self.mapper_catalog = _dsl_catalog(arithmetic_expressions("Item", constants))
        self.predicate_order = tuple(sorted(self.predicate_catalog))
        self.mapper_order = tuple(sorted(self.mapper_catalog))
        if len(self.predicate_order) != 600 or len(self.mapper_order) != 60:
            raise FreshCalibrationError("V3 component catalog sizes differ")
        self.predicate_probability, self.mapper_probability = (
            recursive_grammar_probabilities(self.predicate_order, self.mapper_order)
        )
        self.observed_items = tuple(
            sorted(
                {
                    item
                    for example in self.config.spec.examples
                    for item in cast(list[int], example.input_value)
                }
            )
        )
        if self.observed_items != DOMAIN:
            raise FreshCalibrationError("V3 public evidence must cover all singletons")
        self.item_positions = {
            item: index for index, item in enumerate(self.observed_items)
        }
        self.keep: dict[int, bool] = {}
        self.mapper_facts: dict[int, int] = {}
        for example in self.config.spec.examples:
            inputs = cast(list[int], example.input_value)
            outputs = cast(list[int], example.output_value)
            if len(inputs) != 1:
                continue
            if len(outputs) not in (0, 1):
                raise FreshCalibrationError("singleton output cardinality differs")
            item = inputs[0]
            self.keep[item] = len(outputs) == 1
            if outputs:
                self.mapper_facts[item] = outputs[0]
        if tuple(sorted(self.keep)) != DOMAIN or not self.mapper_facts:
            raise FreshCalibrationError("V3 singleton evidence is incomplete")
        self.complete_representatives_scored = 0

    def predicate_signature(self, dsl: str) -> tuple[bool, ...]:
        expression = cast(Node, self.predicate_catalog[dsl])
        values = tuple(
            evaluate_expression(expression, list(self.observed_items), item=item)
            for item in self.observed_items
        )
        if any(not isinstance(value, bool) for value in values):
            raise FreshCalibrationError("predicate returned a non-boolean value")
        return cast(tuple[bool, ...], values)

    def mapper_signature(self, dsl: str) -> tuple[int, ...]:
        expression = cast(Node, self.mapper_catalog[dsl])
        values = tuple(
            evaluate_expression(expression, list(self.observed_items), item=item)
            for item in self.observed_items
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise FreshCalibrationError("mapper returned a non-integer value")
        return cast(tuple[int, ...], values)

    def semantic_signature(self, key: ProgramKey) -> tuple[int | None, ...]:
        predicate = self.predicate_signature(key.predicate)
        mapper = self.mapper_signature(key.mapper)
        return tuple(
            mapper[index] if predicate[index] else None
            for index in range(len(self.observed_items))
        )

    def score_key(self, key: ProgramKey) -> ScoredProgram:
        if self.complete_representatives_scored >= COMPONENT_POOL_SIZE**2:
            raise FreshCalibrationError("V3 acquisition exceeded 64 complete scores")
        self.complete_representatives_scored += 1
        result = self.scorer.score(
            _assemble_program(
                self.predicate_catalog[key.predicate],
                self.mapper_catalog[key.mapper],
            )
        )
        if not isinstance(result, ScoredProgram):
            raise FreshCalibrationError("V3 public representative was rejected")
        return result

    def key_cost(self, key: ProgramKey) -> int:
        return program_cost(
            cast(
                Node,
                _assemble_program(
                    self.predicate_catalog[key.predicate],
                    self.mapper_catalog[key.mapper],
                ),
            )
        )


def factorized_public_mode_bank(
    binding: TaskBinding,
    *,
    mode_count: int = MODE_COUNT,
    aliases_per_mode: int = ALIASES_PER_MODE,
    component_pool_size: int = COMPONENT_POOL_SIZE,
) -> PublicModeBank:
    """Select the factorized bank before constructing ``TerminalTask``."""

    if min(mode_count, aliases_per_mode, component_pool_size) < 1:
        raise FreshCalibrationError("V3 bank sizes must be positive")
    view = PublicAcquisitionView(binding)
    predicate_groups: dict[tuple[bool, ...], list[str]] = {}
    for dsl in view.predicate_order:
        predicate_groups.setdefault(view.predicate_signature(dsl), []).append(dsl)
    predicate_ranked = sorted(
        predicate_groups.items(),
        key=lambda pair: (
            sum(
                predicted != view.keep[item]
                for item, predicted in zip(DOMAIN, pair[0], strict=True)
            ),
            pair[1][0],
        ),
    )[:component_pool_size]
    mapper_groups: dict[tuple[int, ...], list[str]] = {}
    for dsl in view.mapper_order:
        mapper_groups.setdefault(view.mapper_signature(dsl), []).append(dsl)
    mapper_ranked = sorted(
        mapper_groups.items(),
        key=lambda pair: (
            sum(
                pair[0][view.item_positions[item]] != expected
                for item, expected in view.mapper_facts.items()
            ),
            pair[1][0],
        ),
    )[:component_pool_size]

    candidates: list[
        tuple[tuple[float | int | str, ...], tuple[int | None, ...], str, str]
    ] = []
    seen: set[tuple[int | None, ...]] = set()
    for _, predicate_aliases in predicate_ranked:
        for _, mapper_aliases in mapper_ranked:
            key = ProgramKey(predicate_aliases[0], mapper_aliases[0])
            signature = view.semantic_signature(key)
            if signature in seen:
                continue
            seen.add(signature)
            score = view.score_key(key)
            candidates.append(
                (
                    (score.total_loss, score.cost, key.predicate, key.mapper),
                    signature,
                    key.predicate,
                    key.mapper,
                )
            )
    candidates.sort(key=lambda record: record[0])
    selected = candidates[:mode_count]
    if len(selected) != mode_count:
        raise FreshCalibrationError("V3 factorized cross has too few modes")

    predicate_by_signature = dict(predicate_ranked)
    mapper_by_signature = dict(mapper_ranked)
    groups: list[tuple[ProgramKey, ...]] = []
    for _, semantic, predicate_dsl, mapper_dsl in selected:
        predicate_signature = view.predicate_signature(predicate_dsl)
        mapper_signature = view.mapper_signature(mapper_dsl)
        aliases = [
            ProgramKey(predicate_alias, mapper_alias)
            for predicate_alias in predicate_by_signature[predicate_signature]
            for mapper_alias in mapper_by_signature[mapper_signature]
        ]
        aliases = [key for key in aliases if view.semantic_signature(key) == semantic]
        aliases.sort(key=lambda key: (view.key_cost(key), key.predicate, key.mapper))
        groups.append(tuple(aliases[:aliases_per_mode]))
    if any(not group for group in groups):
        raise FreshCalibrationError("V3 factorized mode has no syntax alias")
    return PublicModeBank(
        groups=tuple(groups),
        metadata={
            "selection": (
                "rank public singleton predicate and mapper behavior classes; cross "
                "the top eight of each; score at most 64 complete representatives; "
                "retain eight finite-domain modes and four cost-ranked aliases"
            ),
            "acquisition_completed_before_reference_materialization": True,
            "hidden_target_used": False,
            "complete_program_catalog_ranked": False,
            "predicate_syntaxes_evaluated": len(view.predicate_order),
            "mapper_syntaxes_evaluated": len(view.mapper_order),
            "component_behavior_classes_crossed": (
                len(predicate_ranked) * len(mapper_ranked)
            ),
            "complete_representatives_scored": view.complete_representatives_scored,
            "retained_semantic_modes": len(groups),
            "retained_syntax_centers": sum(len(group) for group in groups),
            "aliases_per_retained_mode": [len(group) for group in groups],
        },
    )


def materialize_mode_bank(task: TerminalTask, bank: PublicModeBank) -> ModeBank:
    groups = tuple(
        tuple(task.key_index[key] for key in group)
        for group in bank.groups
    )
    flattened = [index for group in groups for index in group]
    metadata = {
        **bank.metadata,
        "reference_materialized_after_acquisition": True,
        "exact_programs_in_bank_postselection": int(task.exact[flattened].sum()),
        "best_bank_loss_postselection": float(task.losses[flattened].min()),
        "worst_bank_loss_postselection": float(task.losses[flattened].max()),
    }
    return ModeBank(groups=groups, metadata=metadata)


def _factorized_q_ledger(
    task: TerminalTask,
    groups: Sequence[Sequence[int]],
) -> tuple[np.ndarray, dict[str, object]]:
    if len(groups) != MODE_COUNT or any(not group for group in groups):
        raise FreshCalibrationError("V3 factorized bank shape differs")
    mapper_count = len(task.mapper_order)
    predicate_count = len(task.predicate_order)
    predicate_probability = np.asarray(
        [float(task.predicate_probability[value]) for value in task.predicate_order]
    )
    mapper_probability = np.asarray(
        [float(task.mapper_probability[value]) for value in task.mapper_order]
    )
    grammar_component = GRAMMAR_MASS * task.grammar.copy()
    local_modes: list[np.ndarray] = []
    predicate_offsets = np.arange(predicate_count, dtype=np.int64) * mapper_count
    for group in groups:
        mode = np.zeros_like(task.grammar)
        per_alias = (1.0 - GRAMMAR_MASS) / MODE_COUNT / len(group)
        for index in group:
            predicate_index, mapper_index = divmod(int(index), mapper_count)
            mode[index] += per_alias * CENTER_MASS
            start = predicate_index * mapper_count
            mode[start : start + mapper_count] += (
                per_alias * 0.125 * mapper_probability
            )
            mode[predicate_offsets + mapper_index] += (
                per_alias * 0.125 * predicate_probability
            )
        local_modes.append(mode)
    raw = grammar_component + np.sum(local_modes, axis=0)
    pre_sum = float(raw.sum())
    if np.any(raw <= 0.0) or abs(pre_sum - 1.0) > IDENTITY_TOLERANCE:
        raise FreshCalibrationError("V3 factorized proposal mass ledger failed")
    local_stack = np.stack(local_modes)
    return raw / pre_sum, {
        "grammar_component_mass": float(grammar_component.sum()),
        "local_component_mass": float(local_stack.sum()),
        "per_mode_component_mass": [float(mode.sum()) for mode in local_modes],
        "center_fraction_within_mode": CENTER_MASS,
        "predicate_slice_fraction_within_mode": 0.125,
        "mapper_slice_fraction_within_mode": 0.125,
        "pre_normalization_probability_sum": pre_sum,
        "pre_normalization_sum_error": pre_sum - 1.0,
        "local_mode_overlap_programs": int(
            np.sum(np.sum(local_stack > 0.0, axis=0) > 1)
        ),
    }


def build_v3_proposal(
    task: TerminalTask,
    arm: ArmSpec,
    bank: ModeBank,
) -> Proposal:
    proposal = build_proposal(task, arm, bank)
    if arm.bank_kind != "sampled-semantic-modes":
        return proposal
    reconstructed, ledger = _factorized_q_ledger(task, bank.groups)
    error = float(np.max(np.abs(reconstructed - proposal.probabilities)))
    if error > IDENTITY_TOLERANCE:
        raise FreshCalibrationError("V3 proposal reconstruction differs")
    census = {
        **proposal.census,
        "factorized_mass_ledger": ledger,
        "factorized_reconstruction_maximum_absolute_error": error,
    }
    return Proposal(
        arm=proposal.arm,
        probabilities=proposal.probabilities,
        cdf=proposal.cdf,
        bank_indices=proposal.bank_indices,
        semantic_mode_count=proposal.semantic_mode_count,
        census=census,
    )


def bridge_telescoping_error(
    task: TerminalTask,
    proposal: Proposal,
    indices: np.ndarray,
    betas: Sequence[float],
) -> float:
    if not betas or betas[0] != 0.0 or betas[-1] != 1.0:
        raise FreshCalibrationError("V3 telescope requires beta endpoints 0 and 1")
    log_ratio = task.log_gamma[indices] - np.log(proposal.probabilities[indices])
    weights = np.ones(len(indices), dtype=np.float64)
    for previous, current in pairwise(betas):
        if not previous < current:
            raise FreshCalibrationError("V3 telescope betas must increase")
        weights *= np.exp((current - previous) * log_ratio)
        weights /= float(weights.sum())
    expected = np.exp(log_ratio - float(log_ratio.max()))
    expected /= float(expected.sum())
    return float(np.max(np.abs(weights - expected)))


def run_v3_repetition(
    task: TerminalTask,
    proposal: Proposal,
    *,
    particles: int,
    repetition: int,
    base_seed: int,
) -> dict[str, object]:
    """Run one V3-native terminal-IS or proposal-bridge repetition."""

    if particles < 2 or repetition not in range(REPETITIONS):
        raise FreshCalibrationError("V3 repetition coordinates differ")
    if base_seed != REPETITION_SEEDS[repetition]:
        raise FreshCalibrationError("V3 repetition base seed differs")
    sample_seed = _v3_seed(
        "smc",
        task.binding.sha256,
        proposal.arm.arm_id,
        particles,
        repetition,
        base_seed,
    )
    rng = np.random.Generator(np.random.PCG64(sample_seed))
    states = _sample_categorical(proposal.cdf, particles, rng)
    weights = np.full(particles, 1.0 / particles, dtype=np.float64)
    resamples = 0
    stages: list[dict[str, object]] = []
    if proposal.arm.algorithm == "terminal-is":
        weights = _normalize(task.target[states] / proposal.probabilities[states])
    elif proposal.arm.algorithm == "proposal-bridge-smc":
        log_ratio = task.log_gamma - np.log(proposal.probabilities)
        beta = 0.0
        for stage in range(1, MAX_ANNEAL_STAGES + 1):
            beta_next = next_beta(
                weights,
                log_ratio[states],
                beta,
                target_fraction=ANNEAL_CONDITIONAL_ESS,
            )
            increment = (beta_next - beta) * log_ratio[states]
            increment -= float(increment.max())
            weights = _normalize(weights * np.exp(increment))
            stage_ess = _ess(weights)
            resampled = beta_next < 1.0 and (
                stage_ess / particles < RESAMPLE_RELATIVE_ESS
            )
            if resampled:
                selected = _systematic_resample(weights, rng)
                states = states[selected]
                weights = np.full(particles, 1.0 / particles, dtype=np.float64)
                resamples += 1
            stages.append(
                {
                    "stage": stage,
                    "beta_previous": beta,
                    "beta_current": beta_next,
                    "relative_ess_before_optional_resampling": stage_ess / particles,
                    "resampled": resampled,
                }
            )
            beta = beta_next
            if beta == 1.0:
                break
        if beta != 1.0:
            raise FreshCalibrationError("V3 bridge exceeded the annealing-stage cap")
    else:
        raise FreshCalibrationError("V3 arm algorithm differs")
    exact_mass = float(weights @ task.exact[states])
    mean_loss = float(weights @ task.losses[states])
    importance_ess = _ess(weights)
    run = {
        "schema": RUN_SCHEMA,
        "task_id": task.binding.task_id,
        "task_sha256": task.binding.sha256,
        "arm_id": proposal.arm.arm_id,
        "algorithm": proposal.arm.algorithm,
        "particles": particles,
        "repetition": repetition,
        "base_seed": base_seed,
        "sample_seed": sample_seed,
        "provider_calls": 0,
        "logical_proposal_draws": particles,
        "estimate": {
            "exact_target_mass": exact_mass,
            "target_mean_loss": mean_loss,
        },
        "reference": {
            "exact_target_mass": task.exact_mass,
            "target_mean_loss": task.target_mean_loss,
        },
        "error": {
            "exact_mass_signed": exact_mass - task.exact_mass,
            "target_mean_loss_signed": mean_loss - task.target_mean_loss,
        },
        "diagnostics": {
            "importance_ess": importance_ess,
            "relative_importance_ess": importance_ess / particles,
            "maximum_normalized_weight": float(weights.max()),
            "unique_terminal_programs": int(np.unique(states).size),
            "exact_terminal_particles": int(task.exact[states].sum()),
            "resampling_events": resamples,
            "mh_attempts": 0,
            "mh_accepted": 0,
            "mh_acceptance_rate": 0.0,
            "annealing_stages": len(stages),
        },
        "stages": stages,
    }
    if not 0.0 <= exact_mass <= 1.0 + IDENTITY_TOLERANCE:
        raise FreshCalibrationError("V3 exact-mass estimate is outside [0,1]")
    if not 1.0 <= importance_ess <= particles + 1e-9:
        raise FreshCalibrationError("V3 importance ESS is outside its range")
    return run


def build_protocol(project_root: Path) -> dict[str, object]:
    failed_v2 = json.loads(
        _resolve_workspace_file(
            project_root,
            "artifacts/calibrated-program-inference-v2-fresh/analysis.json",
        ).read_text()
    )
    if failed_v2.get("primary_gate", {}).get("pass") is not False:
        raise FreshCalibrationError("V3 requires the preserved V2 fresh gate failure")
    diagnostic = json.loads(
        _resolve_workspace_file(
            project_root,
            "artifacts/calibrated-program-inference-v3-factorized-diagnostic/analysis.json",
        ).read_text()
    )
    if diagnostic["diagnostic_rule"]["pass"] is not True:
        raise FreshCalibrationError("V3 method requires the passed diagnostic record")
    return {
        "schema": STUDY_SCHEMA,
        "status": STATUS,
        "frozen_at": "2026-08-14",
        "study_type": "second fresh provider-free finite-support calibration confirmation",
        "question": (
            "Does the factorized evidence-mode V3 method pass the frozen calibration "
            "gate on a second independently generated task suite at N=256?"
        ),
        "authorization": {
            "provider_calls": False,
            "new_model_responses": False,
            "exact_reference_enumeration": True,
            "new_secret_and_suite_required": True,
        },
        "method": {
            "mode_acquisition": (
                "rank predicate behavior classes by singleton keep/drop violations, "
                "rank mapper behavior classes by retained singleton value violations, "
                "cross the top eight of each, and retain eight program behavior modes"
            ),
            "predicate_syntaxes_evaluated": 600,
            "mapper_syntaxes_evaluated": 60,
            "maximum_complete_representatives_scored": COMPONENT_POOL_SIZE**2,
            "complete_program_support_ranked": False,
            "acquisition_before_reference_materialization": True,
            "semantic_modes": 8,
            "maximum_aliases_per_mode": 4,
            "grammar_mass": 0.25,
            "local_center_mass": 0.75,
            "predicate_slice_mass_within_mode": 0.125,
            "mapper_slice_mass_within_mode": 0.125,
            "bridge": "q(p)^(1-beta)*gamma(p)^beta",
            "conditional_ess_fraction": 0.8,
            "resample_relative_ess_threshold": 0.5,
            "maximum_stages": 64,
            "primary_mh_moves": 0,
        },
        "task_distribution": {
            "task_count": TASK_COUNT,
            "task_ids": list(TASK_IDS),
            "constants": list(range(-3, 5)),
            "generator": (
                "same independently normalized rejection-conditioned recursive grammar "
                "as V2, but under a new secret required to differ from V2"
            ),
            "duplicates": "allowed; no cross-task or outcome-dependent replacement",
            "examples": "empty, all singletons, domain list, two permutations, concatenation",
        },
        "design": {
            "arms": [arm.arm_id for arm in FROZEN_ARMS],
            "arm_particles": {
                arm.arm_id: list(ARM_PARTICLES[arm.arm_id]) for arm in FROZEN_ARMS
            },
            "particle_counts": list(PARTICLE_COUNTS),
            "primary_particle_count": PRIMARY_PARTICLE_COUNT,
            "repetitions": REPETITIONS,
            "repetition_seeds": list(REPETITION_SEEDS),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "task_weighting": "equal task weight",
            "expected_run_count": EXPECTED_RUN_COUNT,
            "expected_initial_proposal_draws": EXPECTED_INITIAL_PROPOSAL_DRAWS,
        },
        "primary_gate": {
            "all_required": True,
            "rmse": "task-first bootstrap 95% upper bound is strictly below 0.10",
            "bias": (
                "task-first bootstrap central 90% interval lies strictly inside "
                "(-0.03,+0.03)"
            ),
            "per_task": "every task N=256 exact-mass RMSE is below 0.20",
            "convergence": "point RMSE is nonincreasing from N=256 to 512 to 1024",
            "weight_tail": (
                "N=256 q95 maximum normalized weight below 0.05 and fraction above "
                "0.25 no greater than 0.01"
            ),
            "identity": "maximum finite-state importance identity error <=1e-12",
        },
        "failure_policy": {
            "source_or_seal_drift": "abort before task scoring or Monte Carlo draws",
            "existing_output": "refuse to overwrite",
            "task_replacement": "forbidden",
            "failed_gate": "publish as failure without changing thresholds or tasks",
        },
        "postrun_integrity": {
            "deterministic_public_replay_required_before_unblind": True,
            "external_hash_inputs": [
                "protocol",
                "method seal",
                "custody seal",
            ],
            "replay_scope": (
                "all 32 public banks, references, 10,240 seeded runs, analysis, "
                "and inventory bindings"
            ),
            "replay_private_reveal_read": False,
        },
        "bindings": {
            "harness": _project_record(project_root, HARNESS_PATH),
            "test": _project_record(project_root, TEST_PATH),
            "dependencies": [
                _project_record(project_root, relative) for relative in DEPENDENCY_PATHS
            ],
            "python_tree": _python_tree_record(project_root),
            "research_tree": _research_tree_record(project_root),
            "lockfiles": [
                _project_record(project_root, relative) for relative in LOCKFILE_PATHS
            ],
            "attempt_chain": {
                name: _workspace_record(project_root, relative)
                for name, relative in ATTEMPT_BINDINGS
            },
        },
        "runtime": _runtime_record(),
        "attempt_history": [
            "V2 fresh confirmation failed before V3 was designed.",
            "V3 factorized diagnostic reused V2 tasks and was not confirmatory.",
            "V3 fresh R1 was superseded before secret creation to align the bias-gate "
            "wording with the already strict validator.",
            "This V3 study uses a second secret and independently generated tasks.",
        ],
        "claim_boundary": [
            "Success concerns the declared finite grammar and synthetic task law only.",
            "Factorized primary and matched-q acquisition finish before exact reference "
            "enumeration; the descriptive sticky baseline is reference-derived.",
            "The primary method is provider-free and makes no LLM or large-DSL claim.",
            "V2's failed gate remains reported and is not replaced by this study.",
        ],
    }


def freeze_protocol(project_root: Path, protocol_path: Path) -> str:
    _write_json_exclusive(protocol_path, build_protocol(project_root))
    return _sha256_file(protocol_path)


def _validate_protocol(
    project_root: Path, protocol_path: Path, expected_protocol_sha256: str
) -> dict[str, Any]:
    if _sha256_file(protocol_path) != expected_protocol_sha256:
        raise FreshCalibrationError("V3 protocol SHA-256 differs")
    protocol = cast(dict[str, Any], json.loads(protocol_path.read_text()))
    if protocol != build_protocol(project_root):
        raise FreshCalibrationError("V3 protocol or bound sources differ")
    return protocol


def seal_method(
    project_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    output: Path,
) -> str:
    protocol = _validate_protocol(
        project_root, protocol_path, expected_protocol_sha256
    )
    record = {
        "schema": f"{STUDY_SCHEMA}-method-seal-v1",
        "sealed_before_secret_preparation": True,
        "protocol_sha256": expected_protocol_sha256,
        "method_sha256": _sha256_bytes(canonical_bytes(protocol["method"])),
    }
    _write_json_exclusive(output, record)
    return _sha256_file(output)


def _validate_method_seal(
    project_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
) -> None:
    _validate_protocol(project_root, protocol_path, expected_protocol_sha256)
    if _sha256_file(method_seal_path) != expected_method_seal_sha256:
        raise FreshCalibrationError("V3 method-seal SHA-256 differs")
    expected = {
        "schema": f"{STUDY_SCHEMA}-method-seal-v1",
        "sealed_before_secret_preparation": True,
        "protocol_sha256": expected_protocol_sha256,
        "method_sha256": _sha256_bytes(
            canonical_bytes(
                _validate_protocol(
                    project_root, protocol_path, expected_protocol_sha256
                )["method"]
            )
        ),
    }
    if json.loads(method_seal_path.read_text()) != expected:
        raise FreshCalibrationError("V3 method-seal content differs")


def _draw_hmac(secret: bytes, label: str) -> bytes:
    return hmac.new(secret, DRAW_DOMAIN + label.encode(), hashlib.sha256).digest()


def _uniform_index(secret: bytes, label: str, size: int) -> int:
    if size < 1:
        raise FreshCalibrationError("V3 uniform draw has empty support")
    modulus = 1 << 256
    limit = modulus - modulus % size
    counter = 0
    while True:
        value = int.from_bytes(_draw_hmac(secret, f"{label}\0{counter}"), "big")
        if value < limit:
            return value % size
        counter += 1


def _sample_atom(secret: bytes, label: str) -> str:
    constant = CONSTANTS[_uniform_index(secret, f"{label}\0constant", len(CONSTANTS))]
    template = ("lt(item,c)", "lt(c,item)", "eq(item,c)")[
        _uniform_index(secret, f"{label}\0template", 3)
    ]
    return template.replace("c", str(constant))


def _sample_predicate(secret: bytes, label: str) -> str:
    left = _sample_atom(secret, f"{label}\0left")
    if _uniform_index(secret, f"{label}\0form", 2) == 0:
        return left
    return f"and({left},{_sample_atom(secret, f'{label}\0right')})"


def _sample_mapper(secret: bytes, label: str) -> str:
    form = _uniform_index(secret, f"{label}\0form", 5)
    if form == 0:
        return "item"
    constant = CONSTANTS[_uniform_index(secret, f"{label}\0constant", len(CONSTANTS))]
    if form == 1:
        return str(constant)
    operator = ("add", "sub", "mul")[
        _uniform_index(secret, f"{label}\0operator", 3)
    ]
    if form == 2:
        return f"{operator}(item,item)"
    if form == 3:
        return f"{operator}(item,{constant})"
    return f"{operator}({constant},item)"


def _target_draw_v3(secret: bytes, task_id: str) -> dict[str, object]:
    predicates = _dsl_catalog(filter_predicates(CONSTANTS))
    mappers = _dsl_catalog(arithmetic_expressions("Item", CONSTANTS))
    rejected: list[dict[str, object]] = []
    for attempt in range(1, 1_000_001):
        label = f"{task_id}\0attempt-{attempt}"
        predicate_dsl = _sample_predicate(secret, f"{label}\0predicate")
        mapper_dsl = _sample_mapper(secret, f"{label}\0mapper")
        predicate = cast(Node, predicates[predicate_dsl])
        mapper = cast(Node, mappers[mapper_dsl])
        retained = [
            item
            for item in DOMAIN
            if cast(bool, evaluate_expression(predicate, [], item=item))
        ]
        outputs = [
            cast(int, evaluate_expression(mapper, [], item=item)) for item in retained
        ]
        reasons = []
        if not 3 <= len(retained) <= 5:
            reasons.append("retained-cardinality-outside-3-to-5")
        if len(set(outputs)) < 3:
            reasons.append("fewer-than-three-distinct-retained-outputs")
        if not reasons:
            return {
                "task_id": task_id,
                "predicate_dsl": predicate_dsl,
                "mapper_dsl": mapper_dsl,
                "predicate": predicate,
                "mapper": mapper,
                "accepted_attempt": attempt,
                "rejections": rejected,
            }
        rejected.append(
            {
                "attempt": attempt,
                "predicate_dsl": predicate_dsl,
                "mapper_dsl": mapper_dsl,
                "reasons": reasons,
            }
        )
    raise FreshCalibrationError("V3 rejection sampler exceeded its cap")


def _permutation(secret: bytes, label: str) -> tuple[int, ...]:
    decorated = [
        (_draw_hmac(secret, f"{label}\0{index}\0{item}"), index, item)
        for index, item in enumerate(DOMAIN)
    ]
    return tuple(item for _, _, item in sorted(decorated))


def _encoded(values: Sequence[int]) -> list[str]:
    return [str(value) for value in values]


def _task_document_v3(
    target: Mapping[str, object], secret: bytes
) -> dict[str, object]:
    task_id = cast(str, target["task_id"])
    first = _permutation(secret, f"{task_id}\0permutation-1")
    second = _permutation(secret, f"{task_id}\0permutation-2")
    inputs = (
        (),
        *((item,) for item in DOMAIN),
        DOMAIN,
        first,
        second,
        (*first, *second),
    )
    program = assemble_program(
        cast(Mapping[str, object], target["predicate"]),
        cast(Mapping[str, object], target["mapper"]),
    )
    examples = []
    for input_value in inputs:
        output = evaluate_program(cast(Node, program), list(input_value))
        if not isinstance(output, list):
            raise FreshCalibrationError("V3 target returned non-list output")
        examples.append({"input": _encoded(input_value), "output": _encoded(output)})
    return {
        "name": f"opaque fresh V3 calibration task {task_id}",
        "signature": {"input": "List<Int>", "output": "List<Int>"},
        "examples": examples,
        "integerConstants": _encoded(CONSTANTS),
        "particles": 8,
        "iterations": 4,
        "cloneProbability": 0,
        "essThreshold": 1,
        "seed": 17,
        "lossScale": 0.75,
        "costScale": 0.02,
        "lossCap": 1000,
        "maxCost": 30,
        "maxDepth": 12,
        "maxNodes": 191,
    }


def secret_commitment(secret: bytes) -> str:
    if len(secret) != 32:
        raise FreshCalibrationError("V3 secret must contain exactly 32 bytes")
    return _sha256_bytes(SECRET_DOMAIN + secret)


def _read_secret(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise FreshCalibrationError("V3 secret must be a regular non-symlink file")
    if path.stat().st_mode & 0o077:
        raise FreshCalibrationError("V3 secret permissions are too broad")
    secret = path.read_bytes()
    if len(secret) != 32:
        raise FreshCalibrationError("V3 secret must contain exactly 32 bytes")
    return secret


def prepare_secret(
    project_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
    output: Path,
) -> str:
    _validate_method_seal(
        project_root,
        protocol_path,
        expected_protocol_sha256,
        method_seal_path,
        expected_method_seal_sha256,
    )
    secret = secrets.token_bytes(32)
    commitment = secret_commitment(secret)
    if v2_secret_commitment(secret) == PRIOR_SECRET_COMMITMENT:
        raise FreshCalibrationError("V3 secret bytes reuse the V2 secret")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(secret)
    return commitment


def seal_custody(
    project_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
    secret_path: Path,
    output: Path,
) -> str:
    _validate_method_seal(
        project_root,
        protocol_path,
        expected_protocol_sha256,
        method_seal_path,
        expected_method_seal_sha256,
    )
    secret = _read_secret(secret_path)
    commitment = secret_commitment(secret)
    v2_domain_commitment = v2_secret_commitment(secret)
    if v2_domain_commitment == PRIOR_SECRET_COMMITMENT:
        raise FreshCalibrationError("V3 custody reused V2 secret")
    record = {
        "schema": f"{STUDY_SCHEMA}-custody-seal-v1",
        "sealed_before_task_generation": True,
        "protocol_sha256": expected_protocol_sha256,
        "method_seal_sha256": expected_method_seal_sha256,
        "secret_commitment_sha256": commitment,
        "secret_commitment_under_v2_domain_sha256": v2_domain_commitment,
        "prior_v2_secret_commitment_sha256": PRIOR_SECRET_COMMITMENT,
        "same_domain_secret_nonreuse_verified": True,
    }
    _write_json_exclusive(output, record)
    return _sha256_file(output)


def _validate_custody_seal(
    custody_seal_path: Path,
    expected_custody_seal_sha256: str,
    *,
    protocol_sha256: str,
    method_seal_sha256: str,
    secret: bytes | None = None,
) -> dict[str, Any]:
    if custody_seal_path.is_symlink() or not custody_seal_path.is_file():
        raise FreshCalibrationError("V3 custody seal is not a regular file")
    if _sha256_file(custody_seal_path) != expected_custody_seal_sha256:
        raise FreshCalibrationError("V3 custody-seal SHA-256 differs")
    record = cast(dict[str, Any], json.loads(custody_seal_path.read_text()))
    expected_fields = {
        "schema",
        "sealed_before_task_generation",
        "protocol_sha256",
        "method_seal_sha256",
        "secret_commitment_sha256",
        "secret_commitment_under_v2_domain_sha256",
        "prior_v2_secret_commitment_sha256",
        "same_domain_secret_nonreuse_verified",
    }
    if set(record) != expected_fields:
        raise FreshCalibrationError("V3 custody-seal fields differ")
    required = {
        "schema": f"{STUDY_SCHEMA}-custody-seal-v1",
        "sealed_before_task_generation": True,
        "protocol_sha256": protocol_sha256,
        "method_seal_sha256": method_seal_sha256,
        "prior_v2_secret_commitment_sha256": PRIOR_SECRET_COMMITMENT,
        "same_domain_secret_nonreuse_verified": True,
    }
    if any(record.get(key) != value for key, value in required.items()):
        raise FreshCalibrationError("V3 custody-seal content differs")
    v2_domain = record.get("secret_commitment_under_v2_domain_sha256")
    if v2_domain == PRIOR_SECRET_COMMITMENT:
        raise FreshCalibrationError("V3 custody seal records V2 secret reuse")
    if secret is not None:
        if record.get("secret_commitment_sha256") != secret_commitment(secret):
            raise FreshCalibrationError("V3 custody V3 commitment differs")
        if v2_domain != v2_secret_commitment(secret):
            raise FreshCalibrationError("V3 custody same-domain commitment differs")
    return record


def generate_suite(
    project_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
    custody_seal_path: Path,
    expected_custody_seal_sha256: str,
    secret_path: Path,
    output: Path,
) -> dict[str, object]:
    _validate_method_seal(
        project_root,
        protocol_path,
        expected_protocol_sha256,
        method_seal_path,
        expected_method_seal_sha256,
    )
    secret = _read_secret(secret_path)
    commitment = secret_commitment(secret)
    custody = _validate_custody_seal(
        custody_seal_path,
        expected_custody_seal_sha256,
        protocol_sha256=expected_protocol_sha256,
        method_seal_sha256=expected_method_seal_sha256,
        secret=secret,
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V3 suite: {output}")
    public = output / "public"
    private = output / "private"
    public.mkdir(parents=True)
    private.mkdir(mode=0o700)
    manifest_tasks = []
    reveal_tasks = []
    for task_id in TASK_IDS:
        target = _target_draw_v3(secret, task_id)
        document = _task_document_v3(target, secret)
        task_path = public / f"{task_id}.json"
        task_bytes = canonical_bytes(document) + b"\n"
        task_path.write_bytes(task_bytes)
        nonce = hmac.new(
            secret, TARGET_COMMITMENT_DOMAIN + task_id.encode(), hashlib.sha256
        ).hexdigest()
        preimage = {
            "task_id": task_id,
            "task_sha256": _sha256_bytes(task_bytes),
            "predicate_dsl": target["predicate_dsl"],
            "mapper_dsl": target["mapper_dsl"],
            "nonce": nonce,
        }
        target_commitment = _sha256_bytes(canonical_bytes(preimage))
        manifest_tasks.append(
            {
                "task_id": task_id,
                "path": task_path.name,
                "task_sha256": _sha256_bytes(task_bytes),
                "target_commitment_sha256": target_commitment,
            }
        )
        reveal_tasks.append(
            {
                "task_id": task_id,
                "predicate_dsl": target["predicate_dsl"],
                "mapper_dsl": target["mapper_dsl"],
                "accepted_attempt": target["accepted_attempt"],
                "rejections": target["rejections"],
                "commitment_preimage": preimage,
            }
        )
    manifest = {
        "schema": f"{STUDY_SCHEMA}-public-suite-v1",
        "protocol_sha256": expected_protocol_sha256,
        "method_seal_sha256": expected_method_seal_sha256,
        "custody_seal_sha256": expected_custody_seal_sha256,
        "secret_commitment_sha256": commitment,
        "secret_commitment_under_v2_domain_sha256": custody[
            "secret_commitment_under_v2_domain_sha256"
        ],
        "prior_v2_secret_commitment_sha256": PRIOR_SECRET_COMMITMENT,
        "same_domain_secret_nonreuse_verified": True,
        "task_count": TASK_COUNT,
        "task_ids": list(TASK_IDS),
        "tasks": manifest_tasks,
    }
    reveal = {
        "schema": f"{STUDY_SCHEMA}-private-reveal-v1",
        "secret_hex": secret.hex(),
        "manifest_sha256": _sha256_bytes(canonical_bytes(manifest) + b"\n"),
        "tasks": reveal_tasks,
    }
    _write_json_exclusive(public / "manifest.json", manifest)
    _write_json_exclusive(private / "reveal.json", reveal, mode=0o600)
    return manifest


def _validate_suite(
    suite: Path,
    protocol_sha256: str,
    method_seal_sha256: str,
    custody_seal_sha256: str,
    *,
    require_public_only: bool = True,
) -> tuple[list[TaskBinding], dict[str, Any]]:
    if suite.is_symlink() or not suite.is_dir():
        raise FreshCalibrationError("V3 suite root is invalid")
    root_entries = list(suite.iterdir())
    expected_root_names = {"public"} if require_public_only else {"public", "private"}
    if (
        {path.name for path in root_entries} != expected_root_names
        or any(path.is_symlink() for path in root_entries)
    ):
        if require_public_only:
            raise FreshCalibrationError(
                "V3 run suite must contain only the public directory"
            )
        raise FreshCalibrationError("V3 custody suite root differs")
    if not require_public_only:
        private = suite / "private"
        private_entries = list(private.iterdir()) if private.is_dir() else []
        if (
            len(private_entries) != 1
            or private_entries[0].name != "reveal.json"
            or private_entries[0].is_symlink()
            or not private_entries[0].is_file()
        ):
            raise FreshCalibrationError("V3 custody private file set differs")
    public = suite / "public"
    manifest_path = public / "manifest.json"
    if public.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise FreshCalibrationError("V3 public suite manifest is invalid")
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    expected_manifest_fields = {
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
    }
    if set(manifest) != expected_manifest_fields:
        raise FreshCalibrationError("V3 suite manifest fields differ")
    if manifest.get("schema") != f"{STUDY_SCHEMA}-public-suite-v1":
        raise FreshCalibrationError("V3 suite schema differs")
    if manifest.get("protocol_sha256") != protocol_sha256:
        raise FreshCalibrationError("V3 suite protocol binding differs")
    if manifest.get("method_seal_sha256") != method_seal_sha256:
        raise FreshCalibrationError("V3 suite method binding differs")
    if manifest.get("custody_seal_sha256") != custody_seal_sha256:
        raise FreshCalibrationError("V3 suite custody binding differs")
    if (
        manifest.get("task_count") != TASK_COUNT
        or manifest.get("task_ids") != list(TASK_IDS)
    ):
        raise FreshCalibrationError("V3 suite task design differs")
    _require_digest(
        manifest.get("secret_commitment_sha256"), name="suite V3 commitment"
    )
    v2_domain = _require_digest(
        manifest.get("secret_commitment_under_v2_domain_sha256"),
        name="suite V2-domain commitment",
    )
    if (
        manifest.get("prior_v2_secret_commitment_sha256")
        != PRIOR_SECRET_COMMITMENT
        or manifest.get("same_domain_secret_nonreuse_verified") is not True
        or v2_domain == PRIOR_SECRET_COMMITMENT
    ):
        raise FreshCalibrationError("V3 suite secret-nonreuse record differs")
    records = manifest.get("tasks")
    if not isinstance(records, list) or len(records) != TASK_COUNT:
        raise FreshCalibrationError("V3 suite task count differs")
    bindings = []
    seen_paths: set[str] = set()
    for task_id, record in zip(TASK_IDS, records, strict=True):
        if not isinstance(record, Mapping) or record.get("task_id") != task_id:
            raise FreshCalibrationError("V3 suite task order differs")
        if set(record) != {
            "task_id",
            "path",
            "task_sha256",
            "target_commitment_sha256",
        }:
            raise FreshCalibrationError("V3 suite task fields differ")
        relative = record.get("path")
        if relative != f"{task_id}.json" or relative in seen_paths:
            raise FreshCalibrationError("V3 suite task path differs or repeats")
        seen_paths.add(cast(str, relative))
        path = public / cast(str, relative)
        if path.resolve().parent != public.resolve():
            raise FreshCalibrationError("V3 suite task escapes the public root")
        digest = _sha256_file(path)
        if (
            path.is_symlink()
            or not path.is_file()
            or record.get("task_sha256") != digest
        ):
            raise FreshCalibrationError("V3 suite task bytes differ")
        _require_digest(record.get("target_commitment_sha256"), name="target commitment")
        bindings.append(TaskBinding(task_id, path, digest))
    allowed = {"manifest.json", *(f"{task_id}.json" for task_id in TASK_IDS)}
    public_entries = list(public.iterdir())
    actual = {path.name for path in public_entries}
    if (
        actual != allowed
        or any(path.is_symlink() or not path.is_file() for path in public_entries)
    ):
        raise FreshCalibrationError("V3 public suite file set differs")
    return bindings, manifest


def _summary_markdown(analysis: Mapping[str, object]) -> str:
    gate = cast(Mapping[str, object], analysis["primary_gate"])
    lines = [
        "# Factorized Calibrated Program Inference V3 Fresh Confirmation",
        "",
        f"Frozen primary gate: **{'PASS' if gate['pass'] else 'FAIL'}**.",
        "",
        "This is a second fresh suite after the V2 failure and V3 diagnostic.",
        "",
    ]
    return "\n".join(lines)


def run_study(
    project_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
    custody_seal_path: Path,
    expected_custody_seal_sha256: str,
    suite: Path,
    output: Path,
) -> dict[str, object]:
    protocol = _validate_protocol(project_root, protocol_path, expected_protocol_sha256)
    _validate_method_seal(
        project_root,
        protocol_path,
        expected_protocol_sha256,
        method_seal_path,
        expected_method_seal_sha256,
    )
    custody = _validate_custody_seal(
        custody_seal_path,
        expected_custody_seal_sha256,
        protocol_sha256=expected_protocol_sha256,
        method_seal_sha256=expected_method_seal_sha256,
    )
    bindings, manifest = _validate_suite(
        suite,
        expected_protocol_sha256,
        expected_method_seal_sha256,
        expected_custody_seal_sha256,
    )
    if (
        manifest.get("secret_commitment_sha256")
        != custody.get("secret_commitment_sha256")
        or manifest.get("secret_commitment_under_v2_domain_sha256")
        != custody.get("secret_commitment_under_v2_domain_sha256")
    ):
        raise FreshCalibrationError("V3 public suite commitment differs from custody")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V3 output: {output}")
    staging = output.with_name(f".{output.name}.incomplete")
    if staging.exists():
        raise FileExistsError(f"refusing to reuse V3 staging: {staging}")
    staging.mkdir(parents=True)
    try:
        _write_json_exclusive(staging / "protocol.json", protocol)
        _write_json_exclusive(staging / "public-suite-manifest.json", manifest)
        references: dict[str, Mapping[str, object]] = {}
        runs: list[Mapping[str, object]] = []
        public_banks = {
            binding.task_id: factorized_public_mode_bank(binding)
            for binding in bindings
        }
        for binding in bindings:
            task = TerminalTask(binding)
            bank = materialize_mode_bank(task, public_banks[binding.task_id])
            proposals: dict[str, Proposal] = {
                arm.arm_id: build_v3_proposal(task, arm, bank) for arm in FROZEN_ARMS
            }
            inherited_reference = _reference_record(task, proposals, bank.metadata)
            reference = {
                **inherited_reference,
                "schema": REFERENCE_SCHEMA,
                "provider_calls": 0,
                "mode_bank": inherited_reference["sampled_mode_bank"],
            }
            del reference["sampled_mode_bank"]
            references[binding.task_id] = reference
            _write_json_exclusive(
                staging / "references" / f"{binding.task_id}.json", reference
            )
            for arm in FROZEN_ARMS:
                for particles in ARM_PARTICLES[arm.arm_id]:
                    for repetition, base_seed in enumerate(REPETITION_SEEDS):
                        runs.append(
                            run_v3_repetition(
                                task,
                                proposals[arm.arm_id],
                                particles=particles,
                                repetition=repetition,
                                base_seed=base_seed,
                            )
                        )
        if len(runs) != EXPECTED_RUN_COUNT:
            raise FreshCalibrationError("V3 run ledger count differs")
        if sum(int(run["logical_proposal_draws"]) for run in runs) != (
            EXPECTED_INITIAL_PROPOSAL_DRAWS
        ):
            raise FreshCalibrationError("V3 logical proposal-draw count differs")
        analysis = analyze(runs, references)
        _write_json_exclusive(staging / "runs.json", runs)
        _write_json_exclusive(staging / "analysis.json", analysis)
        (staging / "SUMMARY.md").write_text(
            _summary_markdown(analysis), encoding="utf-8"
        )
        _write_json_exclusive(
            staging / "study-metadata.json",
            {
                "schema": f"{STUDY_SCHEMA}-metadata-v1",
                "protocol_sha256": expected_protocol_sha256,
                "method_seal_sha256": expected_method_seal_sha256,
                "custody_seal_sha256": expected_custody_seal_sha256,
                "public_manifest_sha256": _sha256_file(
                    staging / "public-suite-manifest.json"
                ),
                "provider_calls": 0,
                "private_reveal_read": False,
                "exact_support_materialized_after_mode_bank": True,
                "runtime": _runtime_record(),
            },
        )
        seal_inventory(staging)
        validate_artifact(staging)
        staging.rename(output)
        if validate_artifact(output) != analysis:
            raise FreshCalibrationError("V3 final validation recomputation differs")
        return analysis
    except Exception as error:
        _write_json_exclusive(
            staging / "FAILURE.json",
            {
                "schema": f"{STUDY_SCHEMA}-failure-v1",
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _validate_replay_receipt(
    artifact: Path,
    receipt_path: Path,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Validate the mandatory reveal-free replay receipt before unblinding."""

    expected_digest = _require_digest(
        expected_receipt_sha256, name="replay-receipt SHA-256"
    )
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise FreshCalibrationError("V3 replay receipt is not a regular file")
    if _sha256_file(receipt_path) != expected_digest:
        raise FreshCalibrationError("V3 replay-receipt SHA-256 differs")
    receipt = cast(dict[str, Any], json.loads(receipt_path.read_text()))
    if receipt_path.read_bytes() != canonical_bytes(receipt) + b"\n":
        raise FreshCalibrationError("V3 replay receipt is not canonical JSON")
    if set(receipt) != {
        "schema",
        "status",
        "bindings",
        "replay_counts",
        "provider_calls",
        "private_reveal_read",
        "all_public_banks_acquired_before_reference_materialization",
        "artifact_references_runs_analysis_byte_identical",
    }:
        raise FreshCalibrationError("V3 replay receipt fields differ")
    if (
        receipt.get("schema") != REPLAY_SCHEMA
        or receipt.get("status") != "deterministic-public-replay-passed"
        or receipt.get("provider_calls") != 0
        or isinstance(receipt.get("provider_calls"), bool)
        or receipt.get("private_reveal_read") is not False
        or receipt.get(
            "all_public_banks_acquired_before_reference_materialization"
        )
        is not True
        or receipt.get("artifact_references_runs_analysis_byte_identical") is not True
    ):
        raise FreshCalibrationError("V3 replay receipt status differs")
    manifest = cast(
        dict[str, Any],
        json.loads((artifact / "public-suite-manifest.json").read_text()),
    )
    references = [
        {
            "task_id": task_id,
            "sha256": _sha256_file(artifact / "references" / f"{task_id}.json"),
        }
        for task_id in TASK_IDS
    ]
    expected_bindings = {
        "protocol_sha256": _sha256_file(artifact / "protocol.json"),
        "method_seal_sha256": manifest.get("method_seal_sha256"),
        "custody_seal_sha256": manifest.get("custody_seal_sha256"),
        "public_manifest_sha256": _sha256_file(
            artifact / "public-suite-manifest.json"
        ),
        "runs_sha256": _sha256_file(artifact / "runs.json"),
        "analysis_sha256": _sha256_file(artifact / "analysis.json"),
        "inventory_sha256": _sha256_file(artifact / "inventory.json"),
        "reference_set_sha256": _sha256_bytes(canonical_bytes(references)),
    }
    if receipt.get("bindings") != expected_bindings:
        raise FreshCalibrationError("V3 replay receipt artifact binding differs")
    expected_counts = {
        "tasks": TASK_COUNT,
        "public_banks": TASK_COUNT,
        "references": TASK_COUNT,
        "runs": EXPECTED_RUN_COUNT,
        "logical_proposal_draws": EXPECTED_INITIAL_PROPOSAL_DRAWS,
    }
    if receipt.get("replay_counts") != expected_counts:
        raise FreshCalibrationError("V3 replay receipt counts differ")
    return receipt


def verify_unblind(
    suite: Path,
    artifact: Path,
    secret_path: Path,
    replay_receipt_path: Path,
    expected_replay_receipt_sha256: str,
    output: Path,
) -> dict[str, object]:
    analysis = validate_artifact(artifact)
    _validate_replay_receipt(
        artifact,
        replay_receipt_path,
        expected_replay_receipt_sha256,
    )
    secret = _read_secret(secret_path)
    manifest_path = suite / "public/manifest.json"
    reveal_path = suite / "private/reveal.json"
    if (
        manifest_path.is_symlink()
        or reveal_path.is_symlink()
        or not manifest_path.is_file()
        or not reveal_path.is_file()
    ):
        raise FreshCalibrationError("V3 unblind inputs are not regular files")
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    reveal = cast(dict[str, Any], json.loads(reveal_path.read_text()))
    if set(reveal) != {"schema", "secret_hex", "manifest_sha256", "tasks"}:
        raise FreshCalibrationError("V3 reveal fields differ")
    if reveal.get("schema") != f"{STUDY_SCHEMA}-private-reveal-v1":
        raise FreshCalibrationError("V3 reveal schema differs")
    artifact_manifest_path = artifact / "public-suite-manifest.json"
    if manifest_path.read_bytes() != artifact_manifest_path.read_bytes():
        raise FreshCalibrationError("V3 artifact public manifest differs from custody")
    if reveal.get("secret_hex") != secret.hex():
        raise FreshCalibrationError("V3 reveal secret differs")
    if manifest.get("secret_commitment_sha256") != secret_commitment(secret):
        raise FreshCalibrationError("V3 reveal commitment differs")
    if (
        manifest.get("secret_commitment_under_v2_domain_sha256")
        != v2_secret_commitment(secret)
        or v2_secret_commitment(secret) == PRIOR_SECRET_COMMITMENT
    ):
        raise FreshCalibrationError("V3 reveal same-domain nonreuse check failed")
    manifest_sha256 = _sha256_file(manifest_path)
    if reveal.get("manifest_sha256") != manifest_sha256:
        raise FreshCalibrationError("V3 reveal manifest binding differs")
    reveal_records = reveal.get("tasks")
    manifest_records = manifest.get("tasks")
    if (
        not isinstance(reveal_records, list)
        or not isinstance(manifest_records, list)
        or len(reveal_records) != TASK_COUNT
        or len(manifest_records) != TASK_COUNT
    ):
        raise FreshCalibrationError("V3 reveal records are malformed")
    for task_id, reveal_record, manifest_record in zip(
        TASK_IDS, reveal_records, manifest_records, strict=True
    ):
        if not isinstance(reveal_record, Mapping) or not isinstance(
            manifest_record, Mapping
        ):
            raise FreshCalibrationError("V3 reveal task record is malformed")
        if set(reveal_record) != {
            "task_id",
            "predicate_dsl",
            "mapper_dsl",
            "accepted_attempt",
            "rejections",
            "commitment_preimage",
        }:
            raise FreshCalibrationError("V3 reveal task fields differ")
        target = _target_draw_v3(secret, task_id)
        task_bytes = canonical_bytes(_task_document_v3(target, secret)) + b"\n"
        task_path = suite / "public" / f"{task_id}.json"
        if (
            task_path.read_bytes() != task_bytes
            or _sha256_bytes(task_bytes) != manifest_record["task_sha256"]
        ):
            raise FreshCalibrationError("V3 regenerated task differs")
        nonce = hmac.new(
            secret, TARGET_COMMITMENT_DOMAIN + task_id.encode(), hashlib.sha256
        ).hexdigest()
        expected_preimage = {
            "task_id": task_id,
            "task_sha256": _sha256_bytes(task_bytes),
            "predicate_dsl": target["predicate_dsl"],
            "mapper_dsl": target["mapper_dsl"],
            "nonce": nonce,
        }
        if (
            reveal_record.get("task_id") != task_id
            or reveal_record.get("predicate_dsl") != target["predicate_dsl"]
            or reveal_record.get("mapper_dsl") != target["mapper_dsl"]
            or reveal_record.get("accepted_attempt") != target["accepted_attempt"]
            or reveal_record.get("rejections") != target["rejections"]
            or reveal_record.get("commitment_preimage") != expected_preimage
        ):
            raise FreshCalibrationError("V3 revealed target regeneration differs")
        preimage = reveal_record["commitment_preimage"]
        if _sha256_bytes(canonical_bytes(preimage)) != manifest_record[
            "target_commitment_sha256"
        ]:
            raise FreshCalibrationError("V3 target commitment differs")
    artifact_protocol = artifact / "protocol.json"
    result = {
        "schema": f"{STUDY_SCHEMA}-unblind-verification-v1",
        "passed": True,
        "task_count": TASK_COUNT,
        "secret_commitment_sha256": secret_commitment(secret),
        "secret_commitment_under_v2_domain_sha256": v2_secret_commitment(secret),
        "protocol_sha256": _sha256_file(artifact_protocol),
        "method_seal_sha256": manifest["method_seal_sha256"],
        "custody_seal_sha256": manifest["custody_seal_sha256"],
        "public_manifest_sha256": manifest_sha256,
        "private_reveal_sha256": _sha256_file(reveal_path),
        "analysis_sha256": _sha256_file(artifact / "analysis.json"),
        "inventory_sha256": _sha256_file(artifact / "inventory.json"),
        "replay_receipt_sha256": expected_replay_receipt_sha256,
        "primary_gate_pass": cast(Mapping[str, object], analysis["primary_gate"])[
            "pass"
        ],
    }
    _write_json_exclusive(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--project-root", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    seal = subparsers.add_parser("seal-method")
    seal.add_argument("--project-root", type=Path, required=True)
    seal.add_argument("--protocol", type=Path, required=True)
    seal.add_argument("--expected-protocol-sha256", required=True)
    seal.add_argument("--output", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-secret")
    prepare.add_argument("--project-root", type=Path, required=True)
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--expected-protocol-sha256", required=True)
    prepare.add_argument("--method-seal", type=Path, required=True)
    prepare.add_argument("--expected-method-seal-sha256", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    custody = subparsers.add_parser("seal-custody")
    custody.add_argument("--project-root", type=Path, required=True)
    custody.add_argument("--protocol", type=Path, required=True)
    custody.add_argument("--expected-protocol-sha256", required=True)
    custody.add_argument("--method-seal", type=Path, required=True)
    custody.add_argument("--expected-method-seal-sha256", required=True)
    custody.add_argument("--secret", type=Path, required=True)
    custody.add_argument("--output", type=Path, required=True)
    generate = subparsers.add_parser("generate-suite")
    run = subparsers.add_parser("run")
    for command in (generate, run):
        command.add_argument("--project-root", type=Path, required=True)
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--expected-protocol-sha256", required=True)
        command.add_argument("--method-seal", type=Path, required=True)
        command.add_argument("--expected-method-seal-sha256", required=True)
        command.add_argument("--custody-seal", type=Path, required=True)
        command.add_argument("--expected-custody-seal-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
    generate.add_argument("--secret", type=Path, required=True)
    run.add_argument("--suite", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-unblind")
    verify.add_argument("--suite", type=Path, required=True)
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--secret", type=Path, required=True)
    verify.add_argument("--replay-receipt", type=Path, required=True)
    verify.add_argument("--expected-replay-receipt-sha256", required=True)
    verify.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        print(freeze_protocol(args.project_root.resolve(), args.protocol.resolve()))
    elif args.command == "seal-method":
        print(
            seal_method(
                args.project_root.resolve(),
                args.protocol.resolve(),
                args.expected_protocol_sha256,
                args.output.resolve(),
            )
        )
    elif args.command == "prepare-secret":
        print(
            prepare_secret(
                args.project_root.resolve(),
                args.protocol.resolve(),
                args.expected_protocol_sha256,
                args.method_seal.resolve(),
                args.expected_method_seal_sha256,
                args.output.resolve(),
            )
        )
    elif args.command == "seal-custody":
        print(
            seal_custody(
                args.project_root.resolve(),
                args.protocol.resolve(),
                args.expected_protocol_sha256,
                args.method_seal.resolve(),
                args.expected_method_seal_sha256,
                args.secret.resolve(),
                args.output.resolve(),
            )
        )
    elif args.command == "generate-suite":
        manifest = generate_suite(
            args.project_root.resolve(),
            args.protocol.resolve(),
            args.expected_protocol_sha256,
            args.method_seal.resolve(),
            args.expected_method_seal_sha256,
            args.custody_seal.resolve(),
            args.expected_custody_seal_sha256,
            args.secret.resolve(),
            args.output.resolve(),
        )
        print(_sha256_bytes(canonical_bytes(manifest) + b"\n"))
    elif args.command == "run":
        result = run_study(
            args.project_root.resolve(),
            args.protocol.resolve(),
            args.expected_protocol_sha256,
            args.method_seal.resolve(),
            args.expected_method_seal_sha256,
            args.custody_seal.resolve(),
            args.expected_custody_seal_sha256,
            args.suite.resolve(),
            args.output.resolve(),
        )
        print(json.dumps(result["primary_gate"], sort_keys=True))
    elif args.command == "validate":
        print(json.dumps(validate_artifact(args.output.resolve()), sort_keys=True))
    elif args.command == "verify-unblind":
        print(
            json.dumps(
                verify_unblind(
                    args.suite.resolve(),
                    args.artifact.resolve(),
                    args.secret.resolve(),
                    args.replay_receipt.resolve(),
                    args.expected_replay_receipt_sha256,
                    args.output.resolve(),
                ),
                sort_keys=True,
            )
        )
    else:
        raise AssertionError("unreachable command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Provider-free developmental screen for calibrated program inference V2.

This study is deliberately separate from the frozen particle-calibration V1
and terminal-diagnostic V2 artifacts.  It uses the four existing enumerable
36,000-program tasks to compare exactly evaluable proposal mechanisms before
any fresh tasks or new provider responses are acquired.

The scalable candidate constructs a small bank of semantically distinct modes
from a fixed grammar sample.  It then forms a normalized global mixture from
the grammar prior and local predicate/mapper slices around those modes.  The
same mixture is used both for sampling and for probability evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
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

from modelsmc_pbe.core.evaluate import evaluate_expression
from modelsmc_pbe.core.types import Node
from research.particle_calibration_terminal_diagnostic_v2 import (
    TASK_SPECS,
    TaskBinding,
    TerminalTask,
)

STUDY_SCHEMA = "provider-free-calibrated-program-inference-v2-screen"
RUN_SCHEMA = f"{STUDY_SCHEMA}-run-v1"
REFERENCE_SCHEMA = f"{STUDY_SCHEMA}-reference-v1"
ANALYSIS_SCHEMA = f"{STUDY_SCHEMA}-analysis-v1"
STATUS = "frozen-provider-free-developmental-before-draws"

HARNESS_PATH = "research/calibrated_program_inference_v2_screen.py"
TEST_PATH = "research/tests/test_calibrated_program_inference_v2_screen.py"
DEFAULT_PROTOCOL = "research/protocol-calibrated-program-inference-v2-screen.json"
DEFAULT_OUTPUT = "artifacts/calibrated-program-inference-v2-screen"

PARTICLE_COUNTS = (128, 256, 512, 1024)
REPETITIONS = 128
REPETITION_SEEDS = tuple(range(913_001, 913_001 + REPETITIONS))
DISCOVERY_BUDGET = 4096
MAX_MODE_COUNT = 16
MAX_ALIASES_PER_MODE = 4
LOCAL_CENTER_MASS = 0.25
ANNEAL_CONDITIONAL_ESS = 0.80
RESAMPLE_RELATIVE_ESS = 0.50
MAX_ANNEAL_STAGES = 64
IDENTITY_TOLERANCE = 1e-12

DEPENDENCY_PATHS = (
    "research/evidence_shortlist_smc.py",
    "research/execution_guided_repair.py",
    "research/iterative_beam_experiment.py",
    "research/particle_calibration_terminal_diagnostic_v2.py",
)
LOCKFILE_PATHS = ("pyproject.toml", "uv.lock")


class ScreenError(ValueError):
    """A frozen developmental-screen invariant was violated."""


@dataclass(frozen=True, slots=True)
class ArmSpec:
    arm_id: str
    bank_kind: str
    mode_count: int
    grammar_mass: float
    center_mass: float
    algorithm: str
    mh_moves: int
    role: str


ARMS = (
    ArmSpec(
        "sticky-terminal-is",
        "sticky-exact-parent",
        1,
        0.05,
        1.0,
        "terminal-is",
        0,
        "failed-v1-terminal-baseline",
    ),
    ArmSpec(
        "grammar-terminal-is",
        "grammar-only",
        0,
        1.0,
        1.0,
        "terminal-is",
        0,
        "global-prior-baseline",
    ),
    ArmSpec(
        "sampled-modes-k8-a025-terminal-is",
        "sampled-semantic-modes",
        8,
        0.25,
        0.25,
        "terminal-is",
        0,
        "scalable-mode-mixture",
    ),
    ArmSpec(
        "sampled-modes-k16-a025-terminal-is",
        "sampled-semantic-modes",
        16,
        0.25,
        0.25,
        "terminal-is",
        0,
        "scalable-mode-mixture",
    ),
    ArmSpec(
        "sampled-modes-k16-a050-terminal-is",
        "sampled-semantic-modes",
        16,
        0.50,
        0.25,
        "terminal-is",
        0,
        "grammar-mass-ablation",
    ),
    ArmSpec(
        "sampled-modes-k16-a025-annealed",
        "sampled-semantic-modes",
        16,
        0.25,
        0.25,
        "annealed-smc",
        0,
        "annealing-ablation",
    ),
    ArmSpec(
        "sampled-modes-k16-a025-annealed-mh",
        "sampled-semantic-modes",
        16,
        0.25,
        0.25,
        "annealed-smc",
        1,
        "full-v2-candidate",
    ),
    ArmSpec(
        "exhaustive-top64-a050-terminal-positive-control",
        "exhaustive-top64-point-bank",
        64,
        0.50,
        1.0,
        "terminal-is",
        0,
        "exhaustive-developmental-positive-control",
    ),
    ArmSpec(
        "sampled-modes-k8-a025-e075-terminal-is",
        "sampled-semantic-modes",
        8,
        0.25,
        0.75,
        "terminal-is",
        0,
        "center-mass-ablation",
    ),
    ArmSpec(
        "sampled-modes-k16-a025-e075-terminal-is",
        "sampled-semantic-modes",
        16,
        0.25,
        0.75,
        "terminal-is",
        0,
        "center-mass-ablation",
    ),
    ArmSpec(
        "sampled-modes-k16-a025-e075-annealed-mh",
        "sampled-semantic-modes",
        16,
        0.25,
        0.75,
        "annealed-smc",
        1,
        "high-center-full-v2-candidate",
    ),
    ArmSpec(
        "sampled-modes-k8-a025-e075-proposal-bridge",
        "sampled-semantic-modes",
        8,
        0.25,
        0.75,
        "proposal-bridge-smc",
        0,
        "proposal-bridge-ablation",
    ),
    ArmSpec(
        "sampled-modes-k8-a025-e075-proposal-bridge-mh",
        "sampled-semantic-modes",
        8,
        0.25,
        0.75,
        "proposal-bridge-smc",
        1,
        "proposal-bridge-v2-candidate",
    ),
    ArmSpec(
        "sampled-modes-k16-a025-e075-proposal-bridge-mh",
        "sampled-semantic-modes",
        16,
        0.25,
        0.75,
        "proposal-bridge-smc",
        1,
        "proposal-bridge-mode-count-ablation",
    ),
)


@dataclass(frozen=True, slots=True)
class Proposal:
    arm: ArmSpec
    probabilities: np.ndarray
    cdf: np.ndarray
    bank_indices: tuple[int, ...]
    semantic_mode_count: int
    census: dict[str, object]


@dataclass(frozen=True, slots=True)
class ModeBank:
    groups: tuple[tuple[int, ...], ...]
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class StudyPlan:
    tasks: tuple[TaskBinding, ...]
    arms: tuple[ArmSpec, ...]
    particle_counts: tuple[int, ...]
    repetition_seeds: tuple[int, ...]
    protocol_sha256: str


def canonical_bytes(value: object) -> bytes:
    """Return canonical strict-JSON bytes."""

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


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScreenError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ScreenError(f"{name} must be finite")
    return result


def _seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in (STUDY_SCHEMA, *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _sample_categorical(
    cdf: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count < 1 or cdf.ndim != 1 or cdf.size < 1:
        raise ScreenError("categorical draw has an invalid shape or count")
    draws = np.searchsorted(cdf, rng.random(count), side="right")
    return np.minimum(draws, cdf.size - 1).astype(np.int64, copy=False)


def _cdf(probabilities: np.ndarray) -> np.ndarray:
    result = np.cumsum(probabilities, dtype=np.float64)
    result[-1] = 1.0
    return result


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ScreenError("weights have an invalid normalizer")
    result = values / total
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ScreenError("normalized weights are invalid")
    return result


def _ess(weights: np.ndarray) -> float:
    return 1.0 / float(np.square(weights).sum())


def _systematic_resample(
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    count = weights.size
    positions = (rng.random() + np.arange(count, dtype=np.float64)) / count
    cumulative = np.cumsum(weights, dtype=np.float64)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right").astype(np.int64)


def _semantic_signature(task: TerminalTask, index: int) -> tuple[int | None, ...]:
    key = task.keys[index]
    predicate = cast(Node, task.predicate_catalog[key.predicate])
    mapper_values = task.mapper_signatures[key.mapper]
    result: list[int | None] = []
    for position, item in enumerate(task.observed_items):
        keep = evaluate_expression(
            predicate,
            list(task.observed_items),
            item=item,
        )
        if not isinstance(keep, bool):
            raise ScreenError("predicate returned a non-boolean semantic value")
        result.append(mapper_values[position] if keep else None)
    return tuple(result)


def sampled_semantic_mode_bank(
    task: TerminalTask,
    *,
    mode_count: int = MAX_MODE_COUNT,
    aliases_per_mode: int = MAX_ALIASES_PER_MODE,
    discovery_budget: int = DISCOVERY_BUDGET,
    seed: int | None = None,
) -> ModeBank:
    """Select public-score-ranked semantic modes from a fixed grammar sample."""

    if mode_count < 1 or aliases_per_mode < 1 or discovery_budget < mode_count:
        raise ScreenError("mode-bank count or discovery budget is invalid")
    actual_seed = seed if seed is not None else _seed(task.binding.sha256, "mode-bank")
    rng = np.random.Generator(np.random.PCG64(actual_seed))
    sampled = _sample_categorical(_cdf(task.grammar), discovery_budget, rng)
    unique = tuple(int(index) for index in np.unique(sampled))
    ranked = sorted(
        unique,
        key=lambda index: (
            float(task.losses[index]),
            int(task.costs[index]),
            task.keys[index].predicate,
            task.keys[index].mapper,
        ),
    )
    ordered_signatures: list[tuple[int | None, ...]] = []
    grouped: dict[tuple[int | None, ...], list[int]] = {}
    for index in ranked:
        signature = _semantic_signature(task, index)
        if signature not in grouped:
            grouped[signature] = []
            ordered_signatures.append(signature)
        if len(grouped[signature]) < aliases_per_mode:
            grouped[signature].append(index)
    selected_signatures = ordered_signatures[:mode_count]
    mapper_count = len(task.mapper_order)
    predicate_count = len(task.predicate_order)
    predicate_offsets = np.arange(predicate_count, dtype=np.int64) * mapper_count
    alias_candidates: set[int] = set()
    expanded_groups: list[tuple[int, ...]] = []
    for signature in selected_signatures:
        representative = grouped[signature][0]
        predicate_index, mapper_index = divmod(representative, mapper_count)
        predicate_start = predicate_index * mapper_count
        candidates = set(grouped[signature])
        candidates.update(range(predicate_start, predicate_start + mapper_count))
        candidates.update(int(offset + mapper_index) for offset in predicate_offsets)
        alias_candidates.update(candidates)
        matching = [
            index for index in candidates if _semantic_signature(task, index) == signature
        ]
        matching.sort(
            key=lambda index: (
                float(task.losses[index]),
                int(task.costs[index]),
                task.keys[index].predicate,
                task.keys[index].mapper,
            )
        )
        expanded_groups.append(tuple(matching[:aliases_per_mode]))
    groups = tuple(expanded_groups)
    if len(groups) != mode_count or any(not group for group in groups):
        raise ScreenError("grammar discovery sample did not contain enough semantic modes")
    flattened = tuple(index for group in groups for index in group)
    metadata: dict[str, object] = {
        "selection": (
            "sample from normalized grammar, deduplicate program syntax, rank by public "
            "(loss,cost,predicate,mapper), retain first distinct finite-domain behaviors "
            "with a bounded syntax-alias cloud per behavior"
        ),
        "seed": actual_seed,
        "logical_discovery_draws": discovery_budget,
        "logical_local_alias_candidates": len(alias_candidates),
        "logical_total_bank_construction_candidates": (
            discovery_budget + len(alias_candidates)
        ),
        "unique_sampled_programs": len(unique),
        "retained_semantic_modes": len(groups),
        "maximum_aliases_per_mode": aliases_per_mode,
        "retained_syntax_centers": len(flattened),
        "aliases_per_retained_mode": [len(group) for group in groups],
        "best_bank_loss": float(task.losses[list(flattened)].min()),
        "worst_bank_loss": float(task.losses[list(flattened)].max()),
        "exact_programs_in_bank": int(task.exact[list(flattened)].sum()),
    }
    return ModeBank(groups=groups, metadata=metadata)


def _mode_slice_distribution(
    task: TerminalTask,
    bank_groups: Sequence[Sequence[int]],
    *,
    grammar_mass: float,
    center_mass: float = LOCAL_CENTER_MASS,
) -> np.ndarray:
    if not bank_groups or any(not group for group in bank_groups):
        raise ScreenError("mode-slice mixture requires at least one bank mode")
    if not 0.0 < grammar_mass < 1.0 or not 0.0 <= center_mass <= 1.0:
        raise ScreenError("mode-slice mixture masses are invalid")
    predicate_count = len(task.predicate_order)
    mapper_count = len(task.mapper_order)
    predicate_probability = np.asarray(
        [float(task.predicate_probability[value]) for value in task.predicate_order],
        dtype=np.float64,
    )
    mapper_probability = np.asarray(
        [float(task.mapper_probability[value]) for value in task.mapper_order],
        dtype=np.float64,
    )
    if predicate_count * mapper_count != len(task.keys):
        raise ScreenError("program support does not match predicate/mapper product")
    probabilities = grammar_mass * task.grammar.copy()
    per_mode = (1.0 - grammar_mass) / len(bank_groups)
    slice_mass = (1.0 - center_mass) / 2.0
    predicate_offsets = np.arange(predicate_count, dtype=np.int64) * mapper_count
    for group in bank_groups:
        per_alias = per_mode / len(group)
        for index in group:
            predicate_index, mapper_index = divmod(int(index), mapper_count)
            probabilities[index] += per_alias * center_mass
            predicate_start = predicate_index * mapper_count
            probabilities[predicate_start : predicate_start + mapper_count] += (
                per_alias * slice_mass * mapper_probability
            )
            probabilities[predicate_offsets + mapper_index] += (
                per_alias * slice_mass * predicate_probability
            )
    return _normalize(probabilities)


def build_proposal(
    task: TerminalTask,
    arm: ArmSpec,
    sampled_bank: ModeBank,
) -> Proposal:
    """Construct one normalized proposal and its exact finite-state census."""

    if arm.bank_kind == "sticky-exact-parent":
        bank = (task.fixed_parent_index,)
        semantic_mode_count = 1
        probabilities = arm.grammar_mass * task.grammar.copy()
        probabilities[task.fixed_parent_index] += 1.0 - arm.grammar_mass
    elif arm.bank_kind == "grammar-only":
        bank = ()
        semantic_mode_count = 0
        probabilities = task.grammar.copy()
    elif arm.bank_kind == "sampled-semantic-modes":
        groups = sampled_bank.groups[: arm.mode_count]
        if len(groups) != arm.mode_count:
            raise ScreenError(f"{arm.arm_id} mode bank is incomplete")
        bank = tuple(index for group in groups for index in group)
        semantic_mode_count = len(groups)
        probabilities = _mode_slice_distribution(
            task,
            groups,
            grammar_mass=arm.grammar_mass,
            center_mass=arm.center_mass,
        )
    elif arm.bank_kind == "exhaustive-top64-point-bank":
        bank = tuple(task.global_indices)
        semantic_mode_count = len({_semantic_signature(task, index) for index in bank})
        probabilities = arm.grammar_mass * task.grammar.copy()
        probabilities[list(bank)] += (1.0 - arm.grammar_mass) / len(bank)
    else:
        raise ScreenError(f"unsupported bank kind: {arm.bank_kind}")
    probabilities = _normalize(probabilities)
    if np.any(probabilities <= 0.0):
        raise ScreenError(f"{arm.arm_id} proposal does not have full support")
    ratios = task.target / probabilities
    identity_weight = float(probabilities @ ratios)
    identity_exact = float(probabilities @ (ratios * task.exact))
    identity_loss = float(probabilities @ (ratios * task.losses))
    second_moment = float(probabilities @ np.square(ratios))
    bank_mask = np.zeros(len(task.keys), dtype=bool)
    if bank:
        bank_mask[list(bank)] = True
    census: dict[str, object] = {
        "proposal_probability_sum": float(probabilities.sum()),
        "minimum_proposal_probability": float(probabilities.min()),
        "maximum_proposal_probability": float(probabilities.max()),
        "bank_size": len(bank),
        "semantic_mode_count": semantic_mode_count,
        "bank_programs": [asdict(task.keys[index]) for index in bank],
        "bank_exact_programs": int(task.exact[list(bank)].sum()) if bank else 0,
        "target_mass_on_bank_syntaxes": (
            float(task.target[bank_mask].sum()) if bank else 0.0
        ),
        "proposal_exact_probability": float(probabilities @ task.exact),
        "importance_identity_weight_mean": identity_weight,
        "importance_identity_exact_numerator": identity_exact,
        "importance_identity_loss_numerator": identity_loss,
        "importance_identity_maximum_absolute_error": max(
            abs(identity_weight - 1.0),
            abs(identity_exact - task.exact_mass),
            abs(identity_loss - task.target_mean_loss),
        ),
        "population_importance_ess_fraction": 1.0 / second_moment,
        "chi_square_target_to_proposal": second_moment - 1.0,
        "maximum_normalized_importance_ratio": float(ratios.max()),
    }
    if census["importance_identity_maximum_absolute_error"] > IDENTITY_TOLERANCE:
        raise ScreenError(f"{arm.arm_id} exact importance identity failed")
    return Proposal(
        arm=arm,
        probabilities=probabilities,
        cdf=_cdf(probabilities),
        bank_indices=bank,
        semantic_mode_count=semantic_mode_count,
        census=census,
    )


def _conditional_ess(
    weights: np.ndarray,
    selected_log_likelihood: np.ndarray,
    delta: float,
) -> float:
    log_increment = delta * selected_log_likelihood
    log_increment -= float(log_increment.max())
    increment = np.exp(log_increment)
    numerator = float(weights @ increment) ** 2
    denominator = float(weights @ np.square(increment))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ScreenError("conditional ESS denominator is invalid")
    return weights.size * numerator / denominator


def next_beta(
    weights: np.ndarray,
    selected_log_likelihood: np.ndarray,
    beta: float,
    *,
    target_fraction: float = ANNEAL_CONDITIONAL_ESS,
) -> float:
    """Choose the largest next temperature satisfying a conditional-ESS target."""

    if not 0.0 <= beta < 1.0 or not 0.0 < target_fraction < 1.0:
        raise ScreenError("annealing beta or target fraction is invalid")
    target = target_fraction * weights.size
    if _conditional_ess(weights, selected_log_likelihood, 1.0 - beta) >= target:
        return 1.0
    lower = beta
    upper = 1.0
    for _ in range(60):
        midpoint = (lower + upper) / 2.0
        conditional = _conditional_ess(
            weights,
            selected_log_likelihood,
            midpoint - beta,
        )
        if conditional < target:
            upper = midpoint
        else:
            lower = midpoint
    result = lower
    if result <= beta or result > 1.0:
        raise ScreenError("adaptive annealing failed to advance")
    return result


def independent_mh_move(
    states: np.ndarray,
    *,
    log_tempered_target: np.ndarray,
    proposal: Proposal,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Apply one independent-MH move targeting the declared tempered law."""

    proposed = _sample_categorical(proposal.cdf, states.size, rng)
    log_q = np.log(proposal.probabilities)
    log_acceptance = (
        log_tempered_target[proposed]
        - log_tempered_target[states]
        + log_q[states]
        - log_q[proposed]
    )
    accepted = np.log(rng.random(states.size)) <= np.minimum(log_acceptance, 0.0)
    return np.where(accepted, proposed, states), int(accepted.sum())


def _estimate(
    task: TerminalTask,
    states: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, float], dict[str, float | int]]:
    exact_mass = float(weights @ task.exact[states])
    mean_loss = float(weights @ task.losses[states])
    importance_ess = _ess(weights)
    return (
        {
            "exact_target_mass": exact_mass,
            "target_mean_loss": mean_loss,
        },
        {
            "importance_ess": importance_ess,
            "relative_importance_ess": importance_ess / states.size,
            "maximum_normalized_weight": float(weights.max()),
            "unique_terminal_programs": int(np.unique(states).size),
            "exact_terminal_particles": int(task.exact[states].sum()),
        },
    )


def run_repetition(
    task: TerminalTask,
    proposal: Proposal,
    *,
    particles: int,
    repetition: int,
    base_seed: int,
) -> dict[str, object]:
    """Run one deterministic-seed terminal IS or annealed-SMC repetition."""

    if particles < 2:
        raise ScreenError("particle count must be at least two")
    sample_seed = _seed(
        task.binding.sha256,
        proposal.arm.arm_id,
        particles,
        repetition,
        base_seed,
    )
    rng = np.random.Generator(np.random.PCG64(sample_seed))
    states = _sample_categorical(proposal.cdf, particles, rng)
    logical_draws = particles
    resamples = 0
    mh_attempts = 0
    mh_accepted = 0
    stages: list[dict[str, object]] = []
    if proposal.arm.algorithm == "terminal-is":
        weights = _normalize(task.target[states] / proposal.probabilities[states])
    elif proposal.arm.algorithm in {"annealed-smc", "proposal-bridge-smc"}:
        log_grammar = np.log(task.grammar)
        log_likelihood = task.log_gamma - log_grammar
        log_q = np.log(proposal.probabilities)
        if proposal.arm.algorithm == "annealed-smc":
            log_base = log_grammar
            log_bridge_ratio = log_likelihood
            weights = _normalize(task.grammar[states] / proposal.probabilities[states])
        else:
            log_base = log_q
            log_bridge_ratio = task.log_gamma - log_q
            weights = np.full(particles, 1.0 / particles, dtype=np.float64)
        beta = 0.0
        if _ess(weights) / particles < RESAMPLE_RELATIVE_ESS:
            selected = _systematic_resample(weights, rng)
            states = states[selected]
            weights = np.full(particles, 1.0 / particles, dtype=np.float64)
            resamples += 1
            for _ in range(proposal.arm.mh_moves):
                states, accepted = independent_mh_move(
                    states,
                    log_tempered_target=log_base + beta * log_bridge_ratio,
                    proposal=proposal,
                    rng=rng,
                )
                mh_attempts += particles
                mh_accepted += accepted
                logical_draws += particles
        for stage in range(1, MAX_ANNEAL_STAGES + 1):
            selected_log_bridge = log_bridge_ratio[states]
            beta_next = next_beta(weights, selected_log_bridge, beta)
            log_increment = (beta_next - beta) * selected_log_bridge
            log_increment -= float(log_increment.max())
            weights = _normalize(weights * np.exp(log_increment))
            stage_ess = _ess(weights)
            resampled = beta_next < 1.0 and (
                stage_ess / particles < RESAMPLE_RELATIVE_ESS
            )
            if resampled:
                selected = _systematic_resample(weights, rng)
                states = states[selected]
                weights = np.full(particles, 1.0 / particles, dtype=np.float64)
                resamples += 1
                for _ in range(proposal.arm.mh_moves):
                    states, accepted = independent_mh_move(
                        states,
                        log_tempered_target=(
                            log_base + beta_next * log_bridge_ratio
                        ),
                        proposal=proposal,
                        rng=rng,
                    )
                    mh_attempts += particles
                    mh_accepted += accepted
                    logical_draws += particles
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
            raise ScreenError("annealing did not reach beta=1 within the stage cap")
    else:
        raise ScreenError(f"unsupported algorithm: {proposal.arm.algorithm}")
    estimate, diagnostics = _estimate(task, states, weights)
    diagnostics.update(
        {
            "resampling_events": resamples,
            "mh_attempts": mh_attempts,
            "mh_accepted": mh_accepted,
            "mh_acceptance_rate": mh_accepted / mh_attempts if mh_attempts else 0.0,
            "annealing_stages": len(stages),
        }
    )
    result: dict[str, object] = {
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
        "logical_proposal_draws": logical_draws,
        "estimate": estimate,
        "reference": {
            "exact_target_mass": task.exact_mass,
            "target_mean_loss": task.target_mean_loss,
        },
        "error": {
            "exact_mass_signed": estimate["exact_target_mass"] - task.exact_mass,
            "target_mean_loss_signed": (
                estimate["target_mean_loss"] - task.target_mean_loss
            ),
        },
        "diagnostics": diagnostics,
        "stages": stages,
    }
    validate_run(result)
    return result


def validate_run(run: Mapping[str, object]) -> None:
    if run.get("schema") != RUN_SCHEMA or run.get("provider_calls") != 0:
        raise ScreenError("run schema or provider-free invariant failed")
    particles = run.get("particles")
    if isinstance(particles, bool) or not isinstance(particles, int) or particles < 2:
        raise ScreenError("run particle count is invalid")
    estimate = run.get("estimate")
    diagnostics = run.get("diagnostics")
    if not isinstance(estimate, Mapping) or not isinstance(diagnostics, Mapping):
        raise ScreenError("run estimate or diagnostics is malformed")
    mass = _finite(estimate.get("exact_target_mass"), name="estimated exact mass")
    loss = _finite(estimate.get("target_mean_loss"), name="estimated mean loss")
    ess = _finite(diagnostics.get("importance_ess"), name="importance ESS")
    if not 0.0 <= mass <= 1.0 + 1e-12 or loss < 0.0:
        raise ScreenError("estimated endpoint is outside its valid range")
    if not 1.0 <= ess <= particles + 1e-9:
        raise ScreenError("sample ESS is outside its valid range")


def _sample_sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _quantile(values: Sequence[float], probability: float) -> float:
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
    errors = [
        estimate - reference
        for estimate, reference in zip(estimates, references, strict=True)
    ]
    return {
        "observations": len(errors),
        "estimate_mean": statistics.fmean(estimates),
        "reference_mean": statistics.fmean(references),
        "bias": statistics.fmean(errors),
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors)),
        "mae": statistics.fmean(abs(error) for error in errors),
        "error_sample_sd": _sample_sd(errors),
    }


def _diagnostic_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
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
    logical_draws: list[float] = []
    for record in records:
        estimate = cast(Mapping[str, object], record["estimate"])
        reference = cast(Mapping[str, object], record["reference"])
        diagnostic = cast(Mapping[str, object], record["diagnostics"])
        exact_estimates.append(_finite(estimate["exact_target_mass"], name="mass"))
        exact_references.append(
            _finite(reference["exact_target_mass"], name="mass reference")
        )
        loss_estimates.append(_finite(estimate["target_mean_loss"], name="loss"))
        loss_references.append(
            _finite(reference["target_mean_loss"], name="loss reference")
        )
        relative_ess.append(
            _finite(diagnostic["relative_importance_ess"], name="relative ESS")
        )
        maximum_weights.append(
            _finite(diagnostic["maximum_normalized_weight"], name="maximum weight")
        )
        logical_draws.append(
            _finite(record["logical_proposal_draws"], name="logical draws")
        )
    return {
        "exact_mass": _error_summary(exact_estimates, exact_references),
        "target_mean_loss": _error_summary(loss_estimates, loss_references),
        "relative_importance_ess": _diagnostic_summary(relative_ess),
        "maximum_normalized_weight": _diagnostic_summary(maximum_weights),
        "logical_proposal_draws": _diagnostic_summary(logical_draws),
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
        raise ScreenError(f"analysis received {len(runs)} runs, expected {expected}")
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
                    raise ScreenError(
                        f"{task_id}/{arm.arm_id}/N={particles} is incomplete"
                    )
                by_n[str(particles)] = _group_summary(group)
            task_arms[arm.arm_id] = {"role": arm.role, "by_particle_count": by_n}
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
                raise ScreenError(f"pooled {arm.arm_id}/N={particles} is incomplete")
            by_n[str(particles)] = _group_summary(group)
        pooled[arm.arm_id] = {"role": arm.role, "by_particle_count": by_n}
    identity_errors = [
        _finite(
            cast(Mapping[str, object], reference["proposal_census"])[arm.arm_id][
                "importance_identity_maximum_absolute_error"
            ],
            name="identity error",
        )
        for reference in references.values()
        for arm in arms
    ]
    n_key = str(max(particle_counts))
    comparison = {
        arm.arm_id: cast(Mapping[str, object], pooled[arm.arm_id])["by_particle_count"][
            n_key
        ]["exact_mass"]["rmse"]
        for arm in arms
    }
    population_ess = {
        arm.arm_id: statistics.fmean(
            _finite(
                cast(Mapping[str, object], reference["proposal_census"])[arm.arm_id][
                    "population_importance_ess_fraction"
                ],
                name="population ESS fraction",
            )
            for reference in references.values()
        )
        for arm in arms
    }
    return {
        "schema": ANALYSIS_SCHEMA,
        "status": "developmental-screen-not-a-calibration-gate",
        "run_count": len(runs),
        "task_count": len(task_ids),
        "arm_count": len(arms),
        "particle_counts": list(particle_counts),
        "repetitions_per_task_arm_particle_count": repetitions,
        "per_task": per_task,
        "pooled": pooled,
        "screen_summary": {
            "maximum_importance_identity_error": max(identity_errors),
            "identities_pass": max(identity_errors) <= IDENTITY_TOLERANCE,
            "largest_particle_count": max(particle_counts),
            "largest_n_exact_mass_rmse_by_arm": comparison,
            "mean_exact_population_ess_fraction_by_arm": population_ess,
            "sampled_mode_bank_by_task": {
                task_id: reference["sampled_mode_bank"]
                for task_id, reference in references.items()
            },
            "best_nonexhaustive_arm_by_largest_n_rmse": min(
                (
                    (float(value), arm_id)
                    for arm_id, value in comparison.items()
                    if "exhaustive" not in arm_id
                )
            )[1],
        },
        "claim_boundary": (
            "provider-free reused-task developmental mechanism screen; exact support is "
            "enumerated for references and census; no fresh calibration, provider, or "
            "search-efficiency claim"
        ),
    }


def _resolve_file(project_root: Path, relative: str) -> Path:
    root = project_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ScreenError(f"bound path is not a regular project file: {relative}")
    return path


def _file_record(project_root: Path, relative: str) -> dict[str, str]:
    path = _resolve_file(project_root, relative)
    return {"path": relative, "sha256": _sha256_file(path)}


def _python_tree_record(project_root: Path) -> dict[str, object]:
    relative_root = "src/modelsmc_pbe"
    tree_root = (project_root / relative_root).resolve()
    paths = sorted(
        (path for path in tree_root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(tree_root).as_posix().encode("utf-8"),
    )
    if not paths or any(path.is_symlink() for path in tree_root.rglob("*")):
        raise ScreenError("ModelSMC source tree is empty or contains a symlink")
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
    """Construct the developmental protocol before Monte Carlo draws."""

    tasks = [
        {"id": task_id, **_file_record(project_root, relative)}
        for task_id, relative in TASK_SPECS
    ]
    return {
        "schema": STUDY_SCHEMA,
        "status": STATUS,
        "frozen_at": "2026-08-14",
        "study_type": "provider-free reused-task developmental mechanism screen",
        "question": (
            "Can a fixed-budget sampled semantic-mode mixture improve finite-particle "
            "program-target approximation, and do annealing or independent-MH "
            "rejuvenation add value beyond the mixture alone?"
        ),
        "authorization": {
            "provider_calls": False,
            "new_model_responses": False,
            "fresh_tasks": False,
            "exact_reference_enumeration": True,
        },
        "target": (
            "pi(p) proportional to normalized recursive grammar g(p) times "
            "exp(score.log_target(p))"
        ),
        "mode_bank": {
            "discovery_budget": DISCOVERY_BUDGET,
            "maximum_mode_count": MAX_MODE_COUNT,
            "maximum_syntax_aliases_per_mode": MAX_ALIASES_PER_MODE,
            "semantic_deduplication": "itemwise behavior on the declared finite domain",
            "alias_expansion": (
                "predicate-fixed mapper slice plus mapper-fixed predicate slice around "
                "each retained semantic representative"
            ),
            "selection_uses_hidden_target": False,
        },
        "proposal": {
            "default_center_mass": LOCAL_CENTER_MASS,
            "center_mass_is_arm_specific": True,
            "formula": (
                "alpha*g + (1-alpha)/K sum_k [eta*delta_mode + "
                "(1-eta)/2*predicate_slice + (1-eta)/2*mapper_slice]"
            ),
            "deterministic_mixture_denominator": True,
            "full_support": True,
        },
        "annealing": {
            "conditional_ess_fraction": ANNEAL_CONDITIONAL_ESS,
            "resample_relative_ess": RESAMPLE_RELATIVE_ESS,
            "maximum_stages": MAX_ANNEAL_STAGES,
            "mh_kernel": "independent proposal from the same evaluated global mixture",
            "paths": [
                "prior bridge: g(p) * exp(beta * score.log_target(p))",
                "proposal bridge: q(p)^(1-beta) * gamma(p)^beta",
            ],
        },
        "design": {
            "arms": [asdict(arm) for arm in ARMS],
            "particle_counts": list(PARTICLE_COUNTS),
            "repetitions": REPETITIONS,
            "repetition_seeds": list(REPETITION_SEEDS),
            "task_count": len(TASK_SPECS),
            "total_repetitions": (
                len(TASK_SPECS) * len(ARMS) * len(PARTICLE_COUNTS) * REPETITIONS
            ),
            "identity_tolerance": IDENTITY_TOLERANCE,
        },
        "bindings": {
            "harness": _file_record(project_root, HARNESS_PATH),
            "test": _file_record(project_root, TEST_PATH),
            "dependencies": [
                _file_record(project_root, relative) for relative in DEPENDENCY_PATHS
            ],
            "lockfiles": [
                _file_record(project_root, relative) for relative in LOCKFILE_PATHS
            ],
            "modelsmc_python_tree": _python_tree_record(project_root),
            "tasks": tasks,
        },
        "selection_rule": (
            "Choose no confirmatory method from one run. Report all arms and per-task "
            "results; use this reused-task screen only to freeze a later fresh method."
        ),
        "claim_boundary": [
            "The four tasks were used previously and are not fresh.",
            "Exact 36,000-state enumeration is used for references and audit census.",
            "The exhaustive top-64 arm is a positive control, not a scalable method.",
            "No provider, LLM, fresh-task calibration, or deployment claim is authorized.",
        ],
    }


def freeze_protocol(protocol_path: Path, project_root: Path) -> str:
    if protocol_path.exists():
        raise FileExistsError(f"refusing to overwrite existing protocol: {protocol_path}")
    _write_json(protocol_path, build_frozen_protocol(project_root))
    return _sha256_file(protocol_path)


def _validate_records(
    records: object,
    expected_paths: Sequence[str],
    project_root: Path,
) -> None:
    if not isinstance(records, list) or len(records) != len(expected_paths):
        raise ScreenError("frozen file bundle has the wrong size")
    for record, expected in zip(records, expected_paths, strict=True):
        if not isinstance(record, Mapping) or record.get("path") != expected:
            raise ScreenError("frozen file bundle path or order differs")
        if record.get("sha256") != _sha256_file(_resolve_file(project_root, expected)):
            raise ScreenError(f"frozen SHA-256 differs: {expected}")


def load_frozen_plan(
    protocol_path: Path,
    project_root: Path,
    *,
    expected_protocol_sha256: str,
) -> tuple[StudyPlan, dict[str, object]]:
    actual_sha = _sha256_file(protocol_path)
    if actual_sha != expected_protocol_sha256:
        raise ScreenError("protocol differs from the externally expected SHA-256")
    protocol = cast(dict[str, object], json.loads(protocol_path.read_text()))
    if protocol.get("schema") != STUDY_SCHEMA or protocol.get("status") != STATUS:
        raise ScreenError("protocol schema or frozen status differs")
    if protocol != build_frozen_protocol(project_root):
        raise ScreenError("protocol bytes differ from the current bound implementation")
    bindings = cast(Mapping[str, object], protocol["bindings"])
    _validate_records(bindings["dependencies"], DEPENDENCY_PATHS, project_root)
    _validate_records(bindings["lockfiles"], LOCKFILE_PATHS, project_root)
    if bindings["modelsmc_python_tree"] != _python_tree_record(project_root):
        raise ScreenError("frozen ModelSMC Python tree differs")
    tasks: list[TaskBinding] = []
    task_records = bindings["tasks"]
    if not isinstance(task_records, list) or len(task_records) != len(TASK_SPECS):
        raise ScreenError("frozen task bundle differs")
    for record, (task_id, relative) in zip(task_records, TASK_SPECS, strict=True):
        if not isinstance(record, Mapping) or record.get("id") != task_id:
            raise ScreenError("frozen task id differs")
        path = _resolve_file(project_root, relative)
        digest = _sha256_file(path)
        if record.get("path") != relative or record.get("sha256") != digest:
            raise ScreenError("frozen task path or SHA-256 differs")
        tasks.append(TaskBinding(task_id, path, digest))
    return (
        StudyPlan(
            tasks=tuple(tasks),
            arms=ARMS,
            particle_counts=PARTICLE_COUNTS,
            repetition_seeds=REPETITION_SEEDS,
            protocol_sha256=actual_sha,
        ),
        protocol,
    )


def _reference_record(
    task: TerminalTask,
    proposals: Mapping[str, Proposal],
    mode_metadata: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": REFERENCE_SCHEMA,
        "task_id": task.binding.task_id,
        "task_sha256": task.binding.sha256,
        "program_syntaxes": len(task.keys),
        "exact_program_syntaxes": int(task.exact.sum()),
        "exact_target_mass": task.exact_mass,
        "target_mean_loss": task.target_mean_loss,
        "program_population_sha256": task.program_inventory_sha256,
        "sampled_mode_bank": mode_metadata,
        "proposal_census": {
            arm_id: proposal.census for arm_id, proposal in proposals.items()
        },
    }
    return result


def _summary_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Calibrated Program Inference V2 Developmental Screen",
        "",
        "This is a provider-free, reused-task mechanism screen, not a fresh calibration gate.",
        "",
        "| Arm | N | Exact-mass RMSE | Bias | Mean relative ESS | Mean draws |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    pooled = cast(Mapping[str, object], analysis["pooled"])
    for arm in ARMS:
        by_n = cast(Mapping[str, object], cast(Mapping[str, object], pooled[arm.arm_id])[
            "by_particle_count"
        ])
        for particles in PARTICLE_COUNTS:
            record = cast(Mapping[str, object], by_n[str(particles)])
            exact = cast(Mapping[str, object], record["exact_mass"])
            relative_ess = cast(Mapping[str, object], record["relative_importance_ess"])
            draws = cast(Mapping[str, object], record["logical_proposal_draws"])
            lines.append(
                f"| {arm.arm_id} | {particles} | {float(exact['rmse']):.6g} | "
                f"{float(exact['bias']):.6g} | {float(relative_ess['mean']):.6g} | "
                f"{float(draws['mean']):.6g} |"
            )
    screen = cast(Mapping[str, object], analysis["screen_summary"])
    lines.extend(
        [
            "",
            (
                "Best non-exhaustive largest-N arm: "
                f"**{screen['best_nonexhaustive_arm_by_largest_n_rmse']}**."
            ),
            "",
            "The exhaustive top-64 arm remains a positive control only.",
            "",
        ]
    )
    return "\n".join(lines)


def _seal_inventory(output: Path) -> dict[str, object]:
    files = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "inventory.json"
    )
    if any(path.is_symlink() for path in output.rglob("*")):
        raise ScreenError("artifact inventory rejects symlinks")
    entries = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    inventory = {
        "schema": f"{STUDY_SCHEMA}-inventory-v1",
        "file_count_excluding_inventory": len(entries),
        "entries": entries,
        "entries_sha256": _sha256_bytes(canonical_bytes(entries)),
    }
    _write_json(output / "inventory.json", inventory)
    return inventory


def validate_artifact(output: Path) -> dict[str, object]:
    inventory_path = output / "inventory.json"
    if not inventory_path.is_file() or inventory_path.is_symlink():
        raise ScreenError("artifact inventory is missing")
    saved = cast(dict[str, object], json.loads(inventory_path.read_text()))
    entries = saved.get("entries")
    if not isinstance(entries, list):
        raise ScreenError("artifact inventory entries are malformed")
    actual_paths = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "inventory.json"
    )
    expected_paths = [record["path"] for record in entries if isinstance(record, Mapping)]
    if actual_paths != expected_paths:
        raise ScreenError("artifact inventory paths differ")
    for record in entries:
        if not isinstance(record, Mapping):
            raise ScreenError("artifact inventory record is malformed")
        path = output / cast(str, record["path"])
        if path.is_symlink() or record.get("sha256") != _sha256_file(path):
            raise ScreenError(f"artifact file differs: {path}")
    if saved.get("entries_sha256") != _sha256_bytes(canonical_bytes(entries)):
        raise ScreenError("artifact inventory digest differs")
    return saved


def run_study(
    plan: StudyPlan,
    protocol: Mapping[str, object],
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    staging = output.with_name(f".{output.name}.incomplete")
    if staging.exists():
        raise FileExistsError(f"refusing to reuse existing staging output: {staging}")
    staging.mkdir(parents=True)
    try:
        _write_json(staging / "protocol.json", protocol)
        references: dict[str, Mapping[str, object]] = {}
        runs: list[Mapping[str, object]] = []
        for binding in plan.tasks:
            task = TerminalTask(binding)
            sampled_bank = sampled_semantic_mode_bank(task)
            proposals = {
                arm.arm_id: build_proposal(task, arm, sampled_bank) for arm in plan.arms
            }
            reference = _reference_record(task, proposals, sampled_bank.metadata)
            references[binding.task_id] = reference
            _write_json(staging / "references" / f"{binding.task_id}.json", reference)
            for arm in plan.arms:
                proposal = proposals[arm.arm_id]
                for particles in plan.particle_counts:
                    for repetition, base_seed in enumerate(plan.repetition_seeds):
                        runs.append(
                            run_repetition(
                                task,
                                proposal,
                                particles=particles,
                                repetition=repetition,
                                base_seed=base_seed,
                            )
                        )
        analysis = analyze_runs(
            runs,
            references,
            task_ids=[binding.task_id for binding in plan.tasks],
            arms=plan.arms,
            particle_counts=plan.particle_counts,
            repetitions=len(plan.repetition_seeds),
        )
        _write_json(staging / "runs.json", runs)
        _write_json(staging / "analysis.json", analysis)
        (staging / "SUMMARY.md").write_text(_summary_markdown(analysis), encoding="utf-8")
        _write_json(
            staging / "study-metadata.json",
            {
                "schema": f"{STUDY_SCHEMA}-metadata-v1",
                "protocol_sha256": plan.protocol_sha256,
                "runtime": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "device": "cpu",
                },
                "provider_calls": 0,
                "exact_support_materialized_for_developmental_reference": True,
            },
        )
        _seal_inventory(staging)
        staging.rename(output)
        validate_artifact(output)
        return cast(dict[str, object], analysis)
    except Exception as exc:
        failure = {
            "schema": f"{STUDY_SCHEMA}-failure-v1",
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
        _write_json(staging / "FAILURE.json", failure)
        raise


def run_pilot(
    project_root: Path,
    *,
    task_id: str,
    particles: int,
    repetitions: int,
) -> dict[str, object]:
    """Run an explicitly unfrozen, provider-free smoke pilot without writing artifacts."""

    task_paths = dict(TASK_SPECS)
    relative = task_paths.get(task_id)
    if relative is None:
        raise ScreenError(f"unknown pilot task: {task_id}")
    if particles < 2 or repetitions < 1 or repetitions > len(REPETITION_SEEDS):
        raise ScreenError("pilot particles or repetitions are invalid")
    path = _resolve_file(project_root, relative)
    task = TerminalTask(TaskBinding(task_id, path, _sha256_file(path)))
    sampled_bank = sampled_semantic_mode_bank(task)
    proposals = {
        arm.arm_id: build_proposal(task, arm, sampled_bank) for arm in ARMS
    }
    reference = _reference_record(task, proposals, sampled_bank.metadata)
    runs = [
        run_repetition(
            task,
            proposals[arm.arm_id],
            particles=particles,
            repetition=repetition,
            base_seed=REPETITION_SEEDS[repetition],
        )
        for arm in ARMS
        for repetition in range(repetitions)
    ]
    return analyze_runs(
        runs,
        {task_id: reference},
        task_ids=[task_id],
        arms=ARMS,
        particle_counts=[particles],
        repetitions=repetitions,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--project-root", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--project-root", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--expected-protocol-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--project-root", type=Path, required=True)
    pilot.add_argument("--task-id", default=TASK_SPECS[0][0])
    pilot.add_argument("--particles", type=int, default=256)
    pilot.add_argument("--repetitions", type=int, default=16)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        digest = freeze_protocol(args.protocol.resolve(), args.project_root.resolve())
        print(digest)
        return 0
    if args.command == "run":
        plan, protocol = load_frozen_plan(
            args.protocol.resolve(),
            args.project_root.resolve(),
            expected_protocol_sha256=args.expected_protocol_sha256,
        )
        analysis = run_study(plan, protocol, args.output.resolve())
        print(json.dumps(analysis["screen_summary"], indent=2, sort_keys=True))
        return 0
    if args.command == "pilot":
        analysis = run_pilot(
            args.project_root.resolve(),
            task_id=args.task_id,
            particles=args.particles,
            repetitions=args.repetitions,
        )
        print(json.dumps(analysis["screen_summary"], indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        print(json.dumps(validate_artifact(args.output.resolve()), indent=2, sort_keys=True))
        return 0
    raise ScreenError("unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())

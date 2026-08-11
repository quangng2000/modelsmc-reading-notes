"""Sequential oracle proposal from exact joint execution-target marginals."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any

import torch

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.smc import categorical_sample

from ..records import FamilySupport, ImportanceSMCOptions, ImportanceSupport, ProposedState
from ..score_ledger import LLMScoreWaveLedger
from ..subtree import require_strictly_positive_weights
from ..target import FiniteImportanceTarget
from ..trie import grouped_hole_choices, ordered_families
from .paths import (
    JointPath,
    JointPathSeed,
    positive_probability,
    validate_seeds,
)

type EventEmitter = Callable[..., None]

JOINT_TARGET_PROPOSAL_SOURCE = "finite-joint-target-oracle"


class FiniteJointTargetProposalKernel:
    """Sample the normalized finite target through exact prefix marginals.

    Every complete state has already been assembled and executed by the support
    builder. Exact target mass below each child prefix defines a sequential
    proposal whose conditional probabilities telescope to ``gamma(e) / Z``.

    This is deliberately an oracle, materialized-support control. It is not a
    scalable replacement for a learned proposal.
    """

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        options: ImportanceSMCOptions,
        support: ImportanceSupport,
        target: FiniteImportanceTarget,
        generator: torch.Generator,
        emit: EventEmitter | None = None,
    ) -> None:
        if generator.device.type != "cpu":
            raise ValueError("joint-target proposal sampling requires a CPU generator")
        if config.smc.alpha != 0.0:
            raise ValueError("joint-target proposal requires alpha=0")
        if options.proposal_strategy != "joint-target":
            raise ValueError("joint-target kernel requires proposal_strategy='joint-target'")
        if len(support.states) != target.losses_cpu.numel():
            raise ValueError("joint target and materialized support sizes differ")
        self._config = config
        self._beta_max = options.beta_max
        self._support = support
        self._target = target
        self._generator = generator
        self._emit = emit
        self._emit_ready()

    @property
    def source(self) -> str:
        return JOINT_TARGET_PROPOSAL_SOURCE

    @property
    def scored_candidates(self) -> int:
        """The oracle performs no language-model candidate scoring."""

        return 0

    @property
    def score_ledger(self) -> tuple[LLMScoreWaveLedger, ...]:
        """The oracle has no provider score requests to persist."""

        return ()

    @property
    def final_deduction_guide(self) -> None:
        """The joint oracle does not construct or use a deduction proposal guide."""

        return None

    async def sample_many(
        self,
        ancestor_state_indices: tuple[int, ...],
        *,
        stage: int,
        beta: float,
    ) -> tuple[ProposedState, ...]:
        """Draw one exact normalized-target child for each ancestor context."""

        seeds = tuple(
            JointPathSeed(slot, ancestor_state_index, None)
            for slot, ancestor_state_index in enumerate(ancestor_state_indices)
        )
        return self._propose(seeds, stage=stage, beta=beta)

    async def evaluate_many(
        self,
        ancestor_state_index: int,
        target_state_indices: tuple[int, ...],
        *,
        stage: int,
        beta: float,
    ) -> tuple[ProposedState, ...]:
        """Evaluate exact sequential proposal mass for specified complete states."""

        seeds = tuple(
            JointPathSeed(slot, ancestor_state_index, target_state_index)
            for slot, target_state_index in enumerate(target_state_indices)
        )
        return self._propose(seeds, stage=stage, beta=beta)

    def _propose(
        self,
        seeds: tuple[JointPathSeed, ...],
        *,
        stage: int,
        beta: float,
    ) -> tuple[ProposedState, ...]:
        require_strictly_positive_weights(
            self._target.distribution(beta).weights,
            label="joint-target",
        )
        validate_seeds(seeds, state_count=len(self._support.states))
        paths = self._select_families(seeds, stage=stage, beta=beta)
        maximum_holes = max(len(path.family.hypothesis.holes) for path in paths)
        for hole_index in range(maximum_holes):
            for path in paths:
                if hole_index < len(path.family.hypothesis.holes):
                    self._advance(path, hole_index=hole_index, stage=stage, beta=beta)
        return tuple(self._finish(path, beta=beta) for path in paths)

    def _select_families(
        self,
        seeds: tuple[JointPathSeed, ...],
        *,
        stage: int,
        beta: float,
    ) -> tuple[JointPath, ...]:
        families = ordered_families(self._support)
        probabilities = self._target.subtree_probabilities(
            tuple(family.state_indices for family in families),
            beta=beta,
        )
        paths: list[JointPath] = []
        for seed in seeds:
            selected = self._family_index(seed, families, probabilities)
            probability = positive_probability(probabilities, selected, "family")
            family = families[selected]
            paths.append(
                JointPath(
                    slot=seed.slot,
                    ancestor_state_index=seed.ancestor_state_index,
                    family=family,
                    compatible_state_indices=family.state_indices,
                    target_state_index=seed.target_state_index,
                    log_q_construct=math.log(probability),
                )
            )
            self._event(
                "importance.proposal.joint_target.family_selected",
                message="family selected from exact target marginal",
                level="trace",
                stage=stage,
                slot=seed.slot,
                ancestor_state_index=seed.ancestor_state_index,
                family=family.hypothesis.kind.value,
                probability=probability,
                forced=seed.target_state_index is not None,
            )
        return tuple(paths)

    def _family_index(
        self,
        seed: JointPathSeed,
        families: tuple[FamilySupport, ...],
        probabilities: torch.Tensor,
    ) -> int:
        if seed.target_state_index is None:
            return int(categorical_sample(probabilities, 1, generator=self._generator)[0].item())
        hypothesis_index = self._support.states[seed.target_state_index].hypothesis_index
        return next(
            index
            for index, family in enumerate(families)
            if family.hypothesis_index == hypothesis_index
        )

    def _advance(
        self,
        path: JointPath,
        *,
        hole_index: int,
        stage: int,
        beta: float,
    ) -> None:
        choices = grouped_hole_choices(
            self._support,
            path.family,
            path.compatible_state_indices,
            hole_index=hole_index,
        )
        probabilities = self._target.subtree_probabilities(
            tuple(choice.state_indices for choice in choices),
            beta=beta,
        )
        if path.target_state_index is None:
            selected = int(
                categorical_sample(probabilities, 1, generator=self._generator)[0].item()
            )
        else:
            target = self._support.states[path.target_state_index]
            hole_name = path.family.hypothesis.holes[hole_index].name
            target_key = target.filling(hole_name).key
            selected = next(
                index for index, choice in enumerate(choices) if choice.key == target_key
            )
        probability = positive_probability(probabilities, selected, "hole choice")
        chosen = choices[selected]
        path.log_q_construct += math.log(probability)
        path.compatible_state_indices = chosen.state_indices
        self._event(
            "importance.proposal.joint_target.hole_selected",
            message="hole filling selected from exact conditional target marginal",
            level="trace",
            stage=stage,
            slot=path.slot,
            ancestor_state_index=path.ancestor_state_index,
            family=path.family.hypothesis.kind.value,
            hole=chosen.filling.hole_name,
            candidate_sha256=hashlib.sha256(chosen.key.encode("utf-8")).hexdigest(),
            probability=probability,
            forced=path.target_state_index is not None,
        )

    def _finish(self, path: JointPath, *, beta: float) -> ProposedState:
        if len(path.compatible_state_indices) != 1:
            raise RuntimeError("joint construction trace did not resolve to one complete program")
        state_index = path.compatible_state_indices[0]
        if path.target_state_index is not None and state_index != path.target_state_index:
            raise RuntimeError("joint probability evaluation did not recover its target state")
        distribution = self._target.distribution(beta)
        expected = (
            float(self._target.log_unnormalized(beta)[state_index].item())
            - distribution.log_normalizer
        )
        if not math.isclose(path.log_q_construct, expected, rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError("joint conditional probabilities failed to telescope to target mass")
        return ProposedState(
            state_index=state_index,
            ancestor_state_index=path.ancestor_state_index,
            log_q_construct=path.log_q_construct,
            log_q_mixture=path.log_q_construct,
            cloned=False,
            family=path.family.hypothesis.kind.value,
        )

    def _emit_ready(self) -> None:
        exact_indices = self._target.exact_mask.nonzero(as_tuple=False).flatten()
        final_target = self._target.distribution(self._beta_max).weights
        self._event(
            "importance.proposal.joint_target.ready",
            message="exact joint execution-target oracle constructed",
            level="info",
            support_states=len(self._support.states),
            exact_program_mass=(
                0.0
                if exact_indices.numel() == 0
                else float(final_target[exact_indices].sum().item())
            ),
            alpha=self._config.smc.alpha,
        )

    def _event(
        self,
        name: str,
        *,
        message: str,
        level: str,
        **data: Any,
    ) -> None:
        if self._emit is not None:
            self._emit(name, message=message, level=level, **data)

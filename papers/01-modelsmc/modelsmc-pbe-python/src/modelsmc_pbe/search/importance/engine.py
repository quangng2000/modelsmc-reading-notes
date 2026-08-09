"""Feynman--Kac SMC with an exact finite Qwen-energy proposal density."""

from __future__ import annotations

from typing import Any

import torch

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.core import ProgramScorer
from modelsmc_pbe.observability import RunLogger
from modelsmc_pbe.proposals import CandidateScorer
from modelsmc_pbe.runtime import DeviceInfo
from modelsmc_pbe.smc import (
    categorical_sample,
    effective_sample_size,
    normalize_log_weights,
    relative_effective_sample_size,
    systematic_resample,
)

from .proposal import FiniteLLMProposalKernel
from .records import (
    ImportancePopulation,
    ImportanceSMCOptions,
    ImportanceSMCResult,
    ImportanceStageDiagnostic,
)
from .results import assemble_importance_result
from .support import ImportanceSupportBuilder
from .target import FiniteImportanceTarget


class ImportanceSMCEngine:
    """Approximate a declared finite target with importance-corrected proposals.

    The path target is ``gamma_t(e_1:t) = product_s pi_tilde_beta_s(e_s)``.
    Consequently ``G_t = pi_tilde_beta_t(e_t) / q_t(e_t | history)`` and the
    terminal-program marginal is the normalized finite target at ``beta_t``.
    """

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        options: ImportanceSMCOptions,
        scorer: ProgramScorer,
        candidate_scorer: CandidateScorer,
        device: DeviceInfo,
        generator: torch.Generator,
        logger: RunLogger | None = None,
    ) -> None:
        if generator.device.type != "cpu":
            raise ValueError("importance-smc requires a CPU generator")
        self._config = config
        self._options = options
        self._scorer = scorer
        self._candidate_scorer = candidate_scorer
        self._device = device
        self._generator = generator
        self._logger = logger

    async def run(self) -> ImportanceSMCResult:
        support = ImportanceSupportBuilder(
            spec=self._config.spec,
            smc=self._config.smc,
            options=self._options,
            scorer=self._scorer,
            emit=self._emit,
        ).build()
        target = FiniteImportanceTarget.build(
            support.states,
            smc=self._config.smc,
            device=self._device,
        )
        population = self._initialize(target)
        kernel = FiniteLLMProposalKernel(
            config=self._config,
            options=self._options,
            support=support,
            scorer=self._candidate_scorer,
            generator=self._generator,
            emit=self._emit,
        )
        diagnostics: list[ImportanceStageDiagnostic] = []
        log_path_z_estimate = 0.0
        log_path_z_reference = 0.0
        for stage in range(1, self._config.smc.iterations + 1):
            beta = self._options.beta_max * stage / self._config.smc.iterations
            weights_before = torch.tensor(population.weights, dtype=torch.float64)
            ess_before = effective_sample_size(weights_before)
            relative_ess = relative_effective_sample_size(weights_before)
            ancestor_states, base_weights, resampled = self._ancestors(
                population, relative_ess
            )
            proposals = await kernel.sample_many(
                ancestor_states,
                stage=stage,
                beta=beta,
            )
            log_gamma = target.log_unnormalized(beta)
            log_q = torch.tensor(
                [proposal.log_q_mixture for proposal in proposals], dtype=torch.float64
            )
            state_indices = torch.tensor(
                [proposal.state_index for proposal in proposals], dtype=torch.int64
            )
            incremental = log_gamma[state_indices] - log_q
            log_base = torch.log(base_weights)
            normalized = normalize_log_weights(log_base + incremental)
            log_path_z_estimate += normalized.log_normalizer
            exact_stage = target.distribution(beta)
            log_path_z_reference += exact_stage.log_normalizer
            population = ImportancePopulation(
                state_indices=tuple(int(value) for value in state_indices.tolist()),
                weights=tuple(float(value) for value in normalized.weights.tolist()),
                ancestor_state_indices=ancestor_states,
                log_q_mixture=tuple(float(value) for value in log_q.tolist()),
                log_incremental_weight=tuple(
                    float(value) for value in incremental.tolist()
                ),
                cloned=tuple(proposal.cloned for proposal in proposals),
            )
            diagnostic = ImportanceStageDiagnostic(
                stage=stage,
                beta=beta,
                ess_before=ess_before,
                relative_ess_before=relative_ess,
                resampled=resampled,
                clones=sum(proposal.cloned for proposal in proposals),
                unique_programs=len(set(population.state_indices)),
                exact_programs=sum(
                    support.states[index].score.exact_program
                    for index in population.state_indices
                ),
                ess_after=effective_sample_size(normalized.weights),
                mean_log_q=float(log_q.mean().item()),
                min_log_q=float(log_q.min().item()),
                max_log_importance_ratio=float(incremental.max().item()),
                log_path_z_estimate=log_path_z_estimate,
                log_path_z_reference=log_path_z_reference,
            )
            diagnostics.append(diagnostic)
            self._emit_stage(diagnostic)

        result = assemble_importance_result(
            support=support,
            population=population,
            target=target,
            beta_max=self._options.beta_max,
            proposal_source=kernel.source,
            stages=tuple(diagnostics),
            log_path_z_estimate=log_path_z_estimate,
            log_path_z_reference=log_path_z_reference,
        )
        self._emit(
            "importance_smc.completed",
            message="importance-corrected finite-support SMC completed",
            support_states=result.support_states,
            exact_programs=result.exact_programs,
            particle_exact_mass=result.reference.particle_exact_mass,
            enumeration_exact_mass=result.reference.enumeration_exact_mass,
            total_variation=result.reference.total_variation_distance,
            absolute_log_path_z_error=result.reference.absolute_log_path_z_error,
        )
        return result

    def _initialize(self, target: FiniteImportanceTarget) -> ImportancePopulation:
        count = self._config.smc.particles
        occam = target.distribution(0.0).weights
        state_indices = categorical_sample(
            occam,
            count,
            generator=self._generator,
        )
        values = tuple(int(value) for value in state_indices.tolist())
        uniform = tuple(1.0 / count for _ in range(count))
        self._emit(
            "importance.population.initialized",
            message="initial ancestor contexts sampled from the exact finite Occam prior",
            particles=count,
            unique_programs=len(set(values)),
            support_states=occam.numel(),
        )
        return ImportancePopulation(
            state_indices=values,
            weights=uniform,
            ancestor_state_indices=values,
            log_q_mixture=tuple(0.0 for _ in range(count)),
            log_incremental_weight=tuple(0.0 for _ in range(count)),
            cloned=tuple(False for _ in range(count)),
        )

    def _ancestors(
        self,
        population: ImportancePopulation,
        relative_ess: float,
    ) -> tuple[tuple[int, ...], torch.Tensor, bool]:
        weights = torch.tensor(population.weights, dtype=torch.float64)
        count = len(population.state_indices)
        if relative_ess >= self._config.smc.ess_threshold:
            return population.state_indices, weights, False
        slots = systematic_resample(weights, generator=self._generator)
        ancestors = tuple(population.state_indices[int(slot)] for slot in slots.tolist())
        return ancestors, torch.full((count,), 1.0 / count, dtype=torch.float64), True

    def _emit_stage(self, diagnostic: ImportanceStageDiagnostic) -> None:
        self._emit(
            "importance_smc.stage.completed",
            message="importance-corrected SMC stage completed",
            stage=diagnostic.stage,
            beta=diagnostic.beta,
            ess_before=diagnostic.ess_before,
            resampled=diagnostic.resampled,
            clones=diagnostic.clones,
            unique_programs=diagnostic.unique_programs,
            exact_programs=diagnostic.exact_programs,
            ess_after=diagnostic.ess_after,
            log_path_z_estimate=diagnostic.log_path_z_estimate,
        )

    def _emit(
        self,
        name: str,
        *,
        message: str,
        level: str = "info",
        **data: Any,
    ) -> None:
        if self._logger is not None:
            self._logger.event(name, message=message, level=level, **data)

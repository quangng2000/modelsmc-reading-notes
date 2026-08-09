"""Orchestration for calibrated SMC on a finite program grammar."""

from __future__ import annotations

from typing import Any

import torch

from modelsmc_pbe.config import SMCConfig
from modelsmc_pbe.core import ProgramScorer
from modelsmc_pbe.domain.models import PBESpec
from modelsmc_pbe.observability import RunLogger
from modelsmc_pbe.runtime import DeviceInfo
from modelsmc_pbe.search.grammar_control.records import (
    GrammarSMCOptions,
    GrammarSMCResult,
    GrammarStageDiagnostic,
)
from modelsmc_pbe.search.grammar_control.results import assemble_result
from modelsmc_pbe.search.grammar_control.support import ScoredSupportBuilder
from modelsmc_pbe.search.grammar_control.target import FiniteGrammarTarget
from modelsmc_pbe.search.grammar_control.transition import (
    advance_stage,
    initial_population,
)


class GrammarSMCEngine:
    """Run annealed SMC over one complete, finite AST skeleton catalog."""

    def __init__(
        self,
        *,
        spec: PBESpec,
        smc: SMCConfig,
        options: GrammarSMCOptions,
        scorer: ProgramScorer,
        device: DeviceInfo,
        generator: torch.Generator,
        logger: RunLogger | None = None,
    ) -> None:
        if generator.device.type != "cpu":
            raise ValueError("grammar SMC requires a CPU generator for deterministic decisions")
        self._spec = spec
        self._smc = smc
        self._options = options
        self._scorer = scorer
        self._device = device
        self._generator = generator
        self._logger = logger

    def run(self) -> GrammarSMCResult:
        """Execute the calibrated control and compute exact reference metrics."""

        support = ScoredSupportBuilder(
            spec=self._spec,
            smc=self._smc,
            options=self._options,
            scorer=self._scorer,
            emit=self._event,
        ).build()
        target = FiniteGrammarTarget.build(
            support.states,
            smc=self._smc,
            device=self._device,
        )
        population = initial_population(
            target,
            particle_count=int(self._smc.particles),
            generator=self._generator,
        )
        self._event(
            "grammar_smc.initialized",
            message="particles sampled from normalized Occam prior",
            particles=int(self._smc.particles),
            grammar_states=len(support.states),
            device=self._device.resolved,
            prior_log_normalizer=target.prior_log_normalizer,
        )

        beta_previous = 0.0
        log_z_estimate = 0.0
        diagnostics: list[GrammarStageDiagnostic] = []
        for stage in range(1, int(self._smc.iterations) + 1):
            beta_current = self._options.beta_max * stage / int(self._smc.iterations)
            outcome = advance_stage(
                population,
                stage=stage,
                beta_previous=beta_previous,
                beta_current=beta_current,
                log_z_estimate=log_z_estimate,
                target=target,
                smc=self._smc,
                options=self._options,
                generator=self._generator,
            )
            population = outcome.population
            log_z_estimate = outcome.log_z_estimate
            diagnostics.append(outcome.diagnostic)
            self._emit_stage(outcome.diagnostic)
            beta_previous = beta_current

        result = assemble_result(
            support=support,
            population=population,
            target=target,
            smc=self._smc,
            options=self._options,
            log_z_estimate=log_z_estimate,
            stages=tuple(diagnostics),
        )
        self._event(
            "grammar_smc.completed",
            message="calibrated finite-grammar SMC completed",
            exact=result.exact,
            sampled_best_loss=result.sampled_best.total_loss,
            sampled_best_cost=result.sampled_best.cost,
            particle_exact_mass=result.reference.particle_exact_mass,
            enumeration_exact_mass=result.reference.enumeration_exact_mass,
            log_z_error=result.reference.log_z_error,
            total_variation_distance=result.reference.total_variation_distance,
        )
        return result

    def _emit_stage(self, diagnostic: GrammarStageDiagnostic) -> None:
        self._event(
            "grammar_smc.stage.completed",
            message="annealing stage completed",
            stage=diagnostic.stage,
            stages=int(self._smc.iterations),
            beta=diagnostic.beta_current,
            ess=diagnostic.ess,
            relative_ess=diagnostic.relative_ess,
            resampled=diagnostic.resampled,
            mh_accepted=diagnostic.mh_accepted,
            mh_attempts=diagnostic.mh_attempts,
            log_z_estimate=diagnostic.log_z_estimate,
            particle_exact_mass=diagnostic.particle_exact_mass,
            particle_mean_loss=diagnostic.particle_mean_loss,
        )

    def _event(
        self,
        name: str,
        *,
        message: str,
        level: str = "info",
        **data: Any,
    ) -> None:
        if self._logger is not None:
            self._logger.event(name, message=message, level=level, **data)

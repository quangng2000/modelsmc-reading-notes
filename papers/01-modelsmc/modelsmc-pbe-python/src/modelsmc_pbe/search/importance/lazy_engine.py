"""Visit-only importance-SMC over a factorized typed construction support."""

from __future__ import annotations

from typing import Any

import torch

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.core import ProgramScorer, RejectedProgram, ScoredProgram
from modelsmc_pbe.domain import canonical_key
from modelsmc_pbe.induction import assemble_program
from modelsmc_pbe.observability import RunLogger
from modelsmc_pbe.proposals import CandidateScorer
from modelsmc_pbe.smc import (
    effective_sample_size,
    normalize_log_weights,
    relative_effective_sample_size,
    systematic_resample,
)

from .factorized import (
    sample_prior_trace,
    trace_fillings,
    trace_hole_cost,
    trace_log_prior,
)
from .lazy_proposal import LazyGuidedProposalKernel
from .lazy_records import (
    LAZY_IMPORTANCE_SMC_CLAIM,
    ConstructionTrace,
    FactorizedImportanceSupport,
    LazyFamilySummary,
    LazyImportanceParticle,
    LazyImportancePopulation,
    LazyImportanceSMCResult,
    LazyImportanceState,
    LazySearchMetrics,
    LazyStageDiagnostic,
    LazyStateSummary,
)
from .lazy_support import FactorizedSupportBuilder
from .records import ImportanceSMCOptions


class LazyImportanceSMCEngine:
    """Run SMC while assembling and executing only visited complete traces."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        options: ImportanceSMCOptions,
        scorer: ProgramScorer,
        candidate_scorer: CandidateScorer,
        generator: torch.Generator,
        logger: RunLogger | None = None,
    ) -> None:
        if generator.device.type != "cpu":
            raise ValueError("lazy importance-smc requires a CPU generator")
        if config.smc.alpha >= 1.0:
            raise ValueError("lazy importance-smc requires alpha < 1")
        self._config = config
        self._options = options
        self._scorer = scorer
        self._candidate_scorer = candidate_scorer
        self._generator = generator
        self._logger = logger
        self._states: dict[ConstructionTrace, LazyImportanceState] = {}

    async def run(self) -> LazyImportanceSMCResult:
        support = FactorizedSupportBuilder(
            spec=self._config.spec,
            smc=self._config.smc,
            options=self._options,
            emit=self._emit,
        ).build()
        population, initial_new = self._initialize(support)
        kernel = LazyGuidedProposalKernel(
            config=self._config,
            options=self._options,
            support=support,
            scorer=self._candidate_scorer,
            generator=self._generator,
            emit=self._emit,
        )
        diagnostics: list[LazyStageDiagnostic] = []
        log_path_z_estimate = 0.0
        first_exact_stage = (
            0
            if any(self._states[trace].score.exact_program for trace in population.traces)
            else None
        )
        self._emit(
            "importance.lazy.population.initialized",
            message="prior particles sampled before any complete-state product was built",
            particles=len(population.traces),
            unique_programs=len(set(population.traces)),
            newly_evaluated_programs=initial_new,
            support_states=support.support_states,
        )

        for stage in range(1, self._config.smc.iterations + 1):
            beta = self._options.beta_max * stage / self._config.smc.iterations
            weights_before = torch.tensor(population.weights, dtype=torch.float64)
            ess_before = effective_sample_size(weights_before)
            relative_ess = relative_effective_sample_size(weights_before)
            ancestors, base_weights, resampled = self._ancestors(population, relative_ess)
            ancestor_states = tuple(self._states[trace] for trace in ancestors)
            proposals = await kernel.sample_many(ancestor_states, stage=stage, beta=beta)
            states, newly_evaluated = self._realize(
                support,
                tuple(proposal.trace for proposal in proposals),
            )
            log_q = torch.tensor(
                [proposal.log_q_mixture for proposal in proposals], dtype=torch.float64
            )
            log_gamma = torch.tensor(
                [
                    state.log_prior
                    - beta * float(self._config.smc.loss_scale) * state.score.total_loss
                    for state in states
                ],
                dtype=torch.float64,
            )
            incremental = log_gamma - log_q
            normalized = normalize_log_weights(torch.log(base_weights) + incremental)
            log_path_z_estimate += normalized.log_normalizer
            population = LazyImportancePopulation(
                traces=tuple(proposal.trace for proposal in proposals),
                weights=tuple(float(value) for value in normalized.weights.tolist()),
                ancestor_traces=ancestors,
                log_q_mixture=tuple(float(value) for value in log_q.tolist()),
                log_incremental_weight=tuple(float(value) for value in incremental.tolist()),
                cloned=tuple(proposal.cloned for proposal in proposals),
            )
            exact_mass = sum(
                weight
                for trace, weight in zip(population.traces, population.weights, strict=True)
                if self._states[trace].score.exact_program
            )
            exact_particles = sum(
                self._states[trace].score.exact_program for trace in population.traces
            )
            if first_exact_stage is None and exact_particles:
                first_exact_stage = stage
            diagnostic = LazyStageDiagnostic(
                stage=stage,
                beta=beta,
                ess_before=ess_before,
                relative_ess_before=relative_ess,
                resampled=resampled,
                clones=sum(proposal.cloned for proposal in proposals),
                unique_programs=len(set(population.traces)),
                exact_particles=exact_particles,
                exact_particle_mass=exact_mass,
                ess_after=effective_sample_size(normalized.weights),
                newly_evaluated_programs=newly_evaluated,
                cumulative_evaluated_programs=len(self._states),
                mean_log_q=float(log_q.mean().item()),
                min_log_q=float(log_q.min().item()),
                max_log_importance_ratio=float(incremental.max().item()),
                log_path_z_estimate=log_path_z_estimate,
            )
            diagnostics.append(diagnostic)
            self._emit(
                "importance.lazy.stage.completed",
                message="lazy importance-SMC stage completed",
                stage=diagnostic.stage,
                beta=diagnostic.beta,
                resampled=diagnostic.resampled,
                exact_particle_mass=diagnostic.exact_particle_mass,
                newly_evaluated_programs=diagnostic.newly_evaluated_programs,
                cumulative_evaluated_programs=diagnostic.cumulative_evaluated_programs,
            )

        result = self._result(
            support,
            population,
            kernel,
            tuple(diagnostics),
            first_exact_stage=first_exact_stage,
            log_path_z_estimate=log_path_z_estimate,
        )
        self._emit(
            "importance.lazy.completed",
            message="visit-only importance-SMC completed without exact enumeration",
            support_states=result.support_states,
            evaluated_programs=result.search.evaluated_programs,
            exact_found=result.search.exact_found,
            reference_metrics_available=False,
        )
        return result

    def _initialize(
        self,
        support: FactorizedImportanceSupport,
    ) -> tuple[LazyImportancePopulation, int]:
        traces = tuple(
            sample_prior_trace(
                support,
                cost_scale=float(self._config.smc.cost_scale),
                generator=self._generator,
            )
            for _ in range(self._config.smc.particles)
        )
        _, newly = self._realize(support, traces)
        count = len(traces)
        weights = tuple(1.0 / count for _ in traces)
        return (
            LazyImportancePopulation(
                traces=traces,
                weights=weights,
                ancestor_traces=traces,
                log_q_mixture=tuple(0.0 for _ in traces),
                log_incremental_weight=tuple(0.0 for _ in traces),
                cloned=tuple(False for _ in traces),
            ),
            newly,
        )

    def _realize(
        self,
        support: FactorizedImportanceSupport,
        traces: tuple[ConstructionTrace, ...],
    ) -> tuple[tuple[LazyImportanceState, ...], int]:
        missing = tuple(dict.fromkeys(trace for trace in traces if trace not in self._states))
        programs = []
        metadata = []
        for trace in missing:
            family = support.family(trace.hypothesis_index)
            fillings = trace_fillings(family, trace)
            program = assemble_program(
                family.hypothesis,
                {filling.hole_name: filling.expression for filling in fillings},
                allowed_integer_constants=self._config.spec.integer_constants,
            )
            programs.append(program)
            metadata.append((trace, family, fillings, program))
        results: list[ScoredProgram | RejectedProgram] = []
        size = self._options.score_batch_size
        for start in range(0, len(programs), size):
            results.extend(self._scorer.score_batch(programs[start : start + size]))
        for (trace, family, fillings, program), score in zip(
            metadata,
            results,
            strict=True,
        ):
            if isinstance(score, RejectedProgram):
                raise RuntimeError(
                    "factorized support admitted a sampled program rejected by the semantic "
                    f"core: {score.reason}"
                )
            expected_cost = family.base_cost + trace_hole_cost(family, trace)
            if score.cost != expected_cost:
                raise RuntimeError(
                    "factorized structural cost disagrees with semantic scorer: "
                    f"factorized={expected_cost}, scorer={score.cost}"
                )
            self._states[trace] = LazyImportanceState(
                trace=trace,
                family=family.hypothesis.kind.value,
                fillings=fillings,
                program=program,
                key=canonical_key(program),
                score=score,
                log_prior=trace_log_prior(
                    support,
                    trace,
                    cost_scale=float(self._config.smc.cost_scale),
                ),
            )
        return tuple(self._states[trace] for trace in traces), len(missing)

    def _ancestors(
        self,
        population: LazyImportancePopulation,
        relative_ess: float,
    ) -> tuple[tuple[ConstructionTrace, ...], torch.Tensor, bool]:
        weights = torch.tensor(population.weights, dtype=torch.float64)
        count = len(population.traces)
        if relative_ess >= self._config.smc.ess_threshold:
            return population.traces, weights, False
        slots = systematic_resample(weights, generator=self._generator)
        traces = tuple(population.traces[int(slot)] for slot in slots.tolist())
        return traces, torch.full((count,), 1.0 / count, dtype=torch.float64), True

    def _result(
        self,
        support: FactorizedImportanceSupport,
        population: LazyImportancePopulation,
        kernel: LazyGuidedProposalKernel,
        diagnostics: tuple[LazyStageDiagnostic, ...],
        *,
        first_exact_stage: int | None,
        log_path_z_estimate: float,
    ) -> LazyImportanceSMCResult:
        empirical: dict[ConstructionTrace, float] = {}
        for trace, weight in zip(population.traces, population.weights, strict=True):
            empirical[trace] = empirical.get(trace, 0.0) + weight
        beta = self._options.beta_max
        best_trace = max(
            empirical,
            key=lambda trace: (
                self._states[trace].log_prior
                - beta
                * float(self._config.smc.loss_scale)
                * self._states[trace].score.total_loss,
                self._states[trace].key,
            ),
        )
        best = self._states[best_trace]
        visited_best = min(
            self._states.values(),
            key=lambda state: (
                not state.score.exact_program,
                -(
                    state.log_prior
                    - beta
                    * float(self._config.smc.loss_scale)
                    * state.score.total_loss
                ),
                state.key,
            ),
        )
        final_exact_mass = sum(
            mass for trace, mass in empirical.items() if self._states[trace].score.exact_program
        )
        exact_sampled = sum(state.score.exact_program for state in self._states.values())
        guide = kernel.final_family_guide()
        family_positions = {
            family.hypothesis_index: index for index, family in enumerate(support.families)
        }
        family_particle_mass = {
            family.hypothesis_index: sum(
                mass
                for trace, mass in empirical.items()
                if trace.hypothesis_index == family.hypothesis_index
            )
            for family in support.families
        }
        families = tuple(
            LazyFamilySummary(
                family=family.hypothesis.kind.value,
                support_states=family.support_count,
                prior_mass=1.0 / len(support.families),
                deduction_guide_mass=float(
                    guide[family_positions[family.hypothesis_index]].item()
                ),
                particle_mass=family_particle_mass[family.hypothesis_index],
            )
            for family in support.families
        )
        particles = tuple(
            LazyImportanceParticle(
                particle_index=index,
                trace=trace,
                ancestor_trace=population.ancestor_traces[index],
                family=self._states[trace].family,
                program=self._states[trace].program,
                total_loss=self._states[trace].score.total_loss,
                cost=self._states[trace].score.cost,
                exact_program=self._states[trace].score.exact_program,
                weight=population.weights[index],
                log_q_mixture=population.log_q_mixture[index],
                log_incremental_weight=population.log_incremental_weight[index],
                cloned=population.cloned[index],
            )
            for index, trace in enumerate(population.traces)
        )
        return LazyImportanceSMCResult(
            mode="importance-smc",
            execution="lazy-factorized",
            probabilistic_claim=LAZY_IMPORTANCE_SMC_CLAIM,
            proposal_source=kernel.source,
            deduction_mix=self._options.deduction_mix,
            family_deduction_mix=self._options.resolved_family_deduction_mix,
            hole_deduction_mix=self._options.resolved_hole_deduction_mix,
            deduction_strength=self._options.deduction_strength,
            llm_energy_normalization=self._options.llm_energy_normalization,
            conditioned_skeleton=support.conditioned_skeleton,
            multi_family=support.multi_family,
            support_semantics=(
                "typed construction traces; cross-family AST aliases are retained as "
                "distinct latent traces"
            ),
            support_materialized=False,
            support_states=support.support_states,
            generated_hypotheses=len(support.induction.hypotheses),
            viable_hypotheses=len(support.families),
            refuted_hypotheses=sum(not report.viable for report in support.deductions),
            hole_catalogs=support.hole_catalogs,
            families=families,
            best_visited=LazyStateSummary(
                family=visited_best.family,
                program=visited_best.program,
                total_loss=visited_best.score.total_loss,
                cost=visited_best.score.cost,
                exact_program=visited_best.score.exact_program,
                particle_mass=empirical.get(visited_best.trace, 0.0),
            ),
            sampled_best=LazyStateSummary(
                family=best.family,
                program=best.program,
                total_loss=best.score.total_loss,
                cost=best.score.cost,
                exact_program=best.score.exact_program,
                particle_mass=empirical[best_trace],
            ),
            search=LazySearchMetrics(
                exact_found=bool(exact_sampled),
                final_exact_particle_mass=final_exact_mass,
                exact_sampled_programs=exact_sampled,
                evaluated_programs=len(self._states),
                evaluated_fraction_of_support=len(self._states) / support.support_states,
                first_exact_stage=first_exact_stage,
                log_path_z_estimate=log_path_z_estimate,
            ),
            reference=None,
            stages=diagnostics,
            final_particles=particles,
            scored_candidates=kernel.scored_candidates,
            max_scored_candidates=self._options.max_scored_candidates,
            score_ledger=kernel.score_ledger,
        )

    def _emit(self, name: str, *, message: str, level: str = "info", **data: Any) -> None:
        if self._logger is not None:
            self._logger.event(name, message=message, level=level, **data)

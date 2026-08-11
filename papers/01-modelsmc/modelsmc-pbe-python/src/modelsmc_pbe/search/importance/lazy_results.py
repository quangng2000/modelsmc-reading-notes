"""Result assembly for visit-only importance-SMC."""

from __future__ import annotations

from collections.abc import Mapping

from modelsmc_pbe.config import ExperimentConfig

from .joint_semantic import LazyJointSemanticProposalKernel
from .joint_semantic.ledger import (
    trace_identity,
    validate_final_semantic_selections,
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
from .records import JOINT_SEMANTIC_IMPORTANCE_SMC_CLAIM, ImportanceSMCOptions


def assemble_lazy_result(
    *,
    config: ExperimentConfig,
    options: ImportanceSMCOptions,
    support: FactorizedImportanceSupport,
    population: LazyImportancePopulation,
    kernel: LazyGuidedProposalKernel | LazyJointSemanticProposalKernel,
    diagnostics: tuple[LazyStageDiagnostic, ...],
    states: Mapping[ConstructionTrace, LazyImportanceState],
    first_exact_stage: int | None,
    log_path_z_estimate: float,
) -> LazyImportanceSMCResult:
    """Build the public result without expanding the engine orchestration module."""

    empirical: dict[ConstructionTrace, float] = {}
    for trace, weight in zip(population.traces, population.weights, strict=True):
        empirical[trace] = empirical.get(trace, 0.0) + weight
    beta = options.beta_max
    best_trace = max(
        empirical,
        key=lambda trace: (
            states[trace].log_prior
            - beta * float(config.smc.loss_scale) * states[trace].score.total_loss,
            states[trace].key,
        ),
    )
    best = states[best_trace]
    visited_best = min(
        states.values(),
        key=lambda state: (
            not state.score.exact_program,
            -(state.log_prior - beta * float(config.smc.loss_scale) * state.score.total_loss),
            state.key,
        ),
    )
    final_exact_mass = sum(
        mass for trace, mass in empirical.items() if states[trace].score.exact_program
    )
    exact_sampled = sum(state.score.exact_program for state in states.values())
    if isinstance(kernel, LazyGuidedProposalKernel):
        guided = True
        guide = kernel.final_family_guide()
        proposal = None
        semantic_ledger = None
    else:
        guided = False
        guide = None
        proposal = kernel.final_family_proposal()
        semantic_ledger = kernel.semantic_score_ledger
        validate_final_semantic_selections(
            semantic_ledger,
            iterations=config.smc.iterations,
            final_traces=tuple(
                trace_identity(trace.hypothesis_index, trace.filling_indices)
                for trace in population.traces
            ),
            final_ancestors=tuple(
                trace_identity(trace.hypothesis_index, trace.filling_indices)
                for trace in population.ancestor_traces
            ),
            final_log_q=population.log_q_mixture,
        )
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
            deduction_guide_mass=(
                None
                if guide is None
                else float(guide[family_positions[family.hypothesis_index]].item())
            ),
            particle_mass=family_particle_mass[family.hypothesis_index],
            proposal_mass=(
                None
                if proposal is None
                else float(proposal[family_positions[family.hypothesis_index]].item())
            ),
        )
        for family in support.families
    )
    particles = tuple(
        LazyImportanceParticle(
            particle_index=index,
            trace=trace,
            ancestor_trace=population.ancestor_traces[index],
            family=states[trace].family,
            program=states[trace].program,
            total_loss=states[trace].score.total_loss,
            cost=states[trace].score.cost,
            exact_program=states[trace].score.exact_program,
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
        probabilistic_claim=(
            LAZY_IMPORTANCE_SMC_CLAIM
            if guided
            else JOINT_SEMANTIC_IMPORTANCE_SMC_CLAIM
        ),
        proposal_source=kernel.source,
        deduction_mix=options.deduction_mix if guided else None,
        family_deduction_mix=options.resolved_family_deduction_mix if guided else None,
        hole_deduction_mix=options.resolved_hole_deduction_mix if guided else None,
        deduction_strength=options.deduction_strength if guided else None,
        llm_energy_normalization=options.llm_energy_normalization if guided else None,
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
            evaluated_programs=len(states),
            evaluated_fraction_of_support=len(states) / support.support_states,
            first_exact_stage=first_exact_stage,
            log_path_z_estimate=log_path_z_estimate,
        ),
        reference=None,
        stages=diagnostics,
        final_particles=particles,
        scored_candidates=kernel.scored_candidates,
        max_scored_candidates=options.max_scored_candidates,
        score_ledger=kernel.score_ledger,
        proposal_strategy=options.proposal_strategy,
        proposal_epsilon=options.proposal_epsilon,
        semantic_scale=None if guided else options.semantic_scale,
        semantic_slate_size=(
            None if semantic_ledger is None else semantic_ledger.slate_traces
        ),
        semantic_score_kind=(
            None if guided else "symmetrized-final-label-log-odds"
        ),
        semantic_score_ledger=semantic_ledger,
    )

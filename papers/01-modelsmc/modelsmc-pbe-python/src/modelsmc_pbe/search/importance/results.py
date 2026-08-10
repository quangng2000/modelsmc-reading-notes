"""Exact-reference comparison and public importance-SMC result assembly."""

from __future__ import annotations

import torch

from modelsmc_pbe.proposals import LLMEnergyNormalization

from .records import (
    IMPORTANCE_SMC_CLAIM,
    ImportanceFamilySummary,
    ImportanceHypothesisSummary,
    ImportanceParticle,
    ImportancePopulation,
    ImportanceReferenceMetrics,
    ImportanceSMCResult,
    ImportanceStageDiagnostic,
    ImportanceState,
    ImportanceStateSummary,
    ImportanceSupport,
)
from .score_ledger import LLMScoreWaveLedger
from .target import FiniteImportanceTarget


def _empirical(population: ImportancePopulation, state_count: int) -> torch.Tensor:
    probabilities = torch.zeros(state_count, dtype=torch.float64)
    indices = torch.tensor(population.state_indices, dtype=torch.int64)
    weights = torch.tensor(population.weights, dtype=torch.float64)
    probabilities.scatter_add_(0, indices, weights)
    return probabilities / probabilities.sum()


def _best_state(
    states: tuple[ImportanceState, ...],
    empirical: torch.Tensor,
    final_log_target: torch.Tensor,
) -> ImportanceState:
    present = [state for state in states if float(empirical[state.state_index].item()) > 0]
    return min(
        present,
        key=lambda state: (
            -float(final_log_target[state.state_index].item()),
            state.key,
        ),
    )


def assemble_importance_result(
    *,
    support: ImportanceSupport,
    population: ImportancePopulation,
    target: FiniteImportanceTarget,
    beta_max: float,
    proposal_source: str,
    deduction_mix: float,
    deduction_strength: float,
    llm_energy_normalization: LLMEnergyNormalization,
    deduction_guide: torch.Tensor,
    stages: tuple[ImportanceStageDiagnostic, ...],
    log_path_z_estimate: float,
    log_path_z_reference: float,
    scored_candidates: int,
    max_scored_candidates: int,
    score_ledger: tuple[LLMScoreWaveLedger, ...],
) -> ImportanceSMCResult:
    """Project the final path particles onto programs and compare exactly."""

    exact = target.distribution(beta_max).weights
    prior = target.distribution(0.0).weights
    empirical = _empirical(population, len(support.states))
    particle_exact_mass = float(empirical[target.exact_mask].sum().item())
    enumeration_exact_mass = float(exact[target.exact_mask].sum().item())
    particle_mean_loss = float(torch.dot(empirical, target.losses_cpu).item())
    enumeration_mean_loss = float(torch.dot(exact, target.losses_cpu).item())
    particle_mean_cost = float(torch.dot(empirical, target.costs_cpu).item())
    enumeration_mean_cost = float(torch.dot(exact, target.costs_cpu).item())
    total_variation = float((0.5 * torch.abs(empirical - exact).sum()).item())
    reference = ImportanceReferenceMetrics(
        particle_exact_mass=particle_exact_mass,
        enumeration_exact_mass=enumeration_exact_mass,
        particle_mean_loss=particle_mean_loss,
        enumeration_mean_loss=enumeration_mean_loss,
        particle_mean_cost=particle_mean_cost,
        enumeration_mean_cost=enumeration_mean_cost,
        total_variation_distance=total_variation,
        log_path_z_estimate=log_path_z_estimate,
        log_path_z_enumeration=log_path_z_reference,
        log_path_z_error=log_path_z_estimate - log_path_z_reference,
        absolute_log_path_z_error=abs(log_path_z_estimate - log_path_z_reference),
    )
    best = _best_state(support.states, empirical, target.log_unnormalized(beta_max))
    summary = ImportanceStateSummary(
        state_index=best.state_index,
        family=best.family,
        program=best.program,
        total_loss=best.score.total_loss,
        cost=best.score.cost,
        exact_program=best.score.exact_program,
        empirical_mass=float(empirical[best.state_index].item()),
        enumeration_probability=float(exact[best.state_index].item()),
    )
    particles = tuple(
        ImportanceParticle(
            particle_index=index,
            state_index=state_index,
            ancestor_state_index=population.ancestor_state_indices[index],
            family=support.states[state_index].family,
            program=support.states[state_index].program,
            total_loss=support.states[state_index].score.total_loss,
            cost=support.states[state_index].score.cost,
            exact_program=support.states[state_index].score.exact_program,
            weight=population.weights[index],
            log_q_mixture=population.log_q_mixture[index],
            log_incremental_weight=population.log_incremental_weight[index],
            cloned=population.cloned[index],
        )
        for index, state_index in enumerate(population.state_indices)
    )
    families = tuple(
        ImportanceFamilySummary(
            family=family.hypothesis.kind.value,
            states=len(family.state_indices),
            exact_programs=sum(
                support.states[index].score.exact_program for index in family.state_indices
            ),
            prior_mass=float(prior[list(family.state_indices)].sum().item()),
            deduction_guide_mass=float(
                deduction_guide[list(family.state_indices)].sum().item()
            ),
            posterior_mass=float(exact[list(family.state_indices)].sum().item()),
            particle_mass=float(empirical[list(family.state_indices)].sum().item()),
        )
        for family in support.families
    )
    state_counts = {
        family.hypothesis_index: len(family.state_indices)
        for family in support.families
    }
    hypotheses = tuple(
        ImportanceHypothesisSummary(
            family=report.hypothesis.kind.value,
            viable=report.viable,
            states=state_counts.get(index, 0),
            refutation_kind=(
                None if report.refutation is None else report.refutation.kind.value
            ),
            refutation_sources=(
                () if report.refutation is None else report.refutation.source_examples
            ),
            refutation_detail=(
                None if report.refutation is None else report.refutation.detail
            ),
        )
        for index, report in enumerate(support.deductions)
    )
    return ImportanceSMCResult(
        mode="importance-smc",
        probabilistic_claim=IMPORTANCE_SMC_CLAIM,
        proposal_source=proposal_source,
        deduction_mix=deduction_mix,
        deduction_strength=deduction_strength,
        llm_energy_normalization=llm_energy_normalization,
        conditioned_skeleton=support.conditioned_skeleton,
        multi_family=support.multi_family,
        aliased_programs=support.aliased_programs,
        support_states=len(support.states),
        generated_hypotheses=len(support.induction.hypotheses),
        viable_hypotheses=len(support.families),
        refuted_hypotheses=sum(not report.viable for report in support.deductions),
        exact_programs=support.exact_programs,
        deduction_guide_exact_mass=float(
            deduction_guide[target.exact_mask].sum().item()
        ),
        hole_catalogs=support.hole_catalogs,
        hypotheses=hypotheses,
        families=families,
        sampled_best=summary,
        reference=reference,
        stages=stages,
        final_particles=particles,
        scored_candidates=scored_candidates,
        max_scored_candidates=max_scored_candidates,
        score_ledger=score_ledger,
    )

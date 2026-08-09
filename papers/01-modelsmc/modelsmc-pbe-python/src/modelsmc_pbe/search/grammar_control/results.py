"""Exact reference comparison and final grammar-SMC result assembly."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from modelsmc_pbe.config import SMCConfig
from modelsmc_pbe.domain.ast import clone_program
from modelsmc_pbe.search.grammar_control.records import (
    CALIBRATED_CLAIM,
    GrammarParticle,
    GrammarReferenceMetrics,
    GrammarSMCOptions,
    GrammarSMCResult,
    GrammarStageDiagnostic,
    GrammarStateSummary,
    GrammarSupport,
)
from modelsmc_pbe.search.grammar_control.target import FiniteGrammarTarget
from modelsmc_pbe.search.grammar_control.transition import GrammarPopulation


@dataclass(frozen=True, slots=True)
class ReferenceSnapshot:
    """Exact metrics plus distributions needed to describe sampled states."""

    metrics: GrammarReferenceMetrics
    empirical_probabilities: Tensor
    enumeration_probabilities: Tensor


def compare_with_enumeration(
    population: GrammarPopulation,
    *,
    target: FiniteGrammarTarget,
    beta_max: float,
    loss_scale: float,
    log_z_estimate: float,
) -> ReferenceSnapshot:
    """Compare the particle population with the normalized finite target."""

    exact_distribution = target.exact_distribution(
        beta=beta_max,
        loss_scale=loss_scale,
    )
    enumeration = exact_distribution.weights
    empirical = torch.zeros(enumeration.numel(), dtype=torch.float64)
    empirical.scatter_add_(0, population.state_indices, population.weights)
    empirical /= empirical.sum()

    enumeration_exact_mass = float(enumeration[target.exact_mask].sum().item())
    particle_exact_mass = float(empirical[target.exact_mask].sum().item())
    enumeration_mean_loss = float(torch.dot(enumeration, target.losses_cpu).item())
    particle_mean_loss = float(torch.dot(empirical, target.losses_cpu).item())
    total_variation = float((0.5 * torch.abs(empirical - enumeration).sum()).item())
    log_z_enumeration = exact_distribution.log_normalizer
    log_z_error = log_z_estimate - log_z_enumeration
    metrics = GrammarReferenceMetrics(
        particle_exact_mass=particle_exact_mass,
        enumeration_exact_mass=enumeration_exact_mass,
        particle_mean_loss=particle_mean_loss,
        enumeration_mean_loss=enumeration_mean_loss,
        log_z_estimate=log_z_estimate,
        log_z_enumeration=log_z_enumeration,
        log_z_error=log_z_error,
        absolute_log_z_error=abs(log_z_error),
        total_variation_distance=total_variation,
    )
    return ReferenceSnapshot(
        metrics=metrics,
        empirical_probabilities=empirical,
        enumeration_probabilities=enumeration,
    )


def assemble_result(
    *,
    support: GrammarSupport,
    population: GrammarPopulation,
    target: FiniteGrammarTarget,
    smc: SMCConfig,
    options: GrammarSMCOptions,
    log_z_estimate: float,
    stages: tuple[GrammarStageDiagnostic, ...],
) -> GrammarSMCResult:
    """Build stable public records from the final tensor population."""

    reference = compare_with_enumeration(
        population,
        target=target,
        beta_max=options.beta_max,
        loss_scale=float(smc.loss_scale),
        log_z_estimate=log_z_estimate,
    )
    best = _best_state_summary(support, population, reference)
    particles = tuple(
        _particle_record(index, state_index, support, population)
        for index, state_index in enumerate(population.state_indices.tolist())
    )
    return GrammarSMCResult(
        mode="grammar-smc",
        probabilistic_claim=CALIBRATED_CLAIM,
        skeleton=options.skeleton,
        grammar_states=len(support.states),
        enumerated_asts=support.enumerated_asts,
        rejected_asts=support.rejected_asts,
        over_cost_asts=support.over_cost_asts,
        exact_programs=support.exact_programs,
        beta_max=options.beta_max,
        sampled_best=best,
        reference=reference.metrics,
        stages=stages,
        final_particles=particles,
    )


def _best_state_summary(
    support: GrammarSupport,
    population: GrammarPopulation,
    reference: ReferenceSnapshot,
) -> GrammarStateSummary:
    sampled_indices = sorted(set(int(index) for index in population.state_indices.tolist()))
    best_index = min(
        sampled_indices,
        key=lambda index: (
            support.states[index].target_loss,
            support.states[index].score.cost,
            support.states[index].key,
        ),
    )
    state = support.states[best_index]
    return GrammarStateSummary(
        state_index=best_index,
        program=clone_program(state.program),
        total_loss=state.score.total_loss,
        target_loss=state.target_loss,
        cost=state.score.cost,
        exact_program=state.score.exact_program,
        empirical_mass=float(reference.empirical_probabilities[best_index].item()),
        enumeration_probability=float(reference.enumeration_probabilities[best_index].item()),
    )


def _particle_record(
    particle_index: int,
    state_index: int,
    support: GrammarSupport,
    population: GrammarPopulation,
) -> GrammarParticle:
    state = support.states[int(state_index)]
    return GrammarParticle(
        particle_index=particle_index,
        state_index=int(state_index),
        program=clone_program(state.program),
        total_loss=state.score.total_loss,
        target_loss=state.target_loss,
        cost=state.score.cost,
        exact_program=state.score.exact_program,
        weight=float(population.weights[particle_index].item()),
    )

"""One annealed SMC transition and its independent-MH rejuvenation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from modelsmc_pbe.config import SMCConfig
from modelsmc_pbe.search.grammar_control.records import (
    GrammarSMCOptions,
    GrammarStageDiagnostic,
)
from modelsmc_pbe.search.grammar_control.target import FiniteGrammarTarget
from modelsmc_pbe.smc import (
    categorical_sample,
    effective_sample_size,
    normalize_log_weights,
    systematic_resample,
    tempered_log_potential,
)


@dataclass(frozen=True, slots=True)
class GrammarPopulation:
    """State indices and normalized weights for one particle population."""

    state_indices: Tensor
    weights: Tensor


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """Population and diagnostics produced by one annealing stage."""

    population: GrammarPopulation
    diagnostic: GrammarStageDiagnostic
    log_z_estimate: float


def prior_independent_log_acceptance(
    current_losses: Tensor,
    proposed_losses: Tensor,
    *,
    beta: float,
    loss_scale: float,
) -> Tensor:
    """Return the independent-MH log acceptance for prior proposals.

    The target is ``p0(e) exp(-beta * loss_scale * loss(e))`` and the proposal
    is ``p0`` itself, so the prior and proposal factors cancel.
    """

    if current_losses.shape != proposed_losses.shape:
        raise ValueError("current and proposed losses must have the same shape")
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and nonnegative")
    if not math.isfinite(loss_scale) or loss_scale < 0.0:
        raise ValueError("loss_scale must be finite and nonnegative")
    current = current_losses.detach().to(device="cpu", dtype=torch.float64)
    proposed = proposed_losses.detach().to(device="cpu", dtype=torch.float64)
    if bool(torch.any(~torch.isfinite(current)).item()) or bool(
        torch.any(~torch.isfinite(proposed)).item()
    ):
        raise ValueError("losses must be finite")
    if bool(torch.any(current < 0).item()) or bool(torch.any(proposed < 0).item()):
        raise ValueError("losses must be nonnegative")
    ratio = -beta * loss_scale * (proposed - current)
    return torch.minimum(ratio, torch.zeros_like(ratio))


def initial_population(
    target: FiniteGrammarTarget,
    *,
    particle_count: int,
    generator: torch.Generator,
) -> GrammarPopulation:
    """Sample initial particles from the normalized Occam prior."""

    return GrammarPopulation(
        state_indices=categorical_sample(
            target.prior_probabilities,
            particle_count,
            generator=generator,
        ),
        weights=torch.full(
            (particle_count,),
            1.0 / particle_count,
            dtype=torch.float64,
        ),
    )


def advance_stage(
    population: GrammarPopulation,
    *,
    stage: int,
    beta_previous: float,
    beta_current: float,
    log_z_estimate: float,
    target: FiniteGrammarTarget,
    smc: SMCConfig,
    options: GrammarSMCOptions,
    generator: torch.Generator,
) -> StageOutcome:
    """Reweight, optionally resample, then rejuvenate one population."""

    particle_count = int(population.state_indices.numel())
    selected_losses = target.losses_device[
        population.state_indices.to(device=target.losses_device.device)
    ]
    incremental = tempered_log_potential(
        selected_losses,
        beta_previous=beta_previous,
        beta_current=beta_current,
        loss_scale=float(smc.loss_scale),
    )
    incremental_cpu = incremental.detach().to(device="cpu").to(dtype=torch.float64)
    normalized = normalize_log_weights(torch.log(population.weights) + incremental_cpu)
    weights = normalized.weights
    new_log_z = log_z_estimate + normalized.log_normalizer
    ess = effective_sample_size(weights)
    relative_ess = ess / particle_count
    state_indices = population.state_indices
    resampled = relative_ess < float(smc.ess_threshold)
    if resampled:
        state_indices = state_indices[systematic_resample(weights, generator=generator)]
        weights = torch.full(
            (particle_count,),
            1.0 / particle_count,
            dtype=torch.float64,
        )

    state_indices, accepted, attempts = _rejuvenate(
        state_indices,
        target=target,
        beta=beta_current,
        loss_scale=float(smc.loss_scale),
        moves=options.moves_per_stage,
        generator=generator,
    )
    exact_values = target.exact_mask[state_indices].to(dtype=torch.float64)
    selected_losses_cpu = target.losses_cpu[state_indices]
    diagnostic = GrammarStageDiagnostic(
        stage=stage,
        beta_previous=beta_previous,
        beta_current=beta_current,
        ess=ess,
        relative_ess=relative_ess,
        resampled=resampled,
        mh_accepted=accepted,
        mh_attempts=attempts,
        log_z_estimate=new_log_z,
        particle_exact_mass=float(torch.dot(weights, exact_values).item()),
        particle_mean_loss=float(torch.dot(weights, selected_losses_cpu).item()),
    )
    return StageOutcome(
        population=GrammarPopulation(state_indices=state_indices, weights=weights),
        diagnostic=diagnostic,
        log_z_estimate=new_log_z,
    )


def _rejuvenate(
    state_indices: Tensor,
    *,
    target: FiniteGrammarTarget,
    beta: float,
    loss_scale: float,
    moves: int,
    generator: torch.Generator,
) -> tuple[Tensor, int, int]:
    accepted_total = 0
    attempts = int(state_indices.numel()) * moves
    current = state_indices
    for _move in range(moves):
        proposals = categorical_sample(
            target.prior_probabilities,
            int(current.numel()),
            generator=generator,
        )
        log_acceptance = prior_independent_log_acceptance(
            target.losses_cpu[current],
            target.losses_cpu[proposals],
            beta=beta,
            loss_scale=loss_scale,
        )
        uniforms = torch.rand(
            current.numel(),
            dtype=torch.float64,
            generator=generator,
        )
        accepted = torch.log(uniforms) <= log_acceptance
        current = torch.where(accepted, proposals, current)
        accepted_total += int(accepted.sum().item())
    return current, accepted_total, attempts

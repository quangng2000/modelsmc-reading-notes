"""Objective weighting, ranking, and ancestor selection."""

from __future__ import annotations

from dataclasses import replace

import torch

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.search.models import ParticleRecord
from modelsmc_pbe.smc import (
    gibbs_log_target,
    normalize_log_weights,
    relative_effective_sample_size,
    systematic_resample,
)


def score_weights(
    config: ExperimentConfig,
    particles: list[ParticleRecord],
) -> list[ParticleRecord]:
    """Replace each particle's allocation weight using the fixed objective."""

    losses = torch.tensor(
        [particle.score.total_loss for particle in particles], dtype=torch.float64
    )
    costs = torch.tensor(
        [particle.score.cost for particle in particles], dtype=torch.float64
    )
    log_targets = gibbs_log_target(
        losses,
        costs,
        beta=1.0,
        loss_scale=float(config.smc.loss_scale),
        cost_scale=float(config.smc.cost_scale),
    )
    normalized = normalize_log_weights(log_targets)
    return [
        replace(particle, weight=float(normalized.weights[index].item()))
        for index, particle in enumerate(particles)
    ]


def is_better(candidate: ParticleRecord, current: ParticleRecord) -> bool:
    """Order candidates exactly as the original paper-search implementation."""

    if candidate.score.exact_program != current.score.exact_program:
        return candidate.score.exact_program
    if candidate.score.log_target != current.score.log_target:
        return candidate.score.log_target > current.score.log_target
    return candidate.program_key < current.program_key


def select_ancestors(
    particles: list[ParticleRecord],
    *,
    threshold: float,
    generator: torch.Generator,
) -> tuple[list[ParticleRecord], float, bool]:
    """Apply the ESS policy and return selected ancestors plus diagnostics."""

    weights = torch.tensor(
        [particle.weight for particle in particles], dtype=torch.float64
    )
    relative_ess = relative_effective_sample_size(weights)
    if relative_ess >= threshold:
        return list(particles), relative_ess, False
    indices = systematic_resample(weights, generator=generator)
    return [particles[int(index)] for index in indices.tolist()], relative_ess, True

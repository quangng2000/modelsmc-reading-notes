"""Finite Occam prior and annealed target tensors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from modelsmc_pbe.config import SMCConfig
from modelsmc_pbe.runtime import DeviceInfo
from modelsmc_pbe.search.grammar_control.records import GrammarState
from modelsmc_pbe.smc import NormalizedWeights, gibbs_log_target, normalize_log_weights


@dataclass(frozen=True, slots=True)
class FiniteGrammarTarget:
    """Tensor representation of one normalized finite grammar target."""

    losses_device: Tensor
    losses_cpu: Tensor
    prior_probabilities: Tensor
    log_prior: Tensor
    exact_mask: Tensor
    prior_log_normalizer: float

    @classmethod
    def build(
        cls,
        states: Sequence[GrammarState],
        *,
        smc: SMCConfig,
        device: DeviceInfo,
    ) -> FiniteGrammarTarget:
        """Construct the Occam prior and loss tensors without changing order."""

        compute_dtype = torch.float32 if device.resolved == "mps" else torch.float64
        losses_device = torch.tensor(
            [state.target_loss for state in states],
            dtype=compute_dtype,
            device=device.torch_device,
        )
        losses_cpu = torch.tensor(
            [state.target_loss for state in states],
            dtype=torch.float64,
        )
        costs_cpu = torch.tensor(
            [state.score.cost for state in states],
            dtype=torch.float64,
        )
        prior = normalize_log_weights(
            gibbs_log_target(
                torch.zeros_like(costs_cpu),
                costs_cpu,
                beta=0.0,
                loss_scale=float(smc.loss_scale),
                cost_scale=float(smc.cost_scale),
            )
        )
        return cls(
            losses_device=losses_device,
            losses_cpu=losses_cpu,
            prior_probabilities=prior.weights,
            log_prior=torch.log(prior.weights),
            exact_mask=torch.tensor(
                [state.score.exact_program for state in states],
                dtype=torch.bool,
            ),
            prior_log_normalizer=prior.log_normalizer,
        )

    def exact_distribution(
        self,
        *,
        beta: float,
        loss_scale: float,
    ) -> NormalizedWeights:
        """Normalize the exact finite target at one inverse temperature."""

        return normalize_log_weights(self.log_prior - (beta * loss_scale * self.losses_cpu))

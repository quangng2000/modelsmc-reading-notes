"""Exact finite Gibbs targets used by importance-corrected SMC."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from modelsmc_pbe.config import SMCConfig
from modelsmc_pbe.runtime import DeviceInfo
from modelsmc_pbe.search.importance.records import ImportanceState
from modelsmc_pbe.smc import NormalizedWeights, gibbs_log_target, normalize_log_weights


@dataclass(frozen=True, slots=True)
class FiniteImportanceTarget:
    """On-device unnormalized targets plus CPU exact-reference tensors."""

    losses_device: Tensor
    costs_device: Tensor
    losses_cpu: Tensor
    costs_cpu: Tensor
    exact_mask: Tensor
    smc: SMCConfig

    @classmethod
    def build(
        cls,
        states: Sequence[ImportanceState],
        *,
        smc: SMCConfig,
        device: DeviceInfo,
    ) -> FiniteImportanceTarget:
        dtype = torch.float32 if device.resolved == "mps" else torch.float64
        losses = [state.score.total_loss for state in states]
        costs = [state.score.cost for state in states]
        return cls(
            losses_device=torch.tensor(losses, dtype=dtype, device=device.torch_device),
            costs_device=torch.tensor(costs, dtype=dtype, device=device.torch_device),
            losses_cpu=torch.tensor(losses, dtype=torch.float64),
            costs_cpu=torch.tensor(costs, dtype=torch.float64),
            exact_mask=torch.tensor(
                [state.score.exact_program for state in states], dtype=torch.bool
            ),
            smc=smc,
        )

    def log_unnormalized(self, beta: float) -> Tensor:
        """Return the exact CPU-float64 target in fixed support order.

        The device tensors are retained for future batched proposal diagnostics,
        while normalization and reference comparisons deliberately use the same
        CPU-float64 boundary as the shared SMC numerics.
        """

        return gibbs_log_target(
            self.losses_cpu,
            self.costs_cpu,
            beta=beta,
            loss_scale=float(self.smc.loss_scale),
            cost_scale=float(self.smc.cost_scale),
        )

    def distribution(self, beta: float) -> NormalizedWeights:
        """Exactly normalize the finite target at one inverse temperature."""

        return normalize_log_weights(self.log_unnormalized(beta))

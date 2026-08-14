"""Reproducible random-state setup for Python, NumPy, and Torch."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class SeedState:
    seed: int
    deterministic: bool
    cpu_generator: torch.Generator

    def as_dict(self) -> dict[str, int | bool]:
        return {"seed": self.seed, "deterministic": self.deterministic}


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")


def make_cpu_generator(seed: int) -> torch.Generator:
    """Create the CPU generator used for all SMC sampling decisions."""

    _validate_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def seed_everything(seed: int, *, deterministic: bool = True) -> SeedState:
    """Seed all runtime RNGs and return an isolated CPU SMC generator.

    ``PYTHONHASHSEED`` only controls child processes when set after interpreter
    startup, but recording it here still makes subprocess behavior reproducible.
    SMC draws use the returned CPU generator rather than global Torch state.
    """

    _validate_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = deterministic
        cudnn.benchmark = not deterministic

    return SeedState(
        seed=seed,
        deterministic=deterministic,
        cpu_generator=make_cpu_generator(seed),
    )

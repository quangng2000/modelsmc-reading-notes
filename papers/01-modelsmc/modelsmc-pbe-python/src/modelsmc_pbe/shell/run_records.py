"""Search-engine-neutral shell result records."""

from __future__ import annotations

from dataclasses import dataclass

from modelsmc_pbe.shell.output import ProgramResultView


@dataclass(frozen=True, slots=True)
class CompletedRun:
    """Values needed to persist and present one completed synthesis run."""

    persisted_result: object
    final_particles: tuple[object, ...]
    view: ProgramResultView

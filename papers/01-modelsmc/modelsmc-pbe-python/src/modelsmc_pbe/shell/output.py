"""Human-readable terminal presentation for completed synthesis runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import typer

from modelsmc_pbe.domain import ProgramAst


@dataclass(frozen=True, slots=True)
class ProgramResultView:
    """Small presentation model independent of either search engine."""

    mode: str
    exact: bool
    program: ProgramAst
    total_loss: float
    cost: int
    degraded: bool = False


def print_program_result(result: ProgramResultView, run_dir: Path) -> None:
    """Print the stable result summary consumed by existing CLI users."""

    typer.echo(f"[result] mode: {result.mode}")
    typer.echo(f"[result] exact on every example: {str(result.exact).lower()}")
    typer.echo(f"[result] loss={result.total_loss:g} cost={result.cost}")
    if result.degraded:
        typer.secho(
            "[result] degraded: every provider request failed; ancestors were retained",
            fg=typer.colors.YELLOW,
            err=True,
        )
    typer.echo(
        "[result] body AST: "
        f"{json.dumps(result.program, ensure_ascii=False, sort_keys=True)}"
    )
    typer.echo(f"[result] artifacts: {run_dir}")

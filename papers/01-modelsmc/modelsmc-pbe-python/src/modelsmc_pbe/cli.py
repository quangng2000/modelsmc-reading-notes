"""Minimal Typer application wiring for ``modelsmc-pbe``."""

import typer

from modelsmc_pbe.shell.command import synthesize

app = typer.Typer(
    name="modelsmc-pbe",
    help="Run programming-by-example ModelSMC experiments.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def main() -> None:
    """Choose a command; all synthesis runs create durable research artifacts."""


app.command()(synthesize)


if __name__ == "__main__":
    app()

"""CLI for deterministic publication figures from tidy aggregate outputs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from research.figures.pipeline import FigureConfig, build_figures


def _csv(value: str, name: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError(f"{name} must be a nonempty unique CSV list")
    return result


def _seeds(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in _csv(value, "seeds"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Tidy metrics.json or metrics.csv")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--models",
        default="qwen25-coder-3b,qwen25-coder-7b,qwen25-coder-14b,qwen25-coder-32b",
    )
    parser.add_argument("--tasks", default="map-increment,foldr-signed-window")
    parser.add_argument("--arms", default="Q,QD")
    parser.add_argument("--seeds", default="101")
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument(
        "--latex-prefix",
        help="Path used by future \\includegraphics, relative to paper/main.tex",
    )
    args = parser.parse_args(argv)
    result = build_figures(
        args.inputs,
        args.output,
        FigureConfig(
            models=_csv(args.models, "models"),
            tasks=_csv(args.tasks, "tasks"),
            arms=_csv(args.arms, "arms"),
            seeds=_seeds(args.seeds),
            dpi=args.dpi,
            latex_prefix=args.latex_prefix,
        ),
    )
    print(
        f"[figures] rows={result.selected_rows} complete_grid={result.complete_grid} "
        f"paired={result.paired_figure_generated} files={result.files} "
        f"protocol={result.protocol_sha256} output={result.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

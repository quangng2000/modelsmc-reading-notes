"""Build and validate a self-describing publication-figure bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, cast

from research.figures.data import Row, figure_rows, load_rows, select_rows
from research.figures.plots import (
    outcome_matrix,
    paired_q_qd,
    paired_seed_rows,
    provider_work,
)
from research.heldout import write_json_atomic

DEFAULT_MODELS = (
    "qwen25-coder-3b",
    "qwen25-coder-7b",
    "qwen25-coder-14b",
    "qwen25-coder-32b",
)
DEFAULT_TASKS = ("map-increment", "foldr-signed-window")
DEFAULT_ARMS = ("Q", "QD")
DEFAULT_SEEDS = (101,)


@dataclass(frozen=True, slots=True)
class FigureConfig:
    models: tuple[str, ...] = DEFAULT_MODELS
    tasks: tuple[str, ...] = DEFAULT_TASKS
    arms: tuple[str, ...] = DEFAULT_ARMS
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    dpi: int = 450
    latex_prefix: str | None = None

    def __post_init__(self) -> None:
        for name, values in (
            ("models", self.models),
            ("tasks", self.tasks),
            ("arms", self.arms),
            ("seeds", self.seeds),
        ):
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be nonempty and unique")
        if self.dpi < 300:
            raise ValueError("publication PNG resolution must be at least 300 DPI")


@dataclass(frozen=True, slots=True)
class FigureResult:
    output: Path
    protocol_sha256: str
    selected_rows: int
    complete_grid: bool
    paired_figure_generated: bool
    files: int


def _key(row: Row) -> tuple[str, str, str, int] | None:
    model = row.get("model_id")
    task = row.get("task_id")
    arm = row.get("arm")
    seed = row.get("seed")
    if (
        isinstance(model, str)
        and isinstance(task, str)
        and isinstance(arm, str)
        and isinstance(seed, int)
        and not isinstance(seed, bool)
    ):
        return model, task, arm, seed
    return None


def _required_keys(config: FigureConfig) -> list[tuple[str, str, str, int]]:
    return list(product(config.models, config.tasks, config.arms, config.seeds))


def _missing_cells(rows: list[Row], config: FigureConfig) -> list[str]:
    terminal = {
        key
        for row in rows
        if (key := _key(row)) is not None and row.get("status") != "not_started"
    }
    return [
        "--".join((model, task, arm, f"seed-{seed}"))
        for model, task, arm, seed in _required_keys(config)
        if (model, task, arm, seed) not in terminal
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_artifact(path: Path) -> None:
    content = path.read_bytes()
    if len(content) < 100:
        raise ValueError(f"figure artifact is unexpectedly small: {path}")
    if path.suffix == ".pdf" and not content.startswith(b"%PDF-"):
        raise ValueError(f"figure PDF has no PDF header: {path}")
    if path.suffix == ".svg" and b"<svg" not in content[:1000]:
        raise ValueError(f"figure SVG has no svg element: {path}")
    if path.suffix == ".png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"figure PNG has no PNG signature: {path}")


def validate_bundle(output: Path) -> dict[str, Any]:
    """Validate manifest inventory, checksums, and all declared figure formats."""

    manifest_path = output / "figure_manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("figure manifest must be an object")
    checksums = document.get("sha256")
    if not isinstance(checksums, dict):
        raise ValueError("figure manifest has no checksum object")
    expected = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "figure_manifest.json"
    }
    if set(checksums) != expected:
        raise ValueError("figure manifest checksum inventory does not match files")
    for relative, expected_hash in checksums.items():
        path = output / relative
        if _sha256(path) != expected_hash:
            raise ValueError(f"figure checksum mismatch: {relative}")
        if path.suffix in {".pdf", ".svg", ".png"}:
            _validate_artifact(path)
    return document


def build_figures(
    inputs: list[Path],
    output: Path,
    config: FigureConfig | None = None,
) -> FigureResult:
    """Build vector/raster figures and an auditable, checksum-pinned manifest."""

    if config is None:
        config = FigureConfig()
    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing figure directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    all_rows, input_records = load_rows(inputs)
    rows = select_rows(
        all_rows,
        models=config.models,
        tasks=config.tasks,
        arms=config.arms,
        seeds=config.seeds,
    )
    if not rows:
        raise ValueError("no aggregate rows match the declared figure grid")
    protocol_hashes = {str(row["protocol_sha256"]) for row in rows}
    if len(protocol_hashes) != 1:
        raise ValueError("selected figure rows must share exactly one protocol hash")
    protocol_sha256 = next(iter(protocol_hashes))
    missing = _missing_cells(rows, config)
    complete_grid = not missing
    complete_four_checkpoint_gate2 = (
        complete_grid
        and tuple(config.models) == DEFAULT_MODELS
        and tuple(config.tasks) == DEFAULT_TASKS
        and tuple(config.arms) == DEFAULT_ARMS
        and 101 in config.seeds
    )
    pairs = paired_seed_rows(rows)
    latex_prefix = config.latex_prefix or f"generated/figures/{destination.name}"
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        figures: dict[str, dict[str, Any]] = {}
        figures["outcome_matrix"] = {
            "generated": True,
            "paths": outcome_matrix(
                rows,
                models=config.models,
                tasks=config.tasks,
                arms=config.arms,
                seed_count=len(config.seeds),
                output=temporary,
                dpi=config.dpi,
            ),
            "paragraph_attachment": "P-RES-01",
        }
        figures["provider_work"] = {
            "generated": True,
            "paths": provider_work(
                rows,
                models=config.models,
                tasks=config.tasks,
                arms=config.arms,
                output=temporary,
                dpi=config.dpi,
            ),
            "paragraph_attachment": "P-EXP-02",
        }
        if pairs:
            figures["paired_q_vs_qd"] = {
                "generated": True,
                "paths": paired_q_qd(
                    pairs,
                    models=config.models,
                    tasks=config.tasks,
                    output=temporary,
                    dpi=config.dpi,
                ),
                "paragraph_attachment": "P-RES-01",
            }
        else:
            figures["paired_q_vs_qd"] = {
                "generated": False,
                "paths": {},
                "reason": "requires at least two paired Q/QD seeds for a checkpoint-task",
                "paragraph_attachment": "P-RES-01",
            }
        write_json_atomic(temporary / "figure_data.json", figure_rows(rows))
        file_paths = sorted(path for path in temporary.rglob("*") if path.is_file())
        checksums = {
            path.relative_to(temporary).as_posix(): _sha256(path) for path in file_paths
        }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "pipeline": "modelsmc-pbe-publication-figures-v1",
            "protocol_sha256": protocol_sha256,
            "analysis_status": "exploratory",
            "statistical_unit": "seed",
            "uncertainty_policy": "raw cells/counts only; no confidence intervals",
            "config": asdict(config),
            "inputs": input_records,
            "selected_rows": len(rows),
            "required_cells": len(_required_keys(config)),
            "complete_expected_grid": complete_grid,
            "complete_four_checkpoint_gate2": complete_four_checkpoint_gate2,
            "missing_cells": missing,
            "eligible_for_exploratory_paper_insertion": complete_four_checkpoint_gate2,
            "confirmatory_claim_ready": False,
            "paper_policy": (
                "Do not include preliminary figures in manuscript claims until the "
                "complete expected four-checkpoint Gate-2 grid is present."
            ),
            "latex": {
                name: (
                    f"{latex_prefix}/{name}.pdf"
                    if details.get("generated") is True
                    else None
                )
                for name, details in figures.items()
            },
            "figures": figures,
            "sha256": checksums,
        }
        write_json_atomic(temporary / "figure_manifest.json", manifest)
        validate_bundle(temporary)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validated = validate_bundle(destination)
    files = len(cast(dict[str, str], validated["sha256"])) + 1
    return FigureResult(
        output=destination,
        protocol_sha256=protocol_sha256,
        selected_rows=len(rows),
        complete_grid=complete_grid,
        paired_figure_generated=bool(pairs),
        files=files,
    )

"""Small-multiple plots for outcomes, paired seeds, and provider work."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import EngFormatter, MaxNLocator

from research.figures.data import Row, cache_status
from research.figures.style import (
    BLUE,
    DARK,
    GREEN,
    GREY,
    LIGHT_GREY,
    VERMILLION,
    publication_style,
)

_TASK_LABELS = {
    "negative-int-to-bool": "Sign",
    "map-increment": "Map increment",
    "foldr-signed-window": "Signed window",
    "foldr-bounded-square": "Bounded square",
}
_MODEL_LABELS = {
    "qwen25-coder-3b": "3B",
    "qwen25-coder-7b": "7B",
    "qwen25-coder-14b": "14B",
    "qwen25-coder-32b": "32B",
}
_FORMATS = ("pdf", "svg", "png")


def _task_label(task: str) -> str:
    return _TASK_LABELS.get(task, task.replace("-", " ").title())


def _model_label(model: str) -> str:
    return _MODEL_LABELS.get(model, model)


def _save(fig: Any, output: Path, stem: str, dpi: int) -> dict[str, str]:
    paths: dict[str, str] = {}
    for extension in _FORMATS:
        path = output / f"{stem}.{extension}"
        if extension == "pdf":
            metadata = {
                "Creator": "ModelSMC-PBE research.figures",
                "Producer": "Matplotlib",
                "CreationDate": None,
                "ModDate": None,
            }
        elif extension == "svg":
            metadata = {"Creator": "ModelSMC-PBE research.figures", "Date": None}
        else:
            metadata = {"Software": "ModelSMC-PBE research.figures"}
        fig.savefig(
            path,
            format=extension,
            dpi=dpi if extension == "png" else None,
            bbox_inches="tight",
            pad_inches=0.04,
            metadata=metadata,
        )
        paths[extension] = path.name
    plt.close(fig)
    return paths


def _subtitle(seed_count: int) -> str:
    if seed_count == 1:
        return "Exploratory raw cells · n=1 seed · no confidence intervals"
    return f"Exploratory seed-level counts · n={seed_count} seeds · no confidence intervals"


def _group_rows(
    rows: Sequence[Row],
) -> dict[tuple[str, str, str], list[Row]]:
    groups: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for row in rows:
        model = row.get("model_id")
        task = row.get("task_id")
        arm = row.get("arm")
        if isinstance(model, str) and isinstance(task, str) and isinstance(arm, str):
            groups[(model, task, arm)].append(row)
    return groups


def _outcome_cell(rows: Sequence[Row], *, heldout: bool) -> tuple[int, str]:
    if not rows:
        return 0, "—"
    run_failures = sum(row.get("status") != "completed" for row in rows)
    if heldout:
        evaluable = [row for row in rows if row.get("heldout_evaluable") is True]
        exact = sum(row.get("heldout_correct") is True for row in evaluable)
        unavailable = len(rows) - len(evaluable)
        if len(rows) == 1:
            if run_failures:
                return 1, "run\nfailed"
            if unavailable:
                return 4, "n/a"
            return (3, "✓") if exact else (2, "✗")
        code = 3 if exact == len(rows) else 2 if evaluable else 4
        suffix = f"\n{unavailable} n/a" if unavailable else ""
        return code, f"{exact}/{len(evaluable)}{suffix}"
    exact = sum(row.get("training_exact") is True for row in rows)
    if len(rows) == 1:
        if run_failures:
            return 1, "run\nfailed"
        return (3, "✓") if exact else (2, "✗")
    code = 3 if exact == len(rows) else 2 if run_failures < len(rows) else 1
    suffix = f"\n{run_failures} fail" if run_failures else ""
    return code, f"{exact}/{len(rows)}{suffix}"


def outcome_matrix(
    rows: Sequence[Row],
    *,
    models: Sequence[str],
    tasks: Sequence[str],
    arms: Sequence[str],
    seed_count: int,
    output: Path,
    dpi: int,
) -> dict[str, str]:
    """Render exact and held-out outcomes as auditable raw-cell matrices."""

    groups = _group_rows(rows)
    row_keys = [(task, arm) for task in tasks for arm in arms]
    colors = ["#FFFFFF", GREY, VERMILLION, BLUE, LIGHT_GREY]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), cmap.N)
    with publication_style():
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(max(7.0, len(models) * 1.2 + 3.4), max(3.1, len(row_keys) * 0.55 + 1.5)),
            constrained_layout=True,
        )
        axis_list = list(np.atleast_1d(axes).flat)
        for axis, heldout, title in zip(
            axis_list,
            (False, True),
            ("Training exact discovery", "Held-out exact correctness"),
            strict=True,
        ):
            matrix = np.zeros((len(row_keys), len(models)), dtype=int)
            labels: list[list[str]] = []
            for row_index, (task, arm) in enumerate(row_keys):
                row_labels: list[str] = []
                for model_index, model in enumerate(models):
                    code, label = _outcome_cell(
                        groups.get((model, task, arm), []), heldout=heldout
                    )
                    matrix[row_index, model_index] = code
                    row_labels.append(label)
                labels.append(row_labels)
            axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            axis.set_title(title, loc="left", fontweight="semibold")
            axis.set_xticks(range(len(models)), [_model_label(model) for model in models])
            axis.set_yticks(
                range(len(row_keys)),
                [f"{_task_label(task)} · {arm}" for task, arm in row_keys],
            )
            axis.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(row_keys), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=2)
            axis.tick_params(which="minor", bottom=False, left=False)
            for row_index in range(len(row_keys)):
                for model_index in range(len(models)):
                    code = matrix[row_index, model_index]
                    color = "white" if code in {1, 2, 3} else DARK
                    axis.text(
                        model_index,
                        row_index,
                        labels[row_index][model_index],
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=7.5,
                        fontweight="semibold",
                    )
        fig.suptitle(
            "Exact and held-out outcomes\n" + _subtitle(seed_count),
            x=0.01,
            ha="left",
            fontsize=10.5,
            fontweight="semibold",
        )
        legend = [
            Patch(facecolor=BLUE, label="exact"),
            Patch(facecolor=VERMILLION, label="completed, not exact"),
            Patch(facecolor=GREY, label="run failed"),
            Patch(facecolor=LIGHT_GREY, label="held-out unavailable"),
            Patch(facecolor="white", edgecolor=GREY, label="missing cell"),
        ]
        fig.legend(handles=legend, loc="outside lower center", ncol=5, frameon=False)
        return _save(fig, output, "outcome_matrix", dpi)


def paired_seed_rows(rows: Sequence[Row]) -> list[dict[str, Any]]:
    """Return raw Q/QD seed pairs for groups with at least two paired seeds."""

    lookup: dict[tuple[str, str, int, str], Row] = {}
    for row in rows:
        model = row.get("model_id")
        task = row.get("task_id")
        seed = row.get("seed")
        arm = row.get("arm")
        if (
            isinstance(model, str)
            and isinstance(task, str)
            and isinstance(seed, int)
            and not isinstance(seed, bool)
            and arm in {"Q", "QD"}
        ):
            lookup[(model, task, seed, cast(str, arm))] = row
    provisional: list[dict[str, Any]] = []
    counts: dict[tuple[str, str], int] = defaultdict(int)
    bases = {(model, task, seed) for model, task, seed, _ in lookup}
    for model, task, seed in sorted(bases):
        q = lookup.get((model, task, seed, "Q"))
        qd = lookup.get((model, task, seed, "QD"))
        if q is None or qd is None:
            continue
        provisional.append(
            {
                "model": model,
                "task": task,
                "seed": seed,
                "q": int(q.get("training_exact") is True),
                "qd": int(qd.get("training_exact") is True),
            }
        )
        counts[(model, task)] += 1
    return [
        pair for pair in provisional if counts[(pair["model"], pair["task"])] >= 2
    ]


def _seed_jitter(seed: int, width: float = 0.08) -> float:
    digest = hashlib.sha256(str(seed).encode()).digest()
    unit = int.from_bytes(digest[:2], "big") / 65535
    return (unit - 0.5) * width


def paired_q_qd(
    pairs: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str],
    tasks: Sequence[str],
    output: Path,
    dpi: int,
) -> dict[str, str]:
    """Render paired raw seed transitions; no aggregate interval is inferred."""

    available_tasks = [task for task in tasks if any(pair["task"] == task for pair in pairs)]
    with publication_style():
        fig, axes = plt.subplots(
            1,
            len(available_tasks),
            figsize=(max(6.8, 3.6 * len(available_tasks)), 3.5),
            sharey=True,
            constrained_layout=True,
        )
        axis_list = list(np.atleast_1d(axes).flat)
        for axis, task in zip(axis_list, available_tasks, strict=True):
            for pair in pairs:
                if pair["task"] != task or pair["model"] not in models:
                    continue
                model_index = models.index(cast(str, pair["model"]))
                jitter = _seed_jitter(cast(int, pair["seed"]))
                x_values = [model_index - 0.16 + jitter, model_index + 0.16 + jitter]
                y_values = [cast(int, pair["q"]), cast(int, pair["qd"])]
                axis.plot(x_values, y_values, color=GREY, alpha=0.45, linewidth=0.8)
                axis.scatter(x_values[0], y_values[0], color=BLUE, marker="o", s=22, zorder=3)
                axis.scatter(
                    x_values[1], y_values[1], color=VERMILLION, marker="s", s=22, zorder=3
                )
            axis.set_title(_task_label(task), loc="left", fontweight="semibold")
            axis.set_xticks(range(len(models)), [_model_label(model) for model in models])
            axis.set_ylim(-0.18, 1.18)
            axis.set_yticks([0, 1], ["not exact", "exact"])
            axis.grid(axis="y")
            axis.set_xlabel("Checkpoint")
        axis_list[0].set_ylabel("Training exact discovery")
        fig.suptitle(
            "Paired Q versus QD outcomes by seed\n"
            "Exploratory raw pairs · seed is the unit · no confidence intervals",
            x=0.01,
            ha="left",
            fontsize=10.5,
            fontweight="semibold",
        )
        fig.legend(
            handles=[
                Line2D([], [], color=BLUE, marker="o", linestyle="none", label="Q"),
                Line2D([], [], color=VERMILLION, marker="s", linestyle="none", label="QD"),
            ],
            loc="outside lower center",
            ncol=2,
            frameon=False,
        )
        return _save(fig, output, "paired_q_vs_qd", dpi)


def provider_work(
    rows: Sequence[Row],
    *,
    models: Sequence[str],
    tasks: Sequence[str],
    arms: Sequence[str],
    output: Path,
    dpi: int,
) -> dict[str, str]:
    """Render raw provider tokens, provider-wait seconds, and cache disposition."""

    arm_style = {
        "Q": (BLUE, "o"),
        "QD": (VERMILLION, "s"),
    }
    status_y = {"unavailable": 0, "cold": 1, "mixed": 2, "warm": 3}
    with publication_style():
        fig, axes = plt.subplots(
            len(tasks),
            3,
            figsize=(9.5, max(3.8, 2.35 * len(tasks) + 1.2)),
            squeeze=False,
        )
        fig.subplots_adjust(
            left=0.105,
            right=0.985,
            top=0.80,
            bottom=0.145,
            hspace=0.42,
            wspace=0.38,
        )
        for task_index, task in enumerate(tasks):
            token_axis, time_axis, cache_axis = axes[task_index]
            task_rows = [row for row in rows if row.get("task_id") == task]
            token_values = [
                float(value)
                for row in task_rows
                if isinstance((value := row.get("provider_scored_tokens")), (int, float))
                and not isinstance(value, bool)
            ]
            time_values = [
                float(value)
                for row in task_rows
                if isinstance(
                    (value := row.get("provider_await_wall_seconds")), (int, float)
                )
                and not isinstance(value, bool)
            ]
            for row in rows:
                if row.get("task_id") != task or row.get("model_id") not in models:
                    continue
                arm = row.get("arm")
                seed = row.get("seed")
                if arm not in arms or not isinstance(seed, int) or isinstance(seed, bool):
                    continue
                color, marker = arm_style.get(cast(str, arm), (GREEN, "^"))
                model_index = models.index(cast(str, row["model_id"]))
                arm_offset = -0.10 if arm == "Q" else 0.10
                x = model_index + arm_offset + _seed_jitter(seed, width=0.05)
                tokens = row.get("provider_scored_tokens")
                seconds = row.get("provider_await_wall_seconds")
                if isinstance(tokens, (int, float)) and not isinstance(tokens, bool):
                    token_axis.scatter(
                        x,
                        float(tokens),
                        color=color,
                        marker=marker,
                        s=25,
                        clip_on=False,
                        zorder=3,
                    )
                if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                    time_axis.scatter(
                        x,
                        float(seconds),
                        color=color,
                        marker=marker,
                        s=25,
                        clip_on=False,
                        zorder=3,
                    )
                cache_axis.scatter(
                    x,
                    status_y[cache_status(row)],
                    color=color,
                    marker=marker,
                    s=23,
                )
            for axis in (token_axis, time_axis, cache_axis):
                axis.set_xticks(range(len(models)), [_model_label(model) for model in models])
                axis.set_xlim(-0.45, len(models) - 0.55)
                axis.grid(axis="y")
                if task_index == len(tasks) - 1:
                    axis.set_xlabel("Checkpoint")
            token_top = max(token_values, default=1.0)
            time_top = max(time_values, default=1.0)
            token_axis.set_ylim(0, token_top * 1.14 if token_top > 0 else 1.0)
            time_axis.set_ylim(0, time_top * 1.14 if time_top > 0 else 1.0)
            token_axis.yaxis.set_major_formatter(EngFormatter(sep=""))
            token_axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            time_axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            cache_axis.set_yticks(
                list(status_y.values()),
                ["unavailable", "cold", "mixed", "warm"],
            )
            cache_axis.set_ylim(-0.2, 3.2)
            token_axis.set_ylabel("Token positions")
            time_axis.set_ylabel("Provider wait (s)")
            cache_axis.set_ylabel("Cache status")
            if task_index == 0:
                token_axis.set_title("Scored token positions", loc="left", pad=8)
                time_axis.set_title("Provider wait for misses", loc="left", pad=8)
                cache_axis.set_title("Cache disposition", loc="left", pad=8)
            position = token_axis.get_position()
            fig.text(
                0.015,
                (position.y0 + position.y1) / 2,
                _task_label(task),
                rotation=90,
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="semibold",
            )
        fig.text(
            0.015,
            0.965,
            "Provider work and cache disposition",
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="semibold",
        )
        fig.text(
            0.015,
            0.925,
            "Exploratory raw seed cells · cached reuse is not new provider work",
            ha="left",
            va="top",
            fontsize=8.5,
            color=GREY,
        )
        fig.legend(
            handles=[
                Line2D([], [], color=BLUE, marker="o", linestyle="none", label="Q"),
                Line2D([], [], color=VERMILLION, marker="s", linestyle="none", label="QD"),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.018),
            ncol=2,
            frameon=False,
        )
        return _save(fig, output, "provider_work", dpi)

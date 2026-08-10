from __future__ import annotations

import csv
import json
import struct
from pathlib import Path
from typing import Any

import pytest

from research.figures import FigureConfig, build_figures
from research.figures.data import load_rows
from research.figures.pipeline import validate_bundle


def _row(
    model: str,
    task: str,
    arm: str,
    seed: int,
    *,
    exact: bool,
    protocol: str = "protocol-amended",
    cache_hit: int = 0,
    cache_miss: int = 10,
) -> dict[str, Any]:
    return {
        "protocol_id": "study-v2",
        "protocol_sha256": protocol,
        "cell_id": f"{model}--{task}--{arm}--seed-{seed}",
        "analysis_label": "exploratory",
        "task_id": task,
        "arm": arm,
        "seed": seed,
        "model_id": model,
        "model_total_parameters_billion": 3.0,
        "status": "completed",
        "run_completed": True,
        "success": exact,
        "training_exact": exact,
        "heldout_evaluable": True,
        "heldout_correct": exact,
        "heldout_accuracy": 1.0 if exact else 0.5,
        "provider_candidates": cache_miss,
        "provider_scored_tokens": cache_miss * 12,
        "provider_await_wall_seconds": cache_miss * 0.1,
        "score_cache_hit_candidates": cache_hit,
        "score_cache_miss_candidates": cache_miss,
        "cache_served_scored_tokens": cache_hit * 12,
        "failure_reason": None,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def _png_dpi(path: Path) -> float:
    content = path.read_bytes()
    offset = 8
    while offset < len(content):
        length = struct.unpack(">I", content[offset : offset + 4])[0]
        chunk = content[offset + 4 : offset + 8]
        data = content[offset + 8 : offset + 8 + length]
        if chunk == b"pHYs":
            pixels_per_meter, _, unit = struct.unpack(">IIB", data)
            assert unit == 1
            return float(pixels_per_meter) * 0.0254
        offset += 12 + length
    raise AssertionError("PNG has no physical-resolution chunk")


def test_one_seed_bundle_is_raw_exploratory_and_has_three_formats(tmp_path: Path) -> None:
    inputs = tmp_path / "metrics.json"
    rows = [
        _row(
            model,
            task,
            arm,
            101,
            exact=arm == "QD",
            cache_hit=10 if model == "qwen25-coder-7b" and arm == "QD" else 0,
            cache_miss=0 if model == "qwen25-coder-7b" and arm == "QD" else 10,
        )
        for model in ("qwen25-coder-3b", "qwen25-coder-7b")
        for task in ("map-increment", "foldr-signed-window")
        for arm in ("Q", "QD")
    ]
    _write_rows(inputs, rows)
    output = tmp_path / "figures"

    result = build_figures([inputs], output)
    manifest = validate_bundle(output)

    assert result.complete_grid is False
    assert result.paired_figure_generated is False
    assert manifest["analysis_status"] == "exploratory"
    assert manifest["uncertainty_policy"] == "raw cells/counts only; no confidence intervals"
    assert manifest["eligible_for_exploratory_paper_insertion"] is False
    assert manifest["confirmatory_claim_ready"] is False
    assert manifest["figures"]["paired_q_vs_qd"]["generated"] is False
    assert not list(output.glob("paired_q_vs_qd.*"))
    for stem in ("outcome_matrix", "provider_work"):
        assert (output / f"{stem}.pdf").read_bytes().startswith(b"%PDF-")
        assert b"<svg" in (output / f"{stem}.svg").read_bytes()[:1000]
        assert (output / f"{stem}.png").read_bytes().startswith(b"\x89PNG")
    svg = (output / "outcome_matrix.svg").read_text(encoding="utf-8")
    assert "Exploratory raw cells" in svg
    assert "no confidence intervals" in svg
    provider_svg = (output / "provider_work.svg").read_text(encoding="utf-8")
    assert "Provider work and cache disposition" in provider_svg
    assert "\N{MINUS SIGN}" not in provider_svg
    assert _png_dpi(output / "outcome_matrix.png") == pytest.approx(450, abs=0.1)


def test_repeated_paired_seeds_generate_raw_pair_plot_deterministically(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "metrics.json"
    rows = [
        _row(
            "qwen25-coder-3b",
            "map-increment",
            arm,
            seed,
            exact=(arm == "QD" or seed == 211),
            cache_hit=5 if seed == 211 else 0,
            cache_miss=5 if seed == 211 else 10,
        )
        for seed in (101, 211)
        for arm in ("Q", "QD")
    ]
    _write_rows(inputs, rows)
    config = FigureConfig(
        models=("qwen25-coder-3b",),
        tasks=("map-increment",),
        arms=("Q", "QD"),
        seeds=(101, 211),
        dpi=300,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = build_figures([inputs], first, config)
    build_figures([inputs], second, config)

    assert first_result.complete_grid is True
    assert first_result.paired_figure_generated is True
    assert (first / "paired_q_vs_qd.pdf").is_file()
    assert "Exploratory raw pairs" in (first / "paired_q_vs_qd.svg").read_text(
        encoding="utf-8"
    )
    assert (first / "paired_q_vs_qd.svg").read_bytes() == (
        second / "paired_q_vs_qd.svg"
    ).read_bytes()
    first_manifest = json.loads((first / "figure_manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["complete_expected_grid"] is True
    assert first_manifest["eligible_for_exploratory_paper_insertion"] is False
    assert first_manifest["confirmatory_claim_ready"] is False


def test_mixed_protocols_and_existing_destination_fail_closed(tmp_path: Path) -> None:
    inputs = tmp_path / "mixed.json"
    _write_rows(
        inputs,
        [
            _row("qwen25-coder-3b", "map-increment", "Q", 101, exact=False),
            _row(
                "qwen25-coder-7b",
                "map-increment",
                "Q",
                101,
                exact=False,
                protocol="old-preflight",
            ),
        ],
    )

    with pytest.raises(ValueError, match="cannot be pooled"):
        build_figures([inputs], tmp_path / "mixed-output")

    clean = tmp_path / "clean.json"
    _write_rows(
        clean,
        [_row("qwen25-coder-3b", "map-increment", "Q", 101, exact=False)],
    )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        build_figures([clean], existing)


def test_tidy_csv_input_preserves_boolean_and_provider_metrics(tmp_path: Path) -> None:
    row = _row("qwen25-coder-3b", "map-increment", "Q", 101, exact=True)
    path = tmp_path / "metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    rows, inputs = load_rows([path])

    assert rows[0]["training_exact"] is True
    assert rows[0]["provider_scored_tokens"] == 120
    assert rows[0]["provider_await_wall_seconds"] == pytest.approx(1.0)
    assert inputs[0]["name"] == "metrics.csv"

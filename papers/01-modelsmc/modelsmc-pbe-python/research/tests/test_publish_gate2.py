from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from research.publish_gate2 import build_gate2_release, validate_gate2_release

PROJECT = Path(__file__).parents[2]
PROTOCOL_SHA = "56c590864d456f884c82bf62ada3c11c1c2504d21b021e650bc323007553fc60"
SOURCE_REVISION = "1" * 40


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _score_result(model: dict[str, Any]) -> dict[str, Any]:
    prompt = "finite score prefix\n" + ("candidate context " * 1200)
    wave = {
        "beta": 1.0,
        "cache_hit": False,
        "cache_key_sha256": "2" * 64,
        "candidate_kind": "skeleton",
        "candidates": [
            {
                "canonical_candidate": '"expression"',
                "deduction_probability": 1.0,
                "normalized_energy": -1.0,
                "proposal_probability": 1.0,
                "qwen_probability": 1.0,
                "scored_token_count": 1,
                "token_ids": [42],
                "token_logprobs": [-1.0],
                "total_sequence_logprob": -1.0,
            }
        ],
        "deduction_mix": 0.0,
        "energy_normalization": "mean-full-prompt-conditional-logprob",
        "model": model["model_id"],
        "model_revision": model["model_revision"],
        "prompt_prefix": prompt,
        "prompt_prefix_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "proposal_epsilon": 0.05,
        "request_index": 0,
        "score_origin": "provider",
        "score_semantics": "teacher-forced-full-prompt",
        "selections": [{"selected_index": 0, "selected_probability": 1.0}],
        "source": "vllm-prompt-logprobs",
        "stage": 1,
        "temperature": 0.7,
        "tokenizer_revision": model["tokenizer_revision"],
        "wave": "family",
    }
    return {
        "schema_version": 1,
        "status": "completed",
        "result": {
            "llm_energy_normalization": "mean-full-prompt-conditional-logprob",
            "score_ledger": [wave],
        },
    }


def _make_project(root: Path) -> None:
    tasks = ["map-increment", "foldr-signed-window"]
    arms = ["Q", "QD"]
    models = [
        {
            "model_id": f"qwen25-coder-{size}",
            "model_repository": f"Qwen/Qwen2.5-Coder-{size.upper()}-Instruct",
            "model_revision": str(index + 3) * 40,
            "tokenizer_revision": str(index + 3) * 40,
            "source_path": f"research/outputs/size-gate2-{size}-transport32",
        }
        for index, size in enumerate(("3b", "7b", "14b", "32b"))
    ]
    protocol = b'{"protocol_id":"test-transport32"}\n'
    protocol_sha = hashlib.sha256(protocol).hexdigest()
    protocol_path = root / "research" / "protocol-size-study-transport32.json"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_bytes(protocol)

    manuscript = b"%PDF-1.4\n" + (b"0" * 1200) + b"\n%%EOF\n"
    paper = root / "paper"
    paper.mkdir()
    for filename in (
        "AUTHOR_REVIEW_GUIDE.md",
        "README.md",
        "jmlr2e.sty",
        "main.tex",
        "references.bib",
    ):
        (paper / filename).write_text(f"publication {filename}\n")
    (paper / "main.pdf").write_bytes(manuscript)
    (paper / "generated").mkdir()
    (paper / "generated" / "benchmark_table.tex").write_text("grid table\n")
    figures = paper / "generated" / "figures" / "gate2-size-transport32"
    figure_data = json.dumps({"cells": 16}, indent=2, sort_keys=True) + "\n"
    _write_json(
        figures / "figure_manifest.json",
        {
            "protocol_sha256": protocol_sha,
            "selected_rows": 16,
            "sha256": {"figure_data.json": hashlib.sha256(figure_data.encode()).hexdigest()},
        },
    )
    (figures / "figure_data.json").write_text(figure_data)

    for model in models:
        group = root / model["source_path"]
        planned: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        for task in tasks:
            for arm in arms:
                cell_id = f"{task}--{arm}--{model['model_id']}--seed-101"
                planned.append({"cell_id": cell_id})
                rows.append(
                    {
                        "cell_id": cell_id,
                        "model_id": model["model_id"],
                        "task_id": task,
                        "arm": arm,
                        "seed": 101,
                        "status": "completed",
                        "core_artifact": str(root / "private" / cell_id),
                    }
                )
                cell = group / "cells" / cell_id
                _write_json(
                    cell / "cell.json",
                    {
                        "cell": {
                            "cell_id": cell_id,
                            "model_id": model["model_id"],
                            "model_hf_repository": model["model_repository"],
                            "model_revision": model["model_revision"],
                            "tokenizer_revision": model["tokenizer_revision"],
                        },
                        "command": [str(root / "private" / "modelsmc-pbe")],
                        "protocol_sha256": protocol_sha,
                        "status": "completed",
                    },
                )
                _write_json(cell / "heldout.json", {"accuracy": 1.0, "cases": 1})
                (cell / "stdout.log").write_text("excluded local log\n")
                run = cell / "artifacts" / "run-1"
                _write_json(
                    run / "manifest.json",
                    {
                        "configuration": {
                            "base_url": "https://private-8000.proxy.runpod.net/v1",
                            "llm_energy_normalization": ("mean-full-prompt-conditional-logprob"),
                            "model_repository": model["model_repository"],
                            "model_revision": model["model_revision"],
                            "score_cache_dir": str(root / "cache" / "candidate-scores"),
                            "tokenizer_revision": model["tokenizer_revision"],
                            "vllm_server_config": "vllm=0.11.0;logprobs_mode=processed_logprobs",
                        },
                        "metrics": {
                            "candidate_score_cache": {"hit_requests": 0},
                            "candidate_score_provider": {"http_requests": 1},
                        },
                        "status": "completed",
                    },
                )
                _write_json(run / "result.json", _score_result(model))
                (run / "events.jsonl").write_text('{"event":"complete"}\n')
                (run / "final_particles.jsonl").write_text('{"particle":0}\n')
        _write_json(
            group / "matrix_manifest.json",
            {
                "planned_cells": planned,
                "protocol_sha256": protocol_sha,
                "stage_id": "gate-2-size-pilot",
            },
        )
        (group / "protocol.json").write_bytes(protocol)
        _write_json(group / "analysis" / "metrics.json", rows)
        (group / "analysis" / "metrics.csv").write_text("cell_id,status\n")

    release = {
        "schema_version": 2,
        "release_label": "test-v0.2.0",
        "authors": ["Tri Nguyen", "Thanh-Dat Nguyen"],
        "implementation": {
            "repository": "https://github.com/quangng2000/modelsmc-reading-notes",
            "branch": "python-modelsmc-pbe",
            "revision": None,
        },
        "protocol": {
            "path": "research/protocol-size-study-transport32.json",
            "sha256": protocol_sha,
        },
        "manuscript": {
            "path": "paper/main.pdf",
            "sha256": hashlib.sha256(manuscript).hexdigest(),
        },
        "figures": {
            "path": "paper/generated/figures/gate2-size-transport32",
            "required_manifest": "figure_manifest.json",
        },
        "grid": {
            "stage_id": "gate-2-size-pilot",
            "analysis_population": "intention-to-treat",
            "tasks": tasks,
            "arms": arms,
            "seeds": [101],
            "expected_cells": 16,
            "groups": models,
        },
        "packaging": {
            "compression_threshold_bytes": 10_000,
            "score_cache_blobs_included": False,
            "source_code_included": False,
        },
    }
    _write_json(root / "research" / "gate2_release.json", release)
    card = root / "research" / "huggingface" / "GATE2_README.md"
    card.parent.mkdir()
    card.write_text("publication commit {{SOURCE_REVISION}}\n")


def test_gate2_release_is_data_only_sanitized_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _make_project(project)
    first = tmp_path / "release-1"
    second = tmp_path / "release-2"

    first_result = build_gate2_release(
        project,
        first,
        source_revision=SOURCE_REVISION,
        require_clean_git=False,
    )
    second_result = build_gate2_release(
        project,
        second,
        source_revision=SOURCE_REVISION,
        require_clean_git=False,
    )

    assert first_result == second_result == validate_gate2_release(first)
    assert first_result.cells == first_result.completed_cells == 16
    assert first_result.failed_cells == 0
    assert first_result.score_waves == first_result.scored_candidates == 16
    assert first_result.gzip_files == 16
    assert (first / "SHA256SUMS").read_bytes() == (second / "SHA256SUMS").read_bytes()
    assert not (first / "src").exists()
    assert not (first / "tests").exists()
    assert not (first / "pyproject.toml").exists()
    assert not list(first.rglob("*.log"))
    assert not list(first.rglob("candidate-scores"))
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in first.rglob("*.json") if path.is_file()
    )
    assert str(project) not in text
    assert "proxy.runpod.net" not in text
    assert "<VLLM_ENDPOINT>" in text
    assert SOURCE_REVISION in (first / "README.md").read_text()


def test_gate2_release_fails_closed_on_corrupted_gzip(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _make_project(project)
    release = tmp_path / "release"
    build_gate2_release(
        project,
        release,
        source_revision=SOURCE_REVISION,
        require_clean_git=False,
    )
    result = next(release.rglob("result.json.gz"))
    result.write_bytes(result.read_bytes()[:-8])

    with pytest.raises(ValueError, match="invalid gzip"):
        validate_gate2_release(release)


def test_gate2_release_requires_checked_out_clean_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        build_gate2_release(
            PROJECT,
            tmp_path / "release",
            source_revision="0" * 40,
        )


def test_gate2_manifest_pins_complete_real_source_grid() -> None:
    release = json.loads((PROJECT / "research" / "gate2_release.json").read_text())
    assert release["protocol"]["sha256"] == PROTOCOL_SHA
    assert release["authors"] == ["Tri Nguyen", "Thanh-Dat Nguyen"]
    assert release["grid"]["expected_cells"] == 16
    assert len(release["grid"]["groups"]) == 4
    assert release["implementation"]["revision"] is None

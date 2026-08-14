from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import torch
from rich.console import Console

from modelsmc_pbe.observability import RunLogger
from modelsmc_pbe.runtime import resolve_device


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_successful_run_writes_complete_artifact_set(tmp_path: Path) -> None:
    output = io.StringIO()
    logger = RunLogger.create(
        base_dir=tmp_path / "runs",
        run_name="bounded square",
        config={"particles": 2, "api_key": "must-not-leak"},
        device=resolve_device("cpu"),
        seed=7,
        probabilistic_claim="heuristic search weights",
        console_level="trace",
        run_id="run-123",
        cwd=tmp_path,
        console=Console(file=output, force_terminal=False, color_system=None),
    )
    logger.event(
        "smc.stage",
        message="stage complete",
        level="trace",
        stage=1,
        ess=1.5,
        prompt="private model prompt",
        rationale="private model rationale",
    )
    logger.finish(
        result={"exact": True, "best_loss": 0.0},
        final_particles=[
            {
                "id": 1,
                "weight": torch.tensor(0.75),
                "rationale": "private model rationale",
            }
        ],
    )

    assert "[trace] stage complete" in output.getvalue()
    manifest = read_json(logger.run_dir / "manifest.json")
    assert manifest["status"] == "completed"
    assert manifest["run_id"] == "run-123"
    configuration = manifest["configuration"]
    assert isinstance(configuration, dict)
    assert configuration["api_key"] == "[redacted]"

    events = read_jsonl(logger.run_dir / "events.jsonl")
    assert [event["sequence"] for event in events] == [0, 1, 2]
    assert [event["event"] for event in events] == [
        "run.started",
        "smc.stage",
        "run.completed",
    ]
    data = events[1]["data"]
    assert isinstance(data, dict)
    assert data["prompt"] != "private model prompt"
    assert data["rationale"] != "private model rationale"
    assert "private model prompt" not in (logger.run_dir / "events.jsonl").read_text()
    assert "private model rationale" not in (
        logger.run_dir / "events.jsonl"
    ).read_text()

    result = read_json(logger.run_dir / "result.json")
    assert result["particle_count"] == 1
    particles = read_jsonl(logger.run_dir / "final_particles.jsonl")
    particle = particles[0]["particle"]
    assert isinstance(particle, dict)
    assert particle["id"] == 1
    assert particle["weight"] == 0.75
    assert particle["rationale"] != "private model rationale"
    assert "private model rationale" not in (
        logger.run_dir / "final_particles.jsonl"
    ).read_text()


def test_context_manager_records_failures(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="experiment exploded"):
        with RunLogger.create(
            base_dir=tmp_path,
            run_name="failure",
            config={},
            device=resolve_device("cpu"),
            seed=1,
            probabilistic_claim="test",
            console_level="quiet",
            cwd=tmp_path,
        ) as logger:
            raise RuntimeError("experiment exploded")

    manifest = read_json(logger.run_dir / "manifest.json")
    result = read_json(logger.run_dir / "result.json")
    assert manifest["status"] == "failed"
    assert result["status"] == "failed"
    error = result["error"]
    assert isinstance(error, dict)
    assert error["type"] == "RuntimeError"
    assert (logger.run_dir / "final_particles.jsonl").exists()

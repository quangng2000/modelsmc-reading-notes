from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from modelsmc_pbe.cli import app
from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.proposals import OpenAICompatibleProposer
from modelsmc_pbe.shell.providers import build_candidate_scorer, build_proposer
from modelsmc_pbe.shell.request import SynthesizeRequest

PROJECT_DIR = Path(__file__).resolve().parents[1]
MAP_SPEC = PROJECT_DIR / "examples" / "map-increment.json"
BOOL_SPEC = PROJECT_DIR / "examples" / "negative-int-to-bool.json"


def test_model_backed_provider_does_not_require_a_finite_skeleton() -> None:
    config = load_experiment_config(BOOL_SPEC)
    request = SynthesizeRequest(
        spec=BOOL_SPEC,
        proposal="ollama",
        model="test-model",
        skeleton="auto",
    )

    proposer = build_proposer(request, config)

    assert isinstance(proposer, OpenAICompatibleProposer)


def test_importance_mode_rejects_ollama_as_an_uncorrected_provider() -> None:
    request = SynthesizeRequest(
        spec=BOOL_SPEC,
        mode="importance-smc",
        proposal="ollama",
        model="test-model",
    )

    with pytest.raises(ValueError, match="must be catalog or vllm"):
        build_candidate_scorer(request)


def test_cli_grammar_smoke_seals_completed_artifacts(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "synthesize",
            str(MAP_SPEC),
            "--mode",
            "grammar-smc",
            "--particles",
            "16",
            "--iterations",
            "1",
            "--device",
            "cpu",
            "--artifacts-dir",
            str(tmp_path),
            "--grammar-limit",
            "100",
            "--score-batch-size",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    persisted = json.loads((run_dirs[0] / "result.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert persisted["status"] == "completed"
    assert "[result] artifacts:" in result.output


def test_cli_importance_smoke_uses_an_explicit_uniform_q(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "synthesize",
            str(MAP_SPEC),
            "--mode",
            "importance-smc",
            "--proposal",
            "catalog",
            "--particles",
            "32",
            "--iterations",
            "1",
            "--alpha",
            "0",
            "--hole-max-cost",
            "3",
            "--device",
            "cpu",
            "--artifacts-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    run_dir = next(tmp_path.iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    persisted = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert manifest["probabilistic_claim"] == (
        "importance_corrected_finite_deduction_refuted_target"
    )
    assert persisted["result"]["mode"] == "importance-smc"
    assert persisted["result"]["proposal_source"] == "uniform-finite-candidates"

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
from modelsmc_pbe.shell.skeletons import (
    automatic_skeleton,
    importance_uses_multiple_families,
    resolve_importance_skeleton,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
MAP_SPEC = PROJECT_DIR / "examples" / "map-increment.json"
BOOL_SPEC = PROJECT_DIR / "examples" / "negative-int-to-bool.json"
BOUNDED_SPEC = PROJECT_DIR / "examples" / "foldr-bounded-square.json"


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


def test_grammar_auto_selects_one_structure_but_importance_auto_keeps_many() -> None:
    mapped = load_experiment_config(MAP_SPEC)
    filtered = load_experiment_config(BOUNDED_SPEC)
    scalar_bool = load_experiment_config(BOOL_SPEC)

    assert automatic_skeleton(mapped) == "map-arithmetic"
    assert automatic_skeleton(filtered) == "foldr-filter-map"
    assert importance_uses_multiple_families("auto") is True
    assert importance_uses_multiple_families("general") is False
    assert resolve_importance_skeleton(scalar_bool, "auto") is None
    assert resolve_importance_skeleton(filtered, "general") is None


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
            "--score-batch-size",
            "16",
            "--candidate-batch-size",
            "17",
            "--max-scored-candidates",
            "50000",
            "--temperature",
            "0.9",
            "--llm-energy-normalization",
            "mean-full-prompt-conditional-logprob",
            "--model-revision",
            "b2cff646",
            "--tokenizer-revision",
            "tokenizer-test",
            "--deduction-mix",
            "0.6",
            "--deduction-strength",
            "3.0",
            "--max-tokens",
            "123",
            "--max-concurrency",
            "3",
            "--timeout-seconds",
            "12",
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
        "importance_corrected_lazy_factorized_construction_target"
    )
    assert persisted["result"]["mode"] == "importance-smc"
    assert persisted["result"]["execution"] == "lazy-factorized"
    assert persisted["result"]["support_materialized"] is False
    assert persisted["result"]["reference"] is None
    assert persisted["result"]["proposal_source"] == "uniform-finite-candidates"
    assert persisted["result"]["deduction_mix"] == 0.6
    assert persisted["result"]["deduction_strength"] == 3.0
    assert persisted["result"]["llm_energy_normalization"] == (
        "mean-full-prompt-conditional-logprob"
    )
    assert persisted["result"]["score_ledger"]
    assert all(
        "deduction_guide_mass" in family for family in persisted["result"]["families"]
    )
    assert persisted["result"]["conditioned_skeleton"] is None
    assert persisted["result"]["multi_family"] is True
    assert persisted["result"]["viable_hypotheses"] >= 2
    assert persisted["result"]["search"]["evaluated_programs"] < persisted["result"][
        "support_states"
    ]
    assert manifest["configuration"]["requested_skeleton"] == "auto"
    assert manifest["configuration"]["skeleton"] == "multi-family"
    assert manifest["configuration"]["materialize_reference"] is False
    assert manifest["configuration"]["score_batch_size"] == 16
    assert manifest["configuration"]["candidate_batch_size"] == 17
    assert manifest["configuration"]["max_scored_candidates"] == 50_000
    assert manifest["configuration"]["temperature"] == 0.9
    assert manifest["configuration"]["llm_energy_normalization"] == (
        "mean-full-prompt-conditional-logprob"
    )
    assert manifest["configuration"]["model_revision"] == "b2cff646"
    assert manifest["configuration"]["tokenizer_revision"] == "tokenizer-test"
    assert manifest["configuration"]["deduction_mix"] == 0.6
    assert manifest["configuration"]["deduction_strength"] == 3.0
    assert manifest["configuration"]["max_tokens"] == 123
    assert manifest["configuration"]["max_concurrency"] == 3
    assert manifest["configuration"]["timeout_seconds"] == 12.0

"""CLI contract tests for the materialized joint-target oracle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from modelsmc_pbe.cli import app
from modelsmc_pbe.shell.providers import build_candidate_scorer
from modelsmc_pbe.shell.request import SynthesizeRequest

PROJECT_DIR = Path(__file__).resolve().parents[1]
MAP_SPEC = PROJECT_DIR / "examples" / "map-increment.json"


def test_joint_target_uses_no_provider_and_rejects_cache_configuration(
    tmp_path: Path,
) -> None:
    request = SynthesizeRequest(
        spec=MAP_SPEC,
        mode="importance-smc",
        proposal="joint-target",
    )
    with pytest.raises(ValueError, match="bypasses"):
        build_candidate_scorer(request)

    with pytest.raises(ValueError, match="uses no model provider or score cache"):
        build_candidate_scorer(
            SynthesizeRequest(
                spec=MAP_SPEC,
                mode="importance-smc",
                proposal="joint-target",
                score_cache_mode="read-write",
                score_cache_dir=tmp_path,
            )
        )
    with pytest.raises(ValueError, match="only for importance-smc"):
        build_candidate_scorer(
            SynthesizeRequest(spec=MAP_SPEC, mode="paper-search", proposal="joint-target")
        )
    with pytest.raises(ValueError, match="uses no model provider"):
        build_candidate_scorer(
            SynthesizeRequest(
                spec=MAP_SPEC,
                mode="importance-smc",
                proposal="joint-target",
                model_revision="unused-revision",
            )
        )


def test_joint_target_requires_materialization_and_zero_clone_probability(
    tmp_path: Path,
) -> None:
    common = [
        "synthesize",
        str(MAP_SPEC),
        "--mode",
        "importance-smc",
        "--proposal",
        "joint-target",
        "--artifacts-dir",
        str(tmp_path),
        "--device",
        "cpu",
    ]
    lazy = CliRunner().invoke(app, common)
    cloned = CliRunner().invoke(
        app,
        [*common, "--materialize-reference", "--alpha", "0.25"],
    )

    assert lazy.exit_code == 2
    assert "requires --materialize-reference" in lazy.output
    assert cloned.exit_code == 2
    assert "requires --alpha 0" in cloned.output


def test_joint_target_cli_seals_oracle_identity_and_zero_score_work(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "synthesize",
            str(MAP_SPEC),
            "--mode",
            "importance-smc",
            "--proposal",
            "joint-target",
            "--materialize-reference",
            "--particles",
            "8",
            "--iterations",
            "1",
            "--alpha",
            "0",
            "--device",
            "cpu",
            "--artifacts-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    completed = [path for path in tmp_path.iterdir() if (path / "result.json").exists()]
    assert len(completed) == 1
    run_dir = completed[0]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    persisted = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["result"]

    assert manifest["status"] == "completed"
    assert manifest["probabilistic_claim"] == (
        "oracle_normalized_finite_joint_execution_target"
    )
    assert manifest["configuration"]["proposal"] == "joint-target"
    assert manifest["configuration"]["materialize_reference"] is True
    assert manifest["configuration"]["model"] is None
    assert manifest["configuration"]["temperature"] is None
    assert manifest["configuration"]["llm_energy_normalization"] is None
    assert manifest["configuration"]["proposal_epsilon"] is None
    assert manifest["configuration"]["deduction_mix"] is None
    assert manifest["configuration"]["family_deduction_mix"] is None
    assert manifest["configuration"]["hole_deduction_mix"] is None
    assert manifest["configuration"]["deduction_strength"] is None
    assert manifest["configuration"]["candidate_batch_size"] is None
    assert manifest["configuration"]["max_scored_candidates"] is None
    assert manifest["configuration"]["max_tokens"] is None
    assert manifest["configuration"]["max_concurrency"] is None
    assert manifest["configuration"]["timeout_seconds"] is None
    assert persisted["proposal_source"] == "finite-joint-target-oracle"
    assert persisted["proposal_strategy"] == "joint-target"
    assert persisted["deduction_mix"] is None
    assert persisted["family_deduction_mix"] is None
    assert persisted["hole_deduction_mix"] is None
    assert persisted["deduction_strength"] is None
    assert persisted["llm_energy_normalization"] is None
    assert persisted["deduction_guide_exact_mass"] is None
    assert all(family["deduction_guide_mass"] is None for family in persisted["families"])
    assert persisted["scored_candidates"] == 0
    assert persisted["max_scored_candidates"] is None
    assert persisted["score_ledger"] == []
    assert persisted["reference"]["absolute_log_path_z_error"] == pytest.approx(0.0)
    assert manifest["metrics"]["candidate_score_cache"]["provider_candidates"] == 0
    assert "exact normalized joint execution target (oracle)" in result.output

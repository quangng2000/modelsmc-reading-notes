"""Engine and CLI contracts for the LLM-backed joint-semantic proposal."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

import modelsmc_pbe.shell.execution as execution_module
from modelsmc_pbe.cli import app
from modelsmc_pbe.config import ExperimentConfig, load_experiment_config
from modelsmc_pbe.core import ProgramScorer, ScoreResult
from modelsmc_pbe.proposals import (
    CandidateKind,
    CandidateLogprobSemantics,
    CandidateScoreBatch,
    CandidateScoreOrigin,
    CandidateScoreProvenance,
    CandidateScoreRequest,
    CandidateSequenceScore,
    LLMEnergyNormalization,
    VLLMPromptLogprobScorer,
)
from modelsmc_pbe.runtime import make_cpu_generator, resolve_device
from modelsmc_pbe.search.importance import (
    ImportanceProposalBudgetExceeded,
    ImportanceSMCEngine,
    ImportanceSMCOptions,
    LazyImportanceSMCEngine,
)
from modelsmc_pbe.search.importance.factorized import trace_log_prior
from modelsmc_pbe.search.importance.lazy_support import FactorizedSupportBuilder
from modelsmc_pbe.shell.providers import build_candidate_scorer
from modelsmc_pbe.shell.request import SynthesizeRequest

PROJECT_DIR = Path(__file__).resolve().parents[1]
MAP_SPEC = PROJECT_DIR / "examples" / "map-increment.json"


class _SemanticRawScorer:
    name = "scripted-semantic-labels"

    def __init__(self) -> None:
        self.requests: list[CandidateScoreRequest] = []

    async def score_candidates(self, request: CandidateScoreRequest) -> CandidateScoreBatch:
        return (await self.score_many([request]))[0]

    async def score_many(
        self,
        requests: list[CandidateScoreRequest],
    ) -> list[CandidateScoreBatch]:
        self.requests.extend(requests)
        return [self._batch(request) for request in requests]

    def _batch(self, request: CandidateScoreRequest) -> CandidateScoreBatch:
        assert request.candidate_kind is CandidateKind.LABEL
        assert len(request.candidates) % 4 == 0
        scores = []
        for index, candidate in enumerate(request.candidates):
            position = index % 4
            mapping = position // 2
            label = position % 2
            final_logprob = (-0.1, -0.8, -0.8, -0.1)[position]
            token_logprobs = (-0.25, final_logprob)
            scores.append(
                CandidateSequenceScore(
                    candidate=candidate,
                    expression=None,
                    token_ids=(1000 + 2 * (index // 4) + mapping, 700 + label),
                    token_logprobs=token_logprobs,
                    sequence_logprob=math.fsum(token_logprobs),
                )
            )
        return CandidateScoreBatch(
            scores=tuple(scores),
            source=self.name,
            model="semantic-test-model",
            semantics=CandidateLogprobSemantics.TEACHER_FORCED_FULL_PROMPT,
            model_revision="semantic-model-revision",
            tokenizer_revision="semantic-tokenizer-revision",
            provenance=CandidateScoreProvenance(CandidateScoreOrigin.SYNTHETIC),
        )


class _RecordingProgramScorer(ProgramScorer):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        self.scored_programs = 0

    def score_batch(self, programs: Sequence[object]) -> tuple[ScoreResult, ...]:
        self.scored_programs += len(programs)
        return super().score_batch(programs)


def _config() -> ExperimentConfig:
    return load_experiment_config(
        MAP_SPEC,
        smc_overrides={
            "particles": 4,
            "iterations": 1,
            "clone_probability": 0.0,
            "seed": 31,
        },
    )


def _options(*, budget: int = 64) -> ImportanceSMCOptions:
    return ImportanceSMCOptions(
        conditioned_skeleton="map-arithmetic",
        support_limit=1_000,
        proposal_strategy="joint-semantic",
        proposal_epsilon=0.2,
        semantic_scale=1.5,
        semantic_slate_size=16,
        semantic_candidate_batch_size=8,
        max_scored_candidates=budget,
    )


def test_lazy_engine_executes_particles_not_the_semantic_slate() -> None:
    config = _config()
    options = _options()
    raw = _SemanticRawScorer()
    with _RecordingProgramScorer(config) as scorer:
        result = asyncio.run(
            LazyImportanceSMCEngine(
                config=config,
                options=options,
                scorer=scorer,
                candidate_scorer=raw,
                generator=make_cpu_generator(31),
            ).run()
        )

    assert result.proposal_strategy == "joint-semantic"
    assert result.deduction_mix is None
    assert result.llm_energy_normalization is None
    assert result.score_ledger == ()
    assert result.scored_candidates == 64
    assert result.semantic_slate_size == 16
    ledger = result.semantic_score_ledger
    assert ledger is not None
    assert ledger.slate_traces == 16
    assert ledger.raw_candidate_count == 64
    assert len(ledger.selections) == 4
    assert scorer.scored_programs <= 8
    assert scorer.scored_programs < ledger.slate_traces
    assert math.fsum(
        family.proposal_mass or 0.0 for family in result.families
    ) == pytest.approx(1.0)
    assert [item.log_q_proposal for item in ledger.selections] == pytest.approx(
        [particle.log_q_mixture for particle in result.final_particles]
    )

    support = FactorizedSupportBuilder(
        spec=config.spec,
        smc=config.smc,
        options=options,
    ).build()
    for particle in result.final_particles:
        log_gamma = trace_log_prior(
            support,
            particle.trace,
            cost_scale=float(config.smc.cost_scale),
        ) - float(config.smc.loss_scale) * particle.total_loss
        assert particle.log_q_mixture + particle.log_incremental_weight == pytest.approx(
            log_gamma,
            abs=1e-12,
        )


def test_semantic_budget_fails_before_any_label_provider_request() -> None:
    config = _config()
    raw = _SemanticRawScorer()
    with ProgramScorer(config) as scorer, pytest.raises(
        ImportanceProposalBudgetExceeded,
        match="64 raw candidates",
    ):
        asyncio.run(
            LazyImportanceSMCEngine(
                config=config,
                options=_options(budget=63),
                scorer=scorer,
                candidate_scorer=raw,
                generator=make_cpu_generator(31),
            ).run()
        )
    assert raw.requests == []


def test_joint_semantic_requires_the_lazy_zero_clone_vllm_boundary() -> None:
    common = {
        "spec": MAP_SPEC,
        "mode": "importance-smc",
        "proposal": "joint-semantic",
        "model": "semantic-model",
    }
    with pytest.raises(ValueError, match="only for importance-smc"):
        build_candidate_scorer(
            SynthesizeRequest(**{**common, "mode": "paper-search"})
        )
    with pytest.raises(ValueError, match="remove --materialize-reference"):
        build_candidate_scorer(
            SynthesizeRequest(**common, materialize_reference=True)
        )
    with pytest.raises(ValueError, match="requires --alpha 0"):
        build_candidate_scorer(SynthesizeRequest(**common, alpha=0.2))

    scorer = build_candidate_scorer(SynthesizeRequest(**common, alpha=0.0))
    assert isinstance(scorer, VLLMPromptLogprobScorer)
    assert scorer.config.model == "semantic-model"
    with pytest.raises(ValueError, match="guided proposals cannot"):
        ImportanceSMCOptions(
            llm_energy_normalization=(
                LLMEnergyNormalization.SYMMETRIZED_FINAL_LABEL_LOG_ODDS
            )
        )
    config = _config()
    with ProgramScorer(config) as program_scorer, pytest.raises(
        ValueError,
        match="requires lazy factorized execution",
    ):
        ImportanceSMCEngine(
            config=config,
            options=_options(),
            scorer=program_scorer,
            candidate_scorer=_SemanticRawScorer(),
            device=resolve_device("cpu"),
            generator=make_cpu_generator(31),
        )


def test_joint_semantic_cli_seals_active_and_inactive_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _SemanticRawScorer()
    monkeypatch.setattr(execution_module, "build_candidate_scorer", lambda _request: raw)
    result = CliRunner().invoke(
        app,
        [
            "synthesize",
            str(MAP_SPEC),
            "--mode",
            "importance-smc",
            "--proposal",
            "joint-semantic",
            "--skeleton",
            "map-arithmetic",
            "--particles",
            "4",
            "--iterations",
            "1",
            "--alpha",
            "0",
            "--semantic-scale",
            "1.5",
            "--semantic-slate-size",
            "16",
            "--proposal-epsilon",
            "0.2",
            "--candidate-batch-size",
            "8",
            "--max-scored-candidates",
            "64",
            "--device",
            "cpu",
            "--artifacts-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    run_dir = next(path for path in tmp_path.iterdir() if (path / "result.json").exists())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    persisted = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["result"]
    assert manifest["probabilistic_claim"] == (
        "importance_corrected_joint_llm_semantic_proposal"
    )
    configuration = manifest["configuration"]
    assert configuration["semantic_scale"] == 1.5
    assert configuration["semantic_slate_size"] == 16
    assert configuration["proposal_epsilon"] == 0.2
    assert configuration["temperature"] is None
    assert configuration["llm_energy_normalization"] is None
    assert configuration["deduction_mix"] is None
    assert configuration["max_tokens"] is None
    assert persisted["proposal_strategy"] == "joint-semantic"
    assert persisted["semantic_score_ledger"]["raw_candidate_count"] == 64
    assert persisted["score_ledger"] == []
    assert "symmetrized final-label compatibility" in result.output

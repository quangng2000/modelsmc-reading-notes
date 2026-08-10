from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import modelsmc_pbe.shell.execution as execution_module
from modelsmc_pbe.cli import app
from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.proposals import (
    CachedCandidateScorer,
    CandidateKind,
    CandidateLogprobSemantics,
    CandidateScoreBatch,
    CandidateScoreOrigin,
    CandidateScoreProvenance,
    CandidateScoreRequest,
    CandidateSequenceScore,
    LLMEnergyNormalization,
    OpenAICompatibleProposer,
    ScoreCacheIdentity,
    ScoreCacheMode,
)
from modelsmc_pbe.proposals.json_extract import parse_canonical_expression_content
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


def test_vllm_persistent_cache_requires_and_binds_archival_metadata(
    tmp_path: Path,
) -> None:
    incomplete = SynthesizeRequest(
        spec=BOOL_SPEC,
        mode="importance-smc",
        proposal="vllm",
        model="qwen-coder",
        score_cache_dir=tmp_path,
        score_cache_mode="read-write",
    )
    with pytest.raises(ValueError, match="--model-repository"):
        build_candidate_scorer(incomplete)

    complete = SynthesizeRequest(
        spec=BOOL_SPEC,
        mode="importance-smc",
        proposal="vllm",
        model="qwen-coder",
        model_repository="Qwen/Qwen2.5-Coder-3B-Instruct",
        model_revision="488639f1",
        tokenizer_revision="488639f1",
        vllm_server_config="vllm=0.11.0;logprobs=processed_logprobs",
        score_cache_dir=tmp_path,
        score_cache_mode="read-write",
    )
    scorer = build_candidate_scorer(complete)

    assert isinstance(scorer, CachedCandidateScorer)
    assert scorer.identity.model_alias == "qwen-coder"
    assert scorer.identity.model_repository == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert scorer.identity.server_config == "vllm=0.11.0;logprobs=processed_logprobs"


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


def test_cli_cold_and_replay_cache_artifacts_preserve_budget_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    artifacts_dir = tmp_path / "artifacts"
    provider_calls: list[int] = []

    class _FiniteProvider:
        name = "vllm-prompt-logprobs"

        def __init__(self, *, fail: bool) -> None:
            self.fail = fail
            self.calls = 0

        async def score_candidates(
            self, request: CandidateScoreRequest
        ) -> CandidateScoreBatch:
            return (await self.score_many([request]))[0]

        async def score_many(
            self, requests: list[CandidateScoreRequest]
        ) -> list[CandidateScoreBatch]:
            self.calls += 1
            provider_calls.append(1)
            if self.fail:
                raise AssertionError("warm replay contacted the provider")
            return [self._batch(request) for request in requests]

        @staticmethod
        def _batch(request: CandidateScoreRequest) -> CandidateScoreBatch:
            return CandidateScoreBatch(
                scores=tuple(
                    CandidateSequenceScore(
                        candidate=candidate,
                        expression=(
                            parse_canonical_expression_content(candidate, request)
                            if request.candidate_kind is CandidateKind.EXPRESSION
                            else None
                        ),
                        token_ids=(100 + index,),
                        token_logprobs=(-0.1,),
                        sequence_logprob=-0.1,
                    )
                    for index, candidate in enumerate(request.candidates)
                ),
                source="vllm-prompt-logprobs",
                model="qwen-coder",
                semantics=CandidateLogprobSemantics.TEACHER_FORCED_FULL_PROMPT,
                model_revision="488639f1",
                tokenizer_revision="488639f1",
                provenance=CandidateScoreProvenance(CandidateScoreOrigin.PROVIDER),
            )

    build_count = 0

    def build_cached_scorer(request: SynthesizeRequest) -> CachedCandidateScorer:
        nonlocal build_count
        provider = _FiniteProvider(fail=build_count > 0)
        mode = ScoreCacheMode.READ_WRITE if build_count == 0 else ScoreCacheMode.REPLAY_ONLY
        build_count += 1
        return CachedCandidateScorer(
            provider,
            cache_dir=cache_dir,
            mode=mode,
            identity=ScoreCacheIdentity(
                scorer_name=provider.name,
                model_alias="qwen-coder",
                model_repository="Qwen/Qwen2.5-Coder-3B-Instruct",
                model_revision="488639f1",
                tokenizer_revision="488639f1",
                server_config="vllm=0.11.0;logprobs=processed_logprobs",
                semantics=CandidateLogprobSemantics.TEACHER_FORCED_FULL_PROMPT,
                energy_normalization=LLMEnergyNormalization(
                    request.llm_energy_normalization
                ),
            ),
        )

    monkeypatch.setattr(execution_module, "build_candidate_scorer", build_cached_scorer)

    common = [
        "synthesize",
        str(MAP_SPEC),
        "--mode",
        "importance-smc",
        "--proposal",
        "vllm",
        "--model",
        "qwen-coder",
        "--model-repository",
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        "--model-revision",
        "488639f1",
        "--tokenizer-revision",
        "488639f1",
        "--vllm-server-config",
        "vllm=0.11.0;logprobs=processed_logprobs",
        "--score-cache-dir",
        str(cache_dir),
        "--particles",
        "2",
        "--iterations",
        "1",
        "--alpha",
        "0",
        "--max-scored-candidates",
        "50000",
        "--device",
        "cpu",
        "--artifacts-dir",
        str(artifacts_dir),
    ]
    cold = CliRunner().invoke(app, [*common, "--score-cache-mode", "read-write"])
    warm = CliRunner().invoke(app, [*common, "--score-cache-mode", "replay-only"])
    assert cold.exit_code == 0, cold.output
    assert warm.exit_code == 0, warm.output

    runs_by_mode: dict[str, tuple[dict[str, object], Path]] = {}
    for run in artifacts_dir.iterdir():
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        runs_by_mode[manifest["configuration"]["score_cache_mode"]] = (manifest, run)
    cold_manifest, cold_run = runs_by_mode["read-write"]
    warm_manifest, warm_run = runs_by_mode["replay-only"]
    cold_result = json.loads((cold_run / "result.json").read_text(encoding="utf-8"))["result"]
    warm_result = json.loads((warm_run / "result.json").read_text(encoding="utf-8"))["result"]

    assert cold_result["scored_candidates"] == warm_result["scored_candidates"]
    assert cold_manifest["metrics"]["candidate_score_cache"]["miss_requests"] > 0
    assert cold_manifest["metrics"]["candidate_score_cache"]["provider_scored_tokens"] > 0
    assert warm_manifest["metrics"]["candidate_score_cache"]["hit_requests"] > 0
    assert warm_manifest["metrics"]["candidate_score_cache"]["provider_invocations"] == 0
    assert all(item["score_origin"] == "provider" for item in cold_result["score_ledger"])
    assert all(item["cache_hit"] is False for item in cold_result["score_ledger"])
    assert all(item["score_origin"] == "cache" for item in warm_result["score_ledger"])
    assert all(item["cache_hit"] is True for item in warm_result["score_ledger"])
    assert provider_calls

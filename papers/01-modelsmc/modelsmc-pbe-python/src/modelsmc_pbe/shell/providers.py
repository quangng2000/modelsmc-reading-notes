"""Proposal-provider construction for paper-style search runs."""

from __future__ import annotations

import os

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.grammar import enumerate_skeleton
from modelsmc_pbe.proposals import (
    CachedCandidateScorer,
    CandidateScorer,
    CatalogProposer,
    OpenAICompatibleConfig,
    OpenAICompatibleProposer,
    Proposer,
    ScoreCacheIdentity,
    ScoreCacheMode,
    UniformCandidateScorer,
    VLLMPromptLogprobConfig,
    VLLMPromptLogprobScorer,
)
from modelsmc_pbe.proposals.candidate_scoring import (
    CandidateLogprobSemantics,
    LLMEnergyNormalization,
)
from modelsmc_pbe.proposals.labels import (
    SemanticPromptProtocol,
    compatibility_add_special_tokens,
)
from modelsmc_pbe.shell.request import SynthesizeRequest
from modelsmc_pbe.shell.skeletons import resolve_skeleton

_DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "openai-compatible": "http://localhost:8000/v1",
}


def _api_key(environment_variable: str | None) -> str | None:
    if environment_variable is None:
        return None
    value = os.environ.get(environment_variable)
    if value is None:
        raise ValueError(f"API key environment variable {environment_variable!r} is not set")
    return value


def build_proposer(
    request: SynthesizeRequest,
    config: ExperimentConfig,
) -> Proposer:
    """Build a finite-catalog or OpenAI-compatible proposal adapter.

    A finite skeleton is resolved only for the catalog provider. Model-backed
    providers propose directly in the complete transport AST and therefore do
    not require a finite grammar family.
    """

    if request.proposal == "catalog":
        catalog = enumerate_skeleton(
            resolve_skeleton(config, request.skeleton),
            config.spec.integer_constants,
            request.grammar_limit,
        )
        return CatalogProposer(catalog)
    if request.proposal not in _DEFAULT_BASE_URLS:
        raise ValueError("proposal must be catalog, ollama, vllm, or openai-compatible")
    if not request.model.strip():
        raise ValueError("--model is required for a model-backed proposal")
    return OpenAICompatibleProposer(
        OpenAICompatibleConfig(
            model=request.model,
            base_url=request.base_url or _DEFAULT_BASE_URLS[request.proposal],
            api_key=_api_key(request.api_key_env),
            timeout_seconds=request.timeout_seconds,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            max_concurrency=request.max_concurrency,
        )
    )


def build_candidate_scorer(request: SynthesizeRequest) -> CandidateScorer:
    """Build the strict finite-candidate scorer used by importance-SMC."""

    if request.proposal == "joint-target":
        validate_joint_target_request(request)
        raise ValueError("joint-target bypasses the candidate-scorer interface")
    if request.proposal == "joint-semantic":
        validate_joint_semantic_request(request)
    try:
        cache_mode = ScoreCacheMode(request.score_cache_mode)
    except ValueError as error:
        raise ValueError("score cache mode must be off, read-write, or replay-only") from error
    if request.proposal == "catalog":
        if cache_mode is not ScoreCacheMode.OFF or request.score_cache_dir is not None:
            raise ValueError("persistent candidate-score caching is available only for vllm")
        return UniformCandidateScorer()
    if request.proposal not in {"vllm", "joint-semantic"}:
        raise ValueError(
            "importance-smc proposal must be catalog or vllm, joint-semantic, or the "
            "joint-target oracle; Ollama and free-form "
            "OpenAI-compatible output do not expose the required finite-candidate scores"
        )
    if not request.model.strip():
        raise ValueError("--model is required for vLLM candidate scoring")
    prompt_protocol = SemanticPromptProtocol(request.semantic_prompt_protocol)
    add_special_tokens = (
        compatibility_add_special_tokens(prompt_protocol)
        if request.proposal == "joint-semantic"
        else True
    )
    scorer = VLLMPromptLogprobScorer(
        VLLMPromptLogprobConfig(
            model=request.model,
            base_url=request.base_url or _DEFAULT_BASE_URLS["vllm"],
            api_key=_api_key(request.api_key_env),
            timeout_seconds=request.timeout_seconds,
            max_concurrency=request.max_concurrency,
            max_batch_size=request.candidate_batch_size,
            model_revision=request.model_revision,
            tokenizer_revision=request.tokenizer_revision,
            add_special_tokens=add_special_tokens,
        )
    )
    if cache_mode is ScoreCacheMode.OFF:
        if request.score_cache_dir is not None:
            raise ValueError("--score-cache-dir requires a non-off --score-cache-mode")
        return scorer
    cache_dir = request.score_cache_dir
    model_repository = request.model_repository
    model_revision = request.model_revision
    tokenizer_revision = request.tokenizer_revision
    server_config = request.vllm_server_config
    required = {
        "--score-cache-dir": cache_dir,
        "--model-repository": model_repository,
        "--model-revision": model_revision,
        "--tokenizer-revision": tokenizer_revision,
        "--vllm-server-config": server_config,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "persistent score caching requires reproducibility metadata: " + ", ".join(missing)
        )
    assert cache_dir is not None
    assert model_repository is not None
    assert model_revision is not None
    assert tokenizer_revision is not None
    assert server_config is not None
    return CachedCandidateScorer(
        scorer,
        cache_dir=cache_dir,
        mode=cache_mode,
        identity=ScoreCacheIdentity(
            scorer_name=scorer.name,
            model_alias=request.model,
            model_repository=model_repository,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            server_config=server_config,
            semantics=CandidateLogprobSemantics.TEACHER_FORCED_FULL_PROMPT,
            energy_normalization=(
                LLMEnergyNormalization.SYMMETRIZED_FINAL_LABEL_LOG_SCORE_CONTRAST
                if request.proposal == "joint-semantic"
                else LLMEnergyNormalization(request.llm_energy_normalization)
            ),
            add_special_tokens=add_special_tokens,
        ),
    )


def validate_joint_target_request(request: SynthesizeRequest) -> None:
    """Reject provider/cache settings that cannot affect the local oracle."""

    if request.mode != "importance-smc":
        raise ValueError("joint-target proposal is available only for importance-smc")
    provider_options = {
        "--model-repository": request.model_repository,
        "--model-revision": request.model_revision,
        "--tokenizer-revision": request.tokenizer_revision,
        "--vllm-server-config": request.vllm_server_config,
        "--base-url": request.base_url,
        "--api-key-env": request.api_key_env,
        "--score-cache-dir": request.score_cache_dir,
        "--score-cache-mode": (
            None if request.score_cache_mode == "off" else request.score_cache_mode
        ),
    }
    present = [name for name, value in provider_options.items() if value is not None]
    if present:
        raise ValueError(
            "joint-target uses no model provider or score cache; remove: " + ", ".join(present)
        )


def validate_joint_semantic_request(request: SynthesizeRequest) -> None:
    """Validate the lazy, model-backed semantic proposal boundary."""

    if request.mode != "importance-smc":
        raise ValueError("joint-semantic proposal is available only for importance-smc")
    if request.materialize_reference:
        raise ValueError(
            "joint-semantic executes only sampled programs; remove --materialize-reference"
        )
    if request.alpha not in {None, 0.0}:
        raise ValueError("joint-semantic proposal requires --alpha 0")
    if not request.model.strip():
        raise ValueError("--model is required for joint-semantic scoring")
    try:
        SemanticPromptProtocol(request.semantic_prompt_protocol)
    except ValueError as error:
        raise ValueError("semantic prompt protocol must be raw-v2 or harmony-gpt-oss-v1") from error

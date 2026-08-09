"""Proposal-provider construction for paper-style search runs."""

from __future__ import annotations

import os

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.grammar import enumerate_skeleton
from modelsmc_pbe.proposals import (
    CandidateScorer,
    CatalogProposer,
    OpenAICompatibleConfig,
    OpenAICompatibleProposer,
    Proposer,
    UniformCandidateScorer,
    VLLMPromptLogprobConfig,
    VLLMPromptLogprobScorer,
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
        raise ValueError(
            f"API key environment variable {environment_variable!r} is not set"
        )
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
        raise ValueError(
            "proposal must be catalog, ollama, vllm, or openai-compatible"
        )
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

    if request.proposal == "catalog":
        return UniformCandidateScorer()
    if request.proposal != "vllm":
        raise ValueError(
            "importance-smc proposal must be catalog or vllm; Ollama and free-form "
            "OpenAI-compatible output do not expose the required finite-candidate scores"
        )
    if not request.model.strip():
        raise ValueError("--model is required for vLLM candidate scoring")
    return VLLMPromptLogprobScorer(
        VLLMPromptLogprobConfig(
            model=request.model,
            base_url=request.base_url or _DEFAULT_BASE_URLS["vllm"],
            api_key=_api_key(request.api_key_env),
            timeout_seconds=request.timeout_seconds,
            max_concurrency=request.max_concurrency,
            max_batch_size=request.candidate_batch_size,
        )
    )

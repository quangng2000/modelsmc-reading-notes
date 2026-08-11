"""Reusable Typer metadata for the public ``synthesize`` command."""

from pathlib import Path
from typing import Annotated

import typer

SpecArgument = Annotated[
    Path,
    typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="PBE experiment JSON.",
    ),
]
ModeOption = Annotated[
    str,
    typer.Option(help="paper-search, grammar-smc, or importance-smc."),
]
ProposalOption = Annotated[
    str,
    typer.Option(
        help=(
            "Proposal backend: importance-smc accepts vllm, catalog, the LLM-backed "
            "joint-semantic proposal, or the materialized joint-target oracle; "
            "paper-search also accepts ollama or openai-compatible."
        )
    ),
]
ModelOption = Annotated[str, typer.Option(help="Model served by Ollama or vLLM.")]
ModelRepositoryOption = Annotated[
    str | None,
    typer.Option(help="Exact Hugging Face repository behind the served model alias."),
]
ModelRevisionOption = Annotated[
    str | None,
    typer.Option(help="Archival model revision or commit; does not change provider loading."),
]
TokenizerRevisionOption = Annotated[
    str | None,
    typer.Option(help="Archival tokenizer revision; does not change provider loading."),
]
BaseUrlOption = Annotated[
    str | None,
    typer.Option(help="Override the OpenAI-compatible /v1 base URL."),
]
ApiKeyEnvOption = Annotated[
    str | None,
    typer.Option(help="Environment variable containing the provider API key."),
]
VLLMServerConfigOption = Annotated[
    str | None,
    typer.Option(
        help=(
            "Immutable vLLM/runtime scoring fingerprint, including version and "
            "logprob mode; required when the persistent score cache is enabled."
        )
    ),
]
ScoreCacheDirOption = Annotated[
    Path | None,
    typer.Option(file_okay=False, help="Persistent content-addressed score cache directory."),
]
ScoreCacheModeOption = Annotated[
    str,
    typer.Option(help="Persistent score cache policy: off, read-write, or replay-only."),
]
SkeletonOption = Annotated[
    str,
    typer.Option(
        help=(
            "Skeleton policy: importance auto keeps every viable family; general uses "
            "generic families; an explicit name conditions on one family."
        )
    ),
]
ParticlesOption = Annotated[
    int | None,
    typer.Option(min=1, help="Override the specification particle count."),
]
IterationsOption = Annotated[
    int | None,
    typer.Option(min=1, help="Override the specification iteration count."),
]
AlphaOption = Annotated[
    float | None,
    typer.Option(min=0.0, max=1.0, help="Probability of cloning an ancestor."),
]
EssThresholdOption = Annotated[
    float | None,
    typer.Option(min=0.0, max=1.0, help="Relative ESS resampling threshold."),
]
SeedOption = Annotated[
    int | None,
    typer.Option(min=0, help="Override the reproducible random seed."),
]
BetaMaxOption = Annotated[
    float,
    typer.Option(min=0.0, help="Final inverse temperature for calibrated SMC modes."),
]
MovesPerStageOption = Annotated[
    int,
    typer.Option(min=0, help="Independent prior-MH moves per grammar stage."),
]
GrammarLimitOption = Annotated[
    int,
    typer.Option(min=1, help="Fail if a skeleton exceeds this many ASTs."),
]
ScoreBatchSizeOption = Annotated[
    int,
    typer.Option(min=1, max=10_000, help="Semantic scoring batch size."),
]
CandidateBatchSizeOption = Annotated[
    int,
    typer.Option(min=1, help="Canonical candidates per vLLM scoring request."),
]
MaxScoredCandidatesOption = Annotated[
    int,
    typer.Option(
        min=1,
        help="Abort before total finite candidate scores exceed this run budget.",
    ),
]
HoleMaxCostOption = Annotated[
    int,
    typer.Option(min=1, help="Maximum exact structural cost of each enumerated hole."),
]
HoleStateLimitOption = Annotated[
    int,
    typer.Option(min=1, help="Fail if one complete typed hole catalog exceeds this size."),
]
SupportLimitOption = Annotated[
    int,
    typer.Option(
        min=1,
        help="Fail if complete program construction traces exceed this size.",
    ),
]
MaterializeReferenceOption = Annotated[
    bool,
    typer.Option(
        "--materialize-reference",
        help=(
            "Materialize and score every complete importance-support state to run the "
            "exact finite reference control. Disabled by default."
        ),
    ),
]
ProposalEpsilonOption = Annotated[
    float,
    typer.Option(
        min=0.000001,
        max=1.0,
        help=(
            "Defensive proposal mass: uniform for guided catalogs and the exact "
            "Occam prior for joint-semantic."
        ),
    ),
]
SemanticScaleOption = Annotated[
    float,
    typer.Option(
        min=0.0,
        help="Strength eta of the symmetrized LLM final-label log-score contrast.",
    ),
]
SemanticSlateSizeOption = Annotated[
    int | None,
    typer.Option(
        min=1,
        help=(
            "Deterministic number of complete traces scored by joint-semantic; "
            "omit to score the full bounded support."
        ),
    ),
]
DeductionMixOption = Annotated[
    float,
    typer.Option(
        min=0.0,
        max=1.0,
        help="Mass assigned to the exact deduction guide before the uniform floor.",
    ),
]
FamilyDeductionMixOption = Annotated[
    float | None,
    typer.Option(
        min=0.0,
        max=1.0,
        help="Override deduction-guide mass for structural family choices.",
    ),
]
HoleDeductionMixOption = Annotated[
    float | None,
    typer.Option(
        min=0.0,
        max=1.0,
        help="Override deduction-guide mass for typed hole choices.",
    ),
]
DeductionStrengthOption = Annotated[
    float,
    typer.Option(
        min=0.0,
        help="Final penalty per violated derived hole example in the deduction guide.",
    ),
]
TemperatureOption = Annotated[
    float,
    typer.Option(
        min=0.0,
        max=2.0,
        help="Generation temperature, or local finite-categorical temperature.",
    ),
]
LLMEnergyNormalizationOption = Annotated[
    str,
    typer.Option(
        help=(
            "Qwen finite energy: total-full-prompt-logprob or "
            "mean-full-prompt-conditional-logprob."
        )
    ),
]
MaxTokensOption = Annotated[
    int,
    typer.Option(min=1, help="Maximum completion tokens per proposal."),
]
MaxConcurrencyOption = Annotated[
    int,
    typer.Option(min=1, help="Concurrent proposal requests for server batching."),
]
TimeoutSecondsOption = Annotated[
    float,
    typer.Option(min=0.1, help="Timeout for each LLM proposal request."),
]
DeviceOption = Annotated[
    str | None,
    typer.Option(help="auto, cpu, mps, cuda, or cuda:<index>."),
]
ArtifactsDirOption = Annotated[
    Path | None,
    typer.Option(file_okay=False, help="Directory that receives run artifacts."),
]
TraceOption = Annotated[
    bool,
    typer.Option("--trace", help="Render detailed structured events to the console."),
]

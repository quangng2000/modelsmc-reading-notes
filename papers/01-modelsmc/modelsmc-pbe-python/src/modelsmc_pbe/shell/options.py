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
            "Proposal backend: importance-smc accepts vllm or catalog; "
            "paper-search also accepts ollama or openai-compatible."
        )
    ),
]
ModelOption = Annotated[str, typer.Option(help="Model served by Ollama or vLLM.")]
BaseUrlOption = Annotated[
    str | None,
    typer.Option(help="Override the OpenAI-compatible /v1 base URL."),
]
ApiKeyEnvOption = Annotated[
    str | None,
    typer.Option(help="Environment variable containing the provider API key."),
]
SkeletonOption = Annotated[
    str,
    typer.Option(help="Finite grammar skeleton, or auto."),
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
    typer.Option(min=0.0, help="Final inverse temperature for grammar SMC."),
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
    typer.Option(min=1, help="Fail if complete program construction support exceeds this size."),
]
ProposalEpsilonOption = Annotated[
    float,
    typer.Option(
        min=0.000001,
        max=1.0,
        help="Uniform mixture mass guaranteeing finite proposal support.",
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

"""Typer command adapter for the synthesis application service."""

from __future__ import annotations

import typer

from modelsmc_pbe.shell.options import (
    AlphaOption,
    ApiKeyEnvOption,
    ArtifactsDirOption,
    BaseUrlOption,
    BetaMaxOption,
    DeviceOption,
    EssThresholdOption,
    GrammarLimitOption,
    IterationsOption,
    MaxConcurrencyOption,
    MaxTokensOption,
    ModelOption,
    ModeOption,
    MovesPerStageOption,
    ParticlesOption,
    ProposalOption,
    ScoreBatchSizeOption,
    SeedOption,
    SkeletonOption,
    SpecArgument,
    TemperatureOption,
    TimeoutSecondsOption,
    TraceOption,
)
from modelsmc_pbe.shell.request import SynthesizeRequest
from modelsmc_pbe.shell.runner import run_synthesis


def synthesize(
    spec: SpecArgument,
    mode: ModeOption = "paper-search",
    proposal: ProposalOption = "catalog",
    model: ModelOption = "qwen3-coder:30b-a3b-q8_0",
    base_url: BaseUrlOption = None,
    api_key_env: ApiKeyEnvOption = None,
    skeleton: SkeletonOption = "auto",
    particles: ParticlesOption = None,
    iterations: IterationsOption = None,
    alpha: AlphaOption = None,
    ess_threshold: EssThresholdOption = None,
    seed: SeedOption = None,
    beta_max: BetaMaxOption = 1.0,
    moves_per_stage: MovesPerStageOption = 1,
    grammar_limit: GrammarLimitOption = 250_000,
    score_batch_size: ScoreBatchSizeOption = 512,
    temperature: TemperatureOption = 0.7,
    max_tokens: MaxTokensOption = 4_096,
    max_concurrency: MaxConcurrencyOption = 8,
    timeout_seconds: TimeoutSecondsOption = 300.0,
    device: DeviceOption = None,
    artifacts_dir: ArtifactsDirOption = None,
    trace: TraceOption = False,
) -> None:
    """Synthesize and score a typed program from input/output examples."""

    request = SynthesizeRequest(
        spec=spec,
        mode=mode,
        proposal=proposal,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        skeleton=skeleton,
        particles=particles,
        iterations=iterations,
        alpha=alpha,
        ess_threshold=ess_threshold,
        seed=seed,
        beta_max=beta_max,
        moves_per_stage=moves_per_stage,
        grammar_limit=grammar_limit,
        score_batch_size=score_batch_size,
        temperature=temperature,
        max_tokens=max_tokens,
        max_concurrency=max_concurrency,
        timeout_seconds=timeout_seconds,
        device=device,
        artifacts_dir=artifacts_dir,
        trace=trace,
    )
    try:
        run_synthesis(request)
    except (OSError, RuntimeError, ValueError) as error:
        typer.secho(f"error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error

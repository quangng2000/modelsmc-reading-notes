"""Transport object for one ``synthesize`` command invocation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SynthesizeRequest:
    """CLI values passed to the application runner without Typer dependencies."""

    spec: Path
    mode: str = "paper-search"
    proposal: str = "catalog"
    model: str = "qwen3-coder:30b-a3b-q8_0"
    base_url: str | None = None
    api_key_env: str | None = None
    skeleton: str = "auto"
    particles: int | None = None
    iterations: int | None = None
    alpha: float | None = None
    ess_threshold: float | None = None
    seed: int | None = None
    beta_max: float = 1.0
    moves_per_stage: int = 1
    grammar_limit: int = 250_000
    score_batch_size: int = 512
    temperature: float = 0.7
    max_tokens: int = 4_096
    max_concurrency: int = 8
    timeout_seconds: float = 300.0
    device: str | None = None
    artifacts_dir: Path | None = None
    trace: bool = False

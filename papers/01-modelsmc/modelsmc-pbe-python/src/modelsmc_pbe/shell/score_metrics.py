"""Candidate-score cache and provider accounting for shell executions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict

from modelsmc_pbe.observability import RunLogger
from modelsmc_pbe.proposals import (
    CachedCandidateScorer,
    CandidateScorer,
    ProviderMetricSource,
)


@contextmanager
def candidate_score_metrics(
    scorer: CandidateScorer | None,
    logger: RunLogger,
) -> Iterator[None]:
    """Persist cache/provider accounting even when a scoring run fails."""

    try:
        yield
    finally:
        if isinstance(scorer, ProviderMetricSource):
            provider_metrics = asdict(scorer.provider_metrics())
            logger.record_metrics("candidate_score_provider", provider_metrics)
            logger.event(
                "candidate_score_provider.summary",
                message="candidate-score provider I/O accounting",
                level="info",
                **provider_metrics,
            )
        if isinstance(scorer, CachedCandidateScorer):
            metrics = scorer.metrics()
            summary = {
                "mode": scorer.mode.value,
                "cache_dir": str(scorer.cache_dir),
                **asdict(metrics),
            }
            logger.record_metrics("candidate_score_cache", summary)
            logger.event(
                "candidate_score_cache.summary",
                message="persistent candidate-score cache accounting",
                level="info",
                **summary,
            )
        else:
            logger.record_metrics(
                "candidate_score_cache",
                {
                    "mode": "off",
                    "cache_dir": None,
                    "lookup_requests": 0,
                    "lookup_candidates": 0,
                    "hit_requests": 0,
                    "hit_candidates": 0,
                    "miss_requests": 0,
                    "miss_candidates": 0,
                    "provider_invocations": 0,
                    "provider_score_requests": 0,
                    "provider_candidates": 0,
                    "provider_failures": 0,
                    "cache_served_scored_tokens": 0,
                    "provider_scored_tokens": 0,
                    "provider_await_wall_seconds": 0.0,
                },
            )

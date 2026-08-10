"""Mode-specific application services behind the command-line shell."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.core import ProgramScorer
from modelsmc_pbe.observability import RunLogger
from modelsmc_pbe.proposals import (
    CachedCandidateScorer,
    CandidateScorer,
    LLMEnergyNormalization,
    ProviderMetricSource,
)
from modelsmc_pbe.runtime import DeviceInfo, seed_everything
from modelsmc_pbe.search import PaperSearchEngine
from modelsmc_pbe.search.grammar_smc import GrammarSMCEngine, GrammarSMCOptions
from modelsmc_pbe.search.importance import (
    ImportanceSMCEngine,
    ImportanceSMCOptions,
    LazyImportanceSMCEngine,
)
from modelsmc_pbe.shell.output import ProgramResultView
from modelsmc_pbe.shell.providers import build_candidate_scorer, build_proposer
from modelsmc_pbe.shell.request import SynthesizeRequest
from modelsmc_pbe.shell.skeletons import (
    importance_uses_multiple_families,
    resolve_importance_skeleton,
    resolve_skeleton,
)


@dataclass(frozen=True, slots=True)
class CompletedRun:
    """Search-engine-neutral values needed to finalize and present one run."""

    persisted_result: object
    final_particles: tuple[object, ...]
    view: ProgramResultView


@contextmanager
def _candidate_score_metrics(scorer: CandidateScorer, logger: RunLogger) -> Iterator[None]:
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
            summary = {
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
            }
            logger.record_metrics("candidate_score_cache", summary)


def execute_importance_smc(
    request: SynthesizeRequest,
    config: ExperimentConfig,
    device: DeviceInfo,
    logger: RunLogger,
) -> CompletedRun:
    """Run lazy factorized SMC, or the explicit materialized reference control."""

    seeded = seed_everything(
        config.smc.seed,
        deterministic=config.runtime.deterministic,
    )
    options = ImportanceSMCOptions(
        hole_max_cost=request.hole_max_cost,
        hole_state_limit=request.hole_state_limit,
        support_limit=request.support_limit,
        score_batch_size=request.score_batch_size,
        proposal_temperature=request.temperature,
        proposal_epsilon=request.proposal_epsilon,
        deduction_mix=request.deduction_mix,
        deduction_strength=request.deduction_strength,
        beta_max=request.beta_max,
        max_scored_candidates=request.max_scored_candidates,
        conditioned_skeleton=resolve_importance_skeleton(config, request.skeleton),
        multi_family=importance_uses_multiple_families(request.skeleton),
        llm_energy_normalization=LLMEnergyNormalization(request.llm_energy_normalization),
    )
    candidate_scorer = build_candidate_scorer(request)
    with _candidate_score_metrics(candidate_scorer, logger), ProgramScorer(config) as scorer:
        if not request.materialize_reference:
            lazy = asyncio.run(
                LazyImportanceSMCEngine(
                    config=config,
                    options=options,
                    scorer=scorer,
                    candidate_scorer=candidate_scorer,
                    generator=seeded.cpu_generator,
                    logger=logger,
                ).run()
            )
            lazy_best = lazy.best_visited
            family_details = tuple(
                "family "
                f"{family.family}: traces={family.support_states} "
                f"prior={family.prior_mass:.5g} guide={family.deduction_guide_mass:.5g} "
                f"particles={family.particle_mass:.5g}"
                for family in lazy.families
            )
            return CompletedRun(
                persisted_result=lazy,
                final_particles=lazy.final_particles,
                view=ProgramResultView(
                    mode=lazy.mode,
                    exact=lazy.exact,
                    program=lazy_best.program,
                    total_loss=lazy_best.total_loss,
                    cost=lazy_best.cost,
                    details=(
                        "execution=lazy-factorized (exact reference disabled)",
                        f"proposal={lazy.proposal_source}",
                        f"support traces={lazy.support_states} materialized=false",
                        "visited programs: "
                        f"{lazy.search.evaluated_programs}/{lazy.support_states} "
                        f"({lazy.search.evaluated_fraction_of_support:.3%})",
                        "search success: "
                        f"exact-found={lazy.search.exact_found} "
                        f"final-exact-mass={lazy.search.final_exact_particle_mass:.7g}",
                        "reference metrics=unavailable; rerun with "
                        "--materialize-reference for the external finite control",
                        "candidate scores: "
                        f"used={lazy.scored_candidates} limit={lazy.max_scored_candidates}",
                        *family_details,
                    ),
                ),
            )
        result = asyncio.run(
            ImportanceSMCEngine(
                config=config,
                options=options,
                scorer=scorer,
                candidate_scorer=candidate_scorer,
                device=device,
                generator=seeded.cpu_generator,
                logger=logger,
            ).run()
        )
    reference_best = result.sampled_best
    if result.multi_family:
        family_mode = "auto-multi-family"
    elif result.conditioned_skeleton is not None:
        family_mode = f"conditioned:{result.conditioned_skeleton}"
    else:
        family_mode = "general-generic"
    family_details = tuple(
        "family "
        f"{family.family}: states={family.states} prior={family.prior_mass:.5g} "
        f"guide={family.deduction_guide_mass:.5g} "
        f"target={family.posterior_mass:.5g} particles={family.particle_mass:.5g}"
        for family in result.families
    )
    return CompletedRun(
        persisted_result=result,
        final_particles=result.final_particles,
        view=ProgramResultView(
            mode=result.mode,
            exact=result.exact,
            program=reference_best.program,
            total_loss=reference_best.total_loss,
            cost=reference_best.cost,
            details=(
                f"proposal={result.proposal_source}",
                "proposal mixture: "
                f"deduction={result.deduction_mix:.5g} "
                f"strength={result.deduction_strength:.5g}",
                f"LLM energy={result.llm_energy_normalization.value}",
                f"deduction-guide exact-program mass={result.deduction_guide_exact_mass:.7g}",
                f"family mode={family_mode} aliased-programs={result.aliased_programs}",
                f"support states={result.support_states} exact-programs={result.exact_programs}",
                "candidate scores: "
                f"used={result.scored_candidates} limit={result.max_scored_candidates}",
                "exact-program mass: "
                f"particles={result.reference.particle_exact_mass:.7g} "
                f"enumeration={result.reference.enumeration_exact_mass:.7g}",
                f"total-variation distance={result.reference.total_variation_distance:.7g}",
                "path log-normalizer: "
                f"particles={result.reference.log_path_z_estimate:.7g} "
                f"enumeration={result.reference.log_path_z_enumeration:.7g} "
                f"error={result.reference.log_path_z_error:.7g}",
                *family_details,
            ),
        ),
    )


def execute_grammar_smc(
    request: SynthesizeRequest,
    config: ExperimentConfig,
    device: DeviceInfo,
    logger: RunLogger,
) -> CompletedRun:
    """Run the calibrated finite-skeleton control experiment."""

    seeded = seed_everything(
        config.smc.seed,
        deterministic=config.runtime.deterministic,
    )
    options = GrammarSMCOptions(
        skeleton=resolve_skeleton(config, request.skeleton),
        state_limit=request.grammar_limit,
        beta_max=request.beta_max,
        moves_per_stage=request.moves_per_stage,
        score_batch_size=request.score_batch_size,
    )
    with ProgramScorer(config) as scorer:
        result = GrammarSMCEngine(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
            device=device,
            generator=seeded.cpu_generator,
            logger=logger,
        ).run()
    best = result.sampled_best
    return CompletedRun(
        persisted_result=result,
        final_particles=result.final_particles,
        view=ProgramResultView(
            mode=result.mode,
            exact=result.exact,
            program=best.program,
            total_loss=best.total_loss,
            cost=best.cost,
        ),
    )


def execute_paper_search(
    request: SynthesizeRequest,
    config: ExperimentConfig,
    device: DeviceInfo,
    logger: RunLogger,
) -> CompletedRun:
    """Run the practical ModelSMC-inspired resample-and-revise search."""

    seeded = seed_everything(
        config.smc.seed,
        deterministic=config.runtime.deterministic,
    )
    proposer = build_proposer(request, config)
    with ProgramScorer(config) as scorer:
        result = asyncio.run(
            PaperSearchEngine(
                config=config,
                scorer=scorer,
                proposer=proposer,
                device=device,
                generator=seeded.cpu_generator,
                logger=logger,
            ).run()
        )
    champion = result.champion
    return CompletedRun(
        persisted_result=result.as_dict(),
        final_particles=tuple(result.final_particle_records()),
        view=ProgramResultView(
            mode="paper-search",
            exact=result.exact,
            program=champion.program,
            total_loss=champion.score.total_loss,
            cost=champion.score.cost,
            degraded=result.degraded,
        ),
    )

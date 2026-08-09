"""Mode-specific application services behind the command-line shell."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.core import ProgramScorer
from modelsmc_pbe.observability import RunLogger
from modelsmc_pbe.runtime import DeviceInfo, seed_everything
from modelsmc_pbe.search import PaperSearchEngine
from modelsmc_pbe.search.grammar_smc import GrammarSMCEngine, GrammarSMCOptions
from modelsmc_pbe.search.importance import ImportanceSMCEngine, ImportanceSMCOptions
from modelsmc_pbe.shell.output import ProgramResultView
from modelsmc_pbe.shell.providers import build_candidate_scorer, build_proposer
from modelsmc_pbe.shell.request import SynthesizeRequest
from modelsmc_pbe.shell.skeletons import resolve_skeleton


@dataclass(frozen=True, slots=True)
class CompletedRun:
    """Search-engine-neutral values needed to finalize and present one run."""

    persisted_result: object
    final_particles: tuple[object, ...]
    view: ProgramResultView


def execute_importance_smc(
    request: SynthesizeRequest,
    config: ExperimentConfig,
    device: DeviceInfo,
    logger: RunLogger,
) -> CompletedRun:
    """Run finite-support SMC with an explicitly evaluated proposal density."""

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
        beta_max=request.beta_max,
    )
    candidate_scorer = build_candidate_scorer(request)
    with ProgramScorer(config) as scorer:
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
            details=(
                f"proposal={result.proposal_source}",
                f"support states={result.support_states} exact-programs={result.exact_programs}",
                "exact-program mass: "
                f"particles={result.reference.particle_exact_mass:.7g} "
                f"enumeration={result.reference.enumeration_exact_mass:.7g}",
                f"total-variation distance={result.reference.total_variation_distance:.7g}",
                "path log-normalizer: "
                f"particles={result.reference.log_path_z_estimate:.7g} "
                f"enumeration={result.reference.log_path_z_enumeration:.7g} "
                f"error={result.reference.log_path_z_error:.7g}",
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

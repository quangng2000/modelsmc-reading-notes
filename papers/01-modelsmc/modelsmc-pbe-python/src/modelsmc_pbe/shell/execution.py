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
from modelsmc_pbe.shell.output import ProgramResultView
from modelsmc_pbe.shell.providers import build_proposer
from modelsmc_pbe.shell.request import SynthesizeRequest
from modelsmc_pbe.shell.skeletons import resolve_skeleton


@dataclass(frozen=True, slots=True)
class CompletedRun:
    """Search-engine-neutral values needed to finalize and present one run."""

    persisted_result: object
    final_particles: tuple[object, ...]
    view: ProgramResultView


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

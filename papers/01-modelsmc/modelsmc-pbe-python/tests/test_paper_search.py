from __future__ import annotations

import asyncio
from pathlib import Path

from modelsmc_pbe.config import ExperimentConfig, load_experiment_config
from modelsmc_pbe.core import ProgramScorer
from modelsmc_pbe.proposals import ProposalError, ScriptedProposer
from modelsmc_pbe.runtime import resolve_device, seed_everything
from modelsmc_pbe.search import PaperSearchEngine, PaperSearchResult

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_DIR / "examples" / "map-increment.json"

MAP_INCREMENT = {
    "kind": "MapProgram",
    "mapper": {
        "kind": "Add",
        "left": {"kind": "Item"},
        "right": {"kind": "IntLiteral", "intValue": "1"},
    },
}


def _config() -> ExperimentConfig:
    return load_experiment_config(
        SPEC_PATH,
        smc_overrides={"particles": 2, "iterations": 1, "clone_probability": 0.0},
        runtime_overrides={"device": "cpu"},
    )


def test_paper_search_finds_and_retains_exact_champion() -> None:
    config = _config()
    seed = seed_everything(config.smc.seed)
    proposer = ScriptedProposer([MAP_INCREMENT, MAP_INCREMENT])

    async def run() -> PaperSearchResult:
        with ProgramScorer(config) as scorer:
            return await PaperSearchEngine(
                config=config,
                scorer=scorer,
                proposer=proposer,
                device=resolve_device("cpu"),
                generator=seed.cpu_generator,
            ).run()

    result = asyncio.run(run())

    assert result.exact is True
    assert result.champion.program == MAP_INCREMENT
    assert result.champion.score.total_loss == 0
    assert result.proposal_calls == 2
    assert result.proposal_responses == 2
    assert result.proposal_errors == 0
    assert result.scorer_rejections == 0
    assert result.accepted_proposals == 2
    assert result.degraded is False
    assert result.first_exact_iteration == 1
    assert all(len(particle.lineage) == 2 for particle in result.particles)
    assert sum(particle.weight for particle in result.particles) == 1.0


def test_proposal_failure_and_scorer_rejection_fall_back_to_ancestor() -> None:
    config = _config()
    seed = seed_everything(config.smc.seed)
    wrong_type = {
        "kind": "ExpressionProgram",
        "body": {"kind": "BoolLiteral", "boolValue": True},
    }
    proposer = ScriptedProposer([ProposalError("offline"), wrong_type])

    async def run() -> PaperSearchResult:
        with ProgramScorer(config) as scorer:
            return await PaperSearchEngine(
                config=config,
                scorer=scorer,
                proposer=proposer,
                device=resolve_device("cpu"),
                generator=seed.cpu_generator,
            ).run()

    result = asyncio.run(run())

    assert result.exact is False
    assert {particle.source for particle in result.particles} == {
        "proposal-error-fallback",
        "scorer-rejection-fallback",
    }
    assert result.proposal_calls == 2
    assert result.proposal_responses == 1
    assert result.proposal_errors == 1
    assert result.scorer_rejections == 1
    assert result.accepted_proposals == 0
    assert result.degraded is False
    assert all(particle.program["kind"] == "ExpressionProgram" for particle in result.particles)


def test_all_provider_failures_mark_search_as_degraded() -> None:
    config = _config()
    seed = seed_everything(config.smc.seed)
    proposer = ScriptedProposer(
        [ProposalError("offline-1"), ProposalError("offline-2")]
    )

    async def run() -> PaperSearchResult:
        with ProgramScorer(config) as scorer:
            return await PaperSearchEngine(
                config=config,
                scorer=scorer,
                proposer=proposer,
                device=resolve_device("cpu"),
                generator=seed.cpu_generator,
            ).run()

    result = asyncio.run(run())

    assert result.degraded is True
    assert result.proposal_calls == 2
    assert result.proposal_responses == 0
    assert result.proposal_errors == 2
    assert result.scorer_rejections == 0
    assert result.accepted_proposals == 0

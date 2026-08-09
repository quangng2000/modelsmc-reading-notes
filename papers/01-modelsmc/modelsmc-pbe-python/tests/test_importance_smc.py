from __future__ import annotations

import asyncio
import math
from pathlib import Path

import pytest

from modelsmc_pbe.config import ExperimentConfig, SMCConfig, load_experiment_config
from modelsmc_pbe.core import ProgramScorer
from modelsmc_pbe.proposals import (
    CandidateScoreBatch,
    CandidateScoreRequest,
    ProposalError,
    UniformCandidateScorer,
)
from modelsmc_pbe.runtime import make_cpu_generator, resolve_device
from modelsmc_pbe.search.importance import (
    IMPORTANCE_SMC_CLAIM,
    ImportanceSMCEngine,
    ImportanceSMCOptions,
)
from modelsmc_pbe.search.importance.proposal import FiniteLLMProposalKernel
from modelsmc_pbe.search.importance.support import ImportanceSupportBuilder

PROJECT_DIR = Path(__file__).resolve().parents[1]
MAP_SPEC = PROJECT_DIR / "examples" / "map-increment.json"
BOOL_SPEC = PROJECT_DIR / "examples" / "negative-int-to-bool.json"


def _config(
    path: Path,
    *,
    particles: int = 256,
    iterations: int = 3,
    alpha: float = 0.25,
    seed: int = 19,
) -> ExperimentConfig:
    experiment = load_experiment_config(path)
    smc = SMCConfig.model_validate(
        {
            **experiment.smc.model_dump(),
            "particles": particles,
            "iterations": iterations,
            "clone_probability": alpha,
            "ess_threshold": 0.8,
            "seed": seed,
        }
    )
    return experiment.model_copy(update={"smc": smc})


def test_fixed_support_runs_induction_deduction_and_rejects_aliases() -> None:
    config = _config(MAP_SPEC, particles=8, iterations=1, alpha=0.0)
    options = ImportanceSMCOptions(hole_max_cost=3)

    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()

    assert len(support.induction.hypotheses) == 3
    assert len(support.families) == 3
    assert len(support.states) == 280
    assert support.exact_programs == 3
    assert support.rejected_programs == 0
    assert len({state.key for state in support.states}) == len(support.states)
    catalog_sizes = [
        (item.family, item.hole_name, item.returned_expressions)
        for item in support.hole_catalogs
    ]
    assert catalog_sizes == [
        ("expression", "body", 14),
        ("map", "mapper", 154),
        ("foldr", "initial", 7),
        ("foldr", "reducer", 16),
    ]


def test_clone_mixture_includes_both_routes_to_the_ancestor() -> None:
    config = _config(BOOL_SPEC, particles=8, iterations=1, alpha=0.5, seed=7)
    options = ImportanceSMCOptions(hole_max_cost=1, proposal_epsilon=0.1)
    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()
    assert len(support.families) == 1
    assert len(support.states) == 2
    ancestor = support.states[0].state_index
    kernel = FiniteLLMProposalKernel(
        config=config,
        options=options,
        support=support,
        scorer=UniformCandidateScorer(),
        generator=make_cpu_generator(7),
    )

    proposals = asyncio.run(
        kernel.sample_many(tuple(ancestor for _ in range(32)), stage=1, beta=1.0)
    )
    evaluated = asyncio.run(
        kernel.evaluate_many(
            ancestor,
            tuple(state.state_index for state in support.states),
            stage=1,
            beta=1.0,
        )
    )

    assert any(proposal.cloned for proposal in proposals)
    assert any(proposal.state_index != ancestor for proposal in proposals)
    for proposal in proposals:
        expected = 0.75 if proposal.state_index == ancestor else 0.25
        assert math.exp(proposal.log_q_mixture) == pytest.approx(expected)
        assert math.exp(proposal.log_q_llm) == pytest.approx(0.5)
    assert sum(math.exp(proposal.log_q_mixture) for proposal in evaluated) == pytest.approx(
        1.0
    )


def test_importance_smc_matches_the_declared_finite_target() -> None:
    config = _config(MAP_SPEC, particles=512, iterations=4, alpha=0.25, seed=29)
    options = ImportanceSMCOptions(hole_max_cost=3)

    with ProgramScorer(config) as scorer:
        result = asyncio.run(
            ImportanceSMCEngine(
                config=config,
                options=options,
                scorer=scorer,
                candidate_scorer=UniformCandidateScorer(),
                device=resolve_device("cpu"),
                generator=make_cpu_generator(config.smc.seed),
            ).run()
        )

    assert result.mode == "importance-smc"
    assert result.probabilistic_claim == IMPORTANCE_SMC_CLAIM
    assert result.proposal_source == "uniform-finite-candidates"
    assert result.support_states == 280
    assert result.exact_programs == 3
    assert result.exact
    assert result.reference.enumeration_exact_mass > 0.999
    assert result.reference.particle_exact_mass > 0.99
    assert math.isfinite(result.reference.log_path_z_estimate)
    assert math.isfinite(result.reference.log_path_z_enumeration)
    assert sum(particle.weight for particle in result.final_particles) == pytest.approx(1.0)
    assert any(stage.resampled for stage in result.stages)
    assert any(stage.clones > 0 for stage in result.stages)
    assert all(stage.min_log_q < 0.0 for stage in result.stages)


def test_alpha_one_is_rejected_because_it_destroys_proposal_support() -> None:
    config = _config(BOOL_SPEC, particles=4, iterations=1, alpha=1.0)
    options = ImportanceSMCOptions(hole_max_cost=1)
    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()

    with pytest.raises(ValueError, match="alpha < 1"):
        FiniteLLMProposalKernel(
            config=config,
            options=options,
            support=support,
            scorer=UniformCandidateScorer(),
            generator=make_cpu_generator(1),
        )


def test_provider_failure_aborts_instead_of_falling_back_to_the_ancestor() -> None:
    class FailingCandidateScorer:
        name = "always-fails"

        async def score_candidates(
            self, request: CandidateScoreRequest
        ) -> CandidateScoreBatch:
            del request
            raise ProposalError("provider unavailable")

        async def score_many(
            self, requests: list[CandidateScoreRequest]
        ) -> list[CandidateScoreBatch]:
            del requests
            raise ProposalError("provider unavailable")

    config = _config(BOOL_SPEC, particles=4, iterations=1, alpha=0.0)
    with ProgramScorer(config) as scorer, pytest.raises(
        ProposalError, match="provider unavailable"
    ):
        asyncio.run(
            ImportanceSMCEngine(
                config=config,
                options=ImportanceSMCOptions(hole_max_cost=1),
                scorer=scorer,
                candidate_scorer=FailingCandidateScorer(),
                device=resolve_device("cpu"),
                generator=make_cpu_generator(config.smc.seed),
            ).run()
        )

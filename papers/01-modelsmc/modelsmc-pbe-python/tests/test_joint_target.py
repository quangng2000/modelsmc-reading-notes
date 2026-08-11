"""Mathematical and engine tests for the materialized joint-target oracle."""

from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from modelsmc_pbe.config import ExperimentConfig, SMCConfig, load_experiment_config
from modelsmc_pbe.core import ProgramScorer
from modelsmc_pbe.proposals import UniformCandidateScorer
from modelsmc_pbe.runtime import make_cpu_generator, resolve_device
from modelsmc_pbe.search.importance import (
    JOINT_TARGET_IMPORTANCE_SMC_CLAIM,
    JOINT_TARGET_PROPOSAL_SOURCE,
    FiniteJointTargetProposalKernel,
    ImportanceSMCEngine,
    ImportanceSMCOptions,
    LazyImportanceSMCEngine,
)
from modelsmc_pbe.search.importance.records import ImportanceSupport
from modelsmc_pbe.search.importance.support import ImportanceSupportBuilder
from modelsmc_pbe.search.importance.target import FiniteImportanceTarget

PROJECT_DIR = Path(__file__).resolve().parents[1]
MAP_SPEC = PROJECT_DIR / "examples" / "map-increment.json"
STRESS_SPEC = PROJECT_DIR / "examples" / "foldr-sparse-bounded-square-v2.json"


def _config(
    *,
    particles: int = 4,
    iterations: int = 1,
    alpha: float = 0.0,
    seed: int = 47,
) -> ExperimentConfig:
    experiment = load_experiment_config(MAP_SPEC)
    smc = SMCConfig.model_validate(
        {
            **experiment.smc.model_dump(),
            "particles": particles,
            "iterations": iterations,
            "clone_probability": alpha,
            "seed": seed,
        }
    )
    return experiment.model_copy(update={"smc": smc})


def _materialized(
    config: ExperimentConfig,
    *,
    multi_family: bool = False,
    support_limit: int = 250_000,
) -> tuple[ImportanceSMCOptions, ImportanceSupport, FiniteImportanceTarget]:
    options = ImportanceSMCOptions(
        hole_max_cost=3,
        multi_family=multi_family,
        support_limit=support_limit,
        proposal_strategy="joint-target",
    )
    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()
    target = FiniteImportanceTarget.build(
        support,
        smc=config.smc,
        device=resolve_device("cpu"),
    )
    return options, support, target


def test_subtree_marginals_match_hand_calculation() -> None:
    masses = torch.tensor([2.0, 3.0, 5.0, 10.0], dtype=torch.float64)
    target = FiniteImportanceTarget(
        losses_device=torch.zeros(4, dtype=torch.float64),
        costs_device=torch.zeros(4, dtype=torch.float64),
        losses_cpu=torch.zeros(4, dtype=torch.float64),
        costs_cpu=torch.zeros(4, dtype=torch.float64),
        log_prior_device=torch.log(masses),
        log_prior_cpu=torch.log(masses),
        exact_mask=torch.tensor([False, False, True, True]),
        smc=SMCConfig(clone_probability=0.0, loss_scale=1.0),
    )

    distribution = target.distribution(1.0)
    family = target.subtree_probabilities(((0, 1), (2, 3)), beta=1.0)
    first = target.subtree_probabilities(((0,), (1,)), beta=1.0)
    second = target.subtree_probabilities(((2,), (3,)), beta=1.0)

    assert distribution.log_normalizer == pytest.approx(math.log(20.0))
    assert distribution.weights.tolist() == pytest.approx([0.1, 0.15, 0.25, 0.5])
    assert family.tolist() == pytest.approx([0.25, 0.75])
    assert first.tolist() == pytest.approx([0.4, 0.6])
    assert second.tolist() == pytest.approx([1.0 / 3.0, 2.0 / 3.0])
    assert float(family[0] * first[0]) == pytest.approx(0.1)
    assert float(family[0] * first[1]) == pytest.approx(0.15)
    assert float(family[1] * second[0]) == pytest.approx(0.25)
    assert float(family[1] * second[1]) == pytest.approx(0.5)
    assert float(distribution.weights[target.exact_mask].sum().item()) == pytest.approx(0.75)
    assert 1.0 - (1.0 - 0.75) ** 4 == pytest.approx(0.99609375)

    with pytest.raises(ValueError, match="nonempty"):
        target.subtree_probabilities((), beta=1.0)
    with pytest.raises(ValueError, match="disjoint"):
        target.subtree_probabilities(((0, 1), (1, 2)), beta=1.0)
    with pytest.raises(IndexError, match="outside"):
        target.subtree_probabilities(((4,),), beta=1.0)
    with pytest.raises(ValueError, match="proposal_strategy"):
        ImportanceSMCOptions(proposal_strategy="unknown")  # type: ignore[arg-type]


def test_stress_target_has_expected_exact_mass_and_four_draw_budget() -> None:
    config = load_experiment_config(STRESS_SPEC)
    _, support, target = _materialized(config, multi_family=True)
    normalized = target.distribution(beta=1.0)
    exact_mass = float(normalized.weights[target.exact_mask].sum().item())

    assert len(support.states) == 36_198
    assert support.exact_programs == 2
    assert normalized.log_normalizer == pytest.approx(-10.823547289093016)
    assert exact_mass == pytest.approx(0.9207000842137091)
    assert 1.0 - (1.0 - exact_mass) ** 4 == pytest.approx(0.9999604550615014)


def test_sequential_kernel_equals_direct_target_for_every_state() -> None:
    config = _config()
    options, support, target = _materialized(config)
    kernel = FiniteJointTargetProposalKernel(
        config=config,
        options=options,
        support=support,
        target=target,
        generator=make_cpu_generator(config.smc.seed),
    )
    proposals = asyncio.run(
        kernel.evaluate_many(
            0,
            tuple(range(len(support.states))),
            stage=1,
            beta=1.0,
        )
    )
    direct = target.distribution(1.0)
    log_gamma = target.log_unnormalized(1.0)
    log_q = torch.tensor(
        [proposal.log_q_mixture for proposal in proposals],
        dtype=torch.float64,
    )

    assert [proposal.state_index for proposal in proposals] == list(range(len(support.states)))
    assert torch.exp(log_q).tolist() == pytest.approx(direct.weights.tolist(), abs=1e-12)
    assert float(torch.exp(log_q).sum().item()) == pytest.approx(1.0, abs=1e-12)
    assert (log_gamma - log_q).tolist() == pytest.approx(
        [direct.log_normalizer] * len(support.states),
        abs=1e-10,
    )
    assert kernel.source == JOINT_TARGET_PROPOSAL_SOURCE
    assert kernel.scored_candidates == 0
    assert kernel.score_ledger == ()


def test_joint_target_rejects_float64_support_underflow() -> None:
    config = _config()
    options, support, target = _materialized(config)
    log_prior = target.log_prior_cpu.clone()
    log_prior[0] = -10_000.0
    underflowed = replace(target, log_prior_cpu=log_prior)
    kernel = FiniteJointTargetProposalKernel(
        config=config,
        options=options,
        support=support,
        target=underflowed,
        generator=make_cpu_generator(config.smc.seed),
    )

    with pytest.raises(FloatingPointError, match="underflowed"):
        asyncio.run(kernel.evaluate_many(0, (1,), stage=1, beta=1.0))


def test_engine_has_constant_weights_and_never_scores_candidates() -> None:
    config = _config(particles=16, iterations=2, seed=53)
    options = ImportanceSMCOptions(hole_max_cost=3, proposal_strategy="joint-target")
    with ProgramScorer(config) as scorer:
        result = asyncio.run(
            ImportanceSMCEngine(
                config=config,
                options=options,
                scorer=scorer,
                candidate_scorer=None,
                device=resolve_device("cpu"),
                generator=make_cpu_generator(config.smc.seed),
            ).run()
        )

    assert result.probabilistic_claim == JOINT_TARGET_IMPORTANCE_SMC_CLAIM
    assert result.proposal_source == JOINT_TARGET_PROPOSAL_SOURCE
    assert result.proposal_strategy == "joint-target"
    assert result.deduction_mix is None
    assert result.family_deduction_mix is None
    assert result.hole_deduction_mix is None
    assert result.deduction_strength is None
    assert result.llm_energy_normalization is None
    assert result.deduction_guide_exact_mass is None
    assert all(family.deduction_guide_mass is None for family in result.families)
    assert result.scored_candidates == 0
    assert result.max_scored_candidates is None
    assert result.score_ledger == ()
    assert [particle.weight for particle in result.final_particles] == pytest.approx(
        [1.0 / 16.0] * 16,
        abs=1e-12,
    )
    assert all(stage.ess_after == pytest.approx(16.0) for stage in result.stages)
    assert result.reference.log_path_z_error == pytest.approx(0.0, abs=1e-12)


def test_cloning_and_lazy_execution_are_rejected() -> None:
    config = _config(particles=2, alpha=0.25, seed=59)
    options, support, target = _materialized(config)
    with pytest.raises(ValueError, match="alpha=0"):
        FiniteJointTargetProposalKernel(
            config=config,
            options=options,
            support=support,
            target=target,
            generator=make_cpu_generator(config.smc.seed),
        )
    with ProgramScorer(config) as scorer, pytest.raises(ValueError, match="materialized"):
        LazyImportanceSMCEngine(
            config=config,
            options=options,
            scorer=scorer,
            candidate_scorer=UniformCandidateScorer(),
            generator=make_cpu_generator(config.smc.seed),
        )

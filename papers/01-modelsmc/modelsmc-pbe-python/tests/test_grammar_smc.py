from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from modelsmc_pbe.config import SMCConfig, load_experiment_config
from modelsmc_pbe.core import ProgramScorer
from modelsmc_pbe.observability import RunLogger
from modelsmc_pbe.runtime import make_cpu_generator, resolve_device
from modelsmc_pbe.search.grammar_smc import (
    CALIBRATED_CLAIM,
    EmptyGrammarSupportError,
    GrammarSMCEngine,
    GrammarSMCOptions,
    GrammarSMCResult,
    prior_independent_log_acceptance,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_DIR / "examples" / "map-increment.json"


def _smc_config(*, seed: int, particles: int = 256, iterations: int = 6) -> SMCConfig:
    experiment = load_experiment_config(SPEC_PATH)
    return SMCConfig.model_validate(
        {
            **experiment.smc.model_dump(),
            "particles": particles,
            "iterations": iterations,
            "ess_threshold": 0.8,
            "seed": seed,
        }
    )


def _run_map_control(*, seed: int) -> GrammarSMCResult:
    experiment = load_experiment_config(SPEC_PATH)
    smc = _smc_config(seed=seed)
    config = experiment.model_copy(update={"smc": smc})
    with ProgramScorer(config) as scorer:
        return GrammarSMCEngine(
            spec=experiment.spec,
            smc=smc,
            options=GrammarSMCOptions(skeleton="map-arithmetic"),
            scorer=scorer,
            device=resolve_device("cpu"),
            generator=make_cpu_generator(seed),
        ).run()


def test_prior_independent_mh_acceptance_cancels_the_occam_prior() -> None:
    log_acceptance = prior_independent_log_acceptance(
        torch.tensor([0.0, 3.0, 2.0]),
        torch.tensor([2.0, 1.0, 2.0]),
        beta=0.5,
        loss_scale=2.0,
    )

    assert log_acceptance.tolist() == [-2.0, 0.0, 0.0]


def test_calibrated_control_matches_enumeration_and_is_reproducible() -> None:
    first = _run_map_control(seed=23)
    second = _run_map_control(seed=23)

    assert first.mode == "grammar-smc"
    assert first.probabilistic_claim == CALIBRATED_CLAIM
    assert first.grammar_states == 46
    assert first.enumerated_asts == 46
    assert first.rejected_asts == 0
    assert first.over_cost_asts == 0
    assert first.exact_programs == 3
    assert first.exact
    assert first.sampled_best.total_loss == 0.0
    assert first.sampled_best.cost == 5
    assert first.reference.enumeration_exact_mass > 0.999
    assert first.reference.particle_exact_mass > 0.999
    assert first.reference.absolute_log_z_error < 0.1
    assert first.reference.total_variation_distance < 0.15
    assert sum(particle.weight for particle in first.final_particles) == pytest.approx(1.0)
    assert any(stage.resampled for stage in first.stages)
    assert all(stage.mh_attempts == 256 for stage in first.stages)

    assert first.sampled_best == second.sampled_best
    assert first.reference == second.reference
    assert first.stages == second.stages
    assert [particle.state_index for particle in first.final_particles] == [
        particle.state_index for particle in second.final_particles
    ]
    assert [particle.weight for particle in first.final_particles] == [
        particle.weight for particle in second.final_particles
    ]


def test_empty_bounded_support_is_reported_after_semantic_scoring() -> None:
    experiment = load_experiment_config(SPEC_PATH)
    smc = SMCConfig.model_validate(
        {
            **experiment.smc.model_dump(),
            "particles": 8,
            "iterations": 1,
            "max_cost": 1,
        }
    )

    with (
        ProgramScorer(experiment.model_copy(update={"smc": smc})) as scorer,
        pytest.raises(EmptyGrammarSupportError, match="no accepted states"),
    ):
        GrammarSMCEngine(
            spec=experiment.spec,
            smc=smc,
            options=GrammarSMCOptions(skeleton="map-arithmetic"),
            scorer=scorer,
            device=resolve_device("cpu"),
            generator=make_cpu_generator(7),
        ).run()


def test_over_cost_programs_are_reported_separately_from_semantic_rejections() -> None:
    experiment = load_experiment_config(SPEC_PATH)
    smc = SMCConfig.model_validate(
        {
            **experiment.smc.model_dump(),
            "particles": 32,
            "iterations": 1,
            "max_cost": 3,
        }
    )
    config = experiment.model_copy(update={"smc": smc})

    with ProgramScorer(config) as scorer:
        result = GrammarSMCEngine(
            spec=experiment.spec,
            smc=smc,
            options=GrammarSMCOptions(skeleton="map-arithmetic"),
            scorer=scorer,
            device=resolve_device("cpu"),
            generator=make_cpu_generator(7),
        ).run()

    assert result.enumerated_asts == 46
    assert result.grammar_states == 7
    assert result.over_cost_asts == 39
    assert result.rejected_asts == 0


def test_run_logger_records_support_stage_and_reference_events(tmp_path: Path) -> None:
    experiment = load_experiment_config(SPEC_PATH)
    smc = _smc_config(seed=31, particles=32, iterations=2)
    device = resolve_device("cpu")
    logger = RunLogger.create(
        base_dir=tmp_path,
        run_name="grammar-smc-test",
        config={"spec": experiment.spec, "smc": smc},
        device=device,
        seed=31,
        probabilistic_claim=CALIBRATED_CLAIM,
        console_level="quiet",
        run_id="grammar-smc-test-run",
        cwd=PROJECT_DIR,
    )
    config = experiment.model_copy(update={"smc": smc})
    with ProgramScorer(config) as scorer:
        result = GrammarSMCEngine(
            spec=experiment.spec,
            smc=smc,
            options=GrammarSMCOptions(
                skeleton="map-arithmetic",
                score_batch_size=11,
            ),
            scorer=scorer,
            device=device,
            generator=make_cpu_generator(31),
            logger=logger,
        ).run()
    logger.finish(result=result, final_particles=result.final_particles)

    events = [
        json.loads(line)
        for line in (logger.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    names = [event["event"] for event in events]
    assert "grammar.enumerated" in names
    assert "grammar.support.ready" in names
    assert names.count("grammar.score_batch.completed") == 5
    assert names.count("grammar_smc.stage.completed") == 2
    assert "grammar_smc.completed" in names
    assert (logger.run_dir / "result.json").is_file()
    assert (logger.run_dir / "final_particles.jsonl").is_file()

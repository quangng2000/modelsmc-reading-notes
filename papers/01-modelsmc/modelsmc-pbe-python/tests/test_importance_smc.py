from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from modelsmc_pbe.config import ExperimentConfig, SMCConfig, load_experiment_config
from modelsmc_pbe.core import ProgramScorer, ScoreResult
from modelsmc_pbe.domain import canonical_key
from modelsmc_pbe.enumeration import HoleEnumerationLimitExceeded
from modelsmc_pbe.grammar import bounded_square_target, signed_window_target
from modelsmc_pbe.observability.serialization import jsonable
from modelsmc_pbe.proposals import (
    CandidateKind,
    CandidateScoreBatch,
    CandidateScoreRequest,
    CandidateSequenceScore,
    LLMEnergyNormalization,
    ProposalError,
    UniformCandidateScorer,
)
from modelsmc_pbe.runtime import make_cpu_generator, resolve_device
from modelsmc_pbe.search.importance import (
    IMPORTANCE_SMC_CLAIM,
    FactorizedSupportBuilder,
    ImportanceProposalBudgetExceeded,
    ImportanceSMCEngine,
    ImportanceSMCOptions,
    ImportanceSMCResult,
    ImportanceSupportLimitExceeded,
    LazyImportanceSMCEngine,
    replay_qwen_categorical,
)
from modelsmc_pbe.search.importance.deduction_guide import FiniteDeductionGuide
from modelsmc_pbe.search.importance.lazy_records import ConstructionTrace
from modelsmc_pbe.search.importance.proposal import FiniteGuidedProposalKernel
from modelsmc_pbe.search.importance.proposal_distribution import deduction_mismatch_counts
from modelsmc_pbe.search.importance.support import ImportanceSupportBuilder
from modelsmc_pbe.search.importance.target import FiniteImportanceTarget

PROJECT_DIR = Path(__file__).resolve().parents[1]
MAP_SPEC = PROJECT_DIR / "examples" / "map-increment.json"
BOOL_SPEC = PROJECT_DIR / "examples" / "negative-int-to-bool.json"
BOUNDED_SPEC = PROJECT_DIR / "examples" / "foldr-bounded-square.json"
SIGNED_SPEC = PROJECT_DIR / "examples" / "foldr-signed-window.json"


def test_importance_energy_normalization_remains_total_by_default() -> None:
    assert ImportanceSMCOptions().llm_energy_normalization is (
        LLMEnergyNormalization.TOTAL_FULL_PROMPT_LOGPROB
    )


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


def _bounded_config(
    *,
    particles: int = 8,
    iterations: int = 1,
    alpha: float = 0.0,
    seed: int = 19,
) -> ExperimentConfig:
    config = _config(
        BOUNDED_SPEC,
        particles=particles,
        iterations=iterations,
        alpha=alpha,
        seed=seed,
    )
    spec = config.spec.model_copy(update={"integer_constants": [-2, 3]})
    return config.model_copy(update={"spec": spec})


def _signed_config(
    *,
    particles: int = 8,
    iterations: int = 1,
    alpha: float = 0.0,
    seed: int = 19,
) -> ExperimentConfig:
    config = _config(
        SIGNED_SPEC,
        particles=particles,
        iterations=iterations,
        alpha=alpha,
        seed=seed,
    )
    spec = config.spec.model_copy(update={"integer_constants": [-3, 0, 3]})
    return config.model_copy(update={"spec": spec})


def test_lazy_support_counts_complete_traces_without_assembling_programs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The support count must not hide eager complete-program construction."""

    import modelsmc_pbe.induction.assembly as assembly_module

    def forbidden_assembly(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("lazy support construction assembled a complete program")

    monkeypatch.setattr(assembly_module, "assemble_program", forbidden_assembly)
    config = _signed_config()
    support = FactorizedSupportBuilder(
        spec=config.spec,
        smc=config.smc,
        options=ImportanceSMCOptions(
            conditioned_skeleton="foldr-filter-piecewise-map",
            support_limit=6_000,
        ),
    ).build()

    assert support.support_states == 5_400
    assert support.families[0].support_count == 5_400
    assert [len(catalog.fillings) for catalog in support.families[0].catalogs] == [90, 60]


def test_lazy_exact_program_is_executed_only_after_its_trace_is_sampled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A target program cannot leak into scoring during support construction."""

    import modelsmc_pbe.search.importance.lazy_engine as lazy_engine_module

    config = _config(MAP_SPEC, particles=1, iterations=1, alpha=0.0, seed=3)
    options = ImportanceSMCOptions(
        conditioned_skeleton="map-arithmetic",
        support_limit=1_000,
        deduction_mix=0.0,
    )
    support = FactorizedSupportBuilder(
        spec=config.spec,
        smc=config.smc,
        options=options,
    ).build()
    family = support.families[0]
    mapper = {
        "kind": "Add",
        "left": {"kind": "Item"},
        "right": {"kind": "IntLiteral", "intValue": "1"},
    }
    mapper_key = canonical_key(mapper)
    target_choice = next(
        index
        for index, filling in enumerate(family.catalogs[0].fillings)
        if filling.key == mapper_key
    )
    target_trace = ConstructionTrace(
        hypothesis_index=family.hypothesis_index,
        filling_indices=(target_choice,),
    )
    target_program_key = canonical_key({"kind": "MapProgram", "mapper": mapper})
    trace_sampled = False

    def forced_prior_sample(*_args: object, **_kwargs: object) -> ConstructionTrace:
        nonlocal trace_sampled
        trace_sampled = True
        return target_trace

    monkeypatch.setattr(lazy_engine_module, "sample_prior_trace", forced_prior_sample)

    class GuardedScorer(ProgramScorer):
        def score_batch(self, programs: Sequence[object]) -> tuple[ScoreResult, ...]:
            for program in programs:
                if canonical_key(program) == target_program_key:
                    assert trace_sampled
            return super().score_batch(programs)

    with GuardedScorer(config) as scorer:
        result = asyncio.run(
            LazyImportanceSMCEngine(
                config=config,
                options=options,
                scorer=scorer,
                candidate_scorer=UniformCandidateScorer(),
                generator=make_cpu_generator(3),
            ).run()
        )

    assert result.search.exact_found is True
    assert result.support_materialized is False
    assert result.search.evaluated_programs < result.support_states
    assert result.reference is None
    assert result.score_ledger
    for ledger in result.score_ledger:
        assert replay_qwen_categorical(ledger) == pytest.approx(
            tuple(candidate.qwen_probability for candidate in ledger.candidates)
        )


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
        (item.family, item.hole_name, item.returned_expressions) for item in support.hole_catalogs
    ]
    assert catalog_sizes == [
        ("expression", "body", 14),
        ("map", "mapper", 154),
        ("foldr", "initial", 7),
        ("foldr", "reducer", 16),
    ]


def test_clone_mixture_includes_both_routes_to_the_ancestor() -> None:
    config = _config(BOOL_SPEC, particles=8, iterations=1, alpha=0.5, seed=7)
    options = ImportanceSMCOptions(
        hole_max_cost=1,
        proposal_epsilon=0.1,
        deduction_mix=0.0,
    )
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
    kernel = FiniteGuidedProposalKernel(
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
        assert math.exp(proposal.log_q_construct) == pytest.approx(0.5)
    assert sum(math.exp(proposal.log_q_mixture) for proposal in evaluated) == pytest.approx(1.0)


def test_conditioned_foldr_filter_map_support_is_factorized_and_exact() -> None:
    config = _bounded_config()
    options = ImportanceSMCOptions(
        conditioned_skeleton="foldr-filter-map",
        support_limit=1_000,
    )

    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()

    assert support.conditioned_skeleton == "foldr-filter-map"
    assert len(support.induction.hypotheses) == 1
    assert len(support.families) == 1
    assert len(support.states) == 756
    assert support.exact_programs == 2
    assert support.rejected_programs == 0
    assert [(item.hole_name, item.returned_expressions) for item in support.hole_catalogs] == [
        ("predicate", 42),
        ("mapped_value", 18),
    ]
    assert canonical_key(bounded_square_target()) in {state.key for state in support.states}


def test_conditioned_piecewise_support_contains_the_signed_window_program() -> None:
    config = _signed_config()
    options = ImportanceSMCOptions(
        conditioned_skeleton="foldr-filter-piecewise-map",
        support_limit=6_000,
    )

    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()

    assert support.conditioned_skeleton == "foldr-filter-piecewise-map"
    assert len(support.states) == 5_400
    assert support.exact_programs == 4
    assert support.rejected_programs == 0
    assert [(item.hole_name, item.returned_expressions) for item in support.hole_catalogs] == [
        ("predicate", 90),
        ("piecewise_mapped_value", 60),
    ]
    assert canonical_key(signed_window_target()) in {state.key for state in support.states}


def test_auto_multi_family_selects_piecewise_when_simple_arithmetic_cannot_fit() -> None:
    config = _signed_config()
    events: list[tuple[str, dict[str, object]]] = []

    def emit(name: str, **data: object) -> None:
        events.append((name, data))

    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=ImportanceSMCOptions(multi_family=True, support_limit=6_000),
            scorer=scorer,
            emit=emit,
        ).build()

    family_sizes = {
        family.hypothesis.kind.value: len(family.state_indices) for family in support.families
    }
    assert family_sizes == {
        "expression": 8,
        "foldr-filter-piecewise-map": 5_400,
        "foldr": 40,
    }
    assert len(support.states) == 5_448
    assert support.exact_programs == 4
    selection = next(
        data for name, data in events if name == "importance.family.catalog_selected"
    )
    assert selection["retained_family"] == "foldr-filter-piecewise-map"
    assert selection["excluded_family"] == "foldr-filter-map"
    assert "no compact arithmetic mapping" in str(selection["reason"])


def test_signed_window_deduction_identifies_exact_piecewise_hole_fillings() -> None:
    config = _signed_config()
    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=ImportanceSMCOptions(
                conditioned_skeleton="foldr-filter-piecewise-map",
                support_limit=6_000,
            ),
            scorer=scorer,
        ).build()
    family = support.families[0]

    zero_counts: dict[str, int] = {}
    for hole in family.hypothesis.holes:
        fillings = {
            state.filling(hole.name).key: state.filling(hole.name)
            for state in support.states
        }
        ordered = tuple(fillings[key] for key in sorted(fillings))
        mismatches = deduction_mismatch_counts(
            tuple(filling.expression for filling in ordered),
            family.deduction.examples_for(hole.name),
        )
        zero_counts[hole.name] = mismatches.count(0)

    assert zero_counts == {"predicate": 2, "piecewise_mapped_value": 2}


def test_conditioned_proposal_q_is_the_product_of_both_finite_holes() -> None:
    config = _bounded_config(alpha=0.25)
    options = ImportanceSMCOptions(
        conditioned_skeleton="foldr-filter-map",
        support_limit=1_000,
        proposal_epsilon=0.1,
        deduction_mix=0.0,
    )
    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()
    ancestor = support.states[0].state_index
    target = next(
        state.state_index
        for state in support.states
        if state.key == canonical_key(bounded_square_target())
    )
    events: list[tuple[str, dict[str, object]]] = []

    def emit(name: str, **data: object) -> None:
        events.append((name, data))

    kernel = FiniteGuidedProposalKernel(
        config=config,
        options=options,
        support=support,
        scorer=UniformCandidateScorer(),
        generator=make_cpu_generator(11),
        emit=emit,
    )

    ancestor_q, target_q = asyncio.run(
        kernel.evaluate_many(ancestor, (ancestor, target), stage=1, beta=1.0)
    )

    assert math.exp(ancestor_q.log_q_construct) == pytest.approx(1 / 756)
    assert math.exp(target_q.log_q_construct) == pytest.approx(1 / 756)
    assert math.exp(ancestor_q.log_q_mixture) == pytest.approx(0.25 + 0.75 / 756)
    assert math.exp(target_q.log_q_mixture) == pytest.approx(0.75 / 756)
    target_hole_events = [
        data
        for name, data in events
        if name == "importance.proposal.hole_selected" and data["slot"] == 1
    ]
    assert [event["hole"] for event in target_hole_events] == [
        "predicate",
        "mapped_value",
    ]
    assert all(float(event["probability"]) > 0 for event in target_hole_events)
    assert all(event["forced"] is True for event in target_hole_events)
    assert all(len(str(event["candidate_sha256"])) == 64 for event in target_hole_events)


def test_conditioned_support_limit_fails_before_partial_support_is_returned() -> None:
    config = _bounded_config()
    with (
        ProgramScorer(config) as scorer,
        pytest.raises(
            ImportanceSupportLimitExceeded,
            match="exceeds 755 programs",
        ),
    ):
        ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=ImportanceSMCOptions(
                conditioned_skeleton="foldr-filter-map",
                support_limit=755,
            ),
            scorer=scorer,
        ).build()


def test_conditioned_hole_limit_fails_before_allocating_the_large_catalog() -> None:
    config = _bounded_config()
    with ProgramScorer(config) as scorer, pytest.raises(
        HoleEnumerationLimitExceeded,
        match="state_limit=41",
    ) as raised:
        ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=ImportanceSMCOptions(
                conditioned_skeleton="foldr-filter-map",
                hole_state_limit=41,
                support_limit=1_000,
            ),
            scorer=scorer,
        ).build()

    assert raised.value.attempted_states == 42


def test_multi_family_support_keeps_viable_hypotheses_after_sound_refutation() -> None:
    config = _bounded_config()
    options = ImportanceSMCOptions(
        multi_family=True,
        support_limit=1_000,
    )

    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()

    family_sizes = {
        family.hypothesis.kind.value: len(family.state_indices) for family in support.families
    }
    assert support.multi_family is True
    assert support.conditioned_skeleton is None
    assert family_sizes == {
        "expression": 6,
        "foldr-filter-map": 756,
        "foldr": 24,
    }
    assert len(support.states) == 786
    assert support.exact_programs == 2
    assert support.rejected_programs == 0
    assert support.aliased_programs == 0
    assert len({state.key for state in support.states}) == len(support.states)
    map_report = next(
        report for report in support.deductions if report.hypothesis.kind.value == "map"
    )
    assert map_report.viable is False


def test_multi_family_prior_assigns_equal_mass_to_unequal_catalogs() -> None:
    config = _bounded_config()
    options = ImportanceSMCOptions(multi_family=True, support_limit=1_000)
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
    prior = target.distribution(0.0).weights

    assert sum(float(value) for value in prior.tolist()) == pytest.approx(1.0)
    for family in support.families:
        mass = float(prior[list(family.state_indices)].sum().item())
        assert mass == pytest.approx(1 / 3)


def test_deduction_guide_matches_prior_at_zero_and_telescopes_over_holes() -> None:
    config = _bounded_config()
    options = ImportanceSMCOptions(multi_family=True, support_limit=1_000)
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
    guide = FiniteDeductionGuide.build(
        support,
        smc=config.smc,
        strength=options.deduction_strength,
        beta_max=options.beta_max,
    )

    assert torch.allclose(guide.distribution(0.0), target.distribution(0.0).weights)
    exact_state = next(
        state for state in support.states if state.key == canonical_key(bounded_square_target())
    )
    family_groups = tuple(family.state_indices for family in support.families)
    family_probabilities = guide.subtree_probabilities(family_groups, beta=1.0)
    family_position = next(
        index
        for index, family in enumerate(support.families)
        if family.hypothesis_index == exact_state.hypothesis_index
    )
    probability = float(family_probabilities[family_position].item())
    compatible = support.families[family_position].state_indices
    family = support.families[family_position]
    for hole in family.hypothesis.holes:
        keys = tuple(
            sorted({support.states[index].filling(hole.name).key for index in compatible})
        )
        groups = tuple(
            tuple(
                index
                for index in compatible
                if support.states[index].filling(hole.name).key == key
            )
            for key in keys
        )
        conditional = guide.subtree_probabilities(groups, beta=1.0)
        selected_key = exact_state.filling(hole.name).key
        selected = keys.index(selected_key)
        probability *= float(conditional[selected].item())
        compatible = groups[selected]

    assert compatible == (exact_state.state_index,)
    assert probability == pytest.approx(
        float(guide.distribution(1.0)[exact_state.state_index].item())
    )


def test_bounded_square_deduction_identifies_exact_hole_fillings() -> None:
    config = _bounded_config()
    options = ImportanceSMCOptions(
        conditioned_skeleton="foldr-filter-map",
        support_limit=1_000,
    )
    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()
    family = support.families[0]

    zero_counts: dict[str, int] = {}
    for hole in family.hypothesis.holes:
        fillings = {
            state.filling(hole.name).key: state.filling(hole.name)
            for state in support.states
        }
        ordered = tuple(fillings[key] for key in sorted(fillings))
        mismatches = deduction_mismatch_counts(
            tuple(filling.expression for filling in ordered),
            family.deduction.examples_for(hole.name),
        )
        zero_counts[hole.name] = mismatches.count(0)

    assert zero_counts == {"predicate": 2, "mapped_value": 1}


def test_guided_q_is_normalized_positive_and_promotes_exact_programs() -> None:
    config = _bounded_config(alpha=0.0)
    options = ImportanceSMCOptions(
        multi_family=True,
        support_limit=1_000,
        deduction_mix=0.5,
        deduction_strength=2.0,
        proposal_epsilon=0.05,
    )
    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()
    kernel = FiniteGuidedProposalKernel(
        config=config,
        options=options,
        support=support,
        scorer=UniformCandidateScorer(),
        generator=make_cpu_generator(43),
    )

    evaluated = asyncio.run(
        kernel.evaluate_many(
            0,
            tuple(state.state_index for state in support.states),
            stage=1,
            beta=1.0,
        )
    )
    probabilities = tuple(math.exp(item.log_q_construct) for item in evaluated)
    exact_mass = sum(
        probability
        for probability, state in zip(probabilities, support.states, strict=True)
        if state.score.exact_program
    )
    family_mass = sum(
        probability
        for probability, state in zip(probabilities, support.states, strict=True)
        if state.family == "foldr-filter-map"
    )

    assert all(probability > 0.0 for probability in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)
    assert exact_mass > 0.1
    assert family_mass > 0.6


def test_multi_family_q_and_clone_mixture_are_normalized_over_all_states() -> None:
    config = _bounded_config(alpha=0.25)
    options = ImportanceSMCOptions(
        multi_family=True,
        support_limit=1_000,
        proposal_epsilon=0.1,
        deduction_mix=0.0,
    )
    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()
    target_key = canonical_key(bounded_square_target())
    exact_indices = tuple(state.state_index for state in support.states if state.key == target_key)
    assert len(exact_indices) == 1
    ancestor = exact_indices[0]
    events: list[tuple[str, dict[str, object]]] = []

    def emit(name: str, **data: object) -> None:
        events.append((name, data))

    kernel = FiniteGuidedProposalKernel(
        config=config,
        options=options,
        support=support,
        scorer=UniformCandidateScorer(),
        generator=make_cpu_generator(23),
        emit=emit,
    )

    evaluated = asyncio.run(
        kernel.evaluate_many(
            ancestor,
            tuple(state.state_index for state in support.states),
            stage=1,
            beta=1.0,
        )
    )
    by_state = {proposal.state_index: proposal for proposal in evaluated}
    ancestor_q = by_state[ancestor]

    assert math.exp(ancestor_q.log_q_construct) == pytest.approx(1 / 2_268)
    assert math.exp(ancestor_q.log_q_mixture) == pytest.approx(0.25 + 0.75 / 2_268)
    assert sum(math.exp(item.log_q_construct) for item in evaluated) == pytest.approx(1.0)
    assert sum(math.exp(item.log_q_mixture) for item in evaluated) == pytest.approx(1.0)
    family_events = [
        data for name, data in events if name == "importance.proposal.family_selected"
    ]
    assert len(family_events) == len(support.states)
    assert {event["stage"] for event in family_events} == {1}


def test_epsilon_keeps_every_family_and_hole_reachable_under_extreme_scores() -> None:
    class FirstCandidateOnlyScorer:
        name = "first-candidate-only"

        async def score_candidates(
            self, request: CandidateScoreRequest
        ) -> CandidateScoreBatch:
            parsed = await UniformCandidateScorer().score_candidates(request)
            scores = tuple(
                CandidateSequenceScore(
                    candidate=score.candidate,
                    expression=score.expression,
                    token_ids=(index,),
                    token_logprobs=(0.0 if index == 0 else -1_000.0,),
                    sequence_logprob=0.0 if index == 0 else -1_000.0,
                )
                for index, score in enumerate(parsed.scores)
            )
            return CandidateScoreBatch(
                scores=scores,
                source=self.name,
                model="none",
            )

        async def score_many(
            self, requests: list[CandidateScoreRequest]
        ) -> list[CandidateScoreBatch]:
            return [await self.score_candidates(request) for request in requests]

    epsilon = 0.05
    config = _bounded_config()
    options = ImportanceSMCOptions(
        multi_family=True,
        support_limit=1_000,
        proposal_epsilon=epsilon,
        deduction_mix=0.0,
    )
    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()
    target = next(
        state.state_index
        for state in support.states
        if state.key == canonical_key(bounded_square_target())
    )
    proposal = asyncio.run(
        FiniteGuidedProposalKernel(
            config=config,
            options=options,
            support=support,
            scorer=FirstCandidateOnlyScorer(),
            generator=make_cpu_generator(37),
        ).evaluate_many(target, (target,), stage=1, beta=1.0)
    )[0]

    assert math.isfinite(proposal.log_q_construct)
    assert math.exp(proposal.log_q_construct) >= (epsilon**3 / 2_268) * (1.0 - 1e-12)


def test_persistable_score_ledger_replays_pre_mixture_qwen_categorical() -> None:
    class DeterministicTokenScorer:
        name = "deterministic-token-scorer"

        async def score_candidates(
            self, request: CandidateScoreRequest
        ) -> CandidateScoreBatch:
            parsed = await UniformCandidateScorer().score_candidates(request)
            scores: list[CandidateSequenceScore] = []
            for index, score in enumerate(parsed.scores):
                count = index % 3 + 1
                token_logprobs = tuple(-0.01 * (index + 1) for _ in range(count))
                scores.append(
                    CandidateSequenceScore(
                        candidate=score.candidate,
                        expression=score.expression,
                        token_ids=tuple(index * 10 + offset for offset in range(count)),
                        token_logprobs=token_logprobs,
                        sequence_logprob=math.fsum(token_logprobs),
                    )
                )
            return CandidateScoreBatch(
                scores=tuple(scores),
                source=self.name,
                model="Qwen/test",
                model_revision="b2cff646",
                tokenizer_revision="tokenizer-test",
            )

        async def score_many(
            self, requests: list[CandidateScoreRequest]
        ) -> list[CandidateScoreBatch]:
            return [await self.score_candidates(request) for request in requests]

    config = _bounded_config()
    options = ImportanceSMCOptions(
        conditioned_skeleton="foldr-filter-map",
        support_limit=1_000,
        deduction_mix=0.3,
        proposal_temperature=0.6,
        llm_energy_normalization=(
            LLMEnergyNormalization.MEAN_FULL_PROMPT_CONDITIONAL_LOGPROB
        ),
    )
    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()
    target = next(
        state.state_index
        for state in support.states
        if state.key == canonical_key(bounded_square_target())
    )
    kernel = FiniteGuidedProposalKernel(
        config=config,
        options=options,
        support=support,
        scorer=DeterministicTokenScorer(),
        generator=make_cpu_generator(41),
    )

    asyncio.run(kernel.evaluate_many(target, (target,), stage=1, beta=1.0))

    assert kernel.score_ledger
    for ledger in kernel.score_ledger:
        round_tripped = json.loads(
            json.dumps(jsonable(ledger, include_raw_payloads=True))
        )
        replayed = replay_qwen_categorical(round_tripped)
        assert replayed == pytest.approx(
            tuple(candidate.qwen_probability for candidate in ledger.candidates),
            rel=1e-12,
            abs=1e-12,
        )
        assert ledger.model_revision == "b2cff646"
        assert ledger.tokenizer_revision == "tokenizer-test"
        assert ledger.selections[0].selected_probability > 0.0


def test_proposal_deduplicates_scoring_work_and_skips_one_family_wave() -> None:
    class RecordingUniformScorer:
        name = "recording-uniform"

        def __init__(self) -> None:
            self.waves: list[tuple[CandidateKind, ...]] = []
            self.delegate = UniformCandidateScorer()

        async def score_candidates(self, request: CandidateScoreRequest) -> CandidateScoreBatch:
            return await self.delegate.score_candidates(request)

        async def score_many(
            self, requests: list[CandidateScoreRequest]
        ) -> list[CandidateScoreBatch]:
            self.waves.append(tuple(request.candidate_kind for request in requests))
            return await self.delegate.score_many(requests)

    config = _bounded_config()
    conditioned_options = ImportanceSMCOptions(
        conditioned_skeleton="foldr-filter-map",
        support_limit=1_000,
    )
    multi_options = ImportanceSMCOptions(multi_family=True, support_limit=1_000)
    with ProgramScorer(config) as scorer:
        conditioned = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=conditioned_options,
            scorer=scorer,
        ).build()
        multi = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=multi_options,
            scorer=scorer,
        ).build()
    key = canonical_key(bounded_square_target())
    conditioned_target = next(state.state_index for state in conditioned.states if state.key == key)
    multi_target = next(state.state_index for state in multi.states if state.key == key)
    conditioned_scorer = RecordingUniformScorer()
    multi_scorer = RecordingUniformScorer()

    asyncio.run(
        FiniteGuidedProposalKernel(
            config=config,
            options=conditioned_options,
            support=conditioned,
            scorer=conditioned_scorer,
            generator=make_cpu_generator(29),
        ).evaluate_many(
            conditioned_target,
            tuple(conditioned_target for _ in range(32)),
            stage=1,
            beta=1.0,
        )
    )
    asyncio.run(
        FiniteGuidedProposalKernel(
            config=config,
            options=multi_options,
            support=multi,
            scorer=multi_scorer,
            generator=make_cpu_generator(31),
        ).evaluate_many(
            multi_target,
            tuple(multi_target for _ in range(32)),
            stage=1,
            beta=1.0,
        )
    )

    assert conditioned_scorer.waves == [
        (CandidateKind.EXPRESSION,),
        (CandidateKind.EXPRESSION,),
    ]
    assert multi_scorer.waves == [
        (CandidateKind.SKELETON,),
        (CandidateKind.EXPRESSION,),
        (CandidateKind.EXPRESSION,),
    ]


def test_candidate_prompt_budget_aborts_before_provider_io() -> None:
    class NeverCalledScorer:
        name = "never-called"

        def __init__(self) -> None:
            self.calls = 0

        async def score_candidates(
            self, request: CandidateScoreRequest
        ) -> CandidateScoreBatch:
            del request
            self.calls += 1
            raise AssertionError("provider must not be called after budget rejection")

        async def score_many(
            self, requests: list[CandidateScoreRequest]
        ) -> list[CandidateScoreBatch]:
            del requests
            self.calls += 1
            raise AssertionError("provider must not be called after budget rejection")

    config = _bounded_config()
    options = ImportanceSMCOptions(
        multi_family=True,
        support_limit=1_000,
        max_scored_candidates=2,
    )
    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()
    candidate_scorer = NeverCalledScorer()
    kernel = FiniteGuidedProposalKernel(
        config=config,
        options=options,
        support=support,
        scorer=candidate_scorer,
        generator=make_cpu_generator(41),
    )

    with pytest.raises(ImportanceProposalBudgetExceeded, match="used 0, next wave 3"):
        asyncio.run(kernel.sample_many((0,), stage=1, beta=1.0))

    assert candidate_scorer.calls == 0
    assert kernel.scored_candidates == 0


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


def test_guided_smc_finds_bounded_square_with_a_small_seeded_population() -> None:
    config = _bounded_config(particles=12, iterations=1, alpha=0.0, seed=1)

    def run(deduction_mix: float) -> ImportanceSMCResult:
        options = ImportanceSMCOptions(
            multi_family=True,
            support_limit=1_000,
            max_scored_candidates=8_000,
            deduction_mix=deduction_mix,
            deduction_strength=2.0,
        )
        with ProgramScorer(config) as scorer:
            return asyncio.run(
                ImportanceSMCEngine(
                    config=config,
                    options=options,
                    scorer=scorer,
                    candidate_scorer=UniformCandidateScorer(),
                    device=resolve_device("cpu"),
                    generator=make_cpu_generator(config.smc.seed),
                ).run()
            )

    unguided = run(0.0)
    guided = run(0.5)

    assert not unguided.exact
    assert unguided.reference.particle_exact_mass == 0.0
    assert guided.exact
    assert guided.sampled_best.total_loss == 0.0
    assert guided.reference.particle_exact_mass > 0.99


def test_alpha_one_is_rejected_because_it_destroys_proposal_support() -> None:
    config = _config(BOOL_SPEC, particles=4, iterations=1, alpha=1.0)
    options = ImportanceSMCOptions(hole_max_cost=1)
    with ProgramScorer(config) as scorer:
        with pytest.raises(ValueError, match="alpha < 1"):
            ImportanceSMCEngine(
                config=config,
                options=options,
                scorer=scorer,
                candidate_scorer=UniformCandidateScorer(),
                device=resolve_device("cpu"),
                generator=make_cpu_generator(1),
            )
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()

    with pytest.raises(ValueError, match="alpha < 1"):
        FiniteGuidedProposalKernel(
            config=config,
            options=options,
            support=support,
            scorer=UniformCandidateScorer(),
            generator=make_cpu_generator(1),
        )


def test_provider_failure_aborts_instead_of_falling_back_to_the_ancestor() -> None:
    class FailingCandidateScorer:
        name = "always-fails"

        async def score_candidates(self, request: CandidateScoreRequest) -> CandidateScoreBatch:
            del request
            raise ProposalError("provider unavailable")

        async def score_many(
            self, requests: list[CandidateScoreRequest]
        ) -> list[CandidateScoreBatch]:
            del requests
            raise ProposalError("provider unavailable")

    config = _config(BOOL_SPEC, particles=4, iterations=1, alpha=0.0)
    with (
        ProgramScorer(config) as scorer,
        pytest.raises(ProposalError, match="provider unavailable"),
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

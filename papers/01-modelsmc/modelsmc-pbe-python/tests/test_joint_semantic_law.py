"""Pure probability tests for the lazy joint-semantic proposal law."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from modelsmc_pbe.config import ExperimentConfig, load_experiment_config
from modelsmc_pbe.search.importance.joint_semantic import (
    SemanticStageLaw,
    attach_semantic_scores,
    enumerate_construction_traces,
    select_deterministic_slate,
)
from modelsmc_pbe.search.importance.lazy_records import (
    FactorizedHoleCatalog,
    FactorizedImportanceSupport,
)
from modelsmc_pbe.search.importance.lazy_support import FactorizedSupportBuilder
from modelsmc_pbe.search.importance.records import ImportanceSMCOptions

PROJECT_DIR = Path(__file__).resolve().parents[1]
MAP_SPEC = PROJECT_DIR / "examples" / "map-increment.json"


@pytest.fixture(scope="module")
def full_support() -> tuple[ExperimentConfig, FactorizedImportanceSupport]:
    config = load_experiment_config(MAP_SPEC)
    support = FactorizedSupportBuilder(
        spec=config.spec,
        smc=config.smc,
        options=ImportanceSMCOptions(
            hole_max_cost=3,
            multi_family=True,
            support_limit=20_000,
        ),
    ).build()
    return config, support


def _two_equal_cost_choices(catalog: FactorizedHoleCatalog) -> tuple[int, int]:
    costs = catalog.costs
    for cost in sorted(set(costs)):
        positions = tuple(index for index, value in enumerate(costs) if value == cost)
        if len(positions) >= 2:
            return positions[0], positions[1]
    raise AssertionError("test catalog needs two equal-cost choices")


def _four_leaf_support(support: FactorizedImportanceSupport) -> FactorizedImportanceSupport:
    families = []
    for family in support.families[:2]:
        assert len(family.catalogs) == 1
        catalog = family.catalogs[0]
        selected = _two_equal_cost_choices(catalog)
        reduced_catalog = replace(
            catalog,
            fillings=tuple(catalog.fillings[index] for index in selected),
            costs=tuple(catalog.costs[index] for index in selected),
            deduction_mismatches=tuple(
                catalog.deduction_mismatches[index] for index in selected
            ),
        )
        families.append(replace(family, catalogs=(reduced_catalog,), support_count=2))
    return replace(support, families=tuple(families), support_states=4)


def _real_two_hole_support(
    support: FactorizedImportanceSupport,
) -> FactorizedImportanceSupport:
    family = next(item for item in support.families if item.hypothesis.kind.value == "foldr")
    return replace(support, families=(family,), support_states=family.support_count)


def test_hand_four_leaf_marginals_and_telescope(
    full_support: tuple[ExperimentConfig, FactorizedImportanceSupport],
) -> None:
    config, original = full_support
    support = _four_leaf_support(original)
    traces = enumerate_construction_traces(support)
    slate = attach_semantic_scores(
        support,
        traces,
        tuple(math.log(value) for value in (1.0, 2.0, 3.0, 4.0)),
        cost_scale=config.smc.cost_scale,
    )
    law = SemanticStageLaw(
        support=support,
        slate=slate,
        cost_scale=config.smc.cost_scale,
        semantic_scale=1.0,
        stage_fraction=1.0,
        epsilon=0.2,
    )

    direct = tuple(math.exp(law.direct_log_probability(trace)) for trace in traces)
    family = law.family_distribution()
    first = law.hole_distribution(hypothesis_index=traces[0].hypothesis_index, prefix=())
    second = law.hole_distribution(hypothesis_index=traces[2].hypothesis_index, prefix=())

    assert direct == pytest.approx((0.13, 0.21, 0.29, 0.37), abs=1e-12)
    assert family.probabilities == pytest.approx((0.34, 0.66), abs=1e-12)
    assert first.probabilities == pytest.approx((0.13 / 0.34, 0.21 / 0.34))
    assert second.probabilities == pytest.approx((0.29 / 0.66, 0.37 / 0.66))
    assert tuple(math.exp(law.evaluate(trace).log_probability) for trace in traces) == (
        pytest.approx(direct, abs=1e-12)
    )
    assert math.fsum(direct) == pytest.approx(1.0, abs=1e-12)

    draw = law.sample(generator=torch.Generator(device="cpu").manual_seed(29))
    assert draw.log_probability == pytest.approx(
        law.direct_log_probability(draw.trace), abs=1e-12
    )


def test_real_factorized_floor_normalization_and_importance_identity(
    full_support: tuple[ExperimentConfig, FactorizedImportanceSupport],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import modelsmc_pbe.induction.assembly as assembly_module

    config, original = full_support
    support = _real_two_hole_support(original)

    def forbidden_assembly(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("compact semantic trace enumeration assembled a program")

    monkeypatch.setattr(assembly_module, "assemble_program", forbidden_assembly)
    traces = enumerate_construction_traces(support)
    selected = select_deterministic_slate(traces, size=23, seed=71)
    assert selected == select_deterministic_slate(reversed(traces), size=23, seed=71)
    slate = attach_semantic_scores(
        support,
        selected,
        tuple(float((index % 9) - 4) for index in range(len(selected))),
        cost_scale=config.smc.cost_scale,
    )
    epsilon = 0.07
    law = SemanticStageLaw(
        support=support,
        slate=slate,
        cost_scale=config.smc.cost_scale,
        semantic_scale=3.5,
        stage_fraction=0.6,
        epsilon=epsilon,
    )

    log_q = tuple(law.direct_log_probability(trace) for trace in traces)
    log_prior = tuple(
        attach_semantic_scores(
            support,
            (trace,),
            (0.0,),
            cost_scale=config.smc.cost_scale,
        )[0].log_prior
        for trace in traces
    )
    assert math.fsum(math.exp(value) for value in log_q) == pytest.approx(1.0, abs=1e-12)
    assert all(
        q >= math.log(epsilon) + prior - 1e-12
        for q, prior in zip(log_q, log_prior, strict=True)
    )
    assert all(
        law.evaluate(trace).log_probability == pytest.approx(q, abs=1e-10)
        for trace, q in zip(traces, log_q, strict=True)
    )

    slate_traces = {entry.trace for entry in slate}
    outside = next(trace for trace in traces if trace not in slate_traces)
    outside_position = traces.index(outside)
    assert log_q[outside_position] == pytest.approx(
        math.log(epsilon) + log_prior[outside_position], abs=1e-12
    )

    log_gamma = tuple(prior - 0.25 * (index % 7) for index, prior in enumerate(log_prior))
    importance_sum = math.fsum(
        math.exp(q) * math.exp(gamma - q)
        for q, gamma in zip(log_q, log_gamma, strict=True)
    )
    target_sum = math.fsum(math.exp(value) for value in log_gamma)
    assert importance_sum == pytest.approx(target_sum, abs=1e-12)


def test_extreme_semantic_scores_remain_log_domain_stable(
    full_support: tuple[ExperimentConfig, FactorizedImportanceSupport],
) -> None:
    config, original = full_support
    support = _four_leaf_support(original)
    traces = enumerate_construction_traces(support)
    slate = attach_semantic_scores(
        support,
        traces,
        (-1e300, 1e300, -5e299, 5e299),
        cost_scale=config.smc.cost_scale,
    )
    law = SemanticStageLaw(
        support=support,
        slate=slate,
        cost_scale=config.smc.cost_scale,
        semantic_scale=1e100,
        stage_fraction=1.0,
        epsilon=0.1,
    )

    log_q = tuple(law.log_probability(trace) for trace in traces)
    assert all(math.isfinite(value) for value in log_q)
    assert math.fsum(math.exp(value) for value in log_q) == pytest.approx(1.0, abs=1e-12)
    assert max(item.log_q_a for item in law.slate_probabilities) == pytest.approx(0.0)


def test_semantic_raw_batch_minimum_does_not_restrict_legacy_guided_options() -> None:
    guided = ImportanceSMCOptions(semantic_candidate_batch_size=1)

    assert guided.proposal_strategy == "guided"
    with pytest.raises(ValueError, match="at least four"):
        ImportanceSMCOptions(
            proposal_strategy="joint-semantic",
            semantic_candidate_batch_size=1,
        )

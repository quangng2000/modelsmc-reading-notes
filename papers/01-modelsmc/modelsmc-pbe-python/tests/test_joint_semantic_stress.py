"""Corrected stress-support budget regression for joint-semantic scoring."""

from pathlib import Path

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.search.importance import ImportanceSMCOptions
from modelsmc_pbe.search.importance.joint_semantic import (
    enumerate_construction_traces,
)
from modelsmc_pbe.search.importance.joint_semantic.programs import (
    prepare_semantic_programs,
)
from modelsmc_pbe.search.importance.lazy_support import FactorizedSupportBuilder

PROJECT_DIR = Path(__file__).resolve().parents[1]
STRESS_SPEC = PROJECT_DIR / "examples" / "foldr-sparse-bounded-square-v2.json"


def test_corrected_stress_full_slate_has_the_declared_raw_label_budget() -> None:
    config = load_experiment_config(STRESS_SPEC)
    options = ImportanceSMCOptions(
        hole_max_cost=3,
        support_limit=40_000,
        multi_family=True,
        proposal_strategy="joint-semantic",
    )
    support = FactorizedSupportBuilder(
        spec=config.spec,
        smc=config.smc,
        options=options,
    ).build()
    traces = enumerate_construction_traces(support)
    prepared = prepare_semantic_programs(
        config=config,
        support=support,
        traces=traces,
    )

    assert support.support_states == 36_198
    assert len(traces) == 36_198
    assert len(prepared.programs) == 36_198
    assert 4 * len(prepared.programs) == 144_792

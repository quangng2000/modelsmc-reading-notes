from __future__ import annotations

from pathlib import Path

import pytest

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.core import ProgramScorer, ScoredProgram
from modelsmc_pbe.core.types import child
from modelsmc_pbe.deduction import RefutationKind
from modelsmc_pbe.domain import canonical_key
from modelsmc_pbe.grammar import bounded_square_target
from modelsmc_pbe.search.importance import FactorizedSupportBuilder, ImportanceSMCOptions

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPARSE_SPEC = PROJECT_DIR / "examples" / "foldr-sparse-bounded-square.json"


def test_sparse_bounded_square_is_support_positive_but_deduction_underconstrained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freeze the stress-test structure without materializing 36k programs."""

    import modelsmc_pbe.induction.assembly as assembly_module

    def forbid_complete_assembly(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("factorized support eagerly assembled a complete program")

    monkeypatch.setattr(assembly_module, "assemble_program", forbid_complete_assembly)
    config = load_experiment_config(SPARSE_SPEC)
    support = FactorizedSupportBuilder(
        spec=config.spec,
        smc=config.smc,
        options=ImportanceSMCOptions(
            multi_family=True,
            support_limit=250_000,
            hole_state_limit=250_000,
            hole_max_cost=3,
            deduction_mix=0.75,
            deduction_strength=4.0,
        ),
    ).build()

    assert support.support_states == 36_198
    assert not hasattr(support, "states")
    map_report = next(
        report for report in support.deductions if report.hypothesis.kind.value == "map"
    )
    assert map_report.viable is False
    assert map_report.refutation is not None
    assert map_report.refutation.kind is RefutationKind.MAP_LENGTH_MISMATCH

    family = next(
        family for family in support.families if family.hypothesis.kind.value == "foldr-filter-map"
    )
    assert family.support_count == 36_000
    catalogs = {catalog.hole.name: catalog for catalog in family.catalogs}
    assert {name: len(catalog.fillings) for name, catalog in catalogs.items()} == {
        "predicate": 600,
        "mapped_value": 60,
    }
    for hole_name, catalog in catalogs.items():
        assert family.deduction.examples_for(hole_name) == ()
        assert set(catalog.deduction_mismatches) == {0}

    target = bounded_square_target()
    reducer = child(target, "reducer")
    predicate = child(reducer, "condition")
    mapped_value = child(child(reducer, "thenExpr"), "head")
    assert canonical_key(predicate) in {filling.key for filling in catalogs["predicate"].fillings}
    assert canonical_key(mapped_value) in {
        filling.key for filling in catalogs["mapped_value"].fillings
    }

    with ProgramScorer(config) as scorer:
        result = scorer.score(target)
    assert isinstance(result, ScoredProgram)
    assert result.exact_program is True
    assert result.total_loss == 0.0
    assert result.exact_matches == len(config.spec.examples)

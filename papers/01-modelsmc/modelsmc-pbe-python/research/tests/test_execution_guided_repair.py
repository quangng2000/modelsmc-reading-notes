from __future__ import annotations

from modelsmc_pbe.grammar import arithmetic_expressions, filter_predicates
from research.execution_guided_repair import _dsl_catalog, render_expression_dsl


def test_repair_dsl_is_bijective_over_both_finite_catalogs() -> None:
    constants = tuple(range(-3, 5))
    mappers = arithmetic_expressions("Item", constants)
    predicates = filter_predicates(constants)
    assert len(_dsl_catalog(mappers)) == 60
    assert len(_dsl_catalog(predicates)) == 600


def test_repair_dsl_renders_expected_target_components() -> None:
    item = {"kind": "Item"}
    square = {"kind": "Multiply", "left": item, "right": item}
    predicate = {
        "kind": "And",
        "left": {
            "kind": "LessThan",
            "left": {"kind": "IntLiteral", "intValue": "-2"},
            "right": item,
        },
        "right": {
            "kind": "LessThan",
            "left": item,
            "right": {"kind": "IntLiteral", "intValue": "3"},
        },
    }
    assert render_expression_dsl(square) == "mul(item,item)"
    assert render_expression_dsl(predicate) == "and(lt(-2,item),lt(item,3))"

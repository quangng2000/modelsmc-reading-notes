from __future__ import annotations

from pathlib import Path

import pytest

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.core import ProgramScorer, ScoredProgram
from modelsmc_pbe.domain import ProgramAst
from modelsmc_pbe.domain.ast import AstNode

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _leaf(kind: str) -> AstNode:
    return {"kind": kind}


def _integer(value: int) -> AstNode:
    return {"kind": "IntLiteral", "intValue": str(value)}


def _binary(kind: str, left: AstNode, right: AstNode) -> AstNode:
    return {"kind": kind, "left": left, "right": right}


def _branch(condition: AstNode, then_expr: AstNode, else_expr: AstNode) -> AstNode:
    return {
        "kind": "IfThenElse",
        "condition": condition,
        "thenExpr": then_expr,
        "elseExpr": else_expr,
    }


def _prepend(head: AstNode) -> AstNode:
    return {"kind": "PrependInt", "head": head, "tail": _leaf("Accumulator")}


def _fold(initial: AstNode, reducer: AstNode) -> ProgramAst:
    return {"kind": "FoldRightProgram", "initial": initial, "reducer": reducer}


def _affine() -> ProgramAst:
    return {
        "kind": "ExpressionProgram",
        "body": _binary(
            "Add",
            _binary("Multiply", _integer(2), _leaf("Input")),
            _integer(1),
        ),
    }


def _bounded_square() -> ProgramAst:
    within_bounds = _binary(
        "And",
        _binary("LessThan", _integer(-2), _leaf("Item")),
        _binary("LessThan", _leaf("Item"), _integer(3)),
    )
    square = _binary("Multiply", _leaf("Item"), _leaf("Item"))
    return _fold(
        _leaf("EmptyIntList"),
        _branch(within_bounds, _prepend(square), _leaf("Accumulator")),
    )


def _filter_positive() -> ProgramAst:
    positive = _binary("LessThan", _integer(0), _leaf("Item"))
    return _fold(
        _leaf("EmptyIntList"),
        _branch(positive, _prepend(_leaf("Item")), _leaf("Accumulator")),
    )


def _signed_window() -> ProgramAst:
    within_bounds = _binary(
        "And",
        _binary("LessThan", _integer(-3), _leaf("Item")),
        _binary("LessThan", _leaf("Item"), _integer(3)),
    )
    mapped = _branch(
        _binary("LessThan", _leaf("Item"), _integer(0)),
        _binary("Subtract", _integer(0), _leaf("Item")),
        _binary("Multiply", _leaf("Item"), _leaf("Item")),
    )
    return _fold(
        _leaf("EmptyIntList"),
        _branch(within_bounds, _prepend(mapped), _leaf("Accumulator")),
    )


def _sum() -> ProgramAst:
    return _fold(
        _integer(0),
        _binary("Add", _leaf("Item"), _leaf("Accumulator")),
    )


def _window_penalty_sum() -> ProgramAst:
    def add_to_accumulator(value: AstNode) -> AstNode:
        return _binary("Add", value, _leaf("Accumulator"))

    high_penalty = add_to_accumulator(
        _binary("Subtract", _integer(0), _leaf("Item"))
    )
    square = add_to_accumulator(_binary("Multiply", _leaf("Item"), _leaf("Item")))
    reducer = _branch(
        _binary("LessThan", _leaf("Item"), _integer(-1)),
        _leaf("Accumulator"),
        _branch(
            _binary("LessThan", _integer(2), _leaf("Item")),
            high_penalty,
            square,
        ),
    )
    return _fold(_integer(0), reducer)


def _map_increment() -> ProgramAst:
    return {
        "kind": "MapProgram",
        "mapper": _binary("Add", _leaf("Item"), _integer(1)),
    }


def _negative() -> ProgramAst:
    return {
        "kind": "ExpressionProgram",
        "body": _binary("LessThan", _leaf("Input"), _integer(0)),
    }


@pytest.mark.parametrize(
    ("filename", "target", "expected_cost"),
    [
        ("affine-int.json", _affine(), 5),
        ("foldr-bounded-square.json", _bounded_square(), 18),
        ("foldr-filter-positive.json", _filter_positive(), 12),
        ("foldr-signed-window.json", _signed_window(), 25),
        ("foldr-sum.json", _sum(), 7),
        ("foldr-window-penalty-sum.json", _window_penalty_sum(), 23),
        ("map-increment.json", _map_increment(), 5),
        ("negative-int-to-bool.json", _negative(), 3),
    ],
)
def test_known_target_is_exact_on_every_copied_example(
    filename: str,
    target: ProgramAst,
    expected_cost: int,
) -> None:
    config = load_experiment_config(EXAMPLES / filename)

    result = ProgramScorer(config).score(target)

    assert isinstance(result, ScoredProgram)
    assert result.exact_program
    assert result.exact_matches == len(config.spec.examples)
    assert result.total_loss == 0
    assert result.cost == expected_cost

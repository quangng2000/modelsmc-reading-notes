from __future__ import annotations

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.core import ProgramScorer, ScoredProgram


def _scorer(
    input_type: str,
    output_type: str,
    examples: list[dict[str, object]],
    *,
    constants: list[int] | None = None,
) -> ProgramScorer:
    config = ExperimentConfig.model_validate(
        {
            "spec": {
                "name": "core semantics",
                "signature": {"input": input_type, "output": output_type},
                "examples": examples,
                "integerConstants": constants or [-1, 0, 1, 2, 3],
            },
            "smc": {"maxCost": 64, "maxDepth": 20, "maxNodes": 256},
        }
    )
    return ProgramScorer(config)


def _scored(scorer: ProgramScorer, program: object) -> ScoredProgram:
    result = scorer.score(program)
    assert isinstance(result, ScoredProgram)
    return result


def test_map_can_change_integer_elements_to_booleans() -> None:
    scorer = _scorer(
        "List<Int>",
        "List<Bool>",
        [
            {"input": [], "output": []},
            {"input": [-1, 0, 2], "output": [True, False, False]},
        ],
    )
    program = {
        "kind": "MapProgram",
        "mapper": {
            "kind": "LessThan",
            "left": {"kind": "Item"},
            "right": {"kind": "IntLiteral", "intValue": "0"},
        },
    }

    result = _scored(scorer, program)

    assert result.inferred_type == "BoolListType"
    assert result.exact_program
    assert result.cost == 5
    assert result.evaluations[1].predicted == "[true, false, false]"


def test_foldr_preserves_right_associative_subtraction_order() -> None:
    scorer = _scorer(
        "List<Int>",
        "Int",
        [
            {"input": [], "output": 0},
            {"input": [1, 2, 3], "output": 2},
        ],
    )
    program = {
        "kind": "FoldRightProgram",
        "initial": {"kind": "IntLiteral", "intValue": "0"},
        "reducer": {
            "kind": "Subtract",
            "left": {"kind": "Item"},
            "right": {"kind": "Accumulator"},
        },
    }

    result = _scored(scorer, program)

    assert result.exact_program
    assert result.cost == 7
    assert result.evaluations[1].predicted == "2"


def test_boolean_list_map_and_prepend_are_both_supported() -> None:
    examples: list[dict[str, object]] = [
        {"input": [], "output": []},
        {"input": [True, False, True], "output": [False, True, False]},
    ]
    scorer = _scorer("List<Bool>", "List<Bool>", examples)
    mapper = {
        "kind": "MapProgram",
        "mapper": {"kind": "Not", "operand": {"kind": "Item"}},
    }
    fold = {
        "kind": "FoldRightProgram",
        "initial": {"kind": "EmptyBoolList"},
        "reducer": {
            "kind": "PrependBool",
            "head": {"kind": "Not", "operand": {"kind": "Item"}},
            "tail": {"kind": "Accumulator"},
        },
    }

    mapped = _scored(scorer, mapper)
    folded = _scored(scorer, fold)

    assert mapped.exact_program
    assert folded.exact_program
    assert mapped.evaluations == folded.evaluations


def test_arbitrary_precision_integer_is_evaluated_and_rendered_losslessly() -> None:
    huge = 10**100 + 123456789
    scorer = _scorer(
        "Int",
        "Int",
        [{"input": 0, "output": huge}],
        constants=[huge],
    )
    program = {
        "kind": "ExpressionProgram",
        "body": {"kind": "IntLiteral", "intValue": str(huge)},
    }

    result = _scored(scorer, program)

    assert result.exact_program
    assert result.evaluations[0].predicted == str(huge)
    assert result.total_loss == 0


def test_if_then_else_evaluates_only_the_selected_branch() -> None:
    scorer = _scorer("Int", "Int", [{"input": 4, "output": 1}])
    program = {
        "kind": "ExpressionProgram",
        "body": {
            "kind": "IfThenElse",
            "condition": {"kind": "BoolLiteral", "boolValue": True},
            "thenExpr": {"kind": "IntLiteral", "intValue": "1"},
            "elseExpr": {
                "kind": "Add",
                "left": {"kind": "Input"},
                "right": {"kind": "IntLiteral", "intValue": "1"},
            },
        },
    }

    assert _scored(scorer, program).exact_program

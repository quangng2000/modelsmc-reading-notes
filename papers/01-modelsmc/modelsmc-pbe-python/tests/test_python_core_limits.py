from __future__ import annotations

import pytest

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.core import ProgramScorer, RejectedProgram
from modelsmc_pbe.core.loss import soft_loss
from modelsmc_pbe.core.types import StaticType


def _scorer(**smc: int) -> ProgramScorer:
    settings = {"maxCost": 20, "maxDepth": 10, "maxNodes": 127, **smc}
    return ProgramScorer(
        ExperimentConfig.model_validate(
            {
                "spec": {
                    "signature": {"input": "Int", "output": "Int"},
                    "examples": [{"input": 1, "output": 2}],
                    "integerConstants": [0, 1, 2],
                },
                "smc": settings,
            }
        )
    )


def _add_one() -> dict[str, object]:
    return {
        "kind": "ExpressionProgram",
        "body": {
            "kind": "Add",
            "left": {"kind": "Input"},
            "right": {"kind": "IntLiteral", "intValue": "1"},
        },
    }


@pytest.mark.parametrize(
    ("limits", "reason"),
    [
        ({"maxDepth": 2}, "maximum AST depth 2"),
        ({"maxNodes": 3}, "maximum node count 3"),
        ({"maxCost": 2}, "expression cost 3 exceeds maximum 2"),
    ],
)
def test_structural_limits_reject_candidates(
    limits: dict[str, int], reason: str
) -> None:
    result = _scorer(**limits).score(_add_one())

    assert isinstance(result, RejectedProgram)
    assert reason in result.reason


def test_integer_literal_must_belong_to_the_constant_catalog() -> None:
    result = _scorer().score(
        {
            "kind": "ExpressionProgram",
            "body": {"kind": "IntLiteral", "intValue": "99"},
        }
    )

    assert isinstance(result, RejectedProgram)
    assert "99 is not in the allowed constant catalog" in result.reason


def test_negative_zero_matches_the_semantic_zero_catalog_entry() -> None:
    scorer = ProgramScorer(
        ExperimentConfig.model_validate(
            {
                "spec": {
                    "signature": {"input": "Int", "output": "Int"},
                    "examples": [{"input": 4, "output": 0}],
                    "integerConstants": [0],
                }
            }
        )
    )
    result = scorer.score(
        {
            "kind": "ExpressionProgram",
            "body": {"kind": "IntLiteral", "intValue": "-0"},
        }
    )

    assert not isinstance(result, RejectedProgram)
    assert result.exact_program


def test_scope_and_output_type_errors_are_typed_rejections() -> None:
    unbound_item = {
        "kind": "ExpressionProgram",
        "body": {"kind": "Item"},
    }
    wrong_output = {
        "kind": "ExpressionProgram",
        "body": {"kind": "BoolLiteral", "boolValue": True},
    }

    scope_result, output_result = _scorer().score_batch([unbound_item, wrong_output])

    assert isinstance(scope_result, RejectedProgram)
    assert "type checker" in scope_result.reason
    assert isinstance(output_result, RejectedProgram)
    assert output_result.inferred_type == "BoolType"


def test_list_edit_loss_uses_insertions_and_honors_the_global_cap() -> None:
    shifted = soft_loss([1, 2, 3], [0, 1, 2, 3], StaticType.INT_LIST, cap=100)
    capped = soft_loss([100, 200], [0, 1], StaticType.INT_LIST, cap=1)

    assert shifted == 1
    assert capped == 1


def test_oversized_batch_is_rejected_before_scoring() -> None:
    with pytest.raises(ValueError, match="maximum size 10000"):
        _scorer().score_batch([_add_one()] * 10_001)

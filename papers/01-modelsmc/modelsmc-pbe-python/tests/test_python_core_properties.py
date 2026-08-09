from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.core import ProgramScorer, ScoredProgram
from modelsmc_pbe.core.loss import soft_loss
from modelsmc_pbe.core.types import StaticType


@given(
    input_value=st.integers(min_value=-(10**50), max_value=10**50),
    offset=st.integers(min_value=-(10**20), max_value=10**20),
)
def test_arbitrary_precision_addition_scores_exactly(input_value: int, offset: int) -> None:
    config = ExperimentConfig.model_validate(
        {
            "spec": {
                "name": "arbitrary-precision-addition",
                "signature": {"input": "Int", "output": "Int"},
                "examples": [{"input": str(input_value), "output": str(input_value + offset)}],
                "integerConstants": [str(offset)],
            }
        }
    )
    program = {
        "kind": "ExpressionProgram",
        "body": {
            "kind": "Add",
            "left": {"kind": "Input"},
            "right": {"kind": "IntLiteral", "intValue": str(offset)},
        },
    }

    result = ProgramScorer(config).score(program)

    assert isinstance(result, ScoredProgram)
    assert result.exact_program
    assert result.total_loss == 0


@given(
    predicted=st.lists(st.integers(min_value=-100, max_value=100), max_size=12),
    expected=st.lists(st.integers(min_value=-100, max_value=100), max_size=12),
    cap=st.integers(min_value=1, max_value=50),
)
def test_capped_list_loss_is_zero_exactly_for_equal_lists(
    predicted: list[int], expected: list[int], cap: int
) -> None:
    loss = soft_loss(predicted, expected, StaticType.INT_LIST, cap)

    assert 0 <= loss <= cap
    assert (loss == 0) is (predicted == expected)

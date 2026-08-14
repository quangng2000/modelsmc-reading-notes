from __future__ import annotations

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.domain.models import PBESpec
from research.automatic_repair_feedback import derive_automatic_feedback


def _item() -> dict[str, object]:
    return {"kind": "Item"}


def _constant(value: int) -> dict[str, object]:
    return {"kind": "IntLiteral", "intValue": str(value)}


def _wrong_predicate() -> dict[str, object]:
    return {
        "kind": "And",
        "left": {"kind": "LessThan", "left": _constant(-1), "right": _item()},
        "right": {"kind": "LessThan", "left": _item(), "right": _constant(3)},
    }


def test_automatic_feedback_repairs_both_holes_without_oracle_alignment() -> None:
    config = load_experiment_config("examples/foldr-sparse-bounded-square-v2.json")
    predicate = _wrong_predicate()

    mapper_feedback = derive_automatic_feedback(config.spec, predicate, _item())
    constraints = {(fact.item, fact.expected_value) for fact in mapper_feedback.mapper_constraints}
    assert (1, 1) in constraints
    assert (2, 4) in constraints
    assert not mapper_feedback.zero_loss

    square = {"kind": "Multiply", "left": _item(), "right": _item()}
    predicate_feedback = derive_automatic_feedback(config.spec, predicate, square)
    assert predicate_feedback.predicate_toggles == (predicate_feedback.predicate_toggles[0],)
    toggle = predicate_feedback.predicate_toggles[0]
    assert (toggle.item, toggle.keep, toggle.source_examples) == (-1, True, (2, 4))


def test_no_predicate_toggle_is_claimed_when_one_change_cannot_repair_example() -> None:
    config = load_experiment_config("examples/foldr-sparse-bounded-square-v2.json")
    feedback = derive_automatic_feedback(config.spec, _wrong_predicate(), _constant(3))
    assert feedback.predicate_toggles == ()


def test_predicate_toggle_changes_all_equal_items_for_deterministic_semantics() -> None:
    spec = PBESpec.model_validate(
        {
            "signature": {"input": "List<Int>", "output": "List<Int>"},
            "examples": [{"input": ["-1", "-1"], "output": ["1"]}],
            "integerConstants": ["-1", "0", "1"],
        }
    )
    square = {"kind": "Multiply", "left": _item(), "right": _item()}
    feedback = derive_automatic_feedback(spec, _wrong_predicate(), square)
    assert feedback.predicate_toggles == ()


def test_mapper_constraint_sources_are_deduplicated_for_repeated_items() -> None:
    spec = PBESpec.model_validate(
        {
            "signature": {"input": "List<Int>", "output": "List<Int>"},
            "examples": [{"input": ["1", "1"], "output": ["7", "7"]}],
            "integerConstants": ["1", "7"],
        }
    )
    keep_one = {"kind": "EqualInt", "left": _item(), "right": _constant(1)}
    feedback = derive_automatic_feedback(spec, keep_one, _constant(0))
    assert feedback.mapper_constraints[0].source_examples == (1,)

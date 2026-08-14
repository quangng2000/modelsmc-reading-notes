from __future__ import annotations

import hashlib
import json
from pathlib import Path

from research.heldout import evaluate_result, generate_cases, generate_inputs
from research.protocol import TaskSpec, load_protocol

PROTOCOL = Path(__file__).parents[1] / "protocol.json"


def _task(task_id: str) -> TaskSpec:
    protocol = load_protocol(PROTOCOL)
    return next(task for task in protocol.tasks if task.task_id == task_id)


def test_heldout_generation_is_deterministic_and_excludes_training() -> None:
    task = _task("foldr-signed-window")
    first = generate_inputs(task)
    second = generate_inputs(task)

    assert first == second
    assert len(first) == 96
    corpus = json.dumps(first, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(corpus).hexdigest() == task.heldout.corpus_sha256
    training = json.loads(task.spec_path.read_text(encoding="utf-8"))["examples"]
    training_inputs = {
        json.dumps([int(item) for item in example["input"]], separators=(",", ":"))
        for example in training
    }
    assert all(json.dumps(value, separators=(",", ":")) not in training_inputs for value in first)
    first_input = first[0]
    assert isinstance(first_input, list)
    assert generate_cases(task)[0][1] == [
        (-item if item < 0 else item * item)
        for item in first_input
        if -3 < item < 3
    ]


def test_known_map_program_is_exact_on_heldout(tmp_path: Path) -> None:
    task = _task("map-increment")
    result = {
        "status": "completed",
        "result": {
            "sampled_best": {
                "program": {
                    "kind": "MapProgram",
                    "mapper": {
                        "kind": "Add",
                        "left": {"kind": "Item"},
                        "right": {"kind": "IntLiteral", "intValue": "1"},
                    },
                },
                "exact_program": True,
            }
        },
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    evaluation = evaluate_result(task, result_path)

    assert evaluation["status"] == "completed"
    assert evaluation["passed"] == 64
    assert evaluation["exact"] is True
    assert evaluation["program_source"] == "sampled_best_legacy_fallback"


def test_heldout_prefers_best_visited_over_lost_final_particle(tmp_path: Path) -> None:
    task = _task("map-increment")
    result = {
        "status": "completed",
        "result": {
            "best_visited": {
                "program": {
                    "kind": "MapProgram",
                    "mapper": {
                        "kind": "Add",
                        "left": {"kind": "Item"},
                        "right": {"kind": "IntLiteral", "intValue": "1"},
                    },
                },
                "exact_program": True,
            },
            "sampled_best": {
                "program": {
                    "kind": "ExpressionProgram",
                    "body": {"kind": "Input"},
                },
                "exact_program": False,
            },
        },
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    evaluation = evaluate_result(task, result_path)

    assert evaluation["status"] == "completed"
    assert evaluation["passed"] == 64
    assert evaluation["exact"] is True
    assert evaluation["program_source"] == "best_visited"

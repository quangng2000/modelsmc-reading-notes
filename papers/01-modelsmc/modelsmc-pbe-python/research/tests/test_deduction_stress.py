from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from research.heldout import generate_cases, generate_inputs
from research.protocol import effective_caps, load_protocol
from research.run_matrix import build_plan, command_for, find_stage

PROTOCOL_PATH = Path(__file__).parents[1] / "protocol-deduction-stress-v1.json"


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_deduction_stress_protocol_freezes_paired_d_and_qd_plans() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    assert protocol.caps.max_scored_candidates == 2_652

    preflight = find_stage(protocol, "local-d-preflight")
    assert preflight is not None
    preflight_caps = effective_caps(protocol, preflight)

    d_plans = build_plan(protocol, stage=preflight)
    assert len(d_plans) == 5
    assert [plan.seed for plan in d_plans] == [101, 211, 307, 401, 503]
    assert all(plan.arm == "D" for plan in d_plans)
    assert all(plan.model_scope == "size-invariant" for plan in d_plans)
    assert all(plan.model_id is None for plan in d_plans)
    assert preflight_caps.max_scored_candidates == 2_652
    assert preflight.max_provider_scored_candidates == 0

    paired = find_stage(protocol, "paired-d-qd-32b")
    assert paired is not None
    paired_caps = effective_caps(protocol, paired)
    paired_plans = build_plan(protocol, stage=paired)
    assert len(paired_plans) == 10
    assert sum(plan.model_id is None for plan in paired_plans) == 5
    assert sum(plan.model_id == "qwen25-coder-32b" for plan in paired_plans) == 5
    assert paired.max_provider_scored_candidates == 13_260

    qd_plans = build_plan(
        protocol,
        stage=paired,
        arms={"QD"},
        seeds={101},
        model_ids={"qwen25-coder-32b"},
    )
    assert len(qd_plans) == 1
    plan = qd_plans[0]
    assert plan.arm == "QD"
    assert plan.model_scope == "model-specific"
    assert plan.model_id == "qwen25-coder-32b"
    assert plan.model_hf_repository == "Qwen/Qwen2.5-Coder-32B-Instruct"
    assert plan.model_dtype == "bfloat16"
    assert plan.model_quantization == "none"
    assert plan.model_revision == "381fc969f78efac66bc87ff7ddeadb7e73c218a7"
    assert plan.tokenizer_revision == "381fc969f78efac66bc87ff7ddeadb7e73c218a7"

    task = next(task for task in protocol.tasks if task.task_id == plan.task_id)
    arm = next(arm for arm in protocol.arms if arm.name == plan.arm)
    command = command_for(
        protocol,
        task,
        arm,
        plan,
        caps=paired_caps,
        executable="modelsmc-pbe",
        artifacts_dir=Path("/tmp/deduction-stress-test"),
        base_url="https://provider.invalid/v1",
    )
    assert _value_after(command, "--particles") == "4"
    assert _value_after(command, "--iterations") == "1"
    assert _value_after(command, "--max-scored-candidates") == "2652"
    assert _value_after(command, "--model") == "qwen25-coder-32b"


def test_sparse_bounded_square_heldout_corpus_is_frozen_and_deterministic() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    task = protocol.tasks[0]

    assert task.task_id == "foldr-sparse-bounded-square"
    assert task.heldout.oracle == "bounded_square"
    assert task.heldout.seed == 1401
    assert task.heldout.count == 96
    assert (task.heldout.minimum, task.heldout.maximum) == (-8, 8)
    assert task.heldout.max_length == 10

    first = generate_inputs(task)
    second = generate_inputs(task)
    assert first == second
    assert len(first) == 96
    corpus = json.dumps(first, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(corpus).hexdigest() == (
        "4b1ac93177418236e597c91bef5ebf7f192a7de04f344baf137c7460539271ab"
    )

    for input_value, expected in generate_cases(task):
        values = cast(list[int], input_value)
        assert expected == [item * item for item in values if -2 < item < 3]

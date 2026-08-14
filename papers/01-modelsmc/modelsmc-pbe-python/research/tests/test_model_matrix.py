from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from research.protocol import effective_caps, load_protocol
from research.run_matrix import (
    _base_url_for_plan,
    _initialize_matrix,
    _resolve_executable,
    _validate_resume,
    build_plan,
    command_for,
    find_stage,
    main,
)

PROTOCOL_PATH = Path(__file__).parents[1] / "protocol.json"
AMENDED_PROTOCOL_PATH = (
    Path(__file__).parents[1] / "protocol-size-study-transport32.json"
)


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _write_protocol_copy(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None]
) -> Path:
    document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    for task in document["tasks"]:
        task["spec"] = str((PROTOCOL_PATH.parent / task["spec"]).resolve())
    mutate(document)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_relative_executable_is_canonicalized_before_hashing_and_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "tool"
    executable.write_text("test entry point", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert _resolve_executable("./tool") == str(executable.resolve())


def test_size_matrix_crosses_models_only_for_qwen_arms() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    plans = build_plan(
        protocol,
        task_ids={"negative-int-to-bool"},
        seeds={101},
    )

    assert [model.model_id for model in protocol.models] == [
        "qwen25-coder-3b",
        "qwen25-coder-7b",
        "qwen25-coder-14b",
        "qwen25-coder-32b",
    ]
    assert len(plans) == 10
    assert sum(plan.arm in {"U", "D"} for plan in plans) == 2
    assert sum(plan.arm in {"Q", "QD"} for plan in plans) == 8
    assert all(plan.model_id is None for plan in plans if plan.arm in {"U", "D"})
    assert all(plan.model_id is not None for plan in plans if plan.arm in {"Q", "QD"})
    assert len({plan.cell_id for plan in plans}) == len(plans)


def test_schema_two_rejects_a_mutable_model_revision(tmp_path: Path) -> None:
    def mutate(document: dict[str, Any]) -> None:
        models = document["models"]
        assert isinstance(models, list)
        assert isinstance(models[0], dict)
        models[0]["model_revision"] = "main"

    path = _write_protocol_copy(tmp_path, mutate)

    with pytest.raises(ValueError, match="40-hex commit"):
        load_protocol(path)


def test_protocol_emits_only_explicit_split_deduction_mix_overrides(
    tmp_path: Path,
) -> None:
    def mutate(document: dict[str, Any]) -> None:
        arms = document["arms"]
        assert isinstance(arms, list)
        qd = next(arm for arm in arms if arm["id"] == "QD")
        qd["family_deduction_mix"] = 0.8
        qd["hole_deduction_mix"] = 0.1

    split = load_protocol(_write_protocol_copy(tmp_path, mutate))
    stage = find_stage(split, "gate-2-size-pilot")
    assert stage is not None
    plan = next(
        plan
        for plan in build_plan(split, stage=stage, model_ids={"qwen25-coder-3b"})
        if plan.arm == "QD"
    )
    task = next(task for task in split.tasks if task.task_id == plan.task_id)
    arm = next(arm for arm in split.arms if arm.name == plan.arm)
    command = command_for(
        split,
        task,
        arm,
        plan,
        caps=effective_caps(split, stage),
        executable="modelsmc-pbe",
        artifacts_dir=tmp_path,
        base_url="https://provider.invalid/v1",
    )

    assert _value_after(command, "--deduction-mix") == "0.75"
    assert _value_after(command, "--family-deduction-mix") == "0.8"
    assert _value_after(command, "--hole-deduction-mix") == "0.1"

    frozen = load_protocol(PROTOCOL_PATH)
    frozen_arm = next(arm for arm in frozen.arms if arm.name == "QD")
    assert frozen_arm.family_deduction_mix is None
    assert frozen_arm.hole_deduction_mix is None


def test_transport_amendment_changes_only_batching_and_exploratory_metadata() -> None:
    original = load_protocol(PROTOCOL_PATH)
    amended = load_protocol(AMENDED_PROTOCOL_PATH)
    document = json.loads(AMENDED_PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert amended.protocol_sha256 == (
        "56c590864d456f884c82bf62ada3c11c1c2504d21b021e650bc323007553fc60"
    )
    assert document["amendment"]["classification"] == "exploratory-transport-only"
    assert document["amendment"]["supersedes_protocol_sha256"] == (
        original.protocol_sha256
    )
    assert original.shared_arguments["candidate_batch_size"] == 128
    assert amended.shared_arguments["candidate_batch_size"] == 32
    assert {
        key: value
        for key, value in original.shared_arguments.items()
        if key != "candidate_batch_size"
    } == {
        key: value
        for key, value in amended.shared_arguments.items()
        if key != "candidate_batch_size"
    }
    assert amended.caps == original.caps
    assert amended.provider == original.provider
    assert amended.models == original.models
    assert amended.arms == original.arms
    assert amended.seeds == original.seeds
    assert amended.materialize_reference == original.materialize_reference
    assert all(task.label == "exploratory" for task in amended.tasks)
    assert [
        (task.task_id, task.spec_path, task.seen_during_development, task.heldout)
        for task in amended.tasks
    ] == [
        (task.task_id, task.spec_path, task.seen_during_development, task.heldout)
        for task in original.tasks
    ]
    assert [
        (
            stage.stage_id,
            stage.task_ids,
            stage.arms,
            stage.seeds,
            stage.model_ids,
            stage.caps,
            stage.max_provider_scored_candidates,
        )
        for stage in amended.stages
    ] == [
        (
            stage.stage_id,
            stage.task_ids,
            stage.arms,
            stage.seeds,
            stage.model_ids,
            stage.caps,
            stage.max_provider_scored_candidates,
        )
        for stage in original.stages
    ]


def test_amended_gate_two_32b_is_exactly_four_cost_gated_cells(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    protocol = load_protocol(AMENDED_PROTOCOL_PATH)
    stage = find_stage(protocol, "gate-2-size-pilot")
    assert stage is not None
    caps = effective_caps(protocol, stage)
    plans = build_plan(
        protocol,
        stage=stage,
        model_ids={"qwen25-coder-32b"},
    )

    assert caps.max_scored_candidates == 4000
    assert len(plans) == 4
    assert [(plan.task_id, plan.arm) for plan in plans] == [
        ("map-increment", "Q"),
        ("map-increment", "QD"),
        ("foldr-signed-window", "Q"),
        ("foldr-signed-window", "QD"),
    ]
    for plan in plans:
        task = next(task for task in protocol.tasks if task.task_id == plan.task_id)
        arm = next(arm for arm in protocol.arms if arm.name == plan.arm)
        command = command_for(
            protocol,
            task,
            arm,
            plan,
            caps=caps,
            executable="modelsmc-pbe",
            artifacts_dir=Path("/tmp/amended-gate2"),
            base_url="https://provider.invalid/v1",
        )
        assert _value_after(command, "--candidate-batch-size") == "32"
        assert _value_after(command, "--max-scored-candidates") == "4000"

    assert (
        main(
            [
                "--protocol",
                str(AMENDED_PROTOCOL_PATH),
                "--output",
                str(tmp_path / "unused"),
                "--stage",
                "gate-2-size-pilot",
                "--models",
                "qwen25-coder-32b",
                "--max-provider-cells",
                "4",
                "--max-provider-score-cap",
                "16000",
                "--dry-run",
            ]
        )
        == 0
    )
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["cost_gate"]["provider_cells"] == 4
    assert dry_run["cost_gate"]["provider_candidate_score_cap"] == 16000


def test_gate_one_command_is_pinned_cached_and_mean_normalized(tmp_path: Path) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    stage = find_stage(protocol, "gate-1-score-smoke")
    assert stage is not None
    caps = effective_caps(protocol, stage)
    plan = build_plan(
        protocol,
        stage=stage,
        model_ids={"qwen25-coder-3b"},
    )[0]
    task = next(task for task in protocol.tasks if task.task_id == plan.task_id)
    arm = next(arm for arm in protocol.arms if arm.name == plan.arm)

    command = command_for(
        replace(protocol, materialize_reference=True),
        task,
        arm,
        plan,
        caps=caps,
        executable="modelsmc-pbe",
        artifacts_dir=tmp_path,
        base_url="https://provider.invalid/v1",
    )

    assert _value_after(command, "--model") == "qwen25-coder-3b"
    assert _value_after(command, "--model-repository") == (
        "Qwen/Qwen2.5-Coder-3B-Instruct"
    )
    assert _value_after(command, "--model-revision") == (
        "488639f1ff808d1d3d0ba301aef8c11461451ec5"
    )
    assert _value_after(command, "--tokenizer-revision") == (
        "488639f1ff808d1d3d0ba301aef8c11461451ec5"
    )
    assert _value_after(command, "--llm-energy-normalization") == (
        "mean-full-prompt-conditional-logprob"
    )
    assert _value_after(command, "--score-cache-mode") == "read-write"
    assert _value_after(command, "--max-scored-candidates") == "128"
    assert "processed_logprobs" in _value_after(command, "--vllm-server-config")
    assert "--materialize-reference" in command


def test_model_specific_endpoint_env_and_cost_gated_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    stage = find_stage(protocol, "gate-1-score-smoke")
    assert stage is not None
    plans = build_plan(protocol, stage=stage)
    three_b = next(plan for plan in plans if plan.model_id == "qwen25-coder-3b")
    seven_b = next(plan for plan in plans if plan.model_id == "qwen25-coder-7b")
    monkeypatch.setenv("MODELSMC_VLLM_3B_BASE_URL", "https://three.invalid/v1")
    monkeypatch.setenv("MODELSMC_VLLM_7B_BASE_URL", "https://seven.invalid/v1")

    assert _base_url_for_plan(protocol, three_b, None) == "https://three.invalid/v1"
    assert _base_url_for_plan(protocol, seven_b, None) == "https://seven.invalid/v1"
    assert (
        main(
            [
                "--protocol",
                str(PROTOCOL_PATH),
                "--output",
                str(tmp_path / "unused"),
                "--stage",
                "gate-1-score-smoke",
                "--models",
                "qwen25-coder-3b",
                "--max-provider-cells",
                "1",
                "--max-provider-score-cap",
                "128",
                "--dry-run",
            ]
        )
        == 0
    )
    document = json.loads(capsys.readouterr().out)
    assert document["cost_gate"] == {
        "cells": 1,
        "local_control_cells": 0,
        "provider_cells": 1,
        "provider_candidate_score_cap": 128,
        "models": ["qwen25-coder-3b"],
        "tasks": ["negative-int-to-bool"],
        "arms": ["Q"],
        "seeds": [101],
    }


def test_resume_requires_identical_model_aware_plan(tmp_path: Path) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    stage = find_stage(protocol, "gate-1-score-smoke")
    assert stage is not None
    caps = effective_caps(protocol, stage)
    one_model = build_plan(
        protocol,
        stage=stage,
        model_ids={"qwen25-coder-3b"},
    )
    output = tmp_path / "matrix"
    _initialize_matrix(output, protocol, one_model, stage=stage, caps=caps)

    _validate_resume(output, protocol, one_model, stage=stage, caps=caps)
    two_models = build_plan(
        protocol,
        stage=stage,
        model_ids={"qwen25-coder-3b", "qwen25-coder-7b"},
    )
    with pytest.raises(ValueError, match="selection differs"):
        _validate_resume(output, protocol, two_models, stage=stage, caps=caps)

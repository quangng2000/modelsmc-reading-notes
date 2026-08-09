from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from modelsmc_pbe.config import ExperimentConfig, load_experiment_config
from modelsmc_pbe.domain import ValueType
from modelsmc_pbe.runtime import DeviceResolutionError, resolve_device, seed_everything


def flat_config() -> dict[str, object]:
    return {
        "name": "increment",
        "examples": [
            {"input": "-1", "output": "0"},
            {"input": "0", "output": "1"},
        ],
        "integerConstants": ["0", "1", "-1", "1"],
        "particles": 16,
        "iterations": 4,
        "cloneProbability": 0.2,
        "essThreshold": 0.75,
        "seed": 42,
        "lossScale": 2,
        "costScale": 0.1,
        "lossCap": 1000,
        "maxCost": 12,
        "maxDepth": 8,
        "maxNodes": 63,
    }


def test_flat_config_is_normalized() -> None:
    config = ExperimentConfig.model_validate(flat_config())

    assert config.spec.signature is not None
    assert config.spec.signature.input_type is ValueType.INT
    assert config.spec.signature.output_type is ValueType.INT
    assert config.spec.examples[0].input_value == -1
    assert config.spec.integer_constants == [0, 1, -1]
    assert config.smc.particles == 16
    assert config.smc.alpha == pytest.approx(0.2)


def test_scoring_defaults_are_stable() -> None:
    config = ExperimentConfig.model_validate(
        {"examples": [{"input": "0", "output": "0"}]}
    )

    assert config.spec.name == "unnamed-synthesis-task"
    assert config.smc.iterations == 6
    assert config.smc.alpha == pytest.approx(0.35)
    assert config.smc.loss_cap == 1_000_000
    assert config.smc.max_cost == 20
    assert config.smc.max_depth == 10
    assert config.smc.max_nodes == 127
    assert config.spec.integer_constants == [-2, -1, 0, 1, 2]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("particles", "8"),
        ("seed", "7"),
        ("lossScale", "2.0"),
        ("lossScale", float("inf")),
        ("costScale", float("nan")),
        ("cloneProbability", float("inf")),
    ],
)
def test_numeric_configuration_is_strict_and_finite(field: str, value: object) -> None:
    raw = flat_config()
    raw[field] = value

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)


def test_integer_constant_catalog_must_not_be_empty() -> None:
    raw = flat_config()
    raw["integerConstants"] = []

    with pytest.raises(ValidationError, match="at least one"):
        ExperimentConfig.model_validate(raw)


def test_lossless_integer_strings_are_canonical() -> None:
    raw = flat_config()
    raw["examples"] = [{"input": "+1", "output": "1"}]

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)


def test_loader_applies_validated_overrides(tmp_path: Path) -> None:
    path = tmp_path / "task.json"
    path.write_text(json.dumps(flat_config()), encoding="utf-8")

    config = load_experiment_config(
        path,
        smc_overrides={"particles": 32},
        runtime_overrides={"device": "cpu", "artifacts_dir": tmp_path / "artifacts"},
    )

    assert config.smc.particles == 32
    assert config.runtime.device == "cpu"
    assert config.runtime.artifacts_dir == tmp_path / "artifacts"


def test_signature_rejects_mismatched_examples() -> None:
    raw = flat_config()
    raw["signature"] = {"input": "Int", "output": "Bool"}

    with pytest.raises(ValidationError, match="output does not match signature"):
        ExperimentConfig.model_validate(raw)


def test_all_empty_lists_require_explicit_signature() -> None:
    raw = {"name": "empty", "examples": [{"input": [], "output": []}]}

    with pytest.raises(ValidationError, match="explicit signature"):
        ExperimentConfig.model_validate(raw)


def test_explicit_unavailable_cuda_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(DeviceResolutionError, match="is_available"):
        resolve_device("cuda")


def test_auto_device_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"test-gpu-{index}")

    info = resolve_device("auto")

    assert info.resolved == "cuda:0"
    assert info.accelerator is True
    assert "test-gpu-0" in info.reason


def test_seed_everything_replays_all_cpu_rngs() -> None:
    first = seed_everything(8675309)
    first_values = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(()).item()),
        float(torch.rand((), generator=first.cpu_generator).item()),
    )
    second = seed_everything(8675309)
    second_values = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(()).item()),
        float(torch.rand((), generator=second.cpu_generator).item()),
    )

    assert second_values == first_values

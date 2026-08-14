from __future__ import annotations

import json
import math

import pytest
import torch

from research.direct_json_auxiliary_importance import (
    extract_boundary_distribution,
    mix_with_uniform,
)


def _entry(token: str, *, alternatives: dict[str, float] | None = None) -> dict[str, object]:
    return {
        "token": token,
        "bytes": list(token.encode()),
        "logprob": 0.0,
        "top_logprobs": [
            {"token": candidate, "bytes": list(candidate.encode()), "logprob": value}
            for candidate, value in (alternatives or {}).items()
        ],
    }


def _response(selected: str) -> dict[str, object]:
    probabilities = (0.1, 0.6, 0.2, 0.1)
    final = [
        _entry("<|message|>"),
        _entry('{"selected_repair_id":"'),
        _entry("p"),
        _entry(
            selected[1],
            alternatives={str(index): math.log(value) for index, value in enumerate(probabilities)},
        ),
        _entry('"}'),
        _entry("<|return|>"),
    ]
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps({"selected_repair_id": selected})},
                "logprobs": {"content": final},
            }
        ]
    }


def test_extracts_processed_distribution_at_json_choice_boundary() -> None:
    selected, probabilities, proof = extract_boundary_distribution(
        _response("p1"),
        ("p0", "p1", "p2", "p3"),
    )
    assert selected == "p1"
    assert torch.allclose(
        probabilities,
        torch.tensor((0.1, 0.6, 0.2, 0.1), dtype=torch.float64),
        atol=1e-15,
        rtol=0.0,
    )
    assert proof["selected_repair_id"] == "p1"


def test_uniform_floor_is_positive_and_normalized() -> None:
    actual = mix_with_uniform(
        torch.tensor((0.0, 0.0, 1.0, 0.0), dtype=torch.float64),
        0.05,
    )
    assert torch.allclose(
        actual,
        torch.tensor((0.0125, 0.0125, 0.9625, 0.0125), dtype=torch.float64),
        atol=1e-15,
        rtol=0.0,
    )


def test_rejects_final_json_without_logprob_boundary() -> None:
    body = _response("p1")
    body["choices"][0]["logprobs"]["content"] = [_entry("<|message|>"), _entry("{}")]
    with pytest.raises(ValueError, match="repair-ID field"):
        extract_boundary_distribution(body, ("p0", "p1", "p2", "p3"))

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import research.analyze_blind_filter_map_confirmation_v3 as analyzer
from research.blind_filter_map_confirmation_v3 import TASK_IDS, canonical_bytes


def test_analysis_applies_exact_paired_test_and_practical_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = {
        "reporting": {
            "required_claim_limits": ["bounded frozen-distribution claim only"]
        }
    }
    monkeypatch.setattr(
        analyzer,
        "_validate_method",
        lambda **_: (protocol, {}, "1" * 64, "2" * 64, tmp_path / "harness.py"),
    )
    monkeypatch.setattr(
        analyzer,
        "_validate_provider_boundary",
        lambda **_: (
            {},
            {task_id: tmp_path / f"{task_id}.json" for task_id in TASK_IDS},
            "3" * 64,
            "4" * 64,
        ),
    )

    def synthetic_run(*, task_id: str, arm: str, **_: object) -> dict[str, object]:
        index = TASK_IDS.index(task_id)
        exact = arm == "llm" and index < 6
        return {
            "task_id": task_id,
            "arm": arm,
            "exact_by_21": exact and index < 3,
            "exact_by_29": exact,
            "first_exact_slot": 17 if exact else None,
            "best_loss": 0.0 if exact else 2.0,
            "provider_calls": 0 if arm == "grammar-random" else 7,
        }

    monkeypatch.setattr(analyzer, "_validate_run", synthetic_run)
    result = analyzer.analyze(
        repo_root=tmp_path,
        protocol_path=tmp_path / "protocol.json",
        method_seal_path=tmp_path / "method.json",
        expected_method_seal_sha256="2" * 64,
        custody_seal_path=tmp_path / "custody.json",
        expected_custody_seal_sha256="3" * 64,
        provider_seal_path=tmp_path / "provider.json",
        expected_provider_seal_sha256="4" * 64,
        runs_root=tmp_path / "runs",
    )
    primary = result["primary"]
    assert primary["llm_successes"] == 6
    assert primary["grammar_random_successes"] == 0
    assert primary["one_sided_exact_sign_test_p_value"] == pytest.approx(1 / 64)
    assert primary["practical_threshold_met"] is True
    assert primary["study_success"] is True
    assert len(result["runs"]) == 24
    assert result["private_reveal_read"] is False


def test_provider_request_audit_rejects_explicit_support_cardinality_leak(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    request = run / "provider/round-01/parent-000/request.json"
    request.parent.mkdir(parents=True)
    request.write_bytes(
        canonical_bytes(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": '{"complete_program_count":36000}',
                    }
                ]
            }
        )
        + b"\n"
    )
    response = request.parent / "response.json"
    response.write_text("{}\n", encoding="utf-8")
    provider_result = request.parent / "result.json"
    provider_result.write_text("{}\n", encoding="utf-8")
    records = [
        {
            "path": path.relative_to(run).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted((request, response, provider_result))
    ]
    inventory_sha256 = hashlib.sha256(canonical_bytes(records)).hexdigest()
    (run / "provider-seal.json").write_bytes(
        canonical_bytes(
            {"records": records, "inventory_sha256": inventory_sha256}
        )
        + b"\n"
    )
    result = {
        "provider_inventory_sha256": inventory_sha256,
        "search": {"provider_calls": 1},
    }
    with pytest.raises(ValueError, match="information-boundary leak"):
        analyzer._validate_provider_inventory(run, result, arm="llm")


def test_provider_inventory_rejects_exact_byte_drift(tmp_path: Path) -> None:
    run = tmp_path / "run"
    stage = run / "provider/round-01/parent-000-mapper"
    stage.mkdir(parents=True)
    paths = []
    for name in ("request.json", "response.json", "result.json"):
        path = stage / name
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        paths.append(path)
    records = [
        {
            "path": path.relative_to(run).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(paths)
    ]
    inventory_sha256 = hashlib.sha256(canonical_bytes(records)).hexdigest()
    (run / "provider-seal.json").write_bytes(
        canonical_bytes(
            {"records": records, "inventory_sha256": inventory_sha256}
        )
        + b"\n"
    )
    result = {
        "provider_inventory_sha256": inventory_sha256,
        "search": {"provider_calls": 1},
    }
    (stage / "response.json").write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"byte count|SHA-256"):
        analyzer._validate_provider_inventory(run, result, arm="llm")

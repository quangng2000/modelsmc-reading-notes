from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from research.blind_filter_map_confirmation_v3 import (
    ANALYSIS_SCHEMA,
    RESULT_SCHEMA,
    TASK_IDS,
    canonical_bytes,
    sha256_file,
)
from research.run_blind_filter_map_confirmation_v3 import ARMS
from research.seal_blind_filter_map_confirmation_v3_public import (
    ROUND_FILES,
    seal_public_artifacts,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def _public_fixture(tmp_path: Path) -> tuple[Path, Path]:
    runs = tmp_path / "runs"
    empty_inventory = hashlib.sha256(canonical_bytes([])).hexdigest()
    rows = []
    for task_id in TASK_IDS:
        for arm in ARMS:
            task_root = runs / task_id
            run = task_root / arm
            _write_json(task_root / f"{arm}.invocation.json", {"arm": arm})
            _write_json(task_root / f"{arm}.preflight.json", {"arm": arm})
            _write_json(run / "protocol.json", {"proposal_source": arm})
            _write_json(run / "executions.json", [{"slot": index} for index in range(1, 30)])
            for round_name in ROUND_FILES:
                _write_json(
                    run / round_name,
                    {
                        "round": int(round_name.removeprefix("round-").removesuffix(".json")),
                        "proposal_source": arm,
                    },
                )
            _write_json(
                run / "provider-seal.json",
                {"records": [], "inventory_sha256": empty_inventory},
            )
            result = {
                "schema": RESULT_SCHEMA,
                "provider_inventory_sha256": empty_inventory,
                "search": {"provider_calls": 0},
            }
            _write_json(run / "result.json", result)
            rows.append(
                {
                    "task_id": task_id,
                    "arm": arm,
                    "result_file_sha256": sha256_file(run / "result.json"),
                    "execution_file_sha256": sha256_file(run / "executions.json"),
                    "provider_inventory_sha256": empty_inventory,
                }
            )
    analysis = tmp_path / "analysis.json"
    _write_json(
        analysis,
        {
            "schema": ANALYSIS_SCHEMA,
            "status": "complete-reveal-free-confirmatory-analysis",
            "method_integrity": "valid",
            "private_reveal_read": False,
            "runs": rows,
        },
    )
    return runs, analysis


def test_complete_reveal_free_bundle_is_sealed_once(tmp_path: Path) -> None:
    runs, analysis = _public_fixture(tmp_path)
    output = tmp_path / "public-seal"
    preflight = seal_public_artifacts(
        runs_root=runs,
        analysis_path=analysis,
        output=output,
        preflight_only=True,
    )
    assert not output.exists()
    result = seal_public_artifacts(
        runs_root=runs,
        analysis_path=analysis,
        output=output,
    )
    sums = (output / "SHA256SUMS").read_bytes()
    assert result == preflight
    assert result["file_count"] == len(sums.splitlines())
    assert result["bundle_sha256"] == hashlib.sha256(sums).hexdigest()
    assert (output / "BUNDLE_SHA256").read_text(encoding="ascii").strip() == result[
        "bundle_sha256"
    ]
    with pytest.raises(FileExistsError, match="overwrite"):
        seal_public_artifacts(
            runs_root=runs,
            analysis_path=analysis,
            output=output,
        )


def test_public_bundle_rejects_private_reveal_path(tmp_path: Path) -> None:
    runs, analysis = _public_fixture(tmp_path)
    _write_json(runs / "blind-v3-01/llm/private/reveal.json", {"seed_hex": "00" * 32})
    with pytest.raises(ValueError, match="private path"):
        seal_public_artifacts(
            runs_root=runs,
            analysis_path=analysis,
            output=tmp_path / "public-seal",
            preflight_only=True,
        )


def test_public_bundle_rejects_missing_frozen_round_artifact(tmp_path: Path) -> None:
    runs, analysis = _public_fixture(tmp_path)
    (runs / "blind-v3-01/grammar-random/round-04.json").unlink()
    with pytest.raises(ValueError, match="unexpected/incomplete run entries"):
        seal_public_artifacts(
            runs_root=runs,
            analysis_path=analysis,
            output=tmp_path / "public-seal",
            preflight_only=True,
        )


def test_public_bundle_rejects_extra_round_artifact(tmp_path: Path) -> None:
    runs, analysis = _public_fixture(tmp_path)
    _write_json(
        runs / "blind-v3-01/grammar-random/round-05.json",
        {"round": 5, "proposal_source": "grammar-random"},
    )
    with pytest.raises(ValueError, match="unexpected/incomplete run entries"):
        seal_public_artifacts(
            runs_root=runs,
            analysis_path=analysis,
            output=tmp_path / "public-seal",
            preflight_only=True,
        )

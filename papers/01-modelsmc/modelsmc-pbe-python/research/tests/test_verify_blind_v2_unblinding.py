from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from research.generate_blinded_filter_map_tasks import generate_suite, seed_commitment
from research.seal_blind_evidence_frontier_v2_public import (
    _inventory_bytes as finalized_sealer_inventory_bytes,
)
from research.verify_blind_v2_unblinding import (
    ANALYSIS_SCHEMA,
    PROVIDER_SEAL_SCHEMA,
    PUBLIC_SEALER_PATH,
    PUBLIC_SEALER_SHA256,
    REPORT_SCHEMA,
    STUDY_SCHEMA,
    _public_inventory_bytes,
    canonical_bytes,
    main,
    sha256_bytes,
    sha256_file,
)

SEED = bytes.fromhex("0123456789abcdef" * 4)


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_bytes(value) + b"\n")


def _build_inputs(tmp_path: Path) -> dict[str, Path | str]:
    repo_root = Path(__file__).parents[2]
    generator = repo_root / "research" / "generate_blinded_filter_map_tasks.py"
    secret = tmp_path / "custody.secret"
    secret.write_bytes(SEED)
    os.chmod(secret, 0o600)
    suite = tmp_path / "suite"
    manifest = generate_suite(suite, seed_file=secret, suite_version="v2")
    reveal = suite / "private" / "reveal.json"
    public_manifest = suite / "public" / "manifest.json"
    commitment = seed_commitment(SEED, suite_version="v2")

    study = tmp_path / "study.json"
    _write_json(
        study,
        {
            "schema": STUDY_SCHEMA,
            "protocol_status": "method-frozen-before-task-generation",
            "freeze_requirements": {
                "generator": {
                    "path": "research/generate_blinded_filter_map_tasks.py",
                    "sha256": sha256_file(generator),
                },
                "task_secret_commitment_sha256": commitment,
            },
        },
    )
    provider = tmp_path / "provider-seal.json"
    _write_json(
        provider,
        {
            "schema": PROVIDER_SEAL_SCHEMA,
            "sealed_before_provider_calls": True,
            "study_protocol_sha256": sha256_file(study),
            "task_secret_commitment_sha256": commitment,
            "hidden_target_manifest_sha256": sha256_file(reveal),
            "public_manifest": {
                "path": str(public_manifest),
                "sha256": sha256_file(public_manifest),
            },
        },
    )
    analysis_root = tmp_path / "analysis"
    analysis_root.mkdir()
    analysis = analysis_root / "analysis.json"
    _write_json(
        analysis,
        {
            "schema": ANALYSIS_SCHEMA,
            "blind": True,
            "confirmatory": True,
            "private_reveal_used": False,
            "integrity": {"passed": True},
            "bindings": {
                "study_protocol_sha256": sha256_file(study),
                "provider_call_seal_sha256": sha256_file(provider),
                "public_manifest_sha256": sha256_file(public_manifest),
                "hidden_target_manifest_sha256": sha256_file(reveal),
            },
        },
    )
    (analysis_root / "SUMMARY.md").write_text("# Synthetic reveal-free analysis\n")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _write_json(runs_root / "synthetic-public-run.json", {"private_reveal_used": False})
    analysis_seal = tmp_path / "analysis-seal"
    analysis_seal_sha256 = _write_public_seal(analysis_seal, runs_root, analysis_root)
    assert manifest["task_count"] == 12
    return {
        "repo_root": repo_root,
        "study": study,
        "provider": provider,
        "analysis": analysis,
        "analysis_root": analysis_root,
        "runs_root": runs_root,
        "analysis_seal": analysis_seal,
        "analysis_seal_sha256": analysis_seal_sha256,
        "public_dir": suite / "public",
        "secret": secret,
        "receipt": secret.with_name(f"{secret.name}.used.json"),
        "reveal": reveal,
    }


def _write_public_seal(seal: Path, runs_root: Path, analysis_root: Path) -> str:
    records = [
        (f"runs/{path.relative_to(runs_root).as_posix()}", path)
        for path in runs_root.rglob("*")
        if path.is_file()
    ]
    records.extend(
        (f"analysis/{path.relative_to(analysis_root).as_posix()}", path)
        for path in analysis_root.rglob("*")
        if path.is_file()
    )
    inventory = "".join(
        f"{sha256_file(path)}  {relative}\n" for relative, path in sorted(records)
    ).encode()
    bundle_sha256 = sha256_bytes(inventory)
    seal.mkdir(exist_ok=True)
    (seal / "SHA256SUMS").write_bytes(inventory)
    (seal / "BUNDLE_SHA256").write_text(f"{bundle_sha256}\n", encoding="utf-8")
    return bundle_sha256


def _arguments(inputs: dict[str, Path | str], output: Path) -> list[str]:
    return [
        "--repo-root",
        str(inputs["repo_root"]),
        "--study-protocol",
        str(inputs["study"]),
        "--provider-call-seal",
        str(inputs["provider"]),
        "--analysis-seal",
        str(inputs["analysis_seal"]),
        "--expected-analysis-seal-sha256",
        str(inputs["analysis_seal_sha256"]),
        "--runs-root",
        str(inputs["runs_root"]),
        "--analysis-root",
        str(inputs["analysis_root"]),
        "--public-dir",
        str(inputs["public_dir"]),
        "--secret-file",
        str(inputs["secret"]),
        "--receipt",
        str(inputs["receipt"]),
        "--reveal",
        str(inputs["reveal"]),
        "--output",
        str(output),
    ]


def test_inventory_is_byte_compatible_with_finalized_public_sealer(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[2]
    runs_root = tmp_path / "runs"
    analysis_root = tmp_path / "analysis"
    (runs_root / "nested").mkdir(parents=True)
    analysis_root.mkdir()
    (runs_root / "nested" / "artifact.json").write_bytes(b'{"value":1}\n')
    (analysis_root / "analysis.json").write_bytes(b'{"value":2}\n')
    (analysis_root / "SUMMARY.md").write_bytes(b"# summary\n")

    assert sha256_file(repo_root / PUBLIC_SEALER_PATH) == PUBLIC_SEALER_SHA256
    assert _public_inventory_bytes(runs_root, analysis_root) == (
        finalized_sealer_inventory_bytes(runs_root, analysis_root)
    )


def test_post_unblind_report_verifies_complete_chain_without_emitting_targets(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(tmp_path)
    output = tmp_path / "post-unblind-report.json"

    assert main(_arguments(inputs, output)) == 0

    report: dict[str, Any] = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == REPORT_SCHEMA
    assert all(report["checks"].values())
    assert report["generator_verification"]["verified"] == [
        f"blind-v2-{index:02d}" for index in range(1, 13)
    ]
    assert len(report["target_commitment_sha256"]) == 12
    report_text = output.read_text(encoding="utf-8")
    assert SEED.hex() not in report_text
    assert "target_predicate_dsl" not in report_text
    assert "target_mapper_dsl" not in report_text


def test_private_analysis_fails_before_secret_or_report_is_read(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(tmp_path)
    analysis = Path(inputs["analysis"])
    value = json.loads(analysis.read_text(encoding="utf-8"))
    value["private_reveal_used"] = True
    _write_json(analysis, value)
    analysis_seal = Path(inputs["analysis_seal"])
    inputs["analysis_seal_sha256"] = _write_public_seal(
        analysis_seal,
        Path(inputs["runs_root"]),
        Path(inputs["analysis_root"]),
    )
    Path(inputs["secret"]).unlink()
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="analysis private_reveal_used"):
        main(_arguments(inputs, output))

    assert not output.exists()

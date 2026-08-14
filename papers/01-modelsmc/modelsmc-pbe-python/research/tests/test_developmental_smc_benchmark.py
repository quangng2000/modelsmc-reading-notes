from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from research.analyze_developmental_smc_benchmark import analyze, wilson_interval
from research.run_developmental_smc_benchmark import validate_and_build_command


def _paths() -> tuple[Path, Path, Path]:
    project = Path(__file__).resolve().parents[2]
    protocol = project / "research/protocol-developmental-smc-benchmark-v1.json"
    relative_suite = Path(
        "artifacts/execution-guided-repair/"
        "blind-evidence-frontier-v2/suite/public"
    )
    candidates = tuple(root / relative_suite for root in (project, *project.parents))
    suite = next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])
    return project, protocol, suite


def _protocol_sha256(protocol: Path) -> str:
    return sha256(protocol.read_bytes()).hexdigest()


def test_llm_preflight_builds_only_the_frozen_29_slot_command(tmp_path: Path) -> None:
    project, protocol, suite = _paths()
    command, record = validate_and_build_command(
        repo_root=project,
        protocol_path=protocol,
        expected_protocol_sha256=_protocol_sha256(protocol),
        suite_dir=suite,
        runs_root=tmp_path / "runs",
        task_id="blind-v2-01",
        arm="llm-smc",
        python_executable="/pinned/python",
    )
    assert command[:3] == ["/pinned/python", "-m", "research.evidence_shortlist_smc"]
    assert command[command.index("--proposal-source") + 1] == "llm"
    assert "--random-shortlist-seed" not in command
    assert command[command.index("--parent-count") + 1] == "2"
    assert command[command.index("--offspring-per-parent") + 1] == "4"
    assert command[command.index("--first-round-offspring") + 1] == "4"
    assert record["status"] == "validated-before-provider-call"
    assert record["developmental"] is True


@pytest.mark.parametrize(
    ("arm", "epsilon", "evidence_scale"),
    (("evidence-only", "0.05", "2.0"), ("grammar-only", "1.0", "2.0")),
)
def test_no_llm_control_preflight_is_network_free_and_frozen(
    tmp_path: Path,
    arm: str,
    epsilon: str,
    evidence_scale: str,
) -> None:
    project, protocol, suite = _paths()
    command, _ = validate_and_build_command(
        repo_root=project,
        protocol_path=protocol,
        expected_protocol_sha256=_protocol_sha256(protocol),
        suite_dir=suite,
        runs_root=tmp_path / "runs",
        task_id="blind-v2-12",
        arm=arm,
        python_executable="/pinned/python",
    )
    assert command[command.index("--proposal-source") + 1] == "grammar-random"
    assert "--random-shortlist-seed" in command
    assert command[command.index("--epsilon") + 1] == epsilon
    assert command[command.index("--evidence-scale") + 1] == evidence_scale


def test_every_command_uses_its_frozen_run_evidence_scale(tmp_path: Path) -> None:
    project, protocol, suite = _paths()
    frozen = json.loads(protocol.read_text(encoding="utf-8"))
    for run in frozen["runs"]:
        command, _ = validate_and_build_command(
            repo_root=project,
            protocol_path=protocol,
            expected_protocol_sha256=_protocol_sha256(protocol),
            suite_dir=suite,
            runs_root=tmp_path / "runs",
            task_id=run["task"],
            arm=run["arm"],
            python_executable="/pinned/python",
        )
        assert command[command.index("--evidence-scale") + 1] == str(
            run["evidence_scale"]
        )


def test_preflight_rejects_external_protocol_hash_drift(tmp_path: Path) -> None:
    project, protocol, suite = _paths()
    with pytest.raises(ValueError, match="external protocol SHA-256"):
        validate_and_build_command(
            repo_root=project,
            protocol_path=protocol,
            expected_protocol_sha256="0" * 64,
            suite_dir=suite,
            runs_root=tmp_path / "runs",
            task_id="blind-v2-01",
            arm="llm-smc",
            python_executable="/pinned/python",
        )


def test_analysis_refuses_an_incomplete_matrix(tmp_path: Path) -> None:
    project, protocol, suite = _paths()
    with pytest.raises(ValueError, match="missing regular run directory"):
        analyze(
            repo_root=project,
            protocol_path=protocol,
            expected_protocol_sha256=_protocol_sha256(protocol),
            suite_dir=suite,
            runs_root=tmp_path / "empty-runs",
        )


def test_wilson_interval_validates_counts() -> None:
    lower, upper = wilson_interval(6, 12)
    assert 0.20 < lower < 0.30
    assert 0.70 < upper < 0.80
    with pytest.raises(ValueError, match="invalid binomial counts"):
        wilson_interval(13, 12)

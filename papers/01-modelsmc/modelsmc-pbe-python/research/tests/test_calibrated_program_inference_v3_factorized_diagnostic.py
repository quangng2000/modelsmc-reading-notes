from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.calibrated_program_inference_v2_fresh import PRIMARY_ARM
from research.calibrated_program_inference_v2_screen import build_proposal
from research.calibrated_program_inference_v3_factorized_diagnostic import (
    COMPONENT_POOL_SIZE,
    STATUS,
    STUDY_SCHEMA,
    _resolve_artifact,
    build_protocol,
    factorized_evidence_mode_bank,
)
from research.particle_calibration_terminal_diagnostic_v2 import (
    TaskBinding,
    TerminalTask,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def outlier_task() -> TerminalTask:
    path = _resolve_artifact(
        PROJECT_ROOT,
        "artifacts/calibrated-program-inference-v2-fresh-suite/public/"
        "fresh-cal-v2-05.json",
    )
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return TerminalTask(TaskBinding("fresh-cal-v2-05", path, digest))


def test_protocol_is_explicitly_post_failure_and_nonconfirmatory() -> None:
    protocol = build_protocol(PROJECT_ROOT)
    assert protocol["schema"] == STUDY_SCHEMA
    assert protocol["status"] == STATUS
    assert protocol["authorization"]["confirmatory_gate"] is False
    assert protocol["authorization"]["provider_calls"] is False
    assert protocol["fixed_change"]["maximum_complete_representatives_scored"] == 64
    assert protocol["diagnostic_rule"]["not_a_fresh_gate"] is True


def test_factorized_bank_finds_exact_mode_missed_by_grammar_bank(
    outlier_task: TerminalTask,
) -> None:
    bank = factorized_evidence_mode_bank(outlier_task)
    flattened = [index for group in bank.groups for index in group]
    assert len(bank.groups) == 8
    assert int(outlier_task.exact[flattened].sum()) > 0
    assert bank.metadata["exact_programs_in_bank"] > 0
    assert bank.metadata["best_bank_loss"] == 0.0
    assert bank.metadata["complete_representatives_scored"] <= COMPONENT_POOL_SIZE**2
    assert bank.metadata["complete_program_catalog_ranked"] is False
    assert bank.metadata["hidden_target_used"] is False


def test_factorized_proposal_is_normalized_and_exactly_accounted(
    outlier_task: TerminalTask,
) -> None:
    bank = factorized_evidence_mode_bank(outlier_task)
    proposal = build_proposal(outlier_task, PRIMARY_ARM, bank)
    assert abs(float(proposal.probabilities.sum()) - 1.0) < 1e-12
    assert float(proposal.probabilities.min()) > 0.0
    assert proposal.census["importance_identity_maximum_absolute_error"] <= 1e-12


def test_failed_v2_binding_is_the_immutable_failed_gate() -> None:
    path = _resolve_artifact(
        PROJECT_ROOT,
        "artifacts/calibrated-program-inference-v2-fresh/analysis.json",
    )
    analysis = json.loads(path.read_text())
    assert analysis["primary_gate"]["pass"] is False
    assert analysis["primary_gate"]["rmse_by_particle_count"]["256"] == pytest.approx(
        0.26123545638786344
    )


def test_artifact_locator_is_layout_independent() -> None:
    path = _resolve_artifact(
        PROJECT_ROOT,
        "artifacts/calibrated-program-inference-v2-fresh/inventory.json",
    )
    assert path.name == "inventory.json"
    assert path.is_file()

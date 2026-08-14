"""Reveal-free deterministic replay for calibrated program inference V3.

The replay consumes only the frozen public inputs and a finalized artifact. It
validates every external seal, acquires every public mode bank before creating
an exact-support task, deterministically regenerates the complete reference and
Monte Carlo ledger, and writes one exclusive receipt.  No secret or reveal path
is accepted by this module or its command-line interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from research.calibrated_program_inference_v2_screen import _reference_record
from research.calibrated_program_inference_v3_fresh import (
    ARM_PARTICLES,
    EXPECTED_INITIAL_PROPOSAL_DRAWS,
    EXPECTED_RUN_COUNT,
    FROZEN_ARMS,
    REFERENCE_SCHEMA,
    REPETITION_SEEDS,
    STUDY_SCHEMA,
    TASK_IDS,
    _validate_custody_seal,
    _validate_method_seal,
    _validate_protocol,
    _validate_suite,
    build_v3_proposal,
    factorized_public_mode_bank,
    materialize_mode_bank,
    run_v3_repetition,
)
from research.calibrated_program_inference_v3_validation import (
    analyze,
    canonical_bytes,
    validate_artifact,
    validate_inventory,
)
from research.particle_calibration_terminal_diagnostic_v2 import (
    TaskBinding,
    TerminalTask,
)

REPLAY_SCHEMA = f"{STUDY_SCHEMA}-deterministic-replay-receipt-v1"


class V3ReplayError(ValueError):
    """A reveal-free replay invariant was violated."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _require_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V3ReplayError(f"{name} is not a lowercase SHA-256 digest")
    return value


def _require_external_file(path: Path, expected_sha256: str, *, name: str) -> None:
    expected = _require_digest(expected_sha256, name=f"expected {name} SHA-256")
    if path.is_symlink() or not path.is_file():
        raise V3ReplayError(f"{name} is not a regular non-symlink file")
    if _sha256_file(path) != expected:
        raise V3ReplayError(f"external {name} SHA-256 differs")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V3ReplayError(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise V3ReplayError(f"JSON contains non-finite constant: {value}")


def _read_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise V3ReplayError(f"replay input is not a regular file: {path}")
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise V3ReplayError(f"replay JSON is not UTF-8: {path}") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise V3ReplayError(f"replay input is not strict JSON: {path}") from error


def _require_object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise V3ReplayError(f"{name} is not a string-keyed JSON object")
    return cast(dict[str, Any], value)


def _require_public_only_suite(suite: Path) -> None:
    """Reject custody/private material before invoking any suite reader."""

    if suite.is_symlink() or not suite.is_dir():
        raise V3ReplayError("replay suite root is invalid")
    entries = list(suite.iterdir())
    if (
        len(entries) != 1
        or entries[0].name != "public"
        or entries[0].is_symlink()
        or not entries[0].is_dir()
    ):
        raise V3ReplayError("replay requires a public-only suite with no private data")


def _native_reference(
    task: TerminalTask,
    proposals: Mapping[str, object],
    mode_metadata: Mapping[str, object],
) -> dict[str, object]:
    inherited = _reference_record(task, proposals, mode_metadata)
    reference = {
        **inherited,
        "schema": REFERENCE_SCHEMA,
        "provider_calls": 0,
        "mode_bank": inherited["sampled_mode_bank"],
    }
    del reference["sampled_mode_bank"]
    return reference


def reconstruct_public_ledger(
    bindings: Sequence[TaskBinding],
) -> tuple[dict[str, Mapping[str, object]], list[Mapping[str, object]]]:
    """Reconstruct the frozen V3 ledger using only public task bindings.

    The dictionary comprehension is a deliberate phase barrier: every public
    factorized bank must exist before the first ``TerminalTask`` is created.
    """

    binding_ids = tuple(binding.task_id for binding in bindings)
    if binding_ids != tuple(TASK_IDS):
        raise V3ReplayError("replay binding order differs from the frozen task order")
    public_banks = {
        binding.task_id: factorized_public_mode_bank(binding)
        for binding in bindings
    }
    if tuple(public_banks) != tuple(TASK_IDS):
        raise V3ReplayError("replay public-bank acquisition is incomplete")

    references: dict[str, Mapping[str, object]] = {}
    runs: list[Mapping[str, object]] = []
    for binding in bindings:
        task = TerminalTask(binding)
        bank = materialize_mode_bank(task, public_banks[binding.task_id])
        proposals = {
            arm.arm_id: build_v3_proposal(task, arm, bank) for arm in FROZEN_ARMS
        }
        references[binding.task_id] = _native_reference(
            task,
            proposals,
            bank.metadata,
        )
        for arm in FROZEN_ARMS:
            for particles in ARM_PARTICLES[arm.arm_id]:
                for repetition, base_seed in enumerate(REPETITION_SEEDS):
                    runs.append(
                        run_v3_repetition(
                            task,
                            proposals[arm.arm_id],
                            particles=particles,
                            repetition=repetition,
                            base_seed=base_seed,
                        )
                    )
    if len(runs) != EXPECTED_RUN_COUNT:
        raise V3ReplayError(
            f"replay produced {len(runs)} runs, expected {EXPECTED_RUN_COUNT}"
        )
    proposal_draws = sum(int(run["logical_proposal_draws"]) for run in runs)
    if proposal_draws != EXPECTED_INITIAL_PROPOSAL_DRAWS:
        raise V3ReplayError("replay logical proposal-draw count differs")
    if any(run.get("provider_calls") != 0 for run in runs):
        raise V3ReplayError("replay ledger is not provider-free")
    return references, runs


def _compare_json_bytes(path: Path, expected: object, *, name: str) -> None:
    actual = _read_json(path)
    if actual != expected:
        raise V3ReplayError(f"artifact {name} structure differs from replay")
    if path.read_bytes() != canonical_bytes(expected) + b"\n":
        raise V3ReplayError(f"artifact {name} bytes differ from replay")


def _reference_hash_manifest(
    artifact: Path,
    references: Mapping[str, Mapping[str, object]],
) -> str:
    records = [
        {
            "task_id": task_id,
            "sha256": _sha256_file(artifact / "references" / f"{task_id}.json"),
        }
        for task_id in references
    ]
    return _sha256_bytes(canonical_bytes(records))


def _write_receipt_exclusive(path: Path, receipt: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(receipt) + b"\n")


def replay_calibration(
    project_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
    custody_seal_path: Path,
    expected_custody_seal_sha256: str,
    suite: Path,
    artifact: Path,
    receipt_output: Path,
) -> dict[str, object]:
    """Perform a full reveal-free deterministic replay and issue its receipt."""

    if receipt_output.exists() or receipt_output.is_symlink():
        raise FileExistsError(f"refusing to overwrite replay receipt: {receipt_output}")
    project_root = project_root.resolve()
    protocol_path = protocol_path.absolute()
    method_seal_path = method_seal_path.absolute()
    custody_seal_path = custody_seal_path.absolute()
    suite = suite.absolute()
    artifact = artifact.absolute()

    _require_external_file(
        protocol_path,
        expected_protocol_sha256,
        name="protocol",
    )
    _require_external_file(
        method_seal_path,
        expected_method_seal_sha256,
        name="method seal",
    )
    _require_external_file(
        custody_seal_path,
        expected_custody_seal_sha256,
        name="custody seal",
    )
    _require_public_only_suite(suite)

    _validate_protocol(project_root, protocol_path, expected_protocol_sha256)
    _validate_method_seal(
        project_root,
        protocol_path,
        expected_protocol_sha256,
        method_seal_path,
        expected_method_seal_sha256,
    )
    custody = _validate_custody_seal(
        custody_seal_path,
        expected_custody_seal_sha256,
        protocol_sha256=expected_protocol_sha256,
        method_seal_sha256=expected_method_seal_sha256,
    )
    bindings, manifest = _validate_suite(
        suite,
        expected_protocol_sha256,
        expected_method_seal_sha256,
        expected_custody_seal_sha256,
        require_public_only=True,
    )
    for field in (
        "secret_commitment_sha256",
        "secret_commitment_under_v2_domain_sha256",
    ):
        if manifest.get(field) != custody.get(field):
            raise V3ReplayError(f"public suite {field} differs from custody")

    validated_analysis = validate_artifact(artifact)
    inventory = validate_inventory(artifact)
    artifact_protocol = artifact / "protocol.json"
    artifact_manifest = artifact / "public-suite-manifest.json"
    suite_manifest = suite / "public" / "manifest.json"
    if artifact_protocol.read_bytes() != protocol_path.read_bytes():
        raise V3ReplayError("artifact protocol bytes differ from the frozen protocol")
    if artifact_manifest.read_bytes() != suite_manifest.read_bytes():
        raise V3ReplayError("artifact public manifest bytes differ from the public suite")
    if manifest.get("protocol_sha256") != expected_protocol_sha256:
        raise V3ReplayError("public manifest protocol binding differs")
    if manifest.get("method_seal_sha256") != expected_method_seal_sha256:
        raise V3ReplayError("public manifest method-seal binding differs")
    if manifest.get("custody_seal_sha256") != expected_custody_seal_sha256:
        raise V3ReplayError("public manifest custody-seal binding differs")

    references, runs = reconstruct_public_ledger(bindings)
    for task_id, reference in references.items():
        _compare_json_bytes(
            artifact / "references" / f"{task_id}.json",
            reference,
            name=f"reference {task_id}",
        )
    _compare_json_bytes(artifact / "runs.json", runs, name="runs")
    replay_analysis = analyze(runs, references)
    _compare_json_bytes(artifact / "analysis.json", replay_analysis, name="analysis")
    if validated_analysis != replay_analysis:
        raise V3ReplayError("validated artifact analysis differs from replay analysis")

    entries = inventory.get("entries")
    if not isinstance(entries, list):
        raise V3ReplayError("validated artifact inventory entries are malformed")
    inventory_paths = {
        record.get("path")
        for record in entries
        if isinstance(record, Mapping)
    }
    required_inventory_paths = {
        "protocol.json",
        "public-suite-manifest.json",
        "runs.json",
        "analysis.json",
        *(f"references/{task_id}.json" for task_id in TASK_IDS),
    }
    if not required_inventory_paths.issubset(inventory_paths):
        raise V3ReplayError("artifact inventory omits replay-bound files")

    receipt = {
        "schema": REPLAY_SCHEMA,
        "status": "deterministic-public-replay-passed",
        "bindings": {
            "protocol_sha256": expected_protocol_sha256,
            "method_seal_sha256": expected_method_seal_sha256,
            "custody_seal_sha256": expected_custody_seal_sha256,
            "public_manifest_sha256": _sha256_file(suite_manifest),
            "runs_sha256": _sha256_file(artifact / "runs.json"),
            "analysis_sha256": _sha256_file(artifact / "analysis.json"),
            "inventory_sha256": _sha256_file(artifact / "inventory.json"),
            "reference_set_sha256": _reference_hash_manifest(artifact, references),
        },
        "replay_counts": {
            "tasks": len(bindings),
            "public_banks": len(bindings),
            "references": len(references),
            "runs": len(runs),
            "logical_proposal_draws": sum(
                int(run["logical_proposal_draws"]) for run in runs
            ),
        },
        "provider_calls": 0,
        "private_reveal_read": False,
        "all_public_banks_acquired_before_reference_materialization": True,
        "artifact_references_runs_analysis_byte_identical": True,
    }
    _write_receipt_exclusive(receipt_output, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--method-seal", type=Path, required=True)
    parser.add_argument("--expected-method-seal-sha256", required=True)
    parser.add_argument("--custody-seal", type=Path, required=True)
    parser.add_argument("--expected-custody-seal-sha256", required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = replay_calibration(
        args.project_root,
        args.protocol,
        args.expected_protocol_sha256,
        args.method_seal,
        args.expected_method_seal_sha256,
        args.custody_seal,
        args.expected_custody_seal_sha256,
        args.suite,
        args.artifact,
        args.receipt_output,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

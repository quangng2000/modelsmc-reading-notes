#!/usr/bin/env python3
"""Adjudicate and locally normalize the failed ExeDec V2 evidence tar.

This is a packaging-only, fail-closed remediation.  It never changes the
original transfer, never invokes a remote helper, and never imports an
artifact.  Its only permitted transformation is to copy the already sealed
member payloads into a separately named deterministic tar whose file-header
modes are 0600/0700.  A distinct report and seal record that the original
transfer failed its frozen verifier and that the derivative remains
noncanonical pending an independent audit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Mapping, Sequence


STUDY_NAME = "exedec-deepcoder-ho-debug-benchmark-v2"
EVIDENCE_NAME = f"{STUDY_NAME}-operations-evidence"
ATTEMPT_NAME = "exedec-v2-attempt-001"
STUDY_PROTOCOL_SHA256 = (
    "305cd2d89fbe0918a52b8e0ea6b3c4dfce23ad04ae669c48bcc3fcaa37c15b7d"
)
ORIGINAL_MANIFEST_SHA256 = (
    "a97c4bc6211133d916d8001a195913052dc592fff5df8e53d346622febb614e8"
)
ORIGINAL_METHOD_SEAL_SHA256 = (
    "9dcd5313dfb259b15ff953b5a836ef985597ca217e6d1d95e67308e162f4bc87"
)
ORIGINAL_RECORDS_SHA256 = (
    "ef73f2cf3f79253c9f27a55f614f01d760b3b44284d744325a8502230d40261c"
)
ORIGINAL_MODE_RECORDS_SHA256 = (
    "152d26d0089211a0e418bdd6d5ec390601f2286703af9ae8386763326a462d0c"
)
NORMALIZED_MODE_RECORDS_SHA256 = (
    "4c0e5930862c0a75b68f523eaaa23857a628e28377c5977fc243d1582ed35173"
)
MEMBER_PAYLOAD_RECORDS_SHA256 = (
    "2dc4e3c3fd9df01c2d312d0813af19e23e4183a8ac11be6d5c60d12fc108bf2f"
)
EXPECTED_ORIGINAL_VERIFIER_FAILURE = (
    "operations evidence archive safe mode differs: "
    "files/operations/bin/exedec_v2_driver.py"
)
EXPECTED_DERIVATIVE_WITH_ORIGINAL_SEAL_FAILURE = (
    "operations evidence seal mismatch: evidence_archive_sha256"
)

EVIDENCE_ARCHIVE_FILE = f"{EVIDENCE_NAME}.tar"
EVIDENCE_SEAL_FILE = f"{EVIDENCE_NAME}.seal.json"
STUDY_ARCHIVE_FILE = f"{STUDY_NAME}.tar"
EXPORT_INVENTORY_FILE = "export-inventory.json"
POSTRUN_RECEIPT_FILE = "postrun-receipt.json"

DERIVATIVE_STEM = f"{EVIDENCE_NAME}.mode-header-normalized-derivative-v1"
DERIVATIVE_ARCHIVE_FILE = f"{DERIVATIVE_STEM}.tar"
REMEDIATION_REPORT_FILE = f"{DERIVATIVE_STEM}.remediation-report.json"
REMEDIATION_SEAL_FILE = f"{DERIVATIVE_STEM}.remediation-seal.json"

ORIGINAL_TRANSFER_BINDINGS: dict[str, tuple[int, str]] = {
    EVIDENCE_SEAL_FILE: (
        964,
        "72cc4bb3998380eb12f78e15b7e17f04b6b676adf059b5efe6521ef804150ebe",
    ),
    EVIDENCE_ARCHIVE_FILE: (
        931840,
        "aa5b9fa4a8b60ab318030cfb3ff98882de5889b5200a96ffb1e6ca85a8d96b0a",
    ),
    STUDY_ARCHIVE_FILE: (
        12738560,
        "5e97831610852264f686d0f37d6f9c4aef8d783e45c29e6cc3fd5023dbc7e46c",
    ),
    EXPORT_INVENTORY_FILE: (
        197872,
        "b7c400f6c5f076a383959f0dc2551aa057437cd413b177d49adbc3e0966780a8",
    ),
    POSTRUN_RECEIPT_FILE: (
        3237,
        "abf929c4590535241f8bc77488886cf0893b67ef34d6e97889c82a250e1a5839",
    ),
}

FROZEN_OPERATIONS_BINDINGS: dict[str, tuple[int, str]] = {
    "OPERATIONS_EVIDENCE_RUNBOOK.md": (
        12597,
        "badfcf80358f8332c2b8dc16b62eb08dfc57208d846b391d60dadbfcf885129d",
    ),
    "OPERATIONS_PLAN.md": (
        12306,
        "775f7be7dee92f1a2f5d38fa8216cfa8a262e58be43b5386ccadaf73ae163f22",
    ),
    "exedec_v2_driver.py": (
        20475,
        "67bfc164911578d053b8d7a69e685dd8cf5c9174356a00fd657931a015c547f4",
    ),
    "exedec_v2_monitor.py": (
        5375,
        "4c5f1b704439af51bde21f53968c52f0488ce4d1915ab6adfbc7ccd7e0518f94",
    ),
    "exedec_v2_operations_evidence_common.py": (
        41264,
        "bb3e48ca54fbee66ded9edde924bb8c0f8e1c2d5de4dbd355dcbcb581096b371",
    ),
    "exedec_v2_operations_evidence_method_seal.json": (
        851,
        ORIGINAL_METHOD_SEAL_SHA256,
    ),
    "exedec_v2_operations_evidence_seal.py": (
        8284,
        "673329244901ce7dded5e7dda15420821a63aa2d7eb6f115464bb01a48dd7629",
    ),
    "exedec_v2_postrun.py": (
        18972,
        "c5c1b1a1765e6bc930a9c70caa2d1c09b59a1df8196f80da2e9e0d8709bb46db",
    ),
    "test_exedec_v2_operations_evidence.py": (
        20361,
        "a817311fd1b940ce1ed6547a4d9b098eebb8a49a680011c1fffdb14d73009541",
    ),
    "verify_exedec_v2_operations_evidence.py": (
        14034,
        "936c88f7bc365705d59087b8d02fd77796960176a6932a3184abffde563665f2",
    ),
    "verify_exedec_v2_transfer.py": (
        9004,
        "fdc4ab4523dcfac4da10787ce242898744911b31dee947b580f7e07ddb6bd1e8",
    ),
}

EVIDENCE_HELPER_RELATIVES = {
    "exedec_v2_driver.py": "operations/bin/exedec_v2_driver.py",
    "exedec_v2_monitor.py": "operations/bin/exedec_v2_monitor.py",
    "exedec_v2_operations_evidence_common.py": (
        "operations/bin/exedec_v2_operations_evidence_common.py"
    ),
    "exedec_v2_operations_evidence_method_seal.json": (
        "operations/bin/exedec_v2_operations_evidence_method_seal.json"
    ),
    "exedec_v2_operations_evidence_seal.py": (
        "operations/bin/exedec_v2_operations_evidence_seal.py"
    ),
    "exedec_v2_postrun.py": "operations/bin/exedec_v2_postrun.py",
    "verify_exedec_v2_operations_evidence.py": (
        "operations/bin/verify_exedec_v2_operations_evidence.py"
    ),
    "verify_exedec_v2_transfer.py": "operations/bin/verify_exedec_v2_transfer.py",
}

PROTOCOL_FILE_NAME = (
    "protocol-exedec-v2-operations-evidence-fuse-remediation-v1.json"
)
METHOD_SEAL_FILE_NAME = (
    "protocol-exedec-v2-operations-evidence-fuse-remediation-v1.method-seal.json"
)
TEST_FILE_NAME = "test_exedec_v2_operations_evidence_fuse_remediation.py"


@dataclass(frozen=True)
class TarEntry:
    """One verified regular-file member, relative to the evidence prefix."""

    relative: str
    full_name: str
    value: bytes
    mode: int


@dataclass(frozen=True)
class AuditResult:
    """Read-only validation result used to produce the normalized derivative."""

    entries: tuple[TarEntry, ...]
    manifest: dict[str, Any]
    seal: dict[str, Any]
    payload: dict[str, bytes]
    linkage: dict[str, object]
    original_verifier_failure: str


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json_object(value: bytes, *, name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def _require_regular_file(path: Path, *, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} is absent, non-regular, or a symlink: {path}")


def _validate_bound_files(
    root: Path,
    bindings: Mapping[str, tuple[int, str]],
    *,
    name: str,
) -> dict[str, dict[str, object]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{name} directory is absent or unsafe: {root}")
    records: dict[str, dict[str, object]] = {}
    for relative in sorted(bindings, key=lambda item: item.encode("utf-8")):
        expected_bytes, expected_sha256 = bindings[relative]
        path = root / relative
        _require_regular_file(path, name=f"{name} file")
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
            raise ValueError(f"{name} binding mismatch: {relative}")
        records[relative] = {
            "bytes": actual_bytes,
            "sha256": actual_sha256,
        }
    return records


def _reject_unsafe_relative(relative: str, *, name: str) -> PurePosixPath:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or relative in {"", "."}
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"{name} is unsafe: {relative!r}")
    return pure


def read_regular_tar(path: Path, *, prefix: PurePosixPath) -> tuple[TarEntry, ...]:
    """Read a strict, deterministic regular-file tar without extracting it."""

    _require_regular_file(path, name="tar input")
    entries: list[TarEntry] = []
    with tarfile.open(path, mode="r:") as archive:
        if archive.pax_headers:
            raise ValueError("tar has unexpected global PAX headers")
        members = archive.getmembers()
        if not members:
            raise ValueError("tar is empty")
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("tar contains duplicate member names")
        if names != sorted(names, key=lambda item: item.encode("utf-8")):
            raise ValueError("tar member names are not canonically sorted")
        for member in members:
            if member.type != tarfile.REGTYPE or not member.isfile():
                raise ValueError(f"tar member is not a regular file: {member.name}")
            if member.linkname:
                raise ValueError(f"tar member has a link target: {member.name}")
            if (
                member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mtime != 0
            ):
                raise ValueError(f"tar member metadata is not deterministic: {member.name}")
            if set(member.pax_headers) - {"path"}:
                raise ValueError(f"tar member has unexpected PAX headers: {member.name}")
            if member.pax_headers.get("path", member.name) != member.name:
                raise ValueError(f"tar member PAX path differs: {member.name}")
            pure = PurePosixPath(member.name)
            try:
                relative = pure.relative_to(prefix).as_posix()
            except ValueError as error:
                raise ValueError(f"tar member escaped its exact prefix: {member.name}") from error
            _reject_unsafe_relative(relative, name="tar member relative path")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"tar member cannot be read: {member.name}")
            value = source.read()
            if len(value) != member.size:
                raise ValueError(f"tar member size differs: {member.name}")
            entries.append(
                TarEntry(
                    relative=relative,
                    full_name=member.name,
                    value=value,
                    mode=stat.S_IMODE(member.mode),
                )
            )
    return tuple(entries)


def _mode_records(entries: Sequence[TarEntry]) -> list[dict[str, object]]:
    return [{"mode": entry.mode, "path": entry.relative} for entry in entries]


def _payload_records(entries: Sequence[TarEntry]) -> list[dict[str, object]]:
    return [
        {
            "bytes": len(entry.value),
            "path": entry.relative,
            "sha256": sha256_bytes(entry.value),
        }
        for entry in entries
    ]


def _protocol_expected_document() -> dict[str, object]:
    return {
        "authorization": {
            "canonical_import": False,
            "paper_edit": False,
            "remote_action": False,
            "rerun_or_reseal_original": False,
            "single_local_derivative": True,
        },
        "classification": (
            "local packaging-only adjudication of a shared-FUSE tar-header mode defect; "
            "scientific payload and non-exclusive-custody limits remain unchanged"
        ),
        "expected_original_failure": {
            "exception": "ValueError",
            "message": EXPECTED_ORIGINAL_VERIFIER_FAILURE,
            "verifier_sha256": FROZEN_OPERATIONS_BINDINGS[
                "verify_exedec_v2_operations_evidence.py"
            ][1],
        },
        "frozen_operations_files": {
            key: {"bytes": value[0], "sha256": value[1]}
            for key, value in FROZEN_OPERATIONS_BINDINGS.items()
        },
        "integrity_requirements": {
            "content_hashes": 149,
            "member_payload_records_sha256": MEMBER_PAYLOAD_RECORDS_SHA256,
            "original_archive_members": 150,
            "original_archive_mode_counts": {"0666": 146, "0777": 4},
            "original_mode_records_sha256": ORIGINAL_MODE_RECORDS_SHA256,
            "original_record_mode_counts": {"0444": 1, "0555": 4, "0666": 144},
            "paired_blocks": 32,
            "per_run_logs": 128,
            "provider_calls": 188,
            "runs": 64,
        },
        "normalization": {
            "archive_format": "uncompressed deterministic POSIX PAX tar",
            "derivative_archive_file": DERIVATIVE_ARCHIVE_FILE,
            "derivative_with_original_seal_must_fail": (
                EXPECTED_DERIVATIVE_WITH_ORIGINAL_SEAL_FAILURE
            ),
            "executable_file_header_mode": "0700",
            "nonexecutable_file_header_mode": "0600",
            "normalized_mode_counts": {"0600": 146, "0700": 4},
            "normalized_mode_records_sha256": NORMALIZED_MODE_RECORDS_SHA256,
            "only_changed_semantic_header_field": "mode",
            "payload_bytes_changed": False,
            "remediation_report_file": REMEDIATION_REPORT_FILE,
            "remediation_seal_file": REMEDIATION_SEAL_FILE,
        },
        "original_evidence_bindings": {
            "manifest_sha256": ORIGINAL_MANIFEST_SHA256,
            "method_seal_sha256": ORIGINAL_METHOD_SEAL_SHA256,
            "records_sha256": ORIGINAL_RECORDS_SHA256,
            "study_protocol_sha256": STUDY_PROTOCOL_SHA256,
        },
        "original_transfer_files": {
            key: {"bytes": value[0], "sha256": value[1]}
            for key, value in ORIGINAL_TRANSFER_BINDINGS.items()
        },
        "schema": "exedec-v2-operations-evidence-fuse-remediation-protocol-v1",
        "status": "frozen-before-local-remediation",
    }


def validate_protocol(path: Path) -> tuple[dict[str, Any], str]:
    _require_regular_file(path, name="remediation protocol")
    value = path.read_bytes()
    protocol = read_json_object(value, name="remediation protocol")
    if protocol != _protocol_expected_document():
        raise ValueError("remediation protocol differs from the exact approved document")
    if value != canonical_bytes(protocol) + b"\n":
        raise ValueError("remediation protocol is not canonical JSON plus one newline")
    return protocol, sha256_bytes(value)


def validate_method_seal(
    *,
    path: Path,
    expected_sha256: str,
    protocol_path: Path,
) -> dict[str, Any]:
    _require_regular_file(path, name="remediation method seal")
    if sha256_file(path) != expected_sha256:
        raise ValueError("remediation method-seal SHA-256 differs from the approved hash")
    seal_bytes = path.read_bytes()
    seal = read_json_object(seal_bytes, name="remediation method seal")
    if seal_bytes != canonical_bytes(seal) + b"\n":
        raise ValueError("remediation method seal is not canonical JSON plus one newline")
    if set(seal) != {
        "adversarial_tests",
        "classification",
        "implementation",
        "protocol",
        "schema",
        "status",
    }:
        raise ValueError("remediation method seal has unexpected fields")
    if seal.get("schema") != "exedec-v2-operations-evidence-fuse-remediation-method-seal-v1":
        raise ValueError("remediation method-seal schema differs")
    if seal.get("status") != "frozen-before-local-remediation":
        raise ValueError("remediation method seal was not frozen before use")
    if seal.get("classification") != (
        "local deterministic packaging remediation only; no synthesis, analysis, "
        "remote action, original mutation, canonical import, or paper edit"
    ):
        raise ValueError("remediation method-seal classification differs")

    own_path = Path(__file__).resolve()
    test_path = own_path.parent / "tests" / TEST_FILE_NAME
    expected_records = {
        "protocol": (protocol_path.resolve(), PROTOCOL_FILE_NAME),
        "implementation": (own_path, own_path.name),
        "adversarial_tests": (test_path, f"tests/{TEST_FILE_NAME}"),
    }
    for key, (bound_path, expected_relative) in expected_records.items():
        record = seal.get(key)
        if not isinstance(record, dict) or set(record) != {"bytes", "path", "sha256"}:
            raise ValueError(f"remediation method-seal {key} record is malformed")
        _require_regular_file(bound_path, name=f"remediation method-seal {key}")
        if (
            record.get("path") != f"research/{expected_relative}"
            or record.get("bytes") != bound_path.stat().st_size
            or record.get("sha256") != sha256_file(bound_path)
        ):
            raise ValueError(f"remediation method-seal {key} binding differs")
    return seal


def _load_frozen_modules(frozen_root: Path) -> tuple[ModuleType, ModuleType]:
    """Load exact frozen validators without writing bytecode beside them."""

    common_path = frozen_root / "exedec_v2_operations_evidence_common.py"
    verifier_path = frozen_root / "verify_exedec_v2_operations_evidence.py"
    old_dont_write = sys.dont_write_bytecode
    common_name = "exedec_v2_operations_evidence_common"
    verifier_name = "_frozen_exedec_v2_operations_evidence_verifier"
    previous_common = sys.modules.get(common_name)
    previous_verifier = sys.modules.get(verifier_name)
    try:
        sys.dont_write_bytecode = True
        common_spec = importlib.util.spec_from_file_location(common_name, common_path)
        if common_spec is None or common_spec.loader is None:
            raise ValueError("cannot load the frozen evidence common helper")
        common = importlib.util.module_from_spec(common_spec)
        sys.modules[common_name] = common
        common_spec.loader.exec_module(common)

        verifier_spec = importlib.util.spec_from_file_location(verifier_name, verifier_path)
        if verifier_spec is None or verifier_spec.loader is None:
            raise ValueError("cannot load the frozen evidence verifier")
        verifier = importlib.util.module_from_spec(verifier_spec)
        sys.modules[verifier_name] = verifier
        verifier_spec.loader.exec_module(verifier)
        return common, verifier
    finally:
        sys.dont_write_bytecode = old_dont_write
        if previous_common is None:
            sys.modules.pop(common_name, None)
        else:
            sys.modules[common_name] = previous_common
        if previous_verifier is None:
            sys.modules.pop(verifier_name, None)
        else:
            sys.modules[verifier_name] = previous_verifier


def _expect_frozen_verifier_failure(
    verifier: ModuleType,
    *,
    evidence_archive: Path,
    evidence_seal: Path,
    study_archive: Path,
    expected_message: str,
) -> str:
    try:
        verifier.validate_transfer(
            evidence_archive=evidence_archive,
            evidence_seal=evidence_seal,
            study_archive=study_archive,
            expected_method_seal_sha256=ORIGINAL_METHOD_SEAL_SHA256,
        )
    except ValueError as error:
        if str(error) != expected_message:
            raise ValueError(
                "frozen verifier failed for an unexpected reason: " + str(error)
            ) from error
        return str(error)
    raise ValueError("frozen verifier unexpectedly accepted the bound archive")


def _validate_seal(
    seal: dict[str, Any],
    *,
    evidence_archive: Path,
    study_archive: Path,
) -> None:
    expected = {
        "schema": "exedec-v2-operations-evidence-transfer-seal-v1",
        "status": "operations-evidence-export-sealed",
        "protocol_sha256": STUDY_PROTOCOL_SHA256,
        "study_name": STUDY_NAME,
        "evidence_name": EVIDENCE_NAME,
        "evidence_archive_sha256": sha256_file(evidence_archive),
        "evidence_archive_bytes": evidence_archive.stat().st_size,
        "study_archive_sha256": sha256_file(study_archive),
        "study_archive_bytes": study_archive.stat().st_size,
        "manifest_sha256": ORIGINAL_MANIFEST_SHA256,
        "method_seal_sha256": ORIGINAL_METHOD_SEAL_SHA256,
        "postrun_receipt_sha256": ORIGINAL_TRANSFER_BINDINGS[POSTRUN_RECEIPT_FILE][1],
    }
    for key, expected_value in expected.items():
        if seal.get(key) != expected_value:
            raise ValueError(f"original evidence seal mismatch: {key}")
    if set(seal) != {
        "created_at",
        "driver_receipt_sha256",
        "evidence_archive_bytes",
        "evidence_archive_sha256",
        "evidence_name",
        "manifest_sha256",
        "method_seal_sha256",
        "postrun_receipt_sha256",
        "protocol_sha256",
        "schema",
        "status",
        "study_archive_bytes",
        "study_archive_sha256",
        "study_name",
    }:
        raise ValueError("original evidence seal has unexpected fields")


def _validate_manifest_and_payload(
    *,
    entries: Sequence[TarEntry],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    if len(entries) != 150:
        raise ValueError("original evidence archive does not contain exactly 150 members")
    by_relative = {entry.relative: entry for entry in entries}
    if len(by_relative) != len(entries):
        raise ValueError("original evidence archive has duplicate relative paths")
    manifest_entry = by_relative.get("MANIFEST.json")
    if manifest_entry is None:
        raise ValueError("original evidence manifest is absent")
    if sha256_bytes(manifest_entry.value) != ORIGINAL_MANIFEST_SHA256:
        raise ValueError("original evidence manifest SHA-256 differs")
    if manifest_entry.mode != 0o666:
        raise ValueError("original evidence manifest lacks the exact FUSE-coerced 0666 mode")
    manifest = read_json_object(manifest_entry.value, name="original evidence manifest")
    exact_manifest = {
        "schema": "exedec-v2-operations-evidence-manifest-v1",
        "status": "export-ready-consistency-evidence-sealed",
        "protocol_sha256": STUDY_PROTOCOL_SHA256,
        "study_name": STUDY_NAME,
        "evidence_name": EVIDENCE_NAME,
        "attempt_name": ATTEMPT_NAME,
        "file_count": 149,
        "records_sha256": ORIGINAL_RECORDS_SHA256,
    }
    for key, expected_value in exact_manifest.items():
        if manifest.get(key) != expected_value:
            raise ValueError(f"original evidence manifest mismatch: {key}")
    expected_manifest_keys = {
        "attempt_name",
        "classification",
        "created_at",
        "custody_limit",
        "evidence_name",
        "file_count",
        "linkage",
        "protocol_sha256",
        "records",
        "records_sha256",
        "schema",
        "source_stage",
        "status",
        "study_name",
        "terminal_process_check",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("original evidence manifest has unexpected fields")
    expected_classification = (
        "unbound consistency/hash evidence from non-exclusive shared-FUSE custody; "
        "not exclusive-custody evidence; exact allowlist excludes provider payloads, "
        "study contents, and private debug oracles"
    )
    if manifest.get("classification") != expected_classification:
        raise ValueError("original evidence custody classification differs")
    custody = manifest.get("custody_limit")
    if not isinstance(custody, dict) or custody.get("exclusive_custody") is not False:
        raise ValueError("original evidence lacks its non-exclusive custody limit")
    process_check = manifest.get("terminal_process_check")
    if (
        not isinstance(process_check, dict)
        or process_check.get("driver_pid_alive") is not False
        or process_check.get("postrun_pid_alive") is not False
    ):
        raise ValueError("original evidence terminal-process check differs")

    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 149:
        raise ValueError("original evidence manifest does not contain 149 records")
    if sha256_bytes(canonical_bytes(records)) != ORIGINAL_RECORDS_SHA256:
        raise ValueError("original evidence manifest-record digest differs")
    payload: dict[str, bytes] = {}
    previous = b""
    expected_archive_relatives = {"MANIFEST.json"}
    record_modes: Counter[int] = Counter()
    for index, raw in enumerate(records):
        if not isinstance(raw, dict) or set(raw) != {
            "bytes",
            "ctime_ns",
            "gid",
            "mode",
            "mtime_ns",
            "path",
            "sha256",
            "source_relative_path",
            "uid",
        }:
            raise ValueError(f"original manifest record {index} is malformed")
        relative = raw.get("source_relative_path")
        archive_relative = raw.get("path")
        if (
            not isinstance(relative, str)
            or not isinstance(archive_relative, str)
            or archive_relative != f"files/{relative}"
        ):
            raise ValueError(f"original manifest record {index} path differs")
        _reject_unsafe_relative(relative, name="original evidence source path")
        encoded = relative.encode("utf-8")
        if previous and encoded <= previous:
            raise ValueError("original evidence records are not canonically ordered")
        previous = encoded
        entry = by_relative.get(archive_relative)
        if entry is None:
            raise ValueError(f"original evidence member is absent: {archive_relative}")
        if raw.get("bytes") != len(entry.value) or raw.get("sha256") != sha256_bytes(
            entry.value
        ):
            raise ValueError(f"original evidence content hash differs: {archive_relative}")
        source_mode = raw.get("mode")
        if not isinstance(source_mode, int) or isinstance(source_mode, bool):
            raise ValueError(f"original evidence source mode is invalid: {archive_relative}")
        expected_coerced_mode = 0o777 if source_mode & 0o111 else 0o666
        if entry.mode != expected_coerced_mode:
            raise ValueError(
                f"original evidence FUSE-coerced mode differs: {archive_relative}"
            )
        record_modes[source_mode] += 1
        payload[relative] = entry.value
        expected_archive_relatives.add(archive_relative)
    if set(by_relative) != expected_archive_relatives:
        raise ValueError("original evidence archive contains an unmanifested member")
    if record_modes != Counter({0o666: 144, 0o555: 4, 0o444: 1}):
        raise ValueError("original evidence source-mode distribution differs")
    archive_modes = Counter(entry.mode for entry in entries)
    if archive_modes != Counter({0o666: 146, 0o777: 4}):
        raise ValueError("original evidence FUSE-mode distribution differs")
    if sha256_bytes(canonical_bytes(_mode_records(entries))) != ORIGINAL_MODE_RECORDS_SHA256:
        raise ValueError("original evidence exact mode-record digest differs")
    if sha256_bytes(canonical_bytes(_payload_records(entries))) != MEMBER_PAYLOAD_RECORDS_SHA256:
        raise ValueError("original evidence member-payload digest differs")
    return manifest, payload


def audit_original_transfer(
    *,
    transfer_root: Path,
    frozen_root: Path,
) -> AuditResult:
    """Validate all bound inputs and the original failure without writing."""

    _validate_bound_files(
        transfer_root,
        ORIGINAL_TRANSFER_BINDINGS,
        name="original five-file transfer",
    )
    _validate_bound_files(
        frozen_root,
        FROZEN_OPERATIONS_BINDINGS,
        name="frozen operations helper/runbook",
    )
    evidence_archive = transfer_root / EVIDENCE_ARCHIVE_FILE
    evidence_seal = transfer_root / EVIDENCE_SEAL_FILE
    study_archive = transfer_root / STUDY_ARCHIVE_FILE
    prefix = PurePosixPath("artifacts") / EVIDENCE_NAME
    entries = read_regular_tar(evidence_archive, prefix=prefix)
    manifest, payload = _validate_manifest_and_payload(entries=entries)

    seal = read_json_object(evidence_seal.read_bytes(), name="original evidence seal")
    _validate_seal(
        seal,
        evidence_archive=evidence_archive,
        study_archive=study_archive,
    )
    if (
        payload[f"operations/{ATTEMPT_NAME}/export-inventory.json"]
        != (transfer_root / EXPORT_INVENTORY_FILE).read_bytes()
    ):
        raise ValueError("external export inventory differs from the evidence member")
    if (
        payload[f"operations/{ATTEMPT_NAME}/postrun-receipt.json"]
        != (transfer_root / POSTRUN_RECEIPT_FILE).read_bytes()
    ):
        raise ValueError("external postrun receipt differs from the evidence member")

    for helper_name, evidence_relative in EVIDENCE_HELPER_RELATIVES.items():
        if payload[evidence_relative] != (frozen_root / helper_name).read_bytes():
            raise ValueError(f"evidence helper differs from frozen source: {helper_name}")

    common, verifier = _load_frozen_modules(frozen_root)
    source_stage = manifest.get("source_stage")
    if not isinstance(source_stage, str):
        raise ValueError("original evidence source stage is absent")
    linkage = common.validate_payload(
        payload,
        source_stage=source_stage,
        study_archive=study_archive,
        expected_method_seal_sha256=ORIGINAL_METHOD_SEAL_SHA256,
    )
    if manifest.get("linkage") != linkage:
        raise ValueError("original evidence manifest linkage differs after recomputation")
    if linkage.get("run_count") != 64:
        raise ValueError("original evidence does not validate exactly 64 runs")
    if linkage.get("paired_block_count") != 32:
        raise ValueError("original evidence does not validate exactly 32 paired blocks")
    if linkage.get("per_run_log_count") != 128:
        raise ValueError("original evidence does not validate exactly 128 per-run logs")
    if linkage.get("aggregate_provider_calls") != 188:
        raise ValueError("original evidence does not validate exactly 188 provider calls")
    if seal.get("driver_receipt_sha256") != linkage.get("driver_receipt_sha256"):
        raise ValueError("original evidence driver-receipt seal linkage differs")
    if seal.get("postrun_receipt_sha256") != linkage.get("postrun_receipt_sha256"):
        raise ValueError("original evidence postrun-receipt seal linkage differs")
    failure = _expect_frozen_verifier_failure(
        verifier,
        evidence_archive=evidence_archive,
        evidence_seal=evidence_seal,
        study_archive=study_archive,
        expected_message=EXPECTED_ORIGINAL_VERIFIER_FAILURE,
    )
    return AuditResult(
        entries=entries,
        manifest=manifest,
        seal=seal,
        payload=payload,
        linkage=linkage,
        original_verifier_failure=failure,
    )


def normalized_mode_for(entry: TarEntry, *, manifest: Mapping[str, Any]) -> int:
    if entry.relative == "MANIFEST.json":
        return 0o600
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("manifest records are absent during normalization")
    modes = {
        raw["path"]: raw["mode"]
        for raw in records
        if isinstance(raw, dict) and isinstance(raw.get("path"), str)
    }
    if entry.relative not in modes:
        raise ValueError(f"normalization has no manifest mode: {entry.relative}")
    source_mode = modes[entry.relative]
    if not isinstance(source_mode, int) or isinstance(source_mode, bool):
        raise ValueError(f"normalization source mode is invalid: {entry.relative}")
    return 0o700 if source_mode & 0o111 else 0o600


def write_normalized_tar(
    *,
    path: Path,
    entries: Sequence[TarEntry],
    manifest: Mapping[str, Any],
) -> None:
    """Write one deterministic derivative exclusively; never overwrite."""

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as raw:
            with tarfile.open(fileobj=raw, mode="w:", format=tarfile.PAX_FORMAT) as archive:
                for entry in entries:
                    info = tarfile.TarInfo(entry.full_name)
                    info.size = len(entry.value)
                    info.mode = normalized_mode_for(entry, manifest=manifest)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.type = tarfile.REGTYPE
                    info.linkname = ""
                    info.pax_headers = {}
                    archive.addfile(info, fileobj=_BytesReader(entry.value))
            raw.flush()
            os.fsync(raw.fileno())
    except BaseException:
        # Preserve the incomplete output for forensic review.
        raise
    os.chmod(path, 0o600)


class _BytesReader:
    """Minimal read-only stream accepted by tarfile.addfile."""

    def __init__(self, value: bytes) -> None:
        self._value = value
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._value) - self._offset
        start = self._offset
        stop = min(len(self._value), start + size)
        self._offset = stop
        return self._value[start:stop]


def validate_normalized_derivative(
    *,
    derivative: Path,
    original: AuditResult,
) -> tuple[TarEntry, ...]:
    prefix = PurePosixPath("artifacts") / EVIDENCE_NAME
    entries = read_regular_tar(derivative, prefix=prefix)
    if len(entries) != len(original.entries):
        raise ValueError("normalized derivative member count differs")
    for before, after in zip(original.entries, entries, strict=True):
        if (
            after.relative != before.relative
            or after.full_name != before.full_name
            or after.value != before.value
        ):
            raise ValueError(f"normalized derivative changed a member payload: {before.relative}")
        expected_mode = normalized_mode_for(before, manifest=original.manifest)
        if after.mode != expected_mode:
            raise ValueError(f"normalized derivative mode differs: {before.relative}")
    mode_counts = Counter(entry.mode for entry in entries)
    if mode_counts != Counter({0o600: 146, 0o700: 4}):
        raise ValueError("normalized derivative mode distribution differs")
    if sha256_bytes(canonical_bytes(_mode_records(entries))) != NORMALIZED_MODE_RECORDS_SHA256:
        raise ValueError("normalized derivative exact mode-record digest differs")
    if sha256_bytes(canonical_bytes(_payload_records(entries))) != MEMBER_PAYLOAD_RECORDS_SHA256:
        raise ValueError("normalized derivative member-payload digest differs")
    return entries


def _write_exclusive(path: Path, value: bytes, *, mode: int = 0o600) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, mode)


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
        return True
    except ValueError:
        return False


def _prepare_output_directory(
    output_dir: Path,
    *,
    transfer_root: Path,
    frozen_root: Path,
) -> Path:
    output = output_dir.absolute()
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("output parent is absent or unsafe")
    output_resolved = output.resolve(strict=False)
    transfer_resolved = transfer_root.resolve(strict=True)
    frozen_resolved = frozen_root.resolve(strict=True)
    if _is_within(output_resolved, transfer_resolved) or _is_within(
        output_resolved, frozen_resolved
    ):
        raise ValueError("output directory may not be inside either frozen input")
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing an existing remediation output directory")
    os.mkdir(output, 0o700)
    os.chmod(output, 0o700)
    return output


def remediate_once(
    *,
    transfer_root: Path,
    frozen_root: Path,
    protocol_path: Path,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
    output_dir: Path,
) -> dict[str, object]:
    """Execute the already frozen deterministic local remediation exactly once."""

    protocol, protocol_sha256 = validate_protocol(protocol_path)
    validate_method_seal(
        path=method_seal_path,
        expected_sha256=expected_method_seal_sha256,
        protocol_path=protocol_path,
    )
    before_transfer = _validate_bound_files(
        transfer_root,
        ORIGINAL_TRANSFER_BINDINGS,
        name="original five-file transfer",
    )
    before_frozen = _validate_bound_files(
        frozen_root,
        FROZEN_OPERATIONS_BINDINGS,
        name="frozen operations helper/runbook",
    )
    original = audit_original_transfer(
        transfer_root=transfer_root,
        frozen_root=frozen_root,
    )
    output = _prepare_output_directory(
        output_dir,
        transfer_root=transfer_root,
        frozen_root=frozen_root,
    )
    incomplete = output / f".{DERIVATIVE_ARCHIVE_FILE}.incomplete"
    derivative = output / DERIVATIVE_ARCHIVE_FILE
    write_normalized_tar(
        path=incomplete,
        entries=original.entries,
        manifest=original.manifest,
    )
    validate_normalized_derivative(derivative=incomplete, original=original)
    if derivative.exists() or derivative.is_symlink():
        raise FileExistsError("refusing an existing final derivative archive")
    os.rename(incomplete, derivative)

    common, verifier = _load_frozen_modules(frozen_root)
    del common
    derivative_original_seal_failure = _expect_frozen_verifier_failure(
        verifier,
        evidence_archive=derivative,
        evidence_seal=transfer_root / EVIDENCE_SEAL_FILE,
        study_archive=transfer_root / STUDY_ARCHIVE_FILE,
        expected_message=EXPECTED_DERIVATIVE_WITH_ORIGINAL_SEAL_FAILURE,
    )
    after_transfer = _validate_bound_files(
        transfer_root,
        ORIGINAL_TRANSFER_BINDINGS,
        name="original five-file transfer postcheck",
    )
    after_frozen = _validate_bound_files(
        frozen_root,
        FROZEN_OPERATIONS_BINDINGS,
        name="frozen operations helper/runbook postcheck",
    )
    if before_transfer != after_transfer or before_frozen != after_frozen:
        raise ValueError("a frozen input changed during local remediation")

    derivative_sha256 = sha256_file(derivative)
    derivative_bytes = derivative.stat().st_size
    linkage_sha256 = sha256_bytes(canonical_bytes(original.linkage))
    report: dict[str, object] = {
        "adjudication": {
            "failure_class": "FUSE-coerced-tar-header-mode-packaging-defect",
            "original_failure_preserved": True,
            "original_frozen_verifier_error": original.original_verifier_failure,
            "scientific_content_inconsistency_found": False,
        },
        "canonical_import_authorized": False,
        "classification": protocol["classification"],
        "content_validation": {
            "all_manifested_content_hashes_valid": True,
            "content_hash_count": 149,
            "linkage_sha256": linkage_sha256,
            "manifest_sha256": ORIGINAL_MANIFEST_SHA256,
            "member_payload_records_sha256": MEMBER_PAYLOAD_RECORDS_SHA256,
            "paired_blocks": 32,
            "per_run_logs": 128,
            "provider_calls": 188,
            "runs": 64,
            "shared_content_and_cross_bindings_valid": True,
            "study_archive_sha256": ORIGINAL_TRANSFER_BINDINGS[STUDY_ARCHIVE_FILE][1],
        },
        "derivative": {
            "archive_bytes": derivative_bytes,
            "archive_file": DERIVATIVE_ARCHIVE_FILE,
            "archive_sha256": derivative_sha256,
            "derivative_with_original_seal_failure": derivative_original_seal_failure,
            "member_count": 150,
            "mode_counts": {"0600": 146, "0700": 4},
            "mode_records_sha256": NORMALIZED_MODE_RECORDS_SHA256,
            "payload_bytes_changed": False,
            "tar_header_mode_changes": 150,
        },
        "frozen_method": {
            "method_seal_file": method_seal_path.name,
            "method_seal_sha256": expected_method_seal_sha256,
            "protocol_file": protocol_path.name,
            "protocol_sha256": protocol_sha256,
            "runbook_sha256": FROZEN_OPERATIONS_BINDINGS[
                "OPERATIONS_EVIDENCE_RUNBOOK.md"
            ][1],
        },
        "limits": [
            "The derivative repairs only tar header modes and does not create exclusive custody.",
            "The evidence remains debug-only, public-released-data, and contamination-limited.",
            "The derivative is noncanonical until a separate independent audit authorizes import.",
            "The original archive and original seal remain the authoritative failed transfer.",
        ],
        "original_fuse_modes": {
            "archive_mode_counts": {"0666": 146, "0777": 4},
            "archive_mode_records_sha256": ORIGINAL_MODE_RECORDS_SHA256,
            "manifest_record_mode_counts": {"0444": 1, "0555": 4, "0666": 144},
        },
        "original_transfer_files": before_transfer,
        "schema": "exedec-v2-operations-evidence-fuse-remediation-report-v1",
        "status": "local-normalized-derivative-validated-noncanonical",
    }
    report_bytes = canonical_bytes(report) + b"\n"
    report_path = output / REMEDIATION_REPORT_FILE
    _write_exclusive(report_path, report_bytes)
    remediation_seal: dict[str, object] = {
        "canonical_import_authorized": False,
        "classification": (
            "distinct local normalized derivative of the bound failed original; "
            "not the original operations-evidence export seal"
        ),
        "derivative_archive_bytes": derivative_bytes,
        "derivative_archive_file": DERIVATIVE_ARCHIVE_FILE,
        "derivative_archive_sha256": derivative_sha256,
        "member_payload_records_sha256": MEMBER_PAYLOAD_RECORDS_SHA256,
        "method_seal_sha256": expected_method_seal_sha256,
        "normalized_mode_records_sha256": NORMALIZED_MODE_RECORDS_SHA256,
        "original_evidence_archive_sha256": ORIGINAL_TRANSFER_BINDINGS[
            EVIDENCE_ARCHIVE_FILE
        ][1],
        "original_evidence_seal_sha256": ORIGINAL_TRANSFER_BINDINGS[EVIDENCE_SEAL_FILE][1],
        "original_failure": original.original_verifier_failure,
        "original_manifest_sha256": ORIGINAL_MANIFEST_SHA256,
        "original_study_archive_sha256": ORIGINAL_TRANSFER_BINDINGS[STUDY_ARCHIVE_FILE][1],
        "protocol_sha256": protocol_sha256,
        "remediation_report_sha256": sha256_bytes(report_bytes),
        "runbook_sha256": FROZEN_OPERATIONS_BINDINGS["OPERATIONS_EVIDENCE_RUNBOOK.md"][1],
        "schema": "exedec-v2-operations-evidence-fuse-remediation-seal-v1",
        "status": "local-normalized-derivative-sealed-noncanonical",
    }
    seal_path = output / REMEDIATION_SEAL_FILE
    seal_bytes = canonical_bytes(remediation_seal) + b"\n"
    _write_exclusive(seal_path, seal_bytes)
    return {
        "derivative_archive": derivative.as_posix(),
        "derivative_archive_sha256": derivative_sha256,
        "method_seal_sha256": expected_method_seal_sha256,
        "protocol_sha256": protocol_sha256,
        "remediation_report": report_path.as_posix(),
        "remediation_report_sha256": sha256_bytes(report_bytes),
        "remediation_seal": seal_path.as_posix(),
        "remediation_seal_sha256": sha256_bytes(seal_bytes),
        "status": "local-normalized-derivative-sealed-noncanonical",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-transfer", type=Path, required=True)
    parser.add_argument("--frozen-operations", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--method-seal", type=Path, required=True)
    parser.add_argument("--expected-method-seal-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = remediate_once(
        transfer_root=args.original_transfer.absolute(),
        frozen_root=args.frozen_operations.absolute(),
        protocol_path=args.protocol.absolute(),
        method_seal_path=args.method_seal.absolute(),
        expected_method_seal_sha256=args.expected_method_seal_sha256,
        output_dir=args.output_dir.absolute(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

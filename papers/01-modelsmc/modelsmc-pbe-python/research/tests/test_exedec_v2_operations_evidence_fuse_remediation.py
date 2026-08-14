from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from research import exedec_v2_operations_evidence_fuse_remediation as remediation


REAL_TRANSFER = Path("/private/tmp/exedec-v2-evidence-transfer.uWU96u")
FROZEN_OPERATIONS = Path("/private/tmp/exedec-v2-ops")
RESEARCH_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = RESEARCH_ROOT / remediation.PROTOCOL_FILE_NAME
METHOD_SEAL = RESEARCH_ROOT / remediation.METHOD_SEAL_FILE_NAME


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_raw_tar(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    with tarfile.open(path, mode="x:", format=tarfile.PAX_FORMAT) as archive:
        for info, value in members:
            archive.addfile(info, io.BytesIO(value) if info.isfile() else None)


def regular_info(name: str, value: bytes, *, mode: int = 0o666) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    return info


class FuseRemediationUnitTest(unittest.TestCase):
    def test_protocol_is_exact_canonical_and_disallows_expansive_actions(self) -> None:
        protocol, protocol_sha256 = remediation.validate_protocol(PROTOCOL)
        self.assertEqual(64, len(protocol_sha256))
        self.assertEqual(
            {
                "canonical_import": False,
                "paper_edit": False,
                "remote_action": False,
                "rerun_or_reseal_original": False,
                "single_local_derivative": True,
            },
            protocol["authorization"],
        )

    def test_normalized_tar_is_deterministic_and_changes_only_modes(self) -> None:
        prefix = "artifacts/example/"
        entries = (
            remediation.TarEntry(
                relative="MANIFEST.json",
                full_name=f"{prefix}MANIFEST.json",
                value=b"manifest\n",
                mode=0o666,
            ),
            remediation.TarEntry(
                relative="files/a.txt",
                full_name=f"{prefix}files/a.txt",
                value=b"alpha\n",
                mode=0o666,
            ),
            remediation.TarEntry(
                relative="files/run.py",
                full_name=f"{prefix}files/run.py",
                value=b"print(1)\n",
                mode=0o777,
            ),
        )
        manifest = {
            "records": [
                {"mode": 0o444, "path": "files/a.txt"},
                {"mode": 0o555, "path": "files/run.py"},
            ]
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "first.tar"
            second = root / "second.tar"
            remediation.write_normalized_tar(
                path=first,
                entries=entries,
                manifest=manifest,
            )
            remediation.write_normalized_tar(
                path=second,
                entries=entries,
                manifest=manifest,
            )
            self.assertEqual(digest(first), digest(second))
            normalized = remediation.read_regular_tar(
                first,
                prefix=PurePosixPath("artifacts/example"),
            )
            self.assertEqual([0o600, 0o600, 0o700], [item.mode for item in normalized])
            self.assertEqual(
                [item.value for item in entries],
                [item.value for item in normalized],
            )
            with self.assertRaises(FileExistsError):
                remediation.write_normalized_tar(
                    path=first,
                    entries=entries,
                    manifest=manifest,
                )

    def test_tar_reader_rejects_duplicate_member_names(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "duplicate.tar"
            name = "artifacts/example/MANIFEST.json"
            write_raw_tar(
                path,
                [
                    (regular_info(name, b"one"), b"one"),
                    (regular_info(name, b"two"), b"two"),
                ],
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                remediation.read_regular_tar(
                    path,
                    prefix=PurePosixPath("artifacts/example"),
                )

    def test_tar_reader_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "traversal.tar"
            name = "artifacts/example/../escape"
            write_raw_tar(path, [(regular_info(name, b"x"), b"x")])
            with self.assertRaisesRegex(ValueError, "unsafe"):
                remediation.read_regular_tar(
                    path,
                    prefix=PurePosixPath("artifacts/example"),
                )

    def test_tar_reader_rejects_links(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "link.tar"
            info = tarfile.TarInfo("artifacts/example/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "target"
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            write_raw_tar(path, [(info, b"")])
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                remediation.read_regular_tar(
                    path,
                    prefix=PurePosixPath("artifacts/example"),
                )

    def test_tar_reader_rejects_nondeterministic_owner_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "owner.tar"
            info = regular_info("artifacts/example/file", b"x")
            info.uid = 501
            write_raw_tar(path, [(info, b"x")])
            with self.assertRaisesRegex(ValueError, "metadata is not deterministic"):
                remediation.read_regular_tar(
                    path,
                    prefix=PurePosixPath("artifacts/example"),
                )

    def test_bound_file_validator_rejects_content_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "bound"
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                remediation._validate_bound_files(
                    root,
                    {"bound": (8, "0" * 64)},
                    name="adversarial fixture",
                )

    def test_bound_file_validator_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "target"
            target.write_bytes(b"bound")
            link = root / "link"
            os.symlink(target, link)
            with self.assertRaisesRegex(ValueError, "symlink"):
                remediation._validate_bound_files(
                    root,
                    {"link": (5, hashlib.sha256(b"bound").hexdigest())},
                    name="adversarial fixture",
                )

    def test_output_inside_original_transfer_is_rejected(self) -> None:
        if not REAL_TRANSFER.is_dir() or not FROZEN_OPERATIONS.is_dir():
            self.skipTest("bound local evidence inputs are unavailable")
        with self.assertRaisesRegex(ValueError, "inside either frozen input"):
            remediation._prepare_output_directory(
                REAL_TRANSFER / "forbidden-output",
                transfer_root=REAL_TRANSFER,
                frozen_root=FROZEN_OPERATIONS,
            )

    def test_existing_output_directory_is_rejected(self) -> None:
        if not REAL_TRANSFER.is_dir() or not FROZEN_OPERATIONS.is_dir():
            self.skipTest("bound local evidence inputs are unavailable")
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(FileExistsError, "existing"):
                remediation._prepare_output_directory(
                    Path(raw),
                    transfer_root=REAL_TRANSFER,
                    frozen_root=FROZEN_OPERATIONS,
                )


@unittest.skipUnless(
    REAL_TRANSFER.is_dir() and FROZEN_OPERATIONS.is_dir(),
    "bound local evidence inputs are unavailable",
)
class BoundEvidenceIntegrationTest(unittest.TestCase):
    def test_original_failure_and_every_content_cross_binding_validate_read_only(self) -> None:
        before_transfer = {
            name: digest(REAL_TRANSFER / name)
            for name in remediation.ORIGINAL_TRANSFER_BINDINGS
        }
        before_frozen = {
            name: digest(FROZEN_OPERATIONS / name)
            for name in remediation.FROZEN_OPERATIONS_BINDINGS
        }
        result = remediation.audit_original_transfer(
            transfer_root=REAL_TRANSFER,
            frozen_root=FROZEN_OPERATIONS,
        )
        self.assertEqual(150, len(result.entries))
        self.assertEqual(149, len(result.payload))
        self.assertEqual(64, result.linkage["run_count"])
        self.assertEqual(32, result.linkage["paired_block_count"])
        self.assertEqual(128, result.linkage["per_run_log_count"])
        self.assertEqual(188, result.linkage["aggregate_provider_calls"])
        self.assertEqual(
            remediation.EXPECTED_ORIGINAL_VERIFIER_FAILURE,
            result.original_verifier_failure,
        )
        self.assertEqual(
            before_transfer,
            {
                name: digest(REAL_TRANSFER / name)
                for name in remediation.ORIGINAL_TRANSFER_BINDINGS
            },
        )
        self.assertEqual(
            before_frozen,
            {
                name: digest(FROZEN_OPERATIONS / name)
                for name in remediation.FROZEN_OPERATIONS_BINDINGS
            },
        )

    def test_method_seal_binds_protocol_implementation_and_tests(self) -> None:
        if not METHOD_SEAL.is_file():
            self.skipTest("remediation method seal has not been frozen yet")
        seal_sha256 = digest(METHOD_SEAL)
        remediation.validate_method_seal(
            path=METHOD_SEAL,
            expected_sha256=seal_sha256,
            protocol_path=PROTOCOL,
        )
        with tempfile.TemporaryDirectory() as raw:
            tampered = Path(raw) / METHOD_SEAL.name
            shutil.copyfile(METHOD_SEAL, tampered)
            parsed = json.loads(tampered.read_text())
            parsed["status"] = "tampered"
            tampered.write_text(json.dumps(parsed, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "approved hash"):
                remediation.validate_method_seal(
                    path=tampered,
                    expected_sha256=seal_sha256,
                    protocol_path=PROTOCOL,
                )


if __name__ == "__main__":
    unittest.main()

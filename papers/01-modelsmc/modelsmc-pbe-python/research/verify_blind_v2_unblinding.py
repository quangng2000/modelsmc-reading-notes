"""Verify blind-v2 custody and targets after reveal-free analysis is sealed.

This tool treats the frozen study, provider seal, public suite, custody secret,
one-time-use receipt, and private reveal as immutable inputs.  It regenerates a
copy in a temporary directory and creates one exclusive verification report
only after the complete trust chain passes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

REPORT_SCHEMA = "blind-v2-post-unblind-verification-v1"
ANALYSIS_SCHEMA = "blind-evidence-frontier-confirmation-analysis-v2"
STUDY_SCHEMA = "blind-evidence-frontier-confirmation-v2"
PROVIDER_SEAL_SCHEMA = "blind-v2-provider-call-seal-v1"
SUITE_SCHEMA = "blinded-filter-map-suite-v2"
SECRET_RECEIPT_SCHEMA = "blinded-filter-map-secret-use-v1"
SECRET_COMMITMENT_DOMAIN = b"blind-evidence-frontier-v2\0"
TASK_IDS = tuple(f"blind-v2-{index:02d}" for index in range(1, 13))
PUBLIC_SEALER_PATH = "research/seal_blind_evidence_frontier_v2_public.py"
PUBLIC_SEALER_SHA256 = "5a19d1c01c3c1a55997531fd90c5b15464aa128f1da59698edc54791ec45078f"
_PUBLIC_SEAL_FILENAMES = frozenset({"SHA256SUMS", "BUNDLE_SHA256"})
_FORBIDDEN_PUBLIC_PATH_PARTS = frozenset(
    {"private", "reveal", "reveal.json", "secret", "secret.bin", "seed.bin"}
)
_PRIVATE_ANALYSIS_KEYS = frozenset(
    {
        "commitment_preimage",
        "mapper_family",
        "nonce",
        "predicate_shape",
        "predicate_support",
        "seed_hex",
        "structural_rejections_before_acceptance",
        "target",
        "target_mapper_dsl",
        "target_predicate_dsl",
    }
)


class _GenerateSuite(Protocol):
    def __call__(
        self,
        output: Path,
        *,
        seed_hex: str | None = None,
        seed_file: Path | None = None,
        suite_version: str = "v1",
    ) -> dict[str, object]: ...


class _VerifySuite(Protocol):
    def __call__(self, public_dir: Path, reveal_path: Path) -> dict[str, object]: ...


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def _array(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return cast(list[Any], value)


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _expect(actual: object, expected: object, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _reject_private_analysis(value: object, *, path: str = "analysis") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _PRIVATE_ANALYSIS_KEYS:
                raise ValueError(f"private key {key!r} appears at {path}")
            _reject_private_analysis(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_private_analysis(child, path=f"{path}[{index}]")


def _resolve_binding_path(base_dir: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _public_files(root: Path, *, label: str) -> tuple[Path, ...]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} must be a regular directory")
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        forbidden = {part.lower() for part in relative.parts}.intersection(
            _FORBIDDEN_PUBLIC_PATH_PARTS
        )
        if forbidden:
            raise ValueError(f"private/reveal path appears under {label}: {relative}")
        if path.is_symlink():
            raise ValueError(f"symbolic links are forbidden under {label}: {relative}")
        if path.is_file():
            if path.name in _PUBLIC_SEAL_FILENAMES:
                raise ValueError(f"nested seal file appears under {label}: {relative}")
            files.append(path)
        elif not path.is_dir():
            raise ValueError(f"non-regular artifact appears under {label}: {relative}")
    return tuple(sorted(files, key=lambda path: path.relative_to(root).as_posix()))


def _public_inventory_bytes(runs_root: Path, analysis_root: Path) -> bytes:
    records = [
        (f"runs/{path.relative_to(runs_root).as_posix()}", path)
        for path in _public_files(runs_root, label="runs")
    ]
    records.extend(
        (f"analysis/{path.relative_to(analysis_root).as_posix()}", path)
        for path in _public_files(analysis_root, label="analysis")
    )
    return "".join(
        f"{sha256_file(path)}  {relative}\n" for relative, path in sorted(records)
    ).encode()


def _validate_public_sealer(repo_root: Path) -> Path:
    sealer_path = (repo_root.resolve() / PUBLIC_SEALER_PATH).resolve()
    _expect(sealer_path.is_file(), True, name="finalized public-sealer presence")
    _expect(
        sha256_file(sealer_path),
        PUBLIC_SEALER_SHA256,
        name="finalized public-sealer SHA-256",
    )
    return sealer_path


def _validate_analysis_gate(
    *,
    repo_root: Path,
    analysis_seal_path: Path,
    expected_analysis_seal_sha256: str,
    runs_root: Path,
    analysis_root: Path,
    study_protocol_path: Path,
    provider_call_seal_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    _validate_public_sealer(repo_root)
    seal_root = analysis_seal_path.resolve()
    runs = runs_root.resolve()
    analysis_dir = analysis_root.resolve()
    if not seal_root.is_dir() or seal_root.is_symlink():
        raise ValueError("analysis seal must be a regular directory")
    _expect(
        {path.name for path in seal_root.iterdir()},
        _PUBLIC_SEAL_FILENAMES,
        name="analysis-seal file set",
    )
    for path in seal_root.iterdir():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"analysis seal contains a non-regular file: {path.name}")
    expected_bundle = _digest(
        expected_analysis_seal_sha256, name="expected public-bundle SHA-256"
    )
    sealed_inventory = (seal_root / "SHA256SUMS").read_bytes()
    _expect(
        sha256_bytes(sealed_inventory),
        expected_bundle,
        name="external public-bundle SHA-256",
    )
    _expect(
        (seal_root / "BUNDLE_SHA256").read_bytes(),
        f"{expected_bundle}\n".encode(),
        name="BUNDLE_SHA256",
    )
    if runs == analysis_dir or runs in analysis_dir.parents or analysis_dir in runs.parents:
        raise ValueError("runs and analysis roots must be separate and non-nested")
    current_inventory = _public_inventory_bytes(runs, analysis_dir)
    _expect(
        current_inventory, sealed_inventory, name="sealed reveal-free public artifact bytes"
    )
    analysis_files = {
        path.relative_to(analysis_dir).as_posix()
        for path in _public_files(analysis_dir, label="analysis")
    }
    _expect(analysis_files, {"SUMMARY.md", "analysis.json"}, name="analysis file set")
    analysis_path = analysis_dir / "analysis.json"
    study_sha256 = sha256_file(study_protocol_path.resolve())
    provider_sha256 = sha256_file(provider_call_seal_path.resolve())
    analysis = _read_object(analysis_path)
    _reject_private_analysis(analysis)
    _expect(analysis.get("schema"), ANALYSIS_SCHEMA, name="analysis schema")
    _expect(analysis.get("blind"), True, name="analysis blind flag")
    _expect(analysis.get("confirmatory"), True, name="analysis confirmatory flag")
    _expect(analysis.get("private_reveal_used"), False, name="analysis private_reveal_used")
    integrity = _object(analysis.get("integrity"), name="analysis.integrity")
    _expect(integrity.get("passed"), True, name="analysis integrity")
    bindings = _object(analysis.get("bindings"), name="analysis.bindings")
    _expect(bindings.get("study_protocol_sha256"), study_sha256, name="analysis study")
    _expect(
        bindings.get("provider_call_seal_sha256"),
        provider_sha256,
        name="analysis provider seal",
    )
    gate = {
        "scheme": "SHA256(exact SHA256SUMS bytes)",
        "bundle_sha256": expected_bundle,
        "private_reveal_used": False,
    }
    return analysis, gate, analysis_path


def _validate_public_chain(
    *,
    repo_root: Path,
    study_protocol_path: Path,
    provider_call_seal_path: Path,
    public_dir: Path,
    analysis: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    study_path = study_protocol_path.resolve()
    provider_path = provider_call_seal_path.resolve()
    study = _read_object(study_path)
    provider = _read_object(provider_path)
    manifest_path = (public_dir.resolve() / "manifest.json").resolve()
    manifest = _read_object(manifest_path)
    _expect(study.get("schema"), STUDY_SCHEMA, name="study schema")
    _expect(
        study.get("protocol_status"),
        "method-frozen-before-task-generation",
        name="study protocol status",
    )
    _expect(provider.get("schema"), PROVIDER_SEAL_SCHEMA, name="provider-seal schema")
    _expect(provider.get("sealed_before_provider_calls"), True, name="provider pre-call flag")
    _expect(
        provider.get("study_protocol_sha256"), sha256_file(study_path), name="provider study"
    )
    _expect(manifest.get("schema"), SUITE_SCHEMA, name="manifest schema")
    _expect(manifest.get("task_count"), 12, name="manifest task count")
    manifest_binding = _object(provider.get("public_manifest"), name="provider.public_manifest")
    bound_manifest = _resolve_binding_path(
        repo_root.resolve(),
        manifest_binding.get("path"),
        name="provider manifest path",
    )
    _expect(bound_manifest, manifest_path, name="provider manifest path")
    manifest_file_sha256 = sha256_file(manifest_path)
    _expect(
        manifest_binding.get("sha256"), manifest_file_sha256, name="provider manifest SHA-256"
    )
    bindings = _object(analysis.get("bindings"), name="analysis.bindings")
    _expect(
        bindings.get("public_manifest_sha256"),
        manifest_file_sha256,
        name="analysis manifest",
    )
    _expect(
        bindings.get("hidden_target_manifest_sha256"),
        provider.get("hidden_target_manifest_sha256"),
        name="analysis hidden reveal",
    )
    return study, provider, manifest, manifest_path


def _validate_secret_and_receipt(
    *,
    secret_file: Path,
    receipt_path: Path,
    suite_root: Path,
    study: Mapping[str, Any],
    provider: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[bytes, str, dict[str, Any]]:
    secret_path = secret_file.resolve()
    expected_receipt_path = secret_path.with_name(f"{secret_path.name}.used.json")
    _expect(receipt_path.resolve(), expected_receipt_path, name="one-time receipt path")
    if secret_file.is_symlink() or receipt_path.is_symlink():
        raise ValueError("custody secret and receipt must not be symbolic links")
    if stat.S_IMODE(secret_path.stat().st_mode) & 0o077:
        raise ValueError("custody secret must not be accessible by group or others")
    if stat.S_IMODE(expected_receipt_path.stat().st_mode) & 0o077:
        raise ValueError("one-time receipt must not be accessible by group or others")
    secret = secret_path.read_bytes()
    if len(secret) != 32:
        raise ValueError("custody secret must contain exactly 32 raw bytes")
    commitment = sha256_bytes(SECRET_COMMITMENT_DOMAIN + secret)
    freeze = _object(study.get("freeze_requirements"), name="study.freeze_requirements")
    for name, observed in (
        ("study", freeze.get("task_secret_commitment_sha256")),
        ("provider", provider.get("task_secret_commitment_sha256")),
        ("manifest", manifest.get("seed_commitment_sha256")),
    ):
        _expect(observed, commitment, name=f"{name} secret commitment")
    receipt = _read_object(expected_receipt_path)
    _expect(receipt.get("schema"), SECRET_RECEIPT_SCHEMA, name="receipt schema")
    _expect(receipt.get("suite_schema"), SUITE_SCHEMA, name="receipt suite schema")
    _expect(receipt.get("seed_commitment_sha256"), commitment, name="receipt commitment")
    output = receipt.get("output")
    if not isinstance(output, str):
        raise ValueError("receipt output must be an absolute suite path")
    _expect(Path(output).resolve(), suite_root.resolve(), name="receipt suite output")
    return secret, commitment, receipt


def _load_frozen_generator(
    *, repo_root: Path, study: Mapping[str, Any]
) -> tuple[_GenerateSuite, _VerifySuite, Path]:
    freeze = _object(study.get("freeze_requirements"), name="study.freeze_requirements")
    binding = _object(freeze.get("generator"), name="study.freeze.generator")
    relative = binding.get("path")
    is_safe = (
        isinstance(relative, str)
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts
    )
    if not is_safe:
        raise ValueError("frozen generator path must be repository-relative")
    assert isinstance(relative, str)
    generator_path = (repo_root.resolve() / relative).resolve()
    _expect(
        sha256_file(generator_path),
        _digest(binding.get("sha256"), name="frozen generator SHA-256"),
        name="frozen generator SHA-256",
    )
    module_name = "_blind_v2_frozen_generator_for_unblinding"
    specification = importlib.util.spec_from_file_location(module_name, generator_path)
    if specification is None or specification.loader is None:
        raise ValueError("could not load the frozen generator")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    loaded = cast(ModuleType, module)
    return (
        cast(_GenerateSuite, loaded.generate_suite),
        cast(_VerifySuite, loaded.verify_suite),
        generator_path,
    )


def _file_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _file_payloads(root: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symbolic link appears in public suite: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path.read_bytes()
        elif not path.is_dir():
            raise ValueError(f"non-regular entry appears in public suite: {path}")
    return result


def _validate_regeneration(
    *,
    repo_root: Path,
    public_dir: Path,
    reveal_path: Path,
    secret: bytes,
    study: Mapping[str, Any],
    provider: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, object], Path]:
    reveal_file = reveal_path.resolve()
    if reveal_path.is_symlink() or not reveal_file.is_file():
        raise ValueError("hidden reveal must be a regular non-symbolic file")
    reveal_sha256 = sha256_file(reveal_file)
    _expect(
        reveal_sha256,
        _digest(provider.get("hidden_target_manifest_sha256"), name="sealed reveal SHA-256"),
        name="hidden reveal SHA-256",
    )
    generate_suite, verify_suite, generator_path = _load_frozen_generator(
        repo_root=repo_root, study=study
    )
    with tempfile.TemporaryDirectory(prefix="blind-v2-unblind-verify-") as raw_temp:
        regenerated = Path(raw_temp) / "suite"
        generate_suite(regenerated, seed_hex=secret.hex(), suite_version="v2")
        expected_public = _file_payloads(regenerated / "public")
        observed_public = _file_payloads(public_dir.resolve())
        _expect(observed_public, expected_public, name="regenerated public-suite bytes")
        _expect(
            reveal_file.read_bytes(),
            (regenerated / "private" / "reveal.json").read_bytes(),
            name="regenerated hidden-reveal bytes",
        )
    verification = verify_suite(public_dir.resolve(), reveal_file)
    _expect(verification.get("schema"), SUITE_SCHEMA, name="generator verification schema")
    _expect(verification.get("verified"), list(TASK_IDS), name="verified target order")
    _expect(verification.get("exact"), True, name="target oracle verification")
    return _file_inventory(public_dir.resolve()), verification, generator_path


def verify_post_unblinding(
    *,
    repo_root: Path,
    study_protocol_path: Path,
    provider_call_seal_path: Path,
    analysis_seal_path: Path,
    expected_analysis_seal_sha256: str,
    runs_root: Path,
    analysis_root: Path,
    public_dir: Path,
    secret_file: Path,
    receipt_path: Path,
    reveal_path: Path,
) -> dict[str, object]:
    """Return a report only after the reveal-free gate and full unblind audit pass."""

    analysis, analysis_seal, analysis_path = _validate_analysis_gate(
        repo_root=repo_root,
        analysis_seal_path=analysis_seal_path,
        expected_analysis_seal_sha256=expected_analysis_seal_sha256,
        runs_root=runs_root,
        analysis_root=analysis_root,
        study_protocol_path=study_protocol_path,
        provider_call_seal_path=provider_call_seal_path,
    )
    study, provider, manifest, manifest_path = _validate_public_chain(
        repo_root=repo_root,
        study_protocol_path=study_protocol_path,
        provider_call_seal_path=provider_call_seal_path,
        public_dir=public_dir,
        analysis=analysis,
    )
    secret, commitment, receipt = _validate_secret_and_receipt(
        secret_file=secret_file,
        receipt_path=receipt_path,
        suite_root=public_dir.resolve().parent,
        study=study,
        provider=provider,
        manifest=manifest,
    )
    public_inventory, verification, generator_path = _validate_regeneration(
        repo_root=repo_root,
        public_dir=public_dir,
        reveal_path=reveal_path,
        secret=secret,
        study=study,
        provider=provider,
    )
    tasks = _array(manifest.get("tasks"), name="manifest.tasks")
    commitments = {
        cast(str, task["task_id"]): _digest(
            _object(task, name="manifest task").get("target_commitment_sha256"),
            name="target commitment",
        )
        for task in tasks
    }
    _expect(tuple(commitments), TASK_IDS, name="target commitment order")
    return {
        "schema": REPORT_SCHEMA,
        "status": "POST-UNBLIND VERIFICATION PASSED",
        "checks": {
            "reveal_free_analysis_sealed_before_unblinding": True,
            "secret_commitment_verified": True,
            "one_time_receipt_verified": True,
            "hidden_reveal_sha256_verified": True,
            "public_suite_byte_regeneration_verified": True,
            "hidden_reveal_byte_regeneration_verified": True,
            "all_target_commitments_and_oracles_verified": True,
        },
        "bindings": {
            "study_protocol_sha256": sha256_file(study_protocol_path.resolve()),
            "provider_call_seal_sha256": sha256_file(provider_call_seal_path.resolve()),
            "reveal_free_public_bundle_sha256": analysis_seal["bundle_sha256"],
            "reveal_free_public_sealer_sha256": PUBLIC_SEALER_SHA256,
            "analysis_sha256": sha256_file(analysis_path),
            "public_manifest_sha256": sha256_file(manifest_path),
            "hidden_reveal_sha256": sha256_file(reveal_path.resolve()),
            "frozen_generator_sha256": sha256_file(generator_path),
            "task_secret_commitment_sha256": commitment,
        },
        "analysis_gate": {
            "schema": analysis["schema"],
            "private_reveal_used": analysis["private_reveal_used"],
            "seal_scheme": analysis_seal["scheme"],
        },
        "custody": {
            "receipt_schema": receipt["schema"],
            "secret_material_emitted": False,
            "target_ast_material_emitted": False,
        },
        "public_file_sha256": public_inventory,
        "target_commitment_sha256": commitments,
        "generator_verification": verification,
    }


def _write_exclusive_report(path: Path, report: Mapping[str, object]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(canonical_bytes(report) + b"\n")


def _reject_report_inside_inputs(output: Path, protected_roots: Sequence[Path]) -> None:
    destination = output.resolve()
    for raw_root in protected_roots:
        root = raw_root.resolve()
        if destination == root or root in destination.parents:
            raise ValueError(f"report output must remain outside immutable input tree: {root}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--study-protocol", type=Path, required=True)
    parser.add_argument("--provider-call-seal", type=Path, required=True)
    parser.add_argument("--analysis-seal", type=Path, required=True)
    parser.add_argument("--expected-analysis-seal-sha256", required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--secret-file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve().exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output.resolve()}")
    _reject_report_inside_inputs(
        args.output,
        (
            args.public_dir.resolve().parent,
            args.runs_root,
            args.analysis_root,
            args.analysis_seal,
        ),
    )
    report = verify_post_unblinding(
        repo_root=args.repo_root,
        study_protocol_path=args.study_protocol,
        provider_call_seal_path=args.provider_call_seal,
        analysis_seal_path=args.analysis_seal,
        expected_analysis_seal_sha256=args.expected_analysis_seal_sha256,
        runs_root=args.runs_root,
        analysis_root=args.analysis_root,
        public_dir=args.public_dir,
        secret_file=args.secret_file,
        receipt_path=args.receipt,
        reveal_path=args.reveal,
    )
    _write_exclusive_report(args.output, report)
    print(json.dumps({"status": report["status"], "report": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

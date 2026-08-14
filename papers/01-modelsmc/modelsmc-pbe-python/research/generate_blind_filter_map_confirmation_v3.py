"""Custodial generator and seal builder for the fresh blind-v3 confirmation.

The command ordering is deliberately enforced:

1. an external Stage-1 method seal already binds the immutable method;
2. ``prepare-secret`` creates a private 32-byte secret;
3. ``seal-custody`` binds its commitment before task generation;
4. ``generate`` consumes that binding and creates public tasks plus a private reveal;
5. ``seal-provider`` binds public bytes before any provider call.

No command in this module contacts a model provider.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import secrets
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from research.blind_filter_map_confirmation_v3 import (
    CONSTANTS,
    CUSTODY_SEAL_SCHEMA,
    MANIFEST_SCHEMA,
    METHOD_SCHEMA,
    METHOD_SEAL_SCHEMA,
    METHOD_STATUS,
    PROVIDER_SEAL_SCHEMA,
    TARGET_COMMITMENT_DOMAIN,
    TARGET_SCHEMA,
    TASK_IDS,
    canonical_bytes,
    expect,
    frozen_run_seeds,
    read_object,
    reject_private_keys,
    require_array,
    require_digest,
    require_object,
    safe_repo_path,
    sample_target,
    seed_commitment,
    sha256_bytes,
    sha256_file,
    task_document,
    write_json_exclusive,
)


def _validate_method_boundary(
    *,
    repo_root: Path,
    protocol_path: Path,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    root = repo_root.resolve()
    protocol_file = protocol_path.resolve()
    seal_file = method_seal_path.resolve()
    expected = require_digest(
        expected_method_seal_sha256,
        name="expected method-seal SHA-256",
    )
    method_seal_sha256 = sha256_file(seal_file)
    expect(method_seal_sha256, expected, name="external method-seal SHA-256")
    seal = read_object(seal_file)
    expect(seal.get("schema"), METHOD_SEAL_SCHEMA, name="method-seal schema")
    expect(
        seal.get("sealed_before_secret_preparation"),
        True,
        name="method seal timing",
    )
    binding = require_object(seal.get("protocol"), name="method-seal.protocol")
    bound_protocol = safe_repo_path(root, binding.get("path"), name="method-seal.protocol.path")
    expect(bound_protocol, protocol_file, name="method-seal protocol path")
    protocol_sha256 = sha256_file(protocol_file)
    expect(
        binding.get("sha256"),
        protocol_sha256,
        name="method-seal protocol SHA-256",
    )
    protocol = read_object(protocol_file)
    expect(protocol.get("schema"), METHOD_SCHEMA, name="method protocol schema")
    expect(protocol.get("protocol_status"), METHOD_STATUS, name="method protocol status")
    freeze = require_object(protocol.get("freeze_requirements"), name="freeze_requirements")
    common = require_object(freeze.get("common"), name="freeze_requirements.common")
    common_path = safe_repo_path(root, common.get("path"), name="common.path")
    expect(
        common.get("sha256"),
        sha256_file(common_path),
        name="shared v3 definition SHA-256",
    )
    generator = require_object(freeze.get("generator"), name="freeze_requirements.generator")
    generator_path = safe_repo_path(root, generator.get("path"), name="generator.path")
    expect(generator_path, Path(__file__).resolve(), name="executing generator path")
    expect(generator.get("sha256"), sha256_file(generator_path), name="generator SHA-256")
    return protocol, protocol_sha256, method_seal_sha256


def _read_secret(path: Path) -> bytes:
    secret_file = path.expanduser().resolve()
    if secret_file.stat().st_mode & 0o077:
        raise ValueError("secret file must not be accessible by group or others")
    seed = secret_file.read_bytes()
    if len(seed) != 32:
        raise ValueError("secret file must contain exactly 32 raw bytes")
    return seed


def prepare_secret(
    output: Path,
    *,
    repo_root: Path,
    protocol_path: Path,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
) -> str:
    """Create a custodial secret only after validating the external method seal."""

    _validate_method_boundary(
        repo_root=repo_root,
        protocol_path=protocol_path,
        method_seal_path=method_seal_path,
        expected_method_seal_sha256=expected_method_seal_sha256,
    )
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    seed = secrets.token_bytes(32)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(seed)
    return seed_commitment(seed)


def seal_custody(
    output: Path,
    *,
    secret_file: Path,
    repo_root: Path,
    protocol_path: Path,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
) -> dict[str, object]:
    """Bind the secret commitment before the secret can generate any task."""

    _, protocol_sha256, method_seal_sha256 = _validate_method_boundary(
        repo_root=repo_root,
        protocol_path=protocol_path,
        method_seal_path=method_seal_path,
        expected_method_seal_sha256=expected_method_seal_sha256,
    )
    seed = _read_secret(secret_file)
    seal: dict[str, object] = {
        "schema": CUSTODY_SEAL_SCHEMA,
        "sealed_before_task_generation": True,
        "study_protocol_sha256": protocol_sha256,
        "method_seal_sha256": method_seal_sha256,
        "task_secret_commitment_sha256": seed_commitment(seed),
    }
    write_json_exclusive(output.expanduser().resolve(), seal)
    return seal


def _validate_custody_boundary(
    *,
    custody_seal_path: Path,
    expected_custody_seal_sha256: str,
    protocol_sha256: str,
    method_seal_sha256: str,
    seed: bytes,
) -> tuple[dict[str, Any], str]:
    custody_file = custody_seal_path.resolve()
    custody_sha256 = sha256_file(custody_file)
    expect(
        custody_sha256,
        require_digest(
            expected_custody_seal_sha256,
            name="expected custody-seal SHA-256",
        ),
        name="external custody-seal SHA-256",
    )
    custody = read_object(custody_file)
    expect(custody.get("schema"), CUSTODY_SEAL_SCHEMA, name="custody-seal schema")
    expect(
        custody.get("sealed_before_task_generation"),
        True,
        name="custody seal timing",
    )
    expect(
        custody.get("study_protocol_sha256"),
        protocol_sha256,
        name="custody protocol SHA-256",
    )
    expect(
        custody.get("method_seal_sha256"),
        method_seal_sha256,
        name="custody method-seal SHA-256",
    )
    expect(
        custody.get("task_secret_commitment_sha256"),
        seed_commitment(seed),
        name="custody secret commitment",
    )
    return custody, custody_sha256


def _claim_secret_use(secret_file: Path, *, custody_sha256: str, output: Path) -> None:
    secret_path = secret_file.expanduser().resolve()
    receipt_path = secret_path.with_name(secret_path.name + ".used.json")
    receipt = {
        "schema": "blind-filter-map-confirmation-v3-secret-use-v1",
        "custody_seal_sha256": custody_sha256,
        "output": str(output.resolve()),
    }
    write_json_exclusive(receipt_path, receipt, mode=0o600)


def generate_suite(
    output: Path,
    *,
    secret_file: Path,
    repo_root: Path,
    protocol_path: Path,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
    custody_seal_path: Path,
    expected_custody_seal_sha256: str,
) -> dict[str, object]:
    """Generate the committed suite once, after both pre-generation seals."""

    protocol, protocol_sha256, method_seal_sha256 = _validate_method_boundary(
        repo_root=repo_root,
        protocol_path=protocol_path,
        method_seal_path=method_seal_path,
        expected_method_seal_sha256=expected_method_seal_sha256,
    )
    seed = _read_secret(secret_file)
    custody, custody_sha256 = _validate_custody_boundary(
        custody_seal_path=custody_seal_path,
        expected_custody_seal_sha256=expected_custody_seal_sha256,
        protocol_sha256=protocol_sha256,
        method_seal_sha256=method_seal_sha256,
        seed=seed,
    )
    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    _claim_secret_use(secret_file, custody_sha256=custody_sha256, output=destination)
    public_dir = destination / "public"
    private_dir = destination / "private"
    public_dir.mkdir(parents=True)
    private_dir.mkdir(mode=0o700)

    public_records: list[dict[str, object]] = []
    private_records: list[dict[str, object]] = []
    for task_id in TASK_IDS:
        target = sample_target(seed, task_id)
        task = task_document(task_id, target, seed=seed)
        task_payload = canonical_bytes(task) + b"\n"
        task_path = public_dir / f"{task_id}.json"
        task_path.write_bytes(task_payload)
        task_canonical_sha256 = sha256_bytes(canonical_bytes(task))
        nonce = hmac.new(
            seed,
            f"{TARGET_COMMITMENT_DOMAIN}\0{task_id}\0nonce".encode(),
            hashlib.sha256,
        ).hexdigest()
        preimage = {
            "schema": TARGET_SCHEMA,
            "task_id": task_id,
            "task_canonical_sha256": task_canonical_sha256,
            "target": {"predicate": target.predicate, "mapper": target.mapper},
            "nonce": nonce,
        }
        target_commitment = sha256_bytes(canonical_bytes(preimage))
        public_records.append(
            {
                "task_id": task_id,
                "path": task_path.name,
                "task_file_sha256": sha256_bytes(task_payload),
                "task_canonical_sha256": task_canonical_sha256,
                "target_commitment_sha256": target_commitment,
            }
        )
        private_records.append(
            {
                "task_id": task_id,
                "accepted_attempt": target.accepted_attempt,
                "target_predicate_dsl": target.predicate_dsl,
                "target_mapper_dsl": target.mapper_dsl,
                "rejection_log": list(target.rejected),
                "commitment_preimage": preimage,
                "target_commitment_sha256": target_commitment,
            }
        )

    generator_sha256 = sha256_file(Path(__file__).resolve())
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "task_count": len(TASK_IDS),
        "seed_commitment_sha256": seed_commitment(seed),
        "task_generation": {
            "distribution": "target-independent-recursive-grammar-rejection-v1",
            "constants": [str(value) for value in CONSTANTS],
            "generator_sha256": generator_sha256,
            "study_protocol_sha256": protocol_sha256,
            "custody_seal_sha256": custody_sha256,
            "provider_boundary": "public task JSON only; private/reveal.json is forbidden",
        },
        "tasks": public_records,
    }
    reject_private_keys(manifest)
    for task_id in TASK_IDS:
        reject_private_keys(read_object(public_dir / f"{task_id}.json"), path=task_id)
    manifest_path = public_dir / "manifest.json"
    manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")
    reveal = {
        "schema": MANIFEST_SCHEMA,
        "seed_hex": seed.hex(),
        "seed_commitment_sha256": seed_commitment(seed),
        "study_protocol_sha256": protocol_sha256,
        "method_seal_sha256": method_seal_sha256,
        "custody_seal_sha256": custody_sha256,
        "public_manifest_file_sha256": sha256_file(manifest_path),
        "tasks": private_records,
    }
    write_json_exclusive(private_dir / "reveal.json", reveal, mode=0o600)
    expect(
        custody.get("task_secret_commitment_sha256"),
        manifest.get("seed_commitment_sha256"),
        name="manifest custody commitment",
    )
    expect(
        require_object(protocol.get("task_distribution"), name="task_distribution").get(
            "distribution_id"
        ),
        "target-independent-recursive-grammar-rejection-v1",
        name="frozen task distribution",
    )
    return manifest


def verify_suite(
    *,
    public_dir: Path,
    reveal_path: Path,
    custody_seal_path: Path,
) -> dict[str, object]:
    """After unblinding, deterministically verify every target and commitment."""

    public_root = public_dir.resolve()
    manifest_path = public_root / "manifest.json"
    manifest = read_object(manifest_path)
    reveal = read_object(reveal_path.resolve())
    custody = read_object(custody_seal_path.resolve())
    expect(manifest.get("schema"), MANIFEST_SCHEMA, name="manifest schema")
    expect(reveal.get("schema"), MANIFEST_SCHEMA, name="reveal schema")
    expect(custody.get("schema"), CUSTODY_SEAL_SCHEMA, name="custody schema")
    seed_hex = reveal.get("seed_hex")
    try:
        seed = bytes.fromhex(seed_hex) if isinstance(seed_hex, str) else b""
    except ValueError as error:
        raise ValueError("invalid revealed secret") from error
    if len(seed) != 32:
        raise ValueError("revealed secret must contain exactly 32 bytes")
    commitment = seed_commitment(seed)
    for source, observed in (
        ("manifest", manifest.get("seed_commitment_sha256")),
        ("reveal", reveal.get("seed_commitment_sha256")),
        ("custody", custody.get("task_secret_commitment_sha256")),
    ):
        expect(observed, commitment, name=f"{source} secret commitment")
    expect(
        reveal.get("public_manifest_file_sha256"),
        sha256_file(manifest_path),
        name="reveal public-manifest SHA-256",
    )
    reject_private_keys(manifest)
    public_tasks = require_array(manifest.get("tasks"), name="manifest.tasks")
    private_tasks = require_array(reveal.get("tasks"), name="reveal.tasks")
    expect(len(public_tasks), len(TASK_IDS), name="manifest task count")
    expect(len(private_tasks), len(TASK_IDS), name="reveal task count")
    verified: list[str] = []
    for index, task_id in enumerate(TASK_IDS):
        public = require_object(public_tasks[index], name=f"manifest.tasks[{index}]")
        private = require_object(private_tasks[index], name=f"reveal.tasks[{index}]")
        expect(public.get("task_id"), task_id, name=f"public task {index} ID")
        expect(private.get("task_id"), task_id, name=f"private task {index} ID")
        expect(public.get("path"), f"{task_id}.json", name=f"{task_id} path")
        task_path = public_root / f"{task_id}.json"
        task = read_object(task_path)
        reject_private_keys(task, path=task_id)
        expect(
            public.get("task_file_sha256"),
            sha256_file(task_path),
            name=f"{task_id} file SHA-256",
        )
        task_canonical_sha256 = sha256_bytes(canonical_bytes(task))
        expect(
            public.get("task_canonical_sha256"),
            task_canonical_sha256,
            name=f"{task_id} canonical SHA-256",
        )
        target = sample_target(seed, task_id)
        expect(private.get("accepted_attempt"), target.accepted_attempt, name="accepted attempt")
        expect(private.get("target_predicate_dsl"), target.predicate_dsl, name="predicate DSL")
        expect(private.get("target_mapper_dsl"), target.mapper_dsl, name="mapper DSL")
        expect(private.get("rejection_log"), list(target.rejected), name="rejection log")
        expected_task = task_document(task_id, target, seed=seed)
        expect(task, expected_task, name=f"{task_id} oracle examples")
        preimage = require_object(private.get("commitment_preimage"), name="commitment preimage")
        expected_nonce = hmac.new(
            seed,
            f"{TARGET_COMMITMENT_DOMAIN}\0{task_id}\0nonce".encode(),
            hashlib.sha256,
        ).hexdigest()
        expect(
            preimage,
            {
                "schema": TARGET_SCHEMA,
                "task_id": task_id,
                "task_canonical_sha256": task_canonical_sha256,
                "target": {"predicate": target.predicate, "mapper": target.mapper},
                "nonce": expected_nonce,
            },
            name=f"{task_id} commitment preimage",
        )
        nonce = preimage.get("nonce")
        if not isinstance(nonce, str) or len(nonce) != 64:
            raise ValueError(f"invalid target nonce for {task_id}")
        target_commitment = sha256_bytes(canonical_bytes(preimage))
        expect(
            public.get("target_commitment_sha256"),
            target_commitment,
            name=f"{task_id} public target commitment",
        )
        expect(
            private.get("target_commitment_sha256"),
            target_commitment,
            name=f"{task_id} private target commitment",
        )
        verified.append(task_id)
    return {"schema": MANIFEST_SCHEMA, "verified": verified, "exact": True}


def seal_provider_boundary(
    output: Path,
    *,
    repo_root: Path,
    protocol_path: Path,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
    custody_seal_path: Path,
    expected_custody_seal_sha256: str,
    public_manifest_path: Path,
    hidden_target_manifest_sha256: str,
    endpoint_health_record_path: Path,
) -> dict[str, object]:
    """Bind all public bytes after generation and before any provider request."""

    protocol, protocol_sha256, method_seal_sha256 = _validate_method_boundary(
        repo_root=repo_root,
        protocol_path=protocol_path,
        method_seal_path=method_seal_path,
        expected_method_seal_sha256=expected_method_seal_sha256,
    )
    custody_file = custody_seal_path.resolve()
    custody_sha256 = sha256_file(custody_file)
    expect(
        custody_sha256,
        require_digest(expected_custody_seal_sha256, name="expected custody-seal SHA-256"),
        name="external custody-seal SHA-256",
    )
    custody = read_object(custody_file)
    expect(custody.get("schema"), CUSTODY_SEAL_SCHEMA, name="custody schema")
    expect(custody.get("study_protocol_sha256"), protocol_sha256, name="custody protocol")
    expect(custody.get("method_seal_sha256"), method_seal_sha256, name="custody method seal")
    manifest_file = public_manifest_path.resolve()
    manifest = read_object(manifest_file)
    reject_private_keys(manifest)
    expect(manifest.get("schema"), MANIFEST_SCHEMA, name="manifest schema")
    expect(manifest.get("task_count"), len(TASK_IDS), name="manifest task count")
    expect(
        manifest.get("seed_commitment_sha256"),
        custody.get("task_secret_commitment_sha256"),
        name="manifest custody commitment",
    )
    records = require_array(manifest.get("tasks"), name="manifest.tasks")
    root = repo_root.resolve()
    task_records: list[dict[str, object]] = []
    for index, (task_id, seeds) in enumerate(zip(TASK_IDS, frozen_run_seeds(), strict=True)):
        record = require_object(records[index], name=f"manifest.tasks[{index}]")
        expect(record.get("task_id"), task_id, name=f"task {index} ID")
        relative_name = record.get("path")
        expect(relative_name, f"{task_id}.json", name=f"{task_id} path")
        task_path = manifest_file.parent / cast(str, relative_name)
        expect(sha256_file(task_path), record.get("task_file_sha256"), name=f"{task_id} hash")
        try:
            repo_relative = task_path.resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"public task escaped repository: {task_id}") from error
        task_records.append(
            {
                **{key: record[key] for key in (
                    "task_id",
                    "task_file_sha256",
                    "task_canonical_sha256",
                    "target_commitment_sha256",
                )},
                "path": repo_relative,
                **seeds.to_dict(),
            }
        )
    try:
        manifest_relative = manifest_file.relative_to(root).as_posix()
        endpoint_relative = endpoint_health_record_path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("provider-seal inputs must remain inside the repository") from error
    endpoint = read_object(endpoint_health_record_path.resolve())
    expect(endpoint.get("model"), "gpt-oss-120b", name="endpoint model")
    method_model = require_object(protocol.get("provider"), name="provider")
    expect(endpoint.get("model_revision"), method_model.get("model_revision"), name="revision")
    seal: dict[str, object] = {
        "schema": PROVIDER_SEAL_SCHEMA,
        "endpoint_health_query_precedes_seal": True,
        "sealed_before_completion_calls": True,
        "study_protocol_sha256": protocol_sha256,
        "method_seal_sha256": method_seal_sha256,
        "custody_seal_sha256": custody_sha256,
        "task_secret_commitment_sha256": custody.get("task_secret_commitment_sha256"),
        "hidden_target_manifest_sha256": require_digest(
            hidden_target_manifest_sha256,
            name="hidden target manifest SHA-256",
        ),
        "public_manifest": {
            "path": manifest_relative,
            "sha256": sha256_file(manifest_file),
        },
        "endpoint_health_record": {
            "path": endpoint_relative,
            "sha256": sha256_file(endpoint_health_record_path.resolve()),
        },
        "tasks": task_records,
    }
    write_json_exclusive(output.resolve(), seal)
    return seal


def _add_method_boundary_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--study-protocol", type=Path, required=True)
    parser.add_argument("--method-seal", type=Path, required=True)
    parser.add_argument("--expected-method-seal-sha256", required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-secret")
    _add_method_boundary_arguments(prepare)
    prepare.add_argument("--output", type=Path, required=True)
    custody = commands.add_parser("seal-custody")
    _add_method_boundary_arguments(custody)
    custody.add_argument("--secret-file", type=Path, required=True)
    custody.add_argument("--output", type=Path, required=True)
    generate = commands.add_parser("generate")
    _add_method_boundary_arguments(generate)
    generate.add_argument("--secret-file", type=Path, required=True)
    generate.add_argument("--custody-seal", type=Path, required=True)
    generate.add_argument("--expected-custody-seal-sha256", required=True)
    generate.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--public", type=Path, required=True)
    verify.add_argument("--reveal", type=Path, required=True)
    verify.add_argument("--custody-seal", type=Path, required=True)
    provider = commands.add_parser("seal-provider")
    _add_method_boundary_arguments(provider)
    provider.add_argument("--custody-seal", type=Path, required=True)
    provider.add_argument("--expected-custody-seal-sha256", required=True)
    provider.add_argument("--public-manifest", type=Path, required=True)
    provider.add_argument("--hidden-target-manifest-sha256", required=True)
    provider.add_argument("--endpoint-health-record", type=Path, required=True)
    provider.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-secret":
        commitment = prepare_secret(
            args.output,
            repo_root=args.repo_root,
            protocol_path=args.study_protocol,
            method_seal_path=args.method_seal,
            expected_method_seal_sha256=args.expected_method_seal_sha256,
        )
        print(canonical_bytes({"task_secret_commitment_sha256": commitment}).decode())
        return 0
    if args.command == "seal-custody":
        result = seal_custody(
            args.output,
            secret_file=args.secret_file,
            repo_root=args.repo_root,
            protocol_path=args.study_protocol,
            method_seal_path=args.method_seal,
            expected_method_seal_sha256=args.expected_method_seal_sha256,
        )
    elif args.command == "generate":
        result = generate_suite(
            args.output,
            secret_file=args.secret_file,
            repo_root=args.repo_root,
            protocol_path=args.study_protocol,
            method_seal_path=args.method_seal,
            expected_method_seal_sha256=args.expected_method_seal_sha256,
            custody_seal_path=args.custody_seal,
            expected_custody_seal_sha256=args.expected_custody_seal_sha256,
        )
    elif args.command == "verify":
        result = verify_suite(
            public_dir=args.public,
            reveal_path=args.reveal,
            custody_seal_path=args.custody_seal,
        )
    else:
        result = seal_provider_boundary(
            args.output,
            repo_root=args.repo_root,
            protocol_path=args.study_protocol,
            method_seal_path=args.method_seal,
            expected_method_seal_sha256=args.expected_method_seal_sha256,
            custody_seal_path=args.custody_seal,
            expected_custody_seal_sha256=args.expected_custody_seal_sha256,
            public_manifest_path=args.public_manifest,
            hidden_target_manifest_sha256=args.hidden_target_manifest_sha256,
            endpoint_health_record_path=args.endpoint_health_record,
        )
    print(canonical_bytes(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Seal all reveal-free blind-v3 run and analysis bytes before unblinding.

The helper accepts only the public runs tree and completed reveal-free analysis
file.  It rejects incomplete matched arms, private paths/keys, symlinks,
unsealed provider artifacts, and concurrent byte drift, then writes a sorted
``SHA256SUMS`` plus ``BUNDLE_SHA256`` into a new directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from research.analyze_blind_filter_map_confirmation_v3 import (
    _validate_provider_inventory,
)
from research.blind_filter_map_confirmation_v3 import (
    ANALYSIS_SCHEMA,
    RESULT_SCHEMA,
    TASK_IDS,
    expect,
    read_object,
    require_array,
    require_object,
    sha256_file,
)
from research.run_blind_filter_map_confirmation_v3 import ARMS

SEAL_FILES = frozenset({"SHA256SUMS", "BUNDLE_SHA256"})
ROUND_FILES = frozenset(f"round-{index:02d}.json" for index in range(1, 5))
FORBIDDEN_PATH_PARTS = frozenset(
    {"private", "reveal", "reveal.json", "secret", "secret.bin", "seed.bin"}
)
FORBIDDEN_PRIVATE_KEYS = frozenset(
    {
        "accepted_attempt",
        "commitment_nonce",
        "commitment_preimage",
        "nonce",
        "private_seed",
        "rejection_log",
        "seed_hex",
        "target",
        "target_mapper",
        "target_mapper_dsl",
        "target_predicate",
        "target_predicate_dsl",
    }
)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact {path}: {error}") from error


def _reject_private_values(value: object, *, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_PRIVATE_KEYS:
                raise ValueError(f"private key {key!r} appears at {path}")
            _reject_private_values(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_private_values(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            embedded = json.loads(value)
        except json.JSONDecodeError:
            return
        _reject_private_values(embedded, path=f"{path}<embedded-json>")


def _regular_files(root: Path, *, label: str) -> tuple[Path, ...]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} must be a regular directory: {root}")
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        lowered = {part.lower() for part in relative.parts}
        if lowered.intersection(FORBIDDEN_PATH_PARTS):
            raise ValueError(f"private path appears in {label}: {relative}")
        if path.is_symlink():
            raise ValueError(f"symlinks are forbidden in {label}: {relative}")
        if path.is_file():
            if path.name in SEAL_FILES:
                raise ValueError(f"pre-existing seal file in {label}: {relative}")
            files.append(path)
        elif not path.is_dir():
            raise ValueError(f"nonregular artifact in {label}: {relative}")
    return tuple(sorted(files, key=lambda path: path.relative_to(root).as_posix()))


def _validate_public_json(files: Sequence[Path], root: Path, *, label: str) -> None:
    for path in files:
        if path.suffix.lower() == ".json":
            _reject_private_values(
                _read_json(path),
                path=f"{label}/{path.relative_to(root).as_posix()}",
            )


def validate_public_artifacts(runs_root: Path, analysis_path: Path) -> None:
    """Fail closed unless the complete 12-by-2 public study is present."""

    runs = runs_root.resolve()
    analysis_file = analysis_path.resolve()
    if not analysis_file.is_file() or analysis_file.is_symlink():
        raise ValueError(f"analysis must be a regular file: {analysis_file}")
    run_files = _regular_files(runs, label="runs")
    expect(
        {path.name for path in runs.iterdir()},
        set(TASK_IDS),
        name="runs-root task entries",
    )
    analysis = read_object(analysis_file)
    expect(analysis.get("schema"), ANALYSIS_SCHEMA, name="analysis schema")
    expect(
        analysis.get("status"),
        "complete-reveal-free-confirmatory-analysis",
        name="analysis status",
    )
    expect(analysis.get("method_integrity"), "valid", name="analysis integrity")
    expect(analysis.get("private_reveal_read"), False, name="analysis reveal flag")
    raw_rows = require_array(analysis.get("runs"), name="analysis.runs")
    expected_pairs = [(task_id, arm) for task_id in TASK_IDS for arm in ARMS]
    expect(len(raw_rows), len(expected_pairs), name="analysis run count")
    for index, (task_id, arm) in enumerate(expected_pairs):
        row = require_object(raw_rows[index], name=f"analysis.runs[{index}]")
        expect(row.get("task_id"), task_id, name=f"analysis.runs[{index}].task_id")
        expect(row.get("arm"), arm, name=f"analysis.runs[{index}].arm")
        task_root = runs / task_id
        expected_task_entries = {
            "grammar-random",
            "grammar-random.invocation.json",
            "grammar-random.preflight.json",
            "llm",
            "llm.invocation.json",
            "llm.preflight.json",
        }
        expect(
            {path.name for path in task_root.iterdir()},
            expected_task_entries,
            name=f"{task_id} artifact entries",
        )
        run_dir = task_root / arm
        root_entries = {path.name for path in run_dir.iterdir()}
        required = {
            "protocol.json",
            "executions.json",
            "provider-seal.json",
            "result.json",
            *ROUND_FILES,
        }
        if not required.issubset(root_entries) or root_entries - required - {"provider"}:
            raise ValueError(f"unexpected/incomplete run entries for {task_id}/{arm}")
        for round_name in ROUND_FILES:
            round_path = run_dir / round_name
            if not round_path.is_file() or round_path.is_symlink():
                raise ValueError(f"missing/nonregular round artifact: {round_path}")
        result_path = run_dir / "result.json"
        execution_path = run_dir / "executions.json"
        result = read_object(result_path)
        expect(result.get("schema"), RESULT_SCHEMA, name=f"{result_path} schema")
        expect(
            row.get("result_file_sha256"),
            sha256_file(result_path),
            name=f"{task_id}/{arm} result SHA-256",
        )
        expect(
            row.get("execution_file_sha256"),
            sha256_file(execution_path),
            name=f"{task_id}/{arm} executions SHA-256",
        )
        _validate_provider_inventory(run_dir, result, arm=arm)
        expect(
            row.get("provider_inventory_sha256"),
            result.get("provider_inventory_sha256"),
            name=f"{task_id}/{arm} provider inventory",
        )
    _validate_public_json(run_files, runs, label="runs")
    _reject_private_values(analysis, path=f"analysis/{analysis_file.name}")


def _inventory_bytes(runs_root: Path, analysis_path: Path) -> bytes:
    records = [
        (f"runs/{path.relative_to(runs_root).as_posix()}", path)
        for path in _regular_files(runs_root, label="runs")
    ]
    records.append((f"analysis/{analysis_path.name}", analysis_path))
    records.sort(key=lambda item: item[0])
    return "".join(
        f"{sha256_file(path)}  {logical}\n" for logical, path in records
    ).encode()


def seal_public_artifacts(
    *,
    runs_root: Path,
    analysis_path: Path,
    output: Path,
    preflight_only: bool = False,
) -> dict[str, Any]:
    runs = runs_root.resolve()
    analysis_file = analysis_path.resolve()
    destination = output.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite public seal: {destination}")
    if destination == runs or destination in runs.parents or runs in destination.parents:
        raise ValueError("public seal output and runs root must be separate")
    before = _inventory_bytes(runs, analysis_file)
    validate_public_artifacts(runs, analysis_file)
    after = _inventory_bytes(runs, analysis_file)
    if before != after:
        raise ValueError("public artifact bytes changed during validation")
    bundle_sha256 = hashlib.sha256(after).hexdigest()
    result: dict[str, Any] = {
        "schema": "blind-filter-map-confirmation-v3-public-bundle-v1",
        "status": "validated-reveal-free-public-bundle",
        "file_count": len(after.splitlines()),
        "bundle_sha256": bundle_sha256,
        "private_reveal_read": False,
    }
    if not preflight_only:
        destination.mkdir(parents=True)
        (destination / "SHA256SUMS").write_bytes(after)
        (destination / "BUNDLE_SHA256").write_text(
            bundle_sha256 + "\n",
            encoding="ascii",
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    result = seal_public_artifacts(
        runs_root=args.runs_root,
        analysis_path=args.analysis,
        output=args.output,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

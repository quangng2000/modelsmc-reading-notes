"""Build the data-only, privacy-sanitized Gate-2 Hugging Face release."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from research.publish_pilots import (
    _has_forbidden_content,
    _sanitize,
    _validated_pdf_bytes,
)

_RUN_FILES = frozenset({"manifest.json", "events.jsonl", "final_particles.jsonl", "result.json"})
_CELL_FILES = frozenset({"cell.json", "heldout.json"})
_MODEL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_PACKAGED_TEXT_SUFFIXES = frozenset(
    {".bib", ".csv", ".json", ".jsonl", ".log", ".md", ".sty", ".svg", ".tex"}
)


@dataclass(frozen=True, slots=True)
class EvidenceEntry:
    source_path: str | None
    packaged_path: str
    media_type: str
    compression: str | None
    source_sha256: str | None
    source_bytes: int | None
    sanitized_uncompressed_sha256: str
    sanitized_uncompressed_bytes: int
    packaged_sha256: str
    packaged_bytes: int


@dataclass(frozen=True, slots=True)
class Gate2ValidationResult:
    files: int
    checksums: int
    indexed_evidence_files: int
    cells: int
    completed_cells: int
    failed_cells: int
    score_waves: int
    scored_candidates: int
    gzip_files: int
    packaged_bytes: int
    sanitized_uncompressed_bytes: int


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _safe_relative(raw: object, *, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field} must be a nonempty relative path")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} is unsafe: {raw!r}")
    return path


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _sanitized_text_bytes(source: Path, project_root: Path) -> bytes:
    suffix = source.suffix.lower()
    if suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        return _canonical_json_bytes(_sanitize(value, project_root))
    if suffix == ".jsonl":
        records: list[str] = []
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {source}:{line_number}") from error
            records.append(
                json.dumps(
                    _sanitize(value, project_root),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return (("\n".join(records) + "\n") if records else "").encode()
    text = source.read_text(encoding="utf-8")
    sanitized = _sanitize(text, project_root)
    if not isinstance(sanitized, str):  # pragma: no cover - type-level invariant
        raise TypeError("string sanitizer returned a non-string")
    return sanitized.encode()


def _media_type(path: Path) -> str:
    suffixes = path.suffixes
    if suffixes[-2:] in [[".json", ".gz"], [".jsonl", ".gz"]]:
        return "application/gzip"
    return {
        ".bib": "text/plain",
        ".csv": "text/csv",
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".log": "text/plain",
        ".md": "text/markdown",
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".sty": "text/plain",
        ".svg": "image/svg+xml",
        ".tex": "text/plain",
    }.get(path.suffix.lower(), "application/octet-stream")


def _pack_source_file(
    *,
    source: Path,
    target: Path,
    project_root: Path,
    release_root: Path,
    compression_threshold: int,
    permit_binary: bool = False,
    preserve_text_bytes: bool = False,
) -> EvidenceEntry:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"source artifact must be a regular file: {source}")
    raw = source.read_bytes()
    suffix = source.suffix.lower()
    if preserve_text_bytes:
        if suffix not in _PACKAGED_TEXT_SUFFIXES:
            raise ValueError(f"cannot preserve unsupported text artifact: {source}")
        text = raw.decode()
        if _has_forbidden_content(text):
            raise ValueError(f"preserved artifact fails privacy scan: {source}")
        uncompressed = raw
    elif suffix in _PACKAGED_TEXT_SUFFIXES:
        uncompressed = _sanitized_text_bytes(source, project_root)
    elif suffix == ".pdf" and permit_binary:
        uncompressed = _validated_pdf_bytes(source)
    elif suffix == ".png" and permit_binary and raw.startswith(_PNG_MAGIC):
        uncompressed = raw
    else:
        raise ValueError(f"unsupported publication artifact: {source}")

    if suffix in {".pdf", ".png"} and _has_forbidden_content(uncompressed.decode("latin-1")):
        raise ValueError(f"binary artifact fails privacy scan: {source}")
    compress = suffix in {".json", ".jsonl"} and len(uncompressed) >= compression_threshold
    packaged_target = target.with_suffix(target.suffix + ".gz") if compress else target
    packaged = gzip.compress(uncompressed, compresslevel=9, mtime=0) if compress else uncompressed
    packaged_target.parent.mkdir(parents=True, exist_ok=True)
    packaged_target.write_bytes(packaged)
    return EvidenceEntry(
        source_path=source.relative_to(project_root).as_posix(),
        packaged_path=packaged_target.relative_to(release_root).as_posix(),
        media_type=_media_type(packaged_target),
        compression="gzip-mtime-0" if compress else None,
        source_sha256=_sha256(raw),
        source_bytes=len(raw),
        sanitized_uncompressed_sha256=_sha256(uncompressed),
        sanitized_uncompressed_bytes=len(uncompressed),
        packaged_sha256=_sha256(packaged),
        packaged_bytes=len(packaged),
    )


def _pack_generated_file(
    *,
    content: bytes,
    target: Path,
    release_root: Path,
    media_type: str,
) -> EvidenceEntry:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return EvidenceEntry(
        source_path=None,
        packaged_path=target.relative_to(release_root).as_posix(),
        media_type=media_type,
        compression=None,
        source_sha256=None,
        source_bytes=None,
        sanitized_uncompressed_sha256=_sha256(content),
        sanitized_uncompressed_bytes=len(content),
        packaged_sha256=_sha256(content),
        packaged_bytes=len(content),
    )


def _require_score_ledger(
    result_document: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    context: str,
) -> tuple[int, int]:
    payload = result_document.get("result")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context}: result payload is missing")
    ledger = payload.get("score_ledger")
    if not isinstance(ledger, list) or not ledger:
        raise ValueError(f"{context}: complete score_ledger is missing")
    expected_mode = "mean-full-prompt-conditional-logprob"
    if payload.get("llm_energy_normalization") != expected_mode:
        raise ValueError(f"{context}: unexpected LLM energy normalization")
    candidate_count = 0
    for wave_index, raw_wave in enumerate(ledger):
        if not isinstance(raw_wave, Mapping):
            raise ValueError(f"{context}: score wave {wave_index} is not an object")
        wave_label = f"{context}: score wave {wave_index}"
        required_wave = {
            "cache_hit",
            "cache_key_sha256",
            "candidates",
            "energy_normalization",
            "model",
            "model_revision",
            "prompt_prefix",
            "prompt_prefix_sha256",
            "score_origin",
            "score_semantics",
            "selections",
            "source",
            "temperature",
            "tokenizer_revision",
        }
        if not required_wave.issubset(raw_wave):
            raise ValueError(f"{wave_label}: incomplete ledger metadata")
        if raw_wave["energy_normalization"] != expected_mode:
            raise ValueError(f"{wave_label}: energy normalization mismatch")
        if raw_wave["score_semantics"] != "teacher-forced-full-prompt":
            raise ValueError(f"{wave_label}: score semantics mismatch")
        if raw_wave["model"] != model["model_id"]:
            raise ValueError(f"{wave_label}: model alias mismatch")
        if raw_wave["model_revision"] != model["model_revision"]:
            raise ValueError(f"{wave_label}: model revision mismatch")
        if raw_wave["tokenizer_revision"] != model["tokenizer_revision"]:
            raise ValueError(f"{wave_label}: tokenizer revision mismatch")
        prompt = raw_wave["prompt_prefix"]
        if (
            not isinstance(prompt, str)
            or _sha256(prompt.encode()) != raw_wave["prompt_prefix_sha256"]
        ):
            raise ValueError(f"{wave_label}: prompt hash mismatch")
        key = raw_wave["cache_key_sha256"]
        if not isinstance(key, str) or _SHA256_PATTERN.fullmatch(key) is None:
            raise ValueError(f"{wave_label}: invalid cache key")
        origin = raw_wave["score_origin"]
        cache_hit = raw_wave["cache_hit"]
        if origin == "cache" and cache_hit is not True:
            raise ValueError(f"{wave_label}: cache origin without cache hit")
        if origin == "provider" and cache_hit is not False:
            raise ValueError(f"{wave_label}: provider origin with cache hit")
        if origin not in {"cache", "provider"}:
            raise ValueError(f"{wave_label}: unsupported score origin {origin!r}")
        candidates = raw_wave["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{wave_label}: no finite candidates")
        for candidate_index, raw_candidate in enumerate(candidates):
            candidate_label = f"{wave_label}, candidate {candidate_index}"
            if not isinstance(raw_candidate, Mapping):
                raise ValueError(f"{candidate_label}: candidate is not an object")
            required_candidate = {
                "canonical_candidate",
                "deduction_probability",
                "normalized_energy",
                "proposal_probability",
                "qwen_probability",
                "scored_token_count",
                "token_ids",
                "token_logprobs",
                "total_sequence_logprob",
            }
            if not required_candidate.issubset(raw_candidate):
                raise ValueError(f"{candidate_label}: incomplete score record")
            token_ids = raw_candidate["token_ids"]
            token_logprobs = raw_candidate["token_logprobs"]
            if not isinstance(token_ids, list) or not isinstance(token_logprobs, list):
                raise ValueError(f"{candidate_label}: token records are not lists")
            if (
                len(token_ids) != len(token_logprobs)
                or len(token_ids) != raw_candidate["scored_token_count"]
            ):
                raise ValueError(f"{candidate_label}: scored-token count mismatch")
            if not token_ids:
                raise ValueError(f"{candidate_label}: empty teacher-forced score")
            total = raw_candidate["total_sequence_logprob"]
            energy = raw_candidate["normalized_energy"]
            if not isinstance(total, (int, float)) or not isinstance(energy, (int, float)):
                raise ValueError(f"{candidate_label}: nonnumeric energy")
            if not math.isclose(float(total), sum(token_logprobs), abs_tol=1e-8):
                raise ValueError(f"{candidate_label}: total log probability mismatch")
            if not math.isclose(float(total), float(energy) * len(token_ids), abs_tol=1e-8):
                raise ValueError(f"{candidate_label}: normalized energy mismatch")
            candidate_count += 1
        selections = raw_wave["selections"]
        if not isinstance(selections, list) or not selections:
            raise ValueError(f"{wave_label}: selections are missing")
        for selection in selections:
            if not isinstance(selection, Mapping):
                raise ValueError(f"{wave_label}: selection is not an object")
            selected_index = selection.get("selected_index")
            if not isinstance(selected_index, int) or not 0 <= selected_index < len(candidates):
                raise ValueError(f"{wave_label}: selected index is out of range")
    return len(ledger), candidate_count


def _expected_cell_ids(release: Mapping[str, Any], model_id: str) -> set[str]:
    grid = _release_grid(release)
    tasks, arms, seeds = grid.get("tasks"), grid.get("arms"), grid.get("seeds")
    if not isinstance(tasks, list) or not isinstance(arms, list) or not isinstance(seeds, list):
        raise ValueError("release grid axes are invalid")
    return {
        f"{task}--{arm}--{model_id}--seed-{seed}"
        for task in tasks
        for arm in arms
        for seed in seeds
    }


def _release_grid(release: Mapping[str, Any]) -> Mapping[str, Any]:
    grid = release.get("grid")
    if not isinstance(grid, Mapping):
        raise ValueError("release grid is not an object")
    return grid


def _validate_group_source(
    *,
    group_root: Path,
    group: Mapping[str, Any],
    release: Mapping[str, Any],
) -> tuple[list[Path], list[Mapping[str, Any]], Counter[str]]:
    model_id = group.get("model_id")
    if not isinstance(model_id, str):
        raise ValueError("grid group has no model_id")
    protocol = release["protocol"]
    assert isinstance(protocol, Mapping)
    protocol_sha = protocol["sha256"]
    matrix = _read_json(group_root / "matrix_manifest.json")
    if matrix.get("protocol_sha256") != protocol_sha:
        raise ValueError(f"{group_root}: matrix protocol hash mismatch")
    if matrix.get("stage_id") != _release_grid(release)["stage_id"]:
        raise ValueError(f"{group_root}: stage mismatch")
    planned = matrix.get("planned_cells")
    if not isinstance(planned, list):
        raise ValueError(f"{group_root}: planned_cells is missing")
    expected_ids = _expected_cell_ids(release, model_id)
    planned_ids = {item.get("cell_id") for item in planned if isinstance(item, Mapping)}
    if planned_ids != expected_ids:
        raise ValueError(f"{group_root}: planned cell allowlist mismatch")
    actual_ids = {path.name for path in (group_root / "cells").iterdir() if path.is_dir()}
    if actual_ids != expected_ids:
        raise ValueError(f"{group_root}: actual cell allowlist mismatch")
    protocol_bytes = (group_root / "protocol.json").read_bytes()
    if _sha256(protocol_bytes) != protocol_sha:
        raise ValueError(f"{group_root}: embedded protocol hash mismatch")

    sources = [
        group_root / "matrix_manifest.json",
        group_root / "protocol.json",
        group_root / "analysis" / "metrics.json",
        group_root / "analysis" / "metrics.csv",
    ]
    metric_rows = json.loads((group_root / "analysis" / "metrics.json").read_text())
    if not isinstance(metric_rows, list) or len(metric_rows) != len(expected_ids):
        raise ValueError(f"{group_root}: metrics do not cover the planned cells")
    metric_ids = {row.get("cell_id") for row in metric_rows if isinstance(row, Mapping)}
    if metric_ids != expected_ids:
        raise ValueError(f"{group_root}: metrics cell allowlist mismatch")

    statuses: Counter[str] = Counter()
    for cell_id in sorted(expected_ids):
        cell_root = group_root / "cells" / cell_id
        found_cell_files = {path.name for path in cell_root.iterdir() if path.is_file()}
        if not _CELL_FILES.issubset(found_cell_files):
            raise FileNotFoundError(f"{cell_root}: missing cell artifacts")
        sources.extend(cell_root / name for name in sorted(_CELL_FILES))
        cell_document = _read_json(cell_root / "cell.json")
        cell_spec = cell_document.get("cell")
        if not isinstance(cell_spec, Mapping) or cell_spec.get("cell_id") != cell_id:
            raise ValueError(f"{cell_root}: cell identity mismatch")
        for source_key, group_key in (
            ("model_id", "model_id"),
            ("model_hf_repository", "model_repository"),
            ("model_revision", "model_revision"),
            ("tokenizer_revision", "tokenizer_revision"),
        ):
            if cell_spec.get(source_key) != group.get(group_key):
                raise ValueError(f"{cell_root}: {source_key} mismatch")
        if cell_document.get("protocol_sha256") != protocol_sha:
            raise ValueError(f"{cell_root}: protocol hash mismatch")
        status = cell_document.get("status")
        statuses[str(status)] += 1
        artifacts_root = cell_root / "artifacts"
        run_roots = sorted(path for path in artifacts_root.iterdir() if path.is_dir())
        if status != "completed":
            for run_root in run_roots:
                sources.extend(path for path in run_root.iterdir() if path.name in _RUN_FILES)
            continue
        if len(run_roots) != 1:
            raise ValueError(f"{cell_root}: completed cell must contain exactly one core run")
        run_root = run_roots[0]
        run_files = {path.name for path in run_root.iterdir() if path.is_file()}
        if not _RUN_FILES.issubset(run_files):
            raise FileNotFoundError(f"{run_root}: incomplete core artifacts")
        sources.extend(run_root / name for name in sorted(_RUN_FILES))
        run_manifest = _read_json(run_root / "manifest.json")
        configuration = run_manifest.get("configuration")
        if not isinstance(configuration, Mapping):
            raise ValueError(f"{run_root}: run configuration is missing")
        required_config = {
            "model_repository": group["model_repository"],
            "model_revision": group["model_revision"],
            "tokenizer_revision": group["tokenizer_revision"],
            "llm_energy_normalization": "mean-full-prompt-conditional-logprob",
        }
        for key, expected in required_config.items():
            if configuration.get(key) != expected:
                raise ValueError(f"{run_root}: configuration {key} mismatch")
        if not isinstance(configuration.get("vllm_server_config"), str):
            raise ValueError(f"{run_root}: vLLM server configuration is missing")
        metrics = run_manifest.get("metrics")
        if not isinstance(metrics, Mapping) or not {
            "candidate_score_cache",
            "candidate_score_provider",
        }.issubset(metrics):
            raise ValueError(f"{run_root}: cache/provider metrics are missing")
        result_document = _read_json(run_root / "result.json")
        _require_score_ledger(result_document, model=group, context=cell_id)
    return sources, metric_rows, statuses


def _aggregate_csv(rows: list[Mapping[str, Any]], project_root: Path) -> bytes:
    fields = sorted({str(key) for row in rows for key in row})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        sanitized = _sanitize(row, project_root)
        if not isinstance(sanitized, Mapping):  # pragma: no cover
            raise TypeError("row sanitizer returned a non-mapping")
        writer.writerow(sanitized)
    return stream.getvalue().encode()


def _checksums(root: Path) -> None:
    target = root / "SHA256SUMS"
    entries = [
        f"{_sha256(path.read_bytes())}  {path.relative_to(root).as_posix()}"
        for path in sorted(item for item in root.rglob("*") if item.is_file() and item != target)
    ]
    target.write_text("\n".join(entries) + "\n", encoding="utf-8")


def _git_head(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_publication_revision(project_root: Path, revision: str) -> None:
    if _MODEL_SHA_PATTERN.fullmatch(revision) is None:
        raise ValueError("--source-revision must be a full 40-character Git SHA")
    if _git_head(project_root) != revision:
        raise ValueError("--source-revision does not match the checked-out Git HEAD")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty.strip():
        raise ValueError("publication release requires a clean tracked working tree")


def build_gate2_release(
    project_root: Path,
    destination: Path,
    *,
    source_revision: str,
    require_clean_git: bool = True,
) -> Gate2ValidationResult:
    """Build one exact allowlisted Gate-2 publication package."""

    project_root = project_root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"release destination already exists: {destination}")
    if destination.is_relative_to(project_root):
        raise ValueError("release destination must be outside the source project")
    if require_clean_git:
        _require_clean_publication_revision(project_root, source_revision)
    elif _MODEL_SHA_PATTERN.fullmatch(source_revision) is None:
        raise ValueError("source_revision must be a full 40-character Git SHA")

    template_path = project_root / "research" / "gate2_release.json"
    release = dict(_read_json(template_path))
    implementation = release.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("gate2_release.json has no implementation object")
    implementation = dict(implementation)
    implementation["revision"] = source_revision
    release["implementation"] = implementation
    if release.get("authors") != ["Tri Nguyen", "Thanh-Dat Nguyen"]:
        raise ValueError("release authors must be Tri Nguyen and Thanh-Dat Nguyen")
    protocol = release.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("release protocol is missing")
    protocol_path = project_root / _safe_relative(protocol.get("path"), field="protocol.path")
    if _sha256(protocol_path.read_bytes()) != protocol.get("sha256"):
        raise ValueError("frozen amended protocol hash mismatch")
    manuscript = release.get("manuscript")
    if not isinstance(manuscript, Mapping):
        raise ValueError("release manuscript is missing")
    manuscript_path = project_root / _safe_relative(manuscript.get("path"), field="manuscript.path")
    if _sha256(_validated_pdf_bytes(manuscript_path)) != manuscript.get("sha256"):
        raise ValueError("final manuscript hash mismatch")

    packaging = release.get("packaging")
    if not isinstance(packaging, Mapping):
        raise ValueError("release packaging is missing")
    compression_threshold = packaging.get("compression_threshold_bytes")
    if not isinstance(compression_threshold, int) or compression_threshold < 1:
        raise ValueError("invalid compression threshold")
    grid = _release_grid(release)
    groups = grid.get("groups")
    if not isinstance(groups, list) or len(groups) != 4:
        raise ValueError("release must declare exactly four model groups")

    group_records: list[tuple[Mapping[str, Any], Path, list[Path]]] = []
    aggregate_rows: list[Mapping[str, Any]] = []
    statuses: Counter[str] = Counter()
    for raw_group in groups:
        if not isinstance(raw_group, Mapping):
            raise ValueError("grid group is not an object")
        group_root = project_root / _safe_relative(
            raw_group.get("source_path"), field="grid.groups.source_path"
        )
        sources, metric_rows, group_statuses = _validate_group_source(
            group_root=group_root,
            group=raw_group,
            release=release,
        )
        group_records.append((raw_group, group_root, sources))
        aggregate_rows.extend(metric_rows)
        statuses.update(group_statuses)
    expected_cells = grid["expected_cells"]
    if len(aggregate_rows) != expected_cells or sum(statuses.values()) != expected_cells:
        raise ValueError("declared Gate-2 grid is incomplete")

    destination.mkdir(parents=True, exist_ok=False)
    entries: list[EvidenceEntry] = []
    try:
        card = (project_root / "research" / "huggingface" / "GATE2_README.md").read_text()
        card = card.replace("{{SOURCE_REVISION}}", source_revision)
        if "{{SOURCE_REVISION}}" in card:
            raise ValueError("dataset card source revision placeholder was not resolved")
        (destination / "README.md").write_text(card, encoding="utf-8")
        (destination / "release_manifest.json").write_bytes(_canonical_json_bytes(release))

        entries.append(
            _pack_source_file(
                source=protocol_path,
                target=destination / "protocol" / protocol_path.name,
                project_root=project_root,
                release_root=destination,
                compression_threshold=compression_threshold,
                preserve_text_bytes=True,
            )
        )
        for group, group_root, sources in group_records:
            model_id = str(group["model_id"])
            for source in sources:
                relative = source.relative_to(group_root)
                entries.append(
                    _pack_source_file(
                        source=source,
                        target=destination / "evidence" / "grid" / model_id / relative,
                        project_root=project_root,
                        release_root=destination,
                        compression_threshold=compression_threshold,
                    )
                )

        aggregate_rows.sort(
            key=lambda row: (
                str(row.get("model_id")),
                str(row.get("task_id")),
                str(row.get("arm")),
                int(row.get("seed", 0)),
            )
        )
        sanitized_rows = _sanitize(aggregate_rows, project_root)
        entries.append(
            _pack_generated_file(
                content=_canonical_json_bytes(sanitized_rows),
                target=destination / "aggregate" / "metrics.json",
                release_root=destination,
                media_type="application/json",
            )
        )
        entries.append(
            _pack_generated_file(
                content=_aggregate_csv(aggregate_rows, project_root),
                target=destination / "aggregate" / "metrics.csv",
                release_root=destination,
                media_type="text/csv",
            )
        )
        aggregate_manifest = {
            "schema_version": 1,
            "protocol_sha256": protocol["sha256"],
            "analysis_population": grid["analysis_population"],
            "rows": len(aggregate_rows),
            "status_counts": dict(sorted(statuses.items())),
            "inputs": [
                {
                    "model_id": group["model_id"],
                    "path": f"evidence/grid/{group['model_id']}/analysis/metrics.json",
                    "source_sha256": _sha256((root / "analysis" / "metrics.json").read_bytes()),
                }
                for group, root, _ in group_records
            ],
        }
        entries.append(
            _pack_generated_file(
                content=_canonical_json_bytes(aggregate_manifest),
                target=destination / "aggregate" / "manifest.json",
                release_root=destination,
                media_type="application/json",
            )
        )

        figures = release["figures"]
        assert isinstance(figures, Mapping)
        figure_root = project_root / _safe_relative(figures["path"], field="figures.path")
        figure_manifest = _read_json(figure_root / str(figures["required_manifest"]))
        if figure_manifest.get("protocol_sha256") != protocol["sha256"]:
            raise ValueError("figure protocol hash mismatch")
        if figure_manifest.get("selected_rows") != expected_cells:
            raise ValueError("figure manifest does not cover all cells")
        for source in sorted(path for path in figure_root.iterdir() if path.is_file()):
            entries.append(
                _pack_source_file(
                    source=source,
                    target=destination / "figures" / source.name,
                    project_root=project_root,
                    release_root=destination,
                    compression_threshold=compression_threshold,
                    permit_binary=True,
                    preserve_text_bytes=source.suffix.lower() in _PACKAGED_TEXT_SUFFIXES,
                )
            )

        entries.append(
            _pack_source_file(
                source=manuscript_path,
                target=destination / "paper" / "main.pdf",
                project_root=project_root,
                release_root=destination,
                compression_threshold=compression_threshold,
                permit_binary=True,
            )
        )

        index = {
            "schema_version": 1,
            "release_label": release["release_label"],
            "protocol_sha256": protocol["sha256"],
            "entries": [
                asdict(entry) for entry in sorted(entries, key=lambda item: item.packaged_path)
            ],
        }
        (destination / "EVIDENCE_INDEX.json").write_bytes(_canonical_json_bytes(index))
        _checksums(destination)
        return validate_gate2_release(destination)
    except BaseException:
        shutil.rmtree(destination)
        raise


def _read_packaged_bytes(path: Path) -> bytes:
    content = path.read_bytes()
    if path.suffix == ".gz":
        try:
            return gzip.decompress(content)
        except (EOFError, OSError) as error:
            raise ValueError(f"invalid gzip artifact: {path}") from error
    return content


def _load_packaged_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(_read_packaged_bytes(path).decode())
    if not isinstance(value, Mapping):
        raise ValueError(f"expected packaged JSON object: {path}")
    return value


def _result_path(cell_root: Path) -> Path:
    candidates = list(cell_root.glob("artifacts/*/result.json")) + list(
        cell_root.glob("artifacts/*/result.json.gz")
    )
    if len(candidates) != 1:
        raise ValueError(f"completed packaged cell has {len(candidates)} result artifacts")
    return candidates[0]


def _logical_packaged_name(path: Path) -> str:
    return path.stem if path.suffix == ".gz" else path.name


def validate_gate2_release(root: Path) -> Gate2ValidationResult:
    """Validate privacy, exact grid/schema, gzip contents, and all checksums."""

    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"release directory does not exist: {root}")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("release must not contain symlinks")
    allowed_top_level = {
        "README.md",
        "release_manifest.json",
        "EVIDENCE_INDEX.json",
        "SHA256SUMS",
        "aggregate",
        "evidence",
        "figures",
        "paper",
        "protocol",
    }
    if {path.name for path in root.iterdir()} != allowed_top_level:
        raise ValueError("release top-level allowlist mismatch")
    if any(
        "candidate-scores" in path.parts or "score-cache" in path.parts for path in root.rglob("*")
    ):
        raise ValueError("release contains score-cache blobs")

    files = sorted(path for path in root.rglob("*") if path.is_file())
    forbidden_suffixes = {".lock", ".py", ".pyc", ".sh", ".toml", ".ts"}
    if any(path.suffix.lower() in forbidden_suffixes for path in files):
        raise ValueError("data-only release contains executable source or an environment file")
    gzip_files = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.name == "SHA256SUMS":
            continue
        if path.suffix == ".pdf":
            content = _validated_pdf_bytes(path)
            if _has_forbidden_content(content.decode("latin-1")):
                raise ValueError(f"release privacy scan failed: {relative}")
            continue
        if path.suffix == ".png":
            content = path.read_bytes()
            if not content.startswith(_PNG_MAGIC):
                raise ValueError(f"invalid PNG artifact: {relative}")
            if _has_forbidden_content(content.decode("latin-1")):
                raise ValueError(f"release privacy scan failed: {relative}")
            continue
        content = _read_packaged_bytes(path)
        gzip_files += int(path.suffix == ".gz")
        try:
            text = content.decode()
        except UnicodeDecodeError as error:
            raise ValueError(f"unknown binary artifact: {relative}") from error
        if _has_forbidden_content(text):
            raise ValueError(f"release privacy scan failed: {relative}")
        logical_suffix = Path(path.stem).suffix if path.suffix == ".gz" else path.suffix
        if logical_suffix == ".json":
            json.loads(text)
        elif logical_suffix == ".jsonl":
            for line_number, line in enumerate(text.splitlines(), start=1):
                if line.strip():
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"invalid JSONL at {relative}:{line_number}") from error

    release = _load_packaged_json(root / "release_manifest.json")
    if release.get("authors") != ["Tri Nguyen", "Thanh-Dat Nguyen"]:
        raise ValueError("release authors mismatch")
    implementation = release.get("implementation")
    if not isinstance(implementation, Mapping) or not isinstance(
        implementation.get("revision"), str
    ):
        raise ValueError("release has no clean publication revision")
    revision = str(implementation["revision"])
    if _MODEL_SHA_PATTERN.fullmatch(revision) is None:
        raise ValueError("release publication revision is not a full Git SHA")
    card = (root / "README.md").read_text(encoding="utf-8")
    if "{{SOURCE_REVISION}}" in card or revision not in card:
        raise ValueError("dataset card does not pin the publication revision")
    protocol = release["protocol"]
    if not isinstance(protocol, Mapping):
        raise ValueError("release protocol is invalid")
    protocol_copy = root / "protocol" / Path(str(protocol["path"])).name
    if {path.name for path in (root / "protocol").iterdir()} != {protocol_copy.name}:
        raise ValueError("protocol directory allowlist mismatch")
    if _sha256(protocol_copy.read_bytes()) != protocol["sha256"]:
        raise ValueError("packaged protocol hash mismatch")
    manuscript = release["manuscript"]
    if not isinstance(manuscript, Mapping):
        raise ValueError("release manuscript is invalid")
    if _sha256((root / "paper" / "main.pdf").read_bytes()) != manuscript["sha256"]:
        raise ValueError("packaged manuscript hash mismatch")
    if {path.name for path in (root / "paper").iterdir()} != {"main.pdf"}:
        raise ValueError("paper directory must contain only main.pdf")

    if {path.name for path in (root / "aggregate").iterdir()} != {
        "manifest.json",
        "metrics.csv",
        "metrics.json",
    }:
        raise ValueError("aggregate directory allowlist mismatch")
    figure_manifest = _load_packaged_json(root / "figures" / "figure_manifest.json")
    figure_hashes = figure_manifest.get("sha256")
    if not isinstance(figure_hashes, Mapping) or not figure_hashes:
        raise ValueError("figure manifest has no file checksum inventory")
    expected_figure_files = {
        "figure_manifest.json",
        *{str(name) for name in figure_hashes},
    }
    if {path.name for path in (root / "figures").iterdir()} != expected_figure_files:
        raise ValueError("figure directory allowlist mismatch")
    for name, expected_digest in figure_hashes.items():
        figure = root / "figures" / _safe_relative(name, field="figure_manifest.sha256")
        if _sha256(figure.read_bytes()) != expected_digest:
            raise ValueError(f"figure checksum mismatch: {name}")

    grid = _release_grid(release)
    groups = grid["groups"]
    if not isinstance(groups, list):
        raise ValueError("release model groups are invalid")
    declared_model_ids = {str(group["model_id"]) for group in groups if isinstance(group, Mapping)}
    if {path.name for path in (root / "evidence" / "grid").iterdir()} != declared_model_ids:
        raise ValueError("evidence model-group allowlist mismatch")
    statuses: Counter[str] = Counter()
    score_waves = 0
    scored_candidates = 0
    cell_count = 0
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("packaged group declaration is invalid")
        model_id = str(group["model_id"])
        group_root = root / "evidence" / "grid" / model_id
        if {path.name for path in group_root.iterdir()} != {
            "analysis",
            "cells",
            "matrix_manifest.json",
            "protocol.json",
        }:
            raise ValueError(f"packaged {model_id} group allowlist mismatch")
        if {path.name for path in (group_root / "analysis").iterdir()} != {
            "metrics.csv",
            "metrics.json",
        }:
            raise ValueError(f"packaged {model_id} analysis allowlist mismatch")
        expected_ids = _expected_cell_ids(release, model_id)
        actual_ids = {path.name for path in (group_root / "cells").iterdir() if path.is_dir()}
        if actual_ids != expected_ids:
            raise ValueError(f"packaged {model_id} cell allowlist mismatch")
        matrix = _load_packaged_json(group_root / "matrix_manifest.json")
        if matrix.get("protocol_sha256") != protocol["sha256"]:
            raise ValueError(f"packaged {model_id} matrix protocol mismatch")
        for cell_id in expected_ids:
            cell_count += 1
            cell_root = group_root / "cells" / cell_id
            if {path.name for path in cell_root.iterdir()} != {
                "artifacts",
                "cell.json",
                "heldout.json",
            }:
                raise ValueError(f"packaged {cell_id} cell allowlist mismatch")
            cell = _load_packaged_json(cell_root / "cell.json")
            status = str(cell.get("status"))
            statuses[status] += 1
            if status == "completed":
                run_roots = [path for path in (cell_root / "artifacts").iterdir() if path.is_dir()]
                if len(run_roots) != 1:
                    raise ValueError(f"packaged {cell_id} core-run allowlist mismatch")
                logical_run_files = {
                    _logical_packaged_name(path)
                    for path in run_roots[0].iterdir()
                    if path.is_file()
                }
                if logical_run_files != _RUN_FILES:
                    raise ValueError(f"packaged {cell_id} run-file allowlist mismatch")
                waves, candidates = _require_score_ledger(
                    _load_packaged_json(_result_path(cell_root)),
                    model=group,
                    context=cell_id,
                )
                score_waves += waves
                scored_candidates += candidates
    expected_cells = grid["expected_cells"]
    if cell_count != expected_cells:
        raise ValueError("packaged grid cell count mismatch")
    aggregate = json.loads((root / "aggregate" / "metrics.json").read_text())
    if not isinstance(aggregate, list) or len(aggregate) != expected_cells:
        raise ValueError("aggregate metrics do not cover the exact grid")
    aggregate_ids = {row.get("cell_id") for row in aggregate if isinstance(row, Mapping)}
    expected_all_ids = {
        cell_id
        for group in groups
        if isinstance(group, Mapping)
        for cell_id in _expected_cell_ids(release, str(group["model_id"]))
    }
    if aggregate_ids != expected_all_ids:
        raise ValueError("aggregate metrics cell identities mismatch")

    index = _load_packaged_json(root / "EVIDENCE_INDEX.json")
    raw_entries = index.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("evidence index is empty")
    indexed_paths: set[str] = set()
    packaged_bytes = 0
    uncompressed_bytes = 0
    for item in raw_entries:
        if not isinstance(item, Mapping):
            raise ValueError("invalid evidence index entry")
        packaged_path_value = item.get("packaged_path")
        if not isinstance(packaged_path_value, str) or packaged_path_value in indexed_paths:
            raise ValueError("duplicate or invalid evidence index path")
        relative = packaged_path_value
        path = root / _safe_relative(relative, field="EVIDENCE_INDEX.packaged_path")
        content = path.read_bytes()
        uncompressed = _read_packaged_bytes(path)
        if _sha256(content) != item.get("packaged_sha256") or len(content) != item.get(
            "packaged_bytes"
        ):
            raise ValueError(f"packaged evidence digest mismatch: {relative}")
        if _sha256(uncompressed) != item.get("sanitized_uncompressed_sha256") or len(
            uncompressed
        ) != item.get("sanitized_uncompressed_bytes"):
            raise ValueError(f"uncompressed evidence digest mismatch: {relative}")
        indexed_paths.add(relative)
        packaged_bytes += len(content)
        uncompressed_bytes += len(uncompressed)

    expected_indexed = {
        path.relative_to(root).as_posix()
        for path in files
        if path.name
        not in {
            "README.md",
            "release_manifest.json",
            "EVIDENCE_INDEX.json",
            "SHA256SUMS",
        }
    }
    if indexed_paths != expected_indexed:
        raise ValueError("EVIDENCE_INDEX does not cover the evidence files exactly")

    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise ValueError("release has no SHA256SUMS")
    expected_checksum_paths = {
        path.relative_to(root).as_posix() for path in files if path != checksum_path
    }
    checked: set[str] = set()
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"malformed SHA256SUMS line: {line!r}")
        path = root / _safe_relative(relative, field="SHA256SUMS")
        if relative in checked or not path.is_file() or _sha256(path.read_bytes()) != digest:
            raise ValueError(f"checksum mismatch or duplicate: {relative}")
        checked.add(relative)
    if checked != expected_checksum_paths:
        raise ValueError("SHA256SUMS does not cover the release exactly")

    failed = cell_count - statuses["completed"]
    return Gate2ValidationResult(
        files=len(files),
        checksums=len(checked),
        indexed_evidence_files=len(indexed_paths),
        cells=cell_count,
        completed_cells=statuses["completed"],
        failed_cells=failed,
        score_waves=score_waves,
        scored_candidates=scored_candidates,
        gzip_files=gzip_files,
        packaged_bytes=packaged_bytes,
        sanitized_uncompressed_bytes=uncompressed_bytes,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    result = build_gate2_release(
        project_root,
        args.output,
        source_revision=args.source_revision,
    )
    print(json.dumps({"output": str(args.output.resolve()), **asdict(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

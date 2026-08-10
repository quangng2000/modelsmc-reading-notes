"""Build a privacy-sanitized Hugging Face pilot release directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HOME_PATTERN = re.compile(r"(?:/Users|/home)/[^/\s\"']+")
_WINDOWS_HOME_PATTERN = re.compile(r"[A-Za-z]:\\Users\\[^\\\s\"']+")
_ENDPOINT_PATTERN = re.compile(
    r"https?://[a-z0-9-]+-\d+\.proxy\.runpod\.net(?:/v1)?",
    flags=re.IGNORECASE,
)
_HOST_PATTERN = re.compile(r"[A-Za-z0-9-]+\.local")
_EMAIL_PATTERN = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PUBLICATION_EMAILS = frozenset({"datnguyen@seas.harvard.edu"})
_IPV4_PATTERN = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{12,}=*")
_TOKEN_PATTERN = re.compile(r"\b(?:hf_|sk-)[A-Za-z0-9_-]{12,}\b")
_SENSITIVE_KEY_PATTERN = re.compile(
    r"^(?:api[_-]?key|access[_-]?token|authorization|password|secret|hf[_-]?token|"
    r"runpod[_-]?api[_-]?key)$",
    flags=re.IGNORECASE,
)
_RELEASE_RUN_FILES = frozenset(
    {"manifest.json", "events.jsonl", "final_particles.jsonl", "result.json"}
)
_TREE_IGNORES = (
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    ".venv",
    "dist",
    "build",
    "tmp",
    "output",
    "outputs",
    "generated",
    "tests",
    "archives",
    "cache",
    ".DS_Store",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "*.pdf",
    "*.aux",
    "*.bbl",
    "*.blg",
    "*.log",
    "*.out",
    "*.fls",
    "*.fdb_latexmk",
    "*.synctex.gz",
    "*.toc",
)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    files: int
    json_files: int
    jsonl_records: int
    pdf_files: int
    checksums: int


def _sanitize_string(value: str, project_root: Path) -> str:
    rendered = value.replace(str(project_root), "<PROJECT_ROOT>")
    rendered = _HOME_PATTERN.sub("<LOCAL_HOME>", rendered)
    rendered = _WINDOWS_HOME_PATTERN.sub("<LOCAL_HOME>", rendered)
    rendered = _ENDPOINT_PATTERN.sub("<VLLM_ENDPOINT>", rendered)
    rendered = _HOST_PATTERN.sub("<LOCAL_HOST>", rendered)
    rendered = _EMAIL_PATTERN.sub(
        lambda match: (
            match.group(0)
            if match.group(0).lower() in _PUBLICATION_EMAILS
            else "<REDACTED_EMAIL>"
        ),
        rendered,
    )
    rendered = _IPV4_PATTERN.sub("<REDACTED_IP>", rendered)
    rendered = _BEARER_PATTERN.sub("Bearer <REDACTED>", rendered)
    return _TOKEN_PATTERN.sub("<REDACTED_TOKEN>", rendered)


def _sanitize(value: Any, project_root: Path) -> Any:
    if isinstance(value, str):
        return _sanitize_string(value, project_root)
    if isinstance(value, list):
        return [_sanitize(item, project_root) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<REDACTED>"
                if _SENSITIVE_KEY_PATTERN.fullmatch(str(key))
                else _sanitize(item, project_root)
            )
            for key, item in value.items()
        }
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _declared_run_paths(release: Mapping[str, Any]) -> tuple[str, ...]:
    raw_runs = release.get("matched_runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("pilot_release.json must declare a nonempty matched_runs list")
    paths: list[str] = []
    for index, item in enumerate(raw_runs):
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise ValueError(f"matched_runs[{index}] has no path")
        path = str(item["path"])
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts or len(candidate.parts) != 1:
            raise ValueError(f"matched_runs[{index}] path is unsafe: {path!r}")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("pilot_release.json contains duplicate run paths")
    return tuple(paths)


def _copy_sanitized_runs(
    project_root: Path,
    destination: Path,
    release: Mapping[str, Any],
) -> None:
    runs_root = project_root / "runs"
    if not runs_root.is_dir():
        runs_root = project_root / "pilots" / "runs"
    for run_path in _declared_run_paths(release):
        run_root = runs_root / run_path
        if not run_root.is_dir():
            raise FileNotFoundError(f"declared pilot run does not exist: {run_root}")
        found = {path.name for path in run_root.iterdir() if path.is_file()}
        missing = _RELEASE_RUN_FILES - found
        if missing:
            raise FileNotFoundError(f"declared run {run_path} is missing {sorted(missing)}")
        for filename in sorted(_RELEASE_RUN_FILES):
            source = run_root / filename
            target = destination / "pilots" / "runs" / run_path / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".json":
                document = json.loads(source.read_text(encoding="utf-8"))
                _write_json(target, _sanitize(document, project_root))
                continue
            records = [
                json.dumps(
                    _sanitize(json.loads(line), project_root),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            target.write_text("\n".join(records) + "\n", encoding="utf-8")


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*_TREE_IGNORES),
    )


def _checksums(root: Path) -> None:
    output = root / "SHA256SUMS"
    entries: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != output):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.relative_to(root).as_posix()}")
    output.write_text("\n".join(entries) + "\n", encoding="utf-8")


def _validated_pdf_bytes(path: Path) -> bytes:
    content = path.read_bytes()
    if len(content) < 1024 or not content.startswith(b"%PDF-"):
        raise ValueError(f"manuscript is not a valid PDF container: {path}")
    if not content.rstrip().endswith(b"%%EOF"):
        raise ValueError(f"manuscript PDF has no terminal EOF marker: {path}")
    return content


def _copy_manuscript_pdf(project_root: Path, destination: Path) -> None:
    candidates = (
        project_root / "paper" / "main.pdf",
        project_root / "paper" / "output" / "main.pdf",
    )
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(
            "verified manuscript PDF is missing (expected paper/main.pdf)"
        )
    target = destination / "paper" / "main.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_validated_pdf_bytes(source))


def _has_forbidden_content(text: str) -> bool:
    patterns = (
        _HOME_PATTERN,
        _WINDOWS_HOME_PATTERN,
        _ENDPOINT_PATTERN,
        _HOST_PATTERN,
        _BEARER_PATTERN,
        _TOKEN_PATTERN,
    )
    if any(pattern.search(text) is not None for pattern in patterns):
        return True
    return any(
        match.group(0).lower() not in _PUBLICATION_EMAILS
        for match in _EMAIL_PATTERN.finditer(text)
    )


def validate_release(root: Path) -> ValidationResult:
    """Fail closed if JSON, privacy checks, or the checksum inventory disagree."""

    if not root.is_dir():
        raise ValueError(f"release directory does not exist: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("release must not contain symlinks")
    json_files = 0
    jsonl_records = 0
    pdf_files = 0
    for path in files:
        if path.name == "SHA256SUMS":
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "paper/main.pdf":
            _validated_pdf_bytes(path)
            pdf_files += 1
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"release contains a non-text file: {path}") from error
        if _has_forbidden_content(text):
            raise ValueError(f"release privacy scan failed: {path.relative_to(root)}")
        if path.suffix == ".json":
            json.loads(text)
            json_files += 1
        elif path.suffix == ".jsonl":
            for line_number, line in enumerate(text.splitlines(), start=1):
                if line.strip():
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
                    jsonl_records += 1

    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise ValueError("release has no SHA256SUMS")
    expected_paths = {path.relative_to(root).as_posix() for path in files if path != checksum_path}
    checked_paths: set[str] = set()
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"malformed SHA256SUMS line: {line!r}")
        target = root / relative
        if relative in checked_paths or not target.is_file():
            raise ValueError(f"invalid checksum target: {relative}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"checksum mismatch: {relative}")
        checked_paths.add(relative)
    if checked_paths != expected_paths:
        raise ValueError("SHA256SUMS does not cover the release exactly")
    return ValidationResult(
        files=len(files),
        json_files=json_files,
        jsonl_records=jsonl_records,
        pdf_files=pdf_files,
        checksums=len(checked_paths),
    )


def build_release(project_root: Path, destination: Path) -> ValidationResult:
    project_root = project_root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"release destination already exists: {destination}")
    if destination.is_relative_to(project_root):
        raise ValueError("release destination must be outside the source project")
    release = json.loads((project_root / "research" / "pilot_release.json").read_text())
    if not isinstance(release, Mapping):
        raise ValueError("pilot_release.json must contain a JSON object")
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(project_root / "research" / "huggingface" / "README.md", destination)
    shutil.copy2(project_root / "research" / "pilot_release.json", destination)
    _copy_sanitized_runs(project_root, destination, release)
    _copy_tree(project_root / "research", destination / "research")
    _copy_tree(project_root / "paper", destination / "paper")
    _copy_manuscript_pdf(project_root, destination)
    _copy_tree(project_root / "examples", destination / "examples")
    _copy_tree(project_root / "src", destination / "src")
    _copy_tree(project_root / "tests", destination / "tests")
    for name in ("pyproject.toml", "uv.lock", "DESIGN.md", "NOTICE.md"):
        shutil.copy2(project_root / name, destination / name)
    project_readme = project_root / "PROJECT_README.md"
    if not project_readme.is_file():
        project_readme = project_root / "README.md"
    shutil.copy2(project_readme, destination / "PROJECT_README.md")
    _checksums(destination)
    return validate_release(destination)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    result = build_release(project_root, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "files": result.files,
                "json_files": result.json_files,
                "jsonl_records": result.jsonl_records,
                "pdf_files": result.pdf_files,
                "checksums": result.checksums,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

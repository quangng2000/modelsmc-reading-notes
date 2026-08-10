from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.publish_pilots import build_release, validate_release

PROJECT = Path(__file__).parents[2]


def test_release_is_allowlisted_self_contained_and_valid(tmp_path: Path) -> None:
    destination = tmp_path / "release"

    built = build_release(PROJECT, destination)
    validated = validate_release(destination)

    release = json.loads((destination / "pilot_release.json").read_text(encoding="utf-8"))
    declared = {item["path"] for item in release["matched_runs"]}
    copied = {path.name for path in (destination / "pilots" / "runs").iterdir()}
    assert copied == declared
    assert built == validated
    assert built.checksums == built.files - 1
    assert (destination / "src" / "modelsmc_pbe" / "cli.py").is_file()
    assert (destination / "tests" / "test_importance_smc.py").is_file()
    assert (destination / "research" / "protocol.json").is_file()
    assert (destination / "examples" / "foldr-signed-window.json").is_file()
    assert (destination / "paper" / "main.pdf").read_bytes().startswith(b"%PDF-")
    assert built.pdf_files == 1
    assert not list(destination.rglob("__pycache__"))
    assert not list(destination.rglob("*.log"))


def test_release_builder_refuses_to_replace_existing_directory(tmp_path: Path) -> None:
    destination = tmp_path / "release"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        build_release(PROJECT, destination)

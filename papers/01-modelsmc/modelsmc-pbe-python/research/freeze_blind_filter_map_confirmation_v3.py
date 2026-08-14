"""Create the one-shot Stage-1 method protocol and external method seal.

This command hashes already-existing code and tests.  It cannot create a suite
secret, tasks, custody seal, or provider artifact.  Both outputs are exclusive,
so a method change necessarily creates a visibly new attempted study.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from research.blind_filter_map_confirmation_v3 import (
    BUNDLE_SCHEME,
    METHOD_SCHEMA,
    METHOD_SEAL_SCHEMA,
    METHOD_STATUS,
    canonical_bytes,
    expect,
    frozen_run_seeds,
    read_object,
    sha256_bytes,
    sha256_file,
    write_json_exclusive,
)

SOURCE_PATHS = {
    "common": "research/blind_filter_map_confirmation_v3.py",
    "generator": "research/generate_blind_filter_map_confirmation_v3.py",
    "harness": "research/evidence_shortlist_smc.py",
    "runner": "research/run_blind_filter_map_confirmation_v3.py",
    "analysis": "research/analyze_blind_filter_map_confirmation_v3.py",
    "public_sealer": "research/seal_blind_filter_map_confirmation_v3_public.py",
}
DEPENDENCY_PATHS = (
    "research/__init__.py",
    "research/freeze_blind_filter_map_confirmation_v3.py",
    "research/adaptive_shortlist_experiment.py",
    "research/automatic_shortlist_experiment.py",
    "research/automatic_repair_feedback.py",
    "research/direct_json_repair_choice.py",
    "research/execution_guided_repair.py",
    "research/iterative_beam_experiment.py",
    "research/local_repair_importance.py",
    "research/protocol.py",
    "src/modelsmc_pbe/config.py",
    "src/modelsmc_pbe/core/evaluate.py",
    "src/modelsmc_pbe/core/scorer.py",
    "src/modelsmc_pbe/domain/ast.py",
    "src/modelsmc_pbe/grammar/fragments.py",
    "src/modelsmc_pbe/grammar/skeletons.py",
    "src/modelsmc_pbe/smc/numerics.py",
    "pyproject.toml",
    "uv.lock",
)
PROMPT_PATHS = (
    "research/iterative_beam_experiment.py",
    "research/evidence_shortlist_smc.py",
)
TEST_PATHS = (
    "research/tests/test_blind_filter_map_confirmation_v3.py",
    "research/tests/test_run_blind_filter_map_confirmation_v3.py",
    "research/tests/test_analyze_blind_filter_map_confirmation_v3.py",
    "research/tests/test_evidence_shortlist_smc.py",
    "research/tests/test_seal_blind_filter_map_confirmation_v3_public.py",
)


def _binding(repo_root: Path, relative: str) -> dict[str, str]:
    path = repo_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"cannot freeze missing source: {relative}")
    return {"path": relative, "sha256": sha256_file(path)}


def _validate_research_dependency_closure(repo_root: Path) -> None:
    """Reject a freeze that omits any statically imported research module."""

    bound = frozenset((*SOURCE_PATHS.values(), *DEPENDENCY_PATHS))
    for relative in sorted(bound):
        if not relative.startswith("research/") or not relative.endswith(".py"):
            continue
        tree = ast.parse((repo_root / relative).read_text(encoding="utf-8"), filename=relative)
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
        for module in modules:
            if module == "research":
                imported = "research/__init__.py"
            elif module.startswith("research."):
                imported = module.replace(".", "/") + ".py"
            else:
                continue
            if imported not in bound:
                raise ValueError(
                    f"unbound transitive research import: {relative} imports {imported}"
                )


def _bundle(repo_root: Path, *, domain: str, paths: tuple[str, ...]) -> dict[str, object]:
    material = bytearray(domain.encode() + b"\0")
    hashes: dict[str, str] = {}
    for relative in paths:
        payload = (repo_root / relative).read_bytes()
        hashes[relative] = sha256_bytes(payload)
        material.extend(relative.encode() + b"\0" + payload + b"\0")
    return {
        "scheme": BUNDLE_SCHEME,
        "domain": domain,
        "paths_in_order": list(paths),
        "source_sha256": hashes,
        "bundle_sha256": sha256_bytes(bytes(material)),
    }


def _python_tree_binding(repo_root: Path) -> dict[str, object]:
    relative_root = "src/modelsmc_pbe"
    source_root = repo_root / relative_root
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Python-tree freeze rejects symlink: {path}")
    paths = sorted(
        (path for path in source_root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(source_root).as_posix().encode(),
    )
    entries = [
        {
            "path": path.relative_to(source_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    manifest = {
        "schema": "sha256-python-tree-v1",
        "root": relative_root,
        "include": "**/*.py",
        "entries": entries,
    }
    return {
        "schema": "sha256-python-tree-v1",
        "root": relative_root,
        "include": "**/*.py",
        "file_count": len(entries),
        "manifest_sha256": sha256_bytes(canonical_bytes(manifest)),
    }


def freeze_method(
    *,
    repo_root: Path,
    template_path: Path,
    protocol_output: Path,
    seal_output: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    root = repo_root.resolve()
    template = read_object(template_path.resolve())
    expect(template.get("schema"), METHOD_SCHEMA, name="template schema")
    expect(template.get("protocol_status"), METHOD_STATUS, name="template status")
    if "freeze_requirements" in template:
        raise ValueError("template must not contain precomputed freeze requirements")
    _validate_research_dependency_closure(root)
    prompt = _bundle(
        root,
        domain="blind-filter-map-confirmation-v3-prompt-sources-v1",
        paths=PROMPT_PATHS,
    )
    tests = _bundle(
        root,
        domain="blind-filter-map-confirmation-v3-test-sources-v1",
        paths=TEST_PATHS,
    )
    freeze: dict[str, object] = {
        "protocol_sha256_binding": "external-method-seal-and-derived-run-records",
        **{label: _binding(root, relative) for label, relative in SOURCE_PATHS.items()},
        "dependency_bundle": [_binding(root, relative) for relative in DEPENDENCY_PATHS],
        "modelsmc_python_tree": _python_tree_binding(root),
        "prompt_template_binding": prompt,
        "prompt_template_sha256": prompt["bundle_sha256"],
        "test_bundle": tests,
        "run_seeds": [record.to_dict() for record in frozen_run_seeds()],
    }
    protocol = cast(dict[str, object], template | {"freeze_requirements": freeze})
    write_json_exclusive(protocol_output.resolve(), protocol)
    try:
        relative_protocol = protocol_output.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("frozen protocol must be inside the repository") from error
    seal: dict[str, object] = {
        "schema": METHOD_SEAL_SCHEMA,
        "sealed_before_secret_preparation": True,
        "protocol": {
            "path": relative_protocol,
            "sha256": sha256_file(protocol_output.resolve()),
        },
    }
    write_json_exclusive(seal_output.resolve(), seal)
    return protocol, seal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--protocol-output", type=Path, required=True)
    parser.add_argument("--seal-output", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol, seal = freeze_method(
        repo_root=args.repo_root,
        template_path=args.template,
        protocol_output=args.protocol_output,
        seal_output=args.seal_output,
    )
    print(
        canonical_bytes(
            {
                "status": "method-frozen-before-secret-preparation",
                "study_protocol_sha256": sha256_file(args.protocol_output.resolve()),
                "method_seal_sha256": sha256_file(args.seal_output.resolve()),
                "task_count": cast(dict[str, object], protocol["task_distribution"])[
                    "task_count"
                ],
                "sealed_before_secret_preparation": seal[
                    "sealed_before_secret_preparation"
                ],
            }
        ).decode()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

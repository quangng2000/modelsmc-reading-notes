"""Post-failure diagnostic for a factorized evidence-mode proposal.

The immutable V2 fresh gate is not changed.  This provider-free diagnostic
uses the same 20 public tasks to test one mechanism-level repair: discover
predicate and mapper behavior modes separately from singleton evidence, cross
only small top-ranked component pools, and retain eight semantic program modes.
It never ranks the complete 36,000-program product during mode discovery.
"""

from __future__ import annotations

import argparse
import json
import platform
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from modelsmc_pbe.core.evaluate import evaluate_expression
from modelsmc_pbe.core.types import Node
from research.calibrated_program_inference_v2_fresh import (
    ALIASES_PER_MODE,
    ARM_PARTICLES,
    FROZEN_ARMS,
    MODE_COUNT,
    PARTICLE_COUNTS,
    REPETITION_SEEDS,
    TASK_COUNT,
    TASK_IDS,
    FreshCalibrationError,
    _seal_inventory,
    _sha256_file,
    _validate_suite,
    _write_json_exclusive,
    analyze,
    validate_artifact,
)
from research.calibrated_program_inference_v2_screen import (
    ModeBank,
    Proposal,
    _reference_record,
    _semantic_signature,
    build_proposal,
    run_repetition,
)
from research.particle_calibration_terminal_diagnostic_v2 import TerminalTask

STUDY_SCHEMA = "provider-free-calibrated-program-inference-v3-factorized-diagnostic"
STATUS = "frozen-post-v2-failure-before-diagnostic-draws"
HARNESS_PATH = "research/calibrated_program_inference_v3_factorized_diagnostic.py"
TEST_PATH = "research/tests/test_calibrated_program_inference_v3_factorized_diagnostic.py"
DEFAULT_PROTOCOL = (
    "research/protocol-calibrated-program-inference-v3-factorized-diagnostic.json"
)
DEFAULT_OUTPUT = "artifacts/calibrated-program-inference-v3-factorized-diagnostic"
COMPONENT_POOL_SIZE = 8

DEPENDENCY_PATHS = (
    "research/calibrated_program_inference_v2_fresh.py",
    "research/calibrated_program_inference_v2_screen.py",
    "research/calibrated-program-inference-v2-method-candidate.json",
    "research/particle_calibration_terminal_diagnostic_v2.py",
)
LOCKFILE_PATHS = ("pyproject.toml", "uv.lock")
FAILED_ARTIFACT_BINDINGS = (
    (
        "analysis",
        "artifacts/calibrated-program-inference-v2-fresh/analysis.json",
    ),
    (
        "inventory",
        "artifacts/calibrated-program-inference-v2-fresh/inventory.json",
    ),
    (
        "runs",
        "artifacts/calibrated-program-inference-v2-fresh/runs.json",
    ),
    (
        "public_manifest",
        "artifacts/calibrated-program-inference-v2-fresh-suite/public/manifest.json",
    ),
    (
        "unblind_verification",
        "artifacts/calibrated-program-inference-v2-fresh-unblind-verification.json",
    ),
)


def _resolve_file(project_root: Path, relative: str) -> Path:
    root = project_root.resolve()
    path = (root / relative).resolve()
    workspace = root.parents[2]
    if not path.is_relative_to(workspace) or not path.is_file() or path.is_symlink():
        raise FreshCalibrationError(f"bound diagnostic file is invalid: {relative}")
    return path


def _file_record(project_root: Path, relative: str) -> dict[str, str]:
    path = _resolve_file(project_root, relative)
    return {"path": relative, "sha256": _sha256_file(path)}


def _resolve_artifact(project_root: Path, logical_path: str) -> Path:
    if not logical_path.startswith("artifacts/"):
        raise FreshCalibrationError("artifact locator must start with artifacts/")
    for root in (project_root.resolve(), *project_root.resolve().parents):
        candidate = (root / logical_path).resolve()
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise FreshCalibrationError(f"bound artifact is missing: {logical_path}")


def _artifact_record(project_root: Path, logical_path: str) -> dict[str, str]:
    path = _resolve_artifact(project_root, logical_path)
    return {"path": logical_path, "sha256": _sha256_file(path)}


def _singleton_evidence(task: TerminalTask) -> tuple[dict[int, bool], dict[int, int]]:
    keep: dict[int, bool] = {}
    mapper: dict[int, int] = {}
    for example in task.config.spec.examples:
        input_value = cast(list[int], example.input_value)
        output_value = cast(list[int], example.output_value)
        if len(input_value) != 1:
            continue
        item = input_value[0]
        retained = len(output_value) == 1
        if len(output_value) not in (0, 1):
            raise FreshCalibrationError("singleton output has invalid cardinality")
        keep[item] = retained
        if retained:
            mapper[item] = output_value[0]
    if tuple(sorted(keep)) != tuple(task.observed_items):
        raise FreshCalibrationError("factorized bank requires every observed singleton")
    if len(mapper) < 1:
        raise FreshCalibrationError("factorized bank has no retained mapper facts")
    return keep, mapper


def factorized_evidence_mode_bank(
    task: TerminalTask,
    *,
    mode_count: int = MODE_COUNT,
    aliases_per_mode: int = ALIASES_PER_MODE,
    component_pool_size: int = COMPONENT_POOL_SIZE,
) -> ModeBank:
    """Build a semantic bank from factored predicate and mapper evidence."""

    if min(mode_count, aliases_per_mode, component_pool_size) < 1:
        raise FreshCalibrationError("factorized bank sizes must be positive")
    keep, mapper_facts = _singleton_evidence(task)
    item_positions = {item: index for index, item in enumerate(task.observed_items)}
    predicate_groups: dict[tuple[bool, ...], list[int]] = {}
    for predicate_index, dsl in enumerate(task.predicate_order):
        expression = cast(Node, task.predicate_catalog[dsl])
        signature = tuple(
            cast(
                bool,
                evaluate_expression(
                    expression, list(task.observed_items), item=item
                ),
            )
            for item in task.observed_items
        )
        predicate_groups.setdefault(signature, []).append(predicate_index)
    predicate_ranked = sorted(
        predicate_groups.items(),
        key=lambda pair: (
            sum(
                predicted != keep[item]
                for item, predicted in zip(task.observed_items, pair[0], strict=True)
            ),
            task.predicate_order[pair[1][0]],
        ),
    )[:component_pool_size]

    mapper_groups: dict[tuple[int, ...], list[int]] = {}
    for mapper_index, dsl in enumerate(task.mapper_order):
        signature = task.mapper_signatures[dsl]
        mapper_groups.setdefault(signature, []).append(mapper_index)
    mapper_ranked = sorted(
        mapper_groups.items(),
        key=lambda pair: (
            sum(
                pair[0][item_positions[item]] != expected
                for item, expected in mapper_facts.items()
            ),
            task.mapper_order[pair[1][0]],
        ),
    )[:component_pool_size]

    mapper_count = len(task.mapper_order)
    candidates: list[tuple[tuple[object, ...], tuple[int | None, ...], int, int]] = []
    seen_signatures: set[tuple[int | None, ...]] = set()
    for _, predicate_aliases in predicate_ranked:
        for _, mapper_aliases in mapper_ranked:
            predicate_index = predicate_aliases[0]
            mapper_index = mapper_aliases[0]
            program_index = predicate_index * mapper_count + mapper_index
            signature = _semantic_signature(task, program_index)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            key = task.keys[program_index]
            candidates.append(
                (
                    (
                        float(task.losses[program_index]),
                        int(task.costs[program_index]),
                        key.predicate,
                        key.mapper,
                    ),
                    signature,
                    predicate_index,
                    mapper_index,
                )
            )
    candidates.sort(key=lambda record: record[0])
    selected = candidates[:mode_count]
    if len(selected) != mode_count:
        raise FreshCalibrationError("factorized component cross has too few modes")

    predicate_by_signature = {signature: aliases for signature, aliases in predicate_ranked}
    mapper_by_signature = {signature: aliases for signature, aliases in mapper_ranked}
    groups: list[tuple[int, ...]] = []
    for _, semantic_signature, predicate_index, mapper_index in selected:
        predicate_dsl = task.predicate_order[predicate_index]
        predicate_expression = cast(Node, task.predicate_catalog[predicate_dsl])
        predicate_signature = tuple(
            cast(
                bool,
                evaluate_expression(
                    predicate_expression, list(task.observed_items), item=item
                ),
            )
            for item in task.observed_items
        )
        mapper_signature = task.mapper_signatures[task.mapper_order[mapper_index]]
        alias_indices = [
            predicate_alias * mapper_count + mapper_alias
            for predicate_alias in predicate_by_signature[predicate_signature]
            for mapper_alias in mapper_by_signature[mapper_signature]
        ]
        alias_indices = [
            index
            for index in alias_indices
            if _semantic_signature(task, index) == semantic_signature
        ]
        alias_indices.sort(
            key=lambda index: (
                int(task.costs[index]),
                task.keys[index].predicate,
                task.keys[index].mapper,
            )
        )
        groups.append(tuple(alias_indices[:aliases_per_mode]))
    if any(not group for group in groups):
        raise FreshCalibrationError("factorized mode has no syntax representative")
    flattened = [index for group in groups for index in group]
    metadata = {
        "selection": (
            "rank finite-domain predicate behavior classes by singleton keep/drop "
            "violations and mapper behavior classes by retained singleton value "
            "violations; cross the top eight of each, score at most 64 complete "
            "representatives, retain eight semantic modes and four aliases per mode"
        ),
        "hidden_target_used": False,
        "complete_program_catalog_ranked": False,
        "predicate_syntaxes_evaluated": len(task.predicate_order),
        "mapper_syntaxes_evaluated": len(task.mapper_order),
        "component_behavior_classes_crossed": len(predicate_ranked) * len(mapper_ranked),
        "complete_representatives_scored": len(candidates),
        "retained_semantic_modes": len(groups),
        "retained_syntax_centers": len(flattened),
        "aliases_per_retained_mode": [len(group) for group in groups],
        "exact_programs_in_bank": int(task.exact[flattened].sum()),
        "best_bank_loss": float(task.losses[flattened].min()),
        "worst_bank_loss": float(task.losses[flattened].max()),
    }
    return ModeBank(groups=tuple(groups), metadata=metadata)


def build_protocol(project_root: Path) -> dict[str, object]:
    failed = {
        name: _artifact_record(project_root, relative)
        for name, relative in FAILED_ARTIFACT_BINDINGS
    }
    analysis = json.loads(
        _resolve_artifact(
            project_root,
            "artifacts/calibrated-program-inference-v2-fresh/analysis.json",
        ).read_text()
    )
    if analysis["primary_gate"]["pass"] is not False:
        raise FreshCalibrationError("diagnostic requires the frozen V2 gate failure")
    return {
        "schema": STUDY_SCHEMA,
        "status": STATUS,
        "frozen_at": "2026-08-14",
        "study_type": "post-failure provider-free mechanism diagnostic",
        "question": (
            "Did the fresh V2 gate fail because grammar-sampled mode discovery missed "
            "dominant evidence-consistent semantic modes?"
        ),
        "authorization": {
            "provider_calls": False,
            "new_tasks": False,
            "reused_failed_suite": True,
            "confirmatory_gate": False,
            "exact_reference_enumeration": True,
        },
        "fixed_change": {
            "changed": "semantic mode-bank acquisition only",
            "unchanged": [
                "target law",
                "eight retained semantic modes",
                "four aliases per mode",
                "grammar mass 0.25",
                "center mass 0.75",
                "proposal-to-target bridge",
                "particle counts and repetition seeds",
                "resampling and no-MH primary",
            ],
            "component_pool_size": COMPONENT_POOL_SIZE,
            "maximum_complete_representatives_scored": COMPONENT_POOL_SIZE**2,
            "complete_support_ranked_for_mode_discovery": False,
        },
        "diagnostic_rule": {
            "mechanism_supported_if": [
                "every task factorized bank contains at least one exact syntax",
                "the historical gate calculations all pass numerically",
                "N=256 pooled RMSE is below the frozen V2 value",
            ],
            "not_a_fresh_gate": True,
        },
        "design": {
            "task_count": TASK_COUNT,
            "task_ids": list(TASK_IDS),
            "particle_counts": list(PARTICLE_COUNTS),
            "repetition_seeds": list(REPETITION_SEEDS),
            "arms": [arm.arm_id for arm in FROZEN_ARMS],
        },
        "bindings": {
            "harness": _file_record(project_root, HARNESS_PATH),
            "test": _file_record(project_root, TEST_PATH),
            "dependencies": [
                _file_record(project_root, relative) for relative in DEPENDENCY_PATHS
            ],
            "lockfiles": [
                _file_record(project_root, relative) for relative in LOCKFILE_PATHS
            ],
            "failed_v2": failed,
        },
        "claim_boundary": [
            "This diagnostic was designed after observing the V2 fresh failure.",
            "Numerical success cannot retroactively pass or replace the V2 gate.",
            "A new method candidate requires a second independently generated suite.",
        ],
    }


def freeze_protocol(project_root: Path, protocol_path: Path) -> str:
    _write_json_exclusive(protocol_path, build_protocol(project_root))
    return _sha256_file(protocol_path)


def _validate_protocol(
    project_root: Path, protocol_path: Path, expected_protocol_sha256: str
) -> dict[str, object]:
    if _sha256_file(protocol_path) != expected_protocol_sha256:
        raise FreshCalibrationError("diagnostic protocol SHA-256 differs")
    protocol = cast(dict[str, object], json.loads(protocol_path.read_text()))
    if protocol != build_protocol(project_root):
        raise FreshCalibrationError("diagnostic protocol or bound sources differ")
    return protocol


def _summary_markdown(analysis: Mapping[str, object]) -> str:
    diagnostic = cast(Mapping[str, object], analysis["diagnostic_rule"])
    primary = cast(Mapping[str, object], analysis["primary_gate_calculation"])
    lines = [
        "# Factorized Evidence-Mode Diagnostic",
        "",
        "This is a post-failure diagnostic on the V2 fresh tasks, not a fresh gate.",
        "",
        f"Mechanism-support rule: **{'PASS' if diagnostic['pass'] else 'FAIL'}**.",
        f"Historical gate calculation: **{'PASS' if primary['pass'] else 'FAIL'}**.",
        "",
        "The immutable V2 fresh calibration gate remains failed.",
        "",
    ]
    return "\n".join(lines)


def run_diagnostic(
    project_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    fresh_protocol_path: Path,
    fresh_method_seal_path: Path,
    custody_seal_path: Path,
    suite: Path,
    output: Path,
) -> dict[str, object]:
    protocol = _validate_protocol(
        project_root, protocol_path, expected_protocol_sha256
    )
    fresh_protocol_sha = _sha256_file(fresh_protocol_path)
    fresh_method_sha = _sha256_file(fresh_method_seal_path)
    custody_sha = _sha256_file(custody_seal_path)
    bindings, manifest = _validate_suite(
        suite, fresh_protocol_sha, fresh_method_sha, custody_sha
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic: {output}")
    staging = output.with_name(f".{output.name}.incomplete")
    if staging.exists():
        raise FileExistsError(f"refusing to reuse diagnostic staging: {staging}")
    staging.mkdir(parents=True)
    try:
        _write_json_exclusive(staging / "protocol.json", protocol)
        _write_json_exclusive(staging / "public-suite-manifest.json", manifest)
        references: dict[str, Mapping[str, object]] = {}
        runs: list[Mapping[str, object]] = []
        for binding in bindings:
            task = TerminalTask(binding)
            bank = factorized_evidence_mode_bank(task)
            proposals: dict[str, Proposal] = {
                arm.arm_id: build_proposal(task, arm, bank) for arm in FROZEN_ARMS
            }
            reference = _reference_record(task, proposals, bank.metadata)
            references[binding.task_id] = reference
            _write_json_exclusive(
                staging / "references" / f"{binding.task_id}.json", reference
            )
            for arm in FROZEN_ARMS:
                for particles in ARM_PARTICLES[arm.arm_id]:
                    for repetition, base_seed in enumerate(REPETITION_SEEDS):
                        runs.append(
                            run_repetition(
                                task,
                                proposals[arm.arm_id],
                                particles=particles,
                                repetition=repetition,
                                base_seed=base_seed,
                            )
                        )
        gate_analysis = analyze(runs, references)
        banks_all_exact = all(
            int(reference["sampled_mode_bank"]["exact_programs_in_bank"]) > 0
            for reference in references.values()
        )
        failed_analysis = json.loads(
            _resolve_artifact(
                project_root,
                "artifacts/calibrated-program-inference-v2-fresh/analysis.json",
            ).read_text()
        )
        failed_rmse = float(
            failed_analysis["primary_gate"]["rmse_by_particle_count"]["256"]
        )
        new_rmse = float(
            gate_analysis["primary_gate"]["rmse_by_particle_count"]["256"]
        )
        checks = {
            "all_banks_contain_exact": banks_all_exact,
            "historical_gate_calculation_passes": gate_analysis["primary_gate"]["pass"],
            "n256_rmse_improves": new_rmse < failed_rmse,
        }
        result = {
            "schema": f"{STUDY_SCHEMA}-analysis-v1",
            "status": "completed-post-failure-diagnostic",
            "provider_calls": 0,
            "run_count": len(runs),
            "primary_gate_calculation": gate_analysis["primary_gate"],
            "bootstrap": gate_analysis["bootstrap"],
            "pooled": gate_analysis["pooled"],
            "per_task": gate_analysis["per_task"],
            "diagnostic_rule": {"checks": checks, "pass": all(checks.values())},
            "comparison": {
                "failed_v2_n256_rmse": failed_rmse,
                "factorized_n256_rmse": new_rmse,
            },
            "claim_boundary": (
                "post-failure diagnostic on reused fresh-v2 tasks; not a fresh "
                "calibration confirmation"
            ),
        }
        _write_json_exclusive(staging / "runs.json", runs)
        _write_json_exclusive(staging / "analysis.json", result)
        (staging / "SUMMARY.md").write_text(
            _summary_markdown(result), encoding="utf-8"
        )
        _write_json_exclusive(
            staging / "study-metadata.json",
            {
                "schema": f"{STUDY_SCHEMA}-metadata-v1",
                "protocol_sha256": expected_protocol_sha256,
                "provider_calls": 0,
                "private_reveal_read": False,
                "exact_support_materialized_for_reference": True,
                "runtime": {
                    "python": platform.python_version(),
                    "numpy": importlib_metadata_version("numpy"),
                    "device": "cpu",
                },
            },
        )
        _seal_inventory(staging)
        staging.rename(output)
        validate_artifact(output)
        return result
    except Exception as error:
        _write_json_exclusive(
            staging / "FAILURE.json",
            {
                "schema": f"{STUDY_SCHEMA}-failure-v1",
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        raise


def importlib_metadata_version(name: str) -> str:
    import importlib.metadata

    return importlib.metadata.version(name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--project-root", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--project-root", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--expected-protocol-sha256", required=True)
    run.add_argument("--fresh-protocol", type=Path, required=True)
    run.add_argument("--fresh-method-seal", type=Path, required=True)
    run.add_argument("--custody-seal", type=Path, required=True)
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        print(freeze_protocol(args.project_root.resolve(), args.protocol.resolve()))
    elif args.command == "run":
        result = run_diagnostic(
            args.project_root.resolve(),
            args.protocol.resolve(),
            args.expected_protocol_sha256,
            args.fresh_protocol.resolve(),
            args.fresh_method_seal.resolve(),
            args.custody_seal.resolve(),
            args.suite.resolve(),
            args.output.resolve(),
        )
        print(json.dumps(result["diagnostic_rule"], sort_keys=True))
    elif args.command == "validate":
        print(json.dumps(validate_artifact(args.output.resolve()), sort_keys=True))
    else:
        raise AssertionError("unreachable command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

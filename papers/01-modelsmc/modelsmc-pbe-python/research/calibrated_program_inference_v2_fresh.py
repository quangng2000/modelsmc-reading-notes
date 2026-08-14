"""Fresh provider-free confirmation of calibrated program inference V2.

The method, task distribution, particle schedule, repetition seeds, and gates
are frozen before a custodial secret generates any task.  The public run never
opens the private target reveal.  Exact 36,000-syntax references are used only
to evaluate calibration on this bounded confirmation suite.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.metadata
import json
import os
import platform
import secrets
import statistics
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from modelsmc_pbe.core.evaluate import evaluate_expression, evaluate_program
from modelsmc_pbe.core.types import Node
from modelsmc_pbe.grammar import arithmetic_expressions, filter_predicates
from research.blind_filter_map_confirmation_v3 import assemble_program
from research.calibrated_program_inference_v2_screen import (
    ARMS,
    Proposal,
    _reference_record,
    build_proposal,
    canonical_bytes,
    run_repetition,
    sampled_semantic_mode_bank,
)
from research.execution_guided_repair import _dsl_catalog
from research.particle_calibration_terminal_diagnostic_v2 import (
    TaskBinding,
    TerminalTask,
)

STUDY_SCHEMA = "provider-free-calibrated-program-inference-v2-fresh"
PROTOCOL_STATUS = "method-frozen-before-fresh-secret"
HARNESS_PATH = "research/calibrated_program_inference_v2_fresh.py"
TEST_PATH = "research/tests/test_calibrated_program_inference_v2_fresh.py"
METHOD_CANDIDATE_PATH = (
    "research/calibrated-program-inference-v2-method-candidate.json"
)
SCREEN_PATH = "research/calibrated_program_inference_v2_screen.py"
DEFAULT_PROTOCOL = "research/protocol-calibrated-program-inference-v2-fresh.json"
DEFAULT_METHOD_SEAL = (
    "research/protocol-calibrated-program-inference-v2-fresh.method-seal.json"
)
DEFAULT_SUITE = "artifacts/calibrated-program-inference-v2-fresh-suite"
DEFAULT_OUTPUT = "artifacts/calibrated-program-inference-v2-fresh"

TASK_COUNT = 20
TASK_IDS = tuple(f"fresh-cal-v2-{index:02d}" for index in range(1, TASK_COUNT + 1))
CONSTANTS = tuple(range(-3, 5))
DOMAIN = CONSTANTS
PARTICLE_COUNTS = (256, 512, 1024)
REPETITIONS = 128
REPETITION_SEEDS = tuple(range(924_001, 924_001 + REPETITIONS))
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 925_001
IDENTITY_TOLERANCE = 1e-12
DISCOVERY_BUDGET = 4096
MODE_COUNT = 8
ALIASES_PER_MODE = 4
GRAMMAR_MASS = 0.25
CENTER_MASS = 0.75
PRIMARY_PARTICLE_COUNT = 256

PRIMARY_ARM_ID = "sampled-modes-k8-a025-e075-proposal-bridge"
SECONDARY_ARM_ID = "sampled-modes-k8-a025-e075-proposal-bridge-mh"
LEGACY_ARM_ID = "sticky-terminal-is"
ARMS_BY_ID = {arm.arm_id: arm for arm in ARMS}
PRIMARY_ARM = ARMS_BY_ID[PRIMARY_ARM_ID]
SECONDARY_ARM = ARMS_BY_ID[SECONDARY_ARM_ID]
LEGACY_ARM = ARMS_BY_ID[LEGACY_ARM_ID]
FROZEN_ARMS = (PRIMARY_ARM, SECONDARY_ARM, LEGACY_ARM)
ARM_PARTICLES = {
    PRIMARY_ARM_ID: PARTICLE_COUNTS,
    SECONDARY_ARM_ID: PARTICLE_COUNTS,
    LEGACY_ARM_ID: (PRIMARY_PARTICLE_COUNT,),
}

DEPENDENCY_PATHS = (
    METHOD_CANDIDATE_PATH,
    SCREEN_PATH,
    "research/blind_filter_map_confirmation_v3.py",
    "research/evidence_shortlist_smc.py",
    "research/execution_guided_repair.py",
    "research/iterative_beam_experiment.py",
    "research/particle_calibration_terminal_diagnostic_v2.py",
)
LOCKFILE_PATHS = ("pyproject.toml", "uv.lock")

SECRET_DOMAIN = b"calibrated-program-inference-v2-fresh-secret\0"
DRAW_DOMAIN = b"calibrated-program-inference-v2-fresh-draw\0"
TARGET_COMMITMENT_DOMAIN = b"calibrated-program-inference-v2-target\0"


class FreshCalibrationError(ValueError):
    """A frozen fresh-calibration invariant was violated."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json_exclusive(path: Path, value: object, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FreshCalibrationError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _require_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FreshCalibrationError(f"{name} is not a lowercase SHA-256 digest")
    return value


def _resolve_file(project_root: Path, relative: str) -> Path:
    root = project_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise FreshCalibrationError(f"bound file is invalid: {relative}")
    return path


def _file_record(project_root: Path, relative: str) -> dict[str, str]:
    path = _resolve_file(project_root, relative)
    return {"path": relative, "sha256": _sha256_file(path)}


def _python_tree_record(project_root: Path) -> dict[str, object]:
    relative_root = "src/modelsmc_pbe"
    tree_root = (project_root / relative_root).resolve()
    if any(path.is_symlink() for path in tree_root.rglob("*")):
        raise FreshCalibrationError("ModelSMC Python tree contains a symlink")
    paths = sorted(
        (path for path in tree_root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(tree_root).as_posix().encode("utf-8"),
    )
    entries = [
        {
            "path": path.relative_to(tree_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
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
        "manifest_sha256": _sha256_bytes(canonical_bytes(manifest)),
    }


def _runtime_record() -> dict[str, object]:
    return {
        "python_major_minor": ".".join(platform.python_version_tuple()[:2]),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pydantic", "torch")
        },
        "device": "cpu",
    }


def build_protocol(project_root: Path) -> dict[str, object]:
    """Build the exact method protocol before a fresh secret exists."""

    candidate = _read_object(_resolve_file(project_root, METHOD_CANDIDATE_PATH))
    method = cast(Mapping[str, object], candidate["method"])
    expected_method = {
        "adaptive_conditional_ess_fraction": 0.8,
        "algorithm": "proposal-bridge-smc",
        "bridge": "pi_beta(p) proportional to q(p)^(1-beta) * gamma(p)^beta",
        "discovery_grammar_draws": DISCOVERY_BUDGET,
        "grammar_mass": GRAMMAR_MASS,
        "local_center_mass": CENTER_MASS,
        "maximum_annealing_stages": 64,
        "maximum_syntax_aliases_per_mode": ALIASES_PER_MODE,
        "mh_moves": 0,
        "mode_count": MODE_COUNT,
        "mode_selection": (
            "public-score ranking of a fixed grammar sample with finite-domain "
            "semantic deduplication and local predicate/mapper alias expansion"
        ),
        "primary_particle_count": PRIMARY_PARTICLE_COUNT,
        "resample_relative_ess_threshold": 0.5,
    }
    if dict(method) != expected_method:
        raise FreshCalibrationError("selected developmental method record differs")
    return {
        "schema": STUDY_SCHEMA,
        "status": PROTOCOL_STATUS,
        "frozen_at": "2026-08-14",
        "study_type": "fresh provider-free finite-support calibration confirmation",
        "question": (
            "Does the frozen proposal-bridge V2 method accurately estimate semantic "
            "exact-program mass on fresh tasks at N=256?"
        ),
        "authorization": {
            "provider_calls": False,
            "new_model_responses": False,
            "exact_reference_enumeration": True,
            "fresh_secret_required": True,
        },
        "target": {
            "syntax_law": "gamma(p)=g(p)*exp(score.log_target(p))",
            "normalized_law": "pi(p)=gamma(p)/sum_p gamma(p)",
            "primary_semantic_estimand": (
                "sum of pi(p) over all syntaxes with zero loss on every public "
                "example; singleton completeness makes this finite-domain semantic mass"
            ),
        },
        "method": expected_method,
        "arms": [
            {
                **asdict(arm),
                "particle_counts": list(ARM_PARTICLES[arm.arm_id]),
                "confirmatory_role": (
                    "primary" if arm.arm_id == PRIMARY_ARM_ID else "descriptive"
                ),
            }
            for arm in FROZEN_ARMS
        ],
        "task_distribution": {
            "task_count": TASK_COUNT,
            "task_ids": list(TASK_IDS),
            "constants": list(CONSTANTS),
            "target_draw": (
                "independent normalized recursive predicate and mapper draws, accepting "
                "the first draw retaining 3--5 domain values and producing at least "
                "three distinct retained mapper values"
            ),
            "duplicates": "allowed; no cross-task rejection or replacement",
            "examples": (
                "empty, all eight singletons, sorted domain, two secret-derived "
                "permutations, and their concatenation"
            ),
            "outcome_dependent_replacement": False,
        },
        "design": {
            "repetitions_per_task_arm_particle_count": REPETITIONS,
            "repetition_seeds": list(REPETITION_SEEDS),
            "primary_particle_count": PRIMARY_PARTICLE_COUNT,
            "descriptive_particle_counts": [512, 1024],
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "task_weighting": "equal task weight",
            "provider_calls": 0,
        },
        "primary_gate": {
            "all_required": True,
            "rmse": "task-first bootstrap 95% upper bound is strictly below 0.10",
            "bias": (
                "task-first bootstrap central 90% interval lies entirely within "
                "[-0.03,+0.03]"
            ),
            "per_task": "every task N=256 exact-mass RMSE is below 0.20",
            "convergence": "point RMSE is nonincreasing from N=256 to 512 to 1024",
            "weight_tail": (
                "N=256 q95 maximum normalized weight below 0.05 and fraction above "
                "0.25 no greater than 0.01"
            ),
            "identity": "maximum finite-state importance identity error <=1e-12",
        },
        "failure_policy": {
            "source_or_seal_drift": "abort before task scoring or Monte Carlo draws",
            "existing_output": "refuse to overwrite",
            "task_replacement": "forbidden",
            "invalid_or_nonfinite_run": "abort and retain incomplete failure artifact",
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
            "modelsmc_python_tree": _python_tree_record(project_root),
        },
        "runtime": _runtime_record(),
        "claim_boundary": [
            "The primary evidence is provider-free and does not test LLM mode banks.",
            "Exact support enumeration is used for calibration references and audits.",
            "The claim is limited to this grammar, target law, task distribution, and N.",
            "A failed frozen gate remains a failed result; no task may be replaced.",
        ],
    }


def freeze_protocol(project_root: Path, protocol_path: Path) -> str:
    _write_json_exclusive(protocol_path, build_protocol(project_root))
    return _sha256_file(protocol_path)


def seal_method(
    project_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    output: Path,
) -> str:
    protocol = _validate_protocol(
        project_root, protocol_path, expected_protocol_sha256
    )
    seal = {
        "schema": f"{STUDY_SCHEMA}-method-seal-v1",
        "sealed_before_secret_preparation": True,
        "protocol": {
            "path": protocol_path.resolve().relative_to(project_root.resolve()).as_posix(),
            "sha256": expected_protocol_sha256,
        },
        "method_sha256": _sha256_bytes(canonical_bytes(protocol["method"])),
    }
    _write_json_exclusive(output, seal)
    return _sha256_file(output)


def _validate_protocol(
    project_root: Path, protocol_path: Path, expected_protocol_sha256: str
) -> dict[str, Any]:
    expected = _require_digest(expected_protocol_sha256, name="protocol SHA-256")
    if _sha256_file(protocol_path) != expected:
        raise FreshCalibrationError("protocol SHA-256 differs")
    protocol = _read_object(protocol_path)
    if protocol != build_protocol(project_root):
        raise FreshCalibrationError("protocol does not match current bound sources")
    return protocol


def _validate_method_seal(
    project_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
) -> dict[str, Any]:
    _validate_protocol(project_root, protocol_path, expected_protocol_sha256)
    expected = _require_digest(expected_method_seal_sha256, name="method-seal SHA-256")
    if _sha256_file(method_seal_path) != expected:
        raise FreshCalibrationError("method-seal SHA-256 differs")
    seal = _read_object(method_seal_path)
    if seal.get("schema") != f"{STUDY_SCHEMA}-method-seal-v1":
        raise FreshCalibrationError("method-seal schema differs")
    if seal.get("sealed_before_secret_preparation") is not True:
        raise FreshCalibrationError("method seal does not precede secret")
    binding = cast(Mapping[str, object], seal["protocol"])
    if binding.get("sha256") != expected_protocol_sha256:
        raise FreshCalibrationError("method seal binds another protocol")
    return seal


def secret_commitment(secret: bytes) -> str:
    if len(secret) != 32:
        raise FreshCalibrationError("secret must contain exactly 32 bytes")
    return _sha256_bytes(SECRET_DOMAIN + secret)


def prepare_secret(
    project_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
    output: Path,
) -> str:
    _validate_method_seal(
        project_root,
        protocol_path,
        expected_protocol_sha256,
        method_seal_path,
        expected_method_seal_sha256,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    secret = secrets.token_bytes(32)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(secret)
    return secret_commitment(secret)


def _read_secret(path: Path) -> bytes:
    if path.stat().st_mode & 0o077:
        raise FreshCalibrationError("secret permissions are too broad")
    secret = path.read_bytes()
    if len(secret) != 32:
        raise FreshCalibrationError("secret must contain exactly 32 bytes")
    return secret


def seal_custody(
    project_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
    secret_path: Path,
    output: Path,
) -> str:
    _validate_method_seal(
        project_root,
        protocol_path,
        expected_protocol_sha256,
        method_seal_path,
        expected_method_seal_sha256,
    )
    secret = _read_secret(secret_path)
    record = {
        "schema": f"{STUDY_SCHEMA}-custody-seal-v1",
        "sealed_before_task_generation": True,
        "protocol_sha256": expected_protocol_sha256,
        "method_seal_sha256": expected_method_seal_sha256,
        "secret_commitment_sha256": secret_commitment(secret),
    }
    _write_json_exclusive(output, record)
    return _sha256_file(output)


def _hmac(secret: bytes, label: str) -> bytes:
    return hmac.new(secret, DRAW_DOMAIN + label.encode(), hashlib.sha256).digest()


def _uniform_index(secret: bytes, label: str, size: int) -> int:
    if size < 1:
        raise FreshCalibrationError("uniform draw has empty support")
    modulus = 1 << 256
    limit = modulus - modulus % size
    counter = 0
    while True:
        value = int.from_bytes(_hmac(secret, f"{label}\0{counter}"), "big")
        if value < limit:
            return value % size
        counter += 1


def _sample_atom(secret: bytes, label: str) -> str:
    constant = CONSTANTS[_uniform_index(secret, f"{label}\0constant", len(CONSTANTS))]
    template = ("lt(item,c)", "lt(c,item)", "eq(item,c)")[
        _uniform_index(secret, f"{label}\0template", 3)
    ]
    return template.replace("c", str(constant))


def _sample_predicate(secret: bytes, label: str) -> str:
    left = _sample_atom(secret, f"{label}\0left")
    if _uniform_index(secret, f"{label}\0form", 2) == 0:
        return left
    return f"and({left},{_sample_atom(secret, f'{label}\0right')})"


def _sample_mapper(secret: bytes, label: str) -> str:
    form = _uniform_index(secret, f"{label}\0form", 5)
    if form == 0:
        return "item"
    constant = CONSTANTS[_uniform_index(secret, f"{label}\0constant", len(CONSTANTS))]
    if form == 1:
        return str(constant)
    operator = ("add", "sub", "mul")[
        _uniform_index(secret, f"{label}\0operator", 3)
    ]
    if form == 2:
        return f"{operator}(item,item)"
    if form == 3:
        return f"{operator}(item,{constant})"
    return f"{operator}({constant},item)"


def _target_draw(secret: bytes, task_id: str) -> dict[str, object]:
    predicates = _dsl_catalog(filter_predicates(CONSTANTS))
    mappers = _dsl_catalog(arithmetic_expressions("Item", CONSTANTS))
    rejected: list[dict[str, object]] = []
    for attempt in range(1, 1_000_001):
        label = f"{task_id}\0attempt-{attempt}"
        predicate_dsl = _sample_predicate(secret, f"{label}\0predicate")
        mapper_dsl = _sample_mapper(secret, f"{label}\0mapper")
        predicate = cast(Node, predicates[predicate_dsl])
        mapper = cast(Node, mappers[mapper_dsl])
        retained = [
            item
            for item in DOMAIN
            if cast(bool, evaluate_expression(predicate, [], item=item))
        ]
        outputs = [
            cast(int, evaluate_expression(mapper, [], item=item)) for item in retained
        ]
        reasons = []
        if not 3 <= len(retained) <= 5:
            reasons.append("retained-cardinality-outside-3-to-5")
        if len(set(outputs)) < 3:
            reasons.append("fewer-than-three-distinct-retained-outputs")
        if not reasons:
            return {
                "task_id": task_id,
                "predicate_dsl": predicate_dsl,
                "mapper_dsl": mapper_dsl,
                "predicate": predicate,
                "mapper": mapper,
                "accepted_attempt": attempt,
                "rejections": rejected,
            }
        rejected.append(
            {
                "attempt": attempt,
                "predicate_dsl": predicate_dsl,
                "mapper_dsl": mapper_dsl,
                "reasons": reasons,
            }
        )
    raise FreshCalibrationError("target rejection sampler exceeded frozen cap")


def _permutation(secret: bytes, label: str) -> tuple[int, ...]:
    decorated = [
        (_hmac(secret, f"{label}\0{index}\0{item}"), index, item)
        for index, item in enumerate(DOMAIN)
    ]
    return tuple(item for _, _, item in sorted(decorated))


def _encoded(values: Sequence[int]) -> list[str]:
    return [str(value) for value in values]


def _task_document(target: Mapping[str, object], secret: bytes) -> dict[str, object]:
    task_id = cast(str, target["task_id"])
    first = _permutation(secret, f"{task_id}\0permutation-1")
    second = _permutation(secret, f"{task_id}\0permutation-2")
    inputs = (
        (),
        *((item,) for item in DOMAIN),
        DOMAIN,
        first,
        second,
        (*first, *second),
    )
    program = assemble_program(
        cast(Mapping[str, object], target["predicate"]),
        cast(Mapping[str, object], target["mapper"]),
    )
    examples = []
    for input_value in inputs:
        output = evaluate_program(cast(Node, program), list(input_value))
        if not isinstance(output, list):
            raise FreshCalibrationError("target returned non-list output")
        examples.append({"input": _encoded(input_value), "output": _encoded(output)})
    return {
        "name": f"opaque fresh calibration task {task_id}",
        "signature": {"input": "List<Int>", "output": "List<Int>"},
        "examples": examples,
        "integerConstants": _encoded(CONSTANTS),
        "particles": 8,
        "iterations": 4,
        "cloneProbability": 0,
        "essThreshold": 1,
        "seed": 17,
        "lossScale": 0.75,
        "costScale": 0.02,
        "lossCap": 1000,
        "maxCost": 30,
        "maxDepth": 12,
        "maxNodes": 191,
    }


def generate_suite(
    project_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
    custody_seal_path: Path,
    expected_custody_seal_sha256: str,
    secret_path: Path,
    output: Path,
) -> dict[str, object]:
    _validate_method_seal(
        project_root,
        protocol_path,
        expected_protocol_sha256,
        method_seal_path,
        expected_method_seal_sha256,
    )
    if _sha256_file(custody_seal_path) != _require_digest(
        expected_custody_seal_sha256, name="custody-seal SHA-256"
    ):
        raise FreshCalibrationError("custody-seal SHA-256 differs")
    custody = _read_object(custody_seal_path)
    secret = _read_secret(secret_path)
    expected_custody = {
        "schema": f"{STUDY_SCHEMA}-custody-seal-v1",
        "sealed_before_task_generation": True,
        "protocol_sha256": expected_protocol_sha256,
        "method_seal_sha256": expected_method_seal_sha256,
        "secret_commitment_sha256": secret_commitment(secret),
    }
    if custody != expected_custody:
        raise FreshCalibrationError("custody seal content differs")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite suite: {output}")
    public = output / "public"
    private = output / "private"
    public.mkdir(parents=True)
    private.mkdir(mode=0o700)
    manifest_tasks = []
    reveal_tasks = []
    for task_id in TASK_IDS:
        target = _target_draw(secret, task_id)
        document = _task_document(target, secret)
        task_path = public / f"{task_id}.json"
        task_bytes = canonical_bytes(document) + b"\n"
        task_path.write_bytes(task_bytes)
        nonce = hmac.new(
            secret, TARGET_COMMITMENT_DOMAIN + task_id.encode(), hashlib.sha256
        ).hexdigest()
        target_preimage = {
            "task_id": task_id,
            "task_sha256": _sha256_bytes(task_bytes),
            "predicate_dsl": target["predicate_dsl"],
            "mapper_dsl": target["mapper_dsl"],
            "nonce": nonce,
        }
        commitment = _sha256_bytes(canonical_bytes(target_preimage))
        manifest_tasks.append(
            {
                "task_id": task_id,
                "path": task_path.name,
                "task_sha256": _sha256_bytes(task_bytes),
                "target_commitment_sha256": commitment,
            }
        )
        reveal_tasks.append(
            {
                "task_id": task_id,
                "predicate_dsl": target["predicate_dsl"],
                "mapper_dsl": target["mapper_dsl"],
                "accepted_attempt": target["accepted_attempt"],
                "rejections": target["rejections"],
                "commitment_preimage": target_preimage,
            }
        )
    manifest = {
        "schema": f"{STUDY_SCHEMA}-public-suite-v1",
        "protocol_sha256": expected_protocol_sha256,
        "method_seal_sha256": expected_method_seal_sha256,
        "custody_seal_sha256": expected_custody_seal_sha256,
        "secret_commitment_sha256": secret_commitment(secret),
        "task_count": TASK_COUNT,
        "tasks": manifest_tasks,
    }
    reveal = {
        "schema": f"{STUDY_SCHEMA}-private-reveal-v1",
        "secret_hex": secret.hex(),
        "manifest_sha256": _sha256_bytes(canonical_bytes(manifest) + b"\n"),
        "tasks": reveal_tasks,
    }
    _write_json_exclusive(public / "manifest.json", manifest)
    _write_json_exclusive(private / "reveal.json", reveal, mode=0o600)
    return manifest


def _validate_suite(
    suite: Path,
    protocol_sha256: str,
    method_seal_sha256: str,
    custody_seal_sha256: str,
) -> tuple[list[TaskBinding], dict[str, Any]]:
    public = suite / "public"
    manifest_path = public / "manifest.json"
    manifest = _read_object(manifest_path)
    if manifest.get("schema") != f"{STUDY_SCHEMA}-public-suite-v1":
        raise FreshCalibrationError("public suite schema differs")
    if manifest.get("protocol_sha256") != protocol_sha256:
        raise FreshCalibrationError("suite protocol binding differs")
    if manifest.get("method_seal_sha256") != method_seal_sha256:
        raise FreshCalibrationError("suite method-seal binding differs")
    if manifest.get("custody_seal_sha256") != custody_seal_sha256:
        raise FreshCalibrationError("suite custody binding differs")
    records = manifest.get("tasks")
    if not isinstance(records, list) or len(records) != TASK_COUNT:
        raise FreshCalibrationError("suite task count differs")
    bindings = []
    for expected_id, record in zip(TASK_IDS, records, strict=True):
        if not isinstance(record, Mapping) or record.get("task_id") != expected_id:
            raise FreshCalibrationError("suite task order differs")
        path = public / cast(str, record["path"])
        if path.is_symlink() or not path.is_file():
            raise FreshCalibrationError("suite task is not a regular file")
        digest = _sha256_file(path)
        if record.get("task_sha256") != digest:
            raise FreshCalibrationError("suite task hash differs")
        bindings.append(TaskBinding(expected_id, path, digest))
    return bindings, manifest


def _quantile(values: Sequence[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    exact_errors = np.asarray(
        [cast(Mapping[str, float], record["error"])["exact_mass_signed"] for record in records],
        dtype=np.float64,
    )
    loss_errors = np.asarray(
        [
            cast(Mapping[str, float], record["error"])["target_mean_loss_signed"]
            for record in records
        ],
        dtype=np.float64,
    )
    max_weights = [
        float(cast(Mapping[str, object], record["diagnostics"])["maximum_normalized_weight"])
        for record in records
    ]
    relative_ess = [
        float(cast(Mapping[str, object], record["diagnostics"])["relative_importance_ess"])
        for record in records
    ]
    draws = [float(record["logical_proposal_draws"]) for record in records]
    return {
        "observations": len(records),
        "exact_mass": {
            "bias": float(exact_errors.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(exact_errors)))),
            "mae": float(np.mean(np.abs(exact_errors))),
        },
        "target_mean_loss": {
            "bias": float(loss_errors.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(loss_errors)))),
        },
        "relative_ess": {
            "mean": statistics.fmean(relative_ess),
            "q05": _quantile(relative_ess, 0.05),
        },
        "maximum_normalized_weight": {
            "mean": statistics.fmean(max_weights),
            "q95": _quantile(max_weights, 0.95),
            "maximum": max(max_weights),
            "fraction_above_0_25": sum(value > 0.25 for value in max_weights)
            / len(max_weights),
        },
        "logical_proposal_draws": {
            "mean": statistics.fmean(draws),
            "maximum": max(draws),
        },
    }


def _task_first_bootstrap(
    task_errors: np.ndarray, *, replicates: int, seed: int
) -> dict[str, object]:
    if task_errors.ndim != 2 or task_errors.shape != (TASK_COUNT, REPETITIONS):
        raise FreshCalibrationError("bootstrap error matrix shape differs")
    rng = np.random.Generator(np.random.PCG64(seed))
    biases = np.empty(replicates, dtype=np.float64)
    rmses = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        task_indices = rng.integers(0, TASK_COUNT, size=TASK_COUNT)
        repetition_indices = rng.integers(
            0, REPETITIONS, size=(TASK_COUNT, REPETITIONS)
        )
        sampled = task_errors[
            task_indices[:, None], repetition_indices
        ]
        biases[index] = float(sampled.mean())
        rmses[index] = float(np.sqrt(np.mean(np.square(sampled))))
    return {
        "replicates": replicates,
        "seed": seed,
        "rmse_upper_95": float(np.quantile(rmses, 0.95)),
        "rmse_interval_95": [
            float(np.quantile(rmses, 0.025)),
            float(np.quantile(rmses, 0.975)),
        ],
        "bias_interval_90": [
            float(np.quantile(biases, 0.05)),
            float(np.quantile(biases, 0.95)),
        ],
        "bias_interval_95": [
            float(np.quantile(biases, 0.025)),
            float(np.quantile(biases, 0.975)),
        ],
    }


def analyze(
    runs: Sequence[Mapping[str, object]],
    references: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    expected = TASK_COUNT * REPETITIONS * (
        len(PARTICLE_COUNTS) * 2 + 1
    )
    if len(runs) != expected:
        raise FreshCalibrationError(f"run count {len(runs)} differs from {expected}")
    pooled: dict[str, object] = {}
    per_task: dict[str, object] = {}
    for arm in FROZEN_ARMS:
        by_n = {}
        for particles in ARM_PARTICLES[arm.arm_id]:
            selected = [
                record
                for record in runs
                if record["arm_id"] == arm.arm_id and record["particles"] == particles
            ]
            if len(selected) != TASK_COUNT * REPETITIONS:
                raise FreshCalibrationError("pooled run cell is incomplete")
            by_n[str(particles)] = _summary(selected)
        pooled[arm.arm_id] = {"by_particle_count": by_n}
    for task_id in TASK_IDS:
        task_arms = {}
        for arm in FROZEN_ARMS:
            by_n = {}
            for particles in ARM_PARTICLES[arm.arm_id]:
                selected = [
                    record
                    for record in runs
                    if record["task_id"] == task_id
                    and record["arm_id"] == arm.arm_id
                    and record["particles"] == particles
                ]
                if len(selected) != REPETITIONS:
                    raise FreshCalibrationError("per-task run cell is incomplete")
                by_n[str(particles)] = _summary(selected)
            task_arms[arm.arm_id] = {"by_particle_count": by_n}
        per_task[task_id] = {"arms": task_arms}
    primary_errors = np.empty((TASK_COUNT, REPETITIONS), dtype=np.float64)
    for task_index, task_id in enumerate(TASK_IDS):
        selected = sorted(
            (
                record
                for record in runs
                if record["task_id"] == task_id
                and record["arm_id"] == PRIMARY_ARM_ID
                and record["particles"] == PRIMARY_PARTICLE_COUNT
            ),
            key=lambda record: int(record["repetition"]),
        )
        primary_errors[task_index] = [
            float(cast(Mapping[str, object], record["error"])["exact_mass_signed"])
            for record in selected
        ]
    bootstrap = _task_first_bootstrap(
        primary_errors, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED
    )
    primary_by_n = cast(
        Mapping[str, object], pooled[PRIMARY_ARM_ID]
    )["by_particle_count"]
    primary_256 = cast(Mapping[str, object], primary_by_n["256"])
    rmse_by_n = {
        particles: float(
            cast(
                Mapping[str, object],
                cast(Mapping[str, object], primary_by_n[str(particles)])["exact_mass"],
            )["rmse"]
        )
        for particles in PARTICLE_COUNTS
    }
    max_weight = cast(Mapping[str, object], primary_256["maximum_normalized_weight"])
    task_rmses = {
        task_id: float(
            cast(
                Mapping[str, object],
                cast(
                    Mapping[str, object],
                    cast(
                        Mapping[str, object],
                        cast(Mapping[str, object], per_task[task_id])["arms"],
                    )[PRIMARY_ARM_ID],
                )["by_particle_count"],
            )["256"]["exact_mass"]["rmse"]
        )
        for task_id in TASK_IDS
    }
    identity_error = max(
        float(
            cast(Mapping[str, object], reference["proposal_census"])[PRIMARY_ARM_ID][
                "importance_identity_maximum_absolute_error"
            ]
        )
        for reference in references.values()
    )
    checks = {
        "identity": identity_error <= IDENTITY_TOLERANCE,
        "rmse_upper_95_below_0_10": float(bootstrap["rmse_upper_95"]) < 0.10,
        "bias_interval_90_inside_equivalence": (
            float(cast(list[float], bootstrap["bias_interval_90"])[0]) > -0.03
            and float(cast(list[float], bootstrap["bias_interval_90"])[1]) < 0.03
        ),
        "all_task_rmse_below_0_20": max(task_rmses.values()) < 0.20,
        "rmse_nonincreasing": (
            rmse_by_n[512] <= rmse_by_n[256]
            and rmse_by_n[1024] <= rmse_by_n[512]
        ),
        "weight_tail": (
            float(max_weight["q95"]) < 0.05
            and float(max_weight["fraction_above_0_25"]) <= 0.01
        ),
    }
    return {
        "schema": f"{STUDY_SCHEMA}-analysis-v1",
        "status": "completed-fresh-calibration-analysis",
        "run_count": len(runs),
        "task_count": TASK_COUNT,
        "provider_calls": 0,
        "pooled": pooled,
        "per_task": per_task,
        "bootstrap": bootstrap,
        "primary_gate": {
            "checks": checks,
            "pass": all(checks.values()),
            "primary_arm_id": PRIMARY_ARM_ID,
            "primary_particle_count": PRIMARY_PARTICLE_COUNT,
            "maximum_identity_error": identity_error,
            "task_rmse": task_rmses,
            "rmse_by_particle_count": {str(k): v for k, v in rmse_by_n.items()},
        },
        "claim_boundary": (
            "fresh provider-free finite-support calibration confirmation; exact "
            "enumeration is reference-only; no LLM or large-DSL claim"
        ),
    }


def _seal_inventory(output: Path) -> dict[str, object]:
    if any(path.is_symlink() for path in output.rglob("*")):
        raise FreshCalibrationError("artifact tree contains a symlink")
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "inventory.json"
    )
    entries = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]
    inventory = {
        "schema": f"{STUDY_SCHEMA}-inventory-v1",
        "file_count_excluding_inventory": len(entries),
        "entries": entries,
        "entries_sha256": _sha256_bytes(canonical_bytes(entries)),
    }
    _write_json_exclusive(output / "inventory.json", inventory)
    return inventory


def validate_artifact(output: Path) -> dict[str, object]:
    inventory = _read_object(output / "inventory.json")
    entries = inventory.get("entries")
    if not isinstance(entries, list):
        raise FreshCalibrationError("inventory entries are malformed")
    actual = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "inventory.json"
    )
    expected = [cast(str, record["path"]) for record in entries]
    if actual != expected:
        raise FreshCalibrationError("artifact path inventory differs")
    for record in entries:
        path = output / cast(str, record["path"])
        if path.is_symlink() or _sha256_file(path) != record["sha256"]:
            raise FreshCalibrationError(f"artifact file differs: {path}")
    if inventory.get("entries_sha256") != _sha256_bytes(canonical_bytes(entries)):
        raise FreshCalibrationError("inventory entries digest differs")
    return inventory


def _summary_markdown(analysis: Mapping[str, object]) -> str:
    primary = cast(Mapping[str, object], analysis["primary_gate"])
    pooled = cast(Mapping[str, object], analysis["pooled"])
    primary_n = cast(
        Mapping[str, object], cast(Mapping[str, object], pooled[PRIMARY_ARM_ID])[
            "by_particle_count"
        ]
    )
    lines = [
        "# Fresh Calibrated Program Inference V2",
        "",
        f"Primary frozen gate: **{'PASS' if primary['pass'] else 'FAIL'}**.",
        "",
        "| N | Exact-mass RMSE | Bias | Mean relative ESS | q95 max weight |",
        "|---:|---:|---:|---:|---:|",
    ]
    for particles in PARTICLE_COUNTS:
        record = cast(Mapping[str, object], primary_n[str(particles)])
        exact = cast(Mapping[str, object], record["exact_mass"])
        ess = cast(Mapping[str, object], record["relative_ess"])
        weight = cast(Mapping[str, object], record["maximum_normalized_weight"])
        lines.append(
            f"| {particles} | {float(exact['rmse']):.6g} | "
            f"{float(exact['bias']):.6g} | {float(ess['mean']):.6g} | "
            f"{float(weight['q95']):.6g} |"
        )
    lines.extend(["", "All tasks and gate components are retained in analysis.json.", ""])
    return "\n".join(lines)


def run_study(
    project_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
    custody_seal_path: Path,
    expected_custody_seal_sha256: str,
    suite: Path,
    output: Path,
) -> dict[str, object]:
    protocol = _validate_protocol(project_root, protocol_path, expected_protocol_sha256)
    _validate_method_seal(
        project_root,
        protocol_path,
        expected_protocol_sha256,
        method_seal_path,
        expected_method_seal_sha256,
    )
    if _sha256_file(custody_seal_path) != expected_custody_seal_sha256:
        raise FreshCalibrationError("custody-seal SHA-256 differs")
    bindings, manifest = _validate_suite(
        suite,
        expected_protocol_sha256,
        expected_method_seal_sha256,
        expected_custody_seal_sha256,
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    staging = output.with_name(f".{output.name}.incomplete")
    if staging.exists():
        raise FileExistsError(f"refusing to reuse staging: {staging}")
    staging.mkdir(parents=True)
    try:
        _write_json_exclusive(staging / "protocol.json", protocol)
        _write_json_exclusive(staging / "public-suite-manifest.json", manifest)
        references: dict[str, Mapping[str, object]] = {}
        runs: list[Mapping[str, object]] = []
        for binding in bindings:
            task = TerminalTask(binding)
            bank = sampled_semantic_mode_bank(
                task,
                mode_count=MODE_COUNT,
                aliases_per_mode=ALIASES_PER_MODE,
                discovery_budget=DISCOVERY_BUDGET,
            )
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
        analysis = analyze(runs, references)
        _write_json_exclusive(staging / "runs.json", runs)
        _write_json_exclusive(staging / "analysis.json", analysis)
        (staging / "SUMMARY.md").write_text(
            _summary_markdown(analysis), encoding="utf-8"
        )
        _write_json_exclusive(
            staging / "study-metadata.json",
            {
                "schema": f"{STUDY_SCHEMA}-metadata-v1",
                "protocol_sha256": expected_protocol_sha256,
                "method_seal_sha256": expected_method_seal_sha256,
                "custody_seal_sha256": expected_custody_seal_sha256,
                "public_manifest_sha256": _sha256_file(suite / "public/manifest.json"),
                "provider_calls": 0,
                "private_reveal_read": False,
                "exact_support_materialized_for_reference": True,
                "runtime": _runtime_record(),
            },
        )
        _seal_inventory(staging)
        staging.rename(output)
        validate_artifact(output)
        return analysis
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


def verify_unblind(
    suite: Path,
    artifact: Path,
    secret_path: Path,
    output: Path,
) -> dict[str, object]:
    validate_artifact(artifact)
    secret = _read_secret(secret_path)
    manifest = _read_object(suite / "public/manifest.json")
    reveal = _read_object(suite / "private/reveal.json")
    if reveal.get("secret_hex") != secret.hex():
        raise FreshCalibrationError("reveal secret differs")
    if manifest.get("secret_commitment_sha256") != secret_commitment(secret):
        raise FreshCalibrationError("secret commitment differs")
    reveal_records = reveal.get("tasks")
    manifest_records = manifest.get("tasks")
    if not isinstance(reveal_records, list) or not isinstance(manifest_records, list):
        raise FreshCalibrationError("reveal or manifest records are malformed")
    for task_id, reveal_record, manifest_record in zip(
        TASK_IDS, reveal_records, manifest_records, strict=True
    ):
        if not isinstance(reveal_record, Mapping) or not isinstance(
            manifest_record, Mapping
        ):
            raise FreshCalibrationError("reveal task record is malformed")
        target = _target_draw(secret, task_id)
        document = _task_document(target, secret)
        task_bytes = canonical_bytes(document) + b"\n"
        if _sha256_bytes(task_bytes) != manifest_record["task_sha256"]:
            raise FreshCalibrationError("regenerated task differs")
        preimage = reveal_record["commitment_preimage"]
        if _sha256_bytes(canonical_bytes(preimage)) != manifest_record[
            "target_commitment_sha256"
        ]:
            raise FreshCalibrationError("target commitment differs")
        if target["predicate_dsl"] != reveal_record["predicate_dsl"] or target[
            "mapper_dsl"
        ] != reveal_record["mapper_dsl"]:
            raise FreshCalibrationError("regenerated target differs")
    result = {
        "schema": f"{STUDY_SCHEMA}-unblind-verification-v1",
        "passed": True,
        "task_count": TASK_COUNT,
        "secret_commitment_sha256": secret_commitment(secret),
        "artifact_inventory_sha256": _sha256_file(artifact / "inventory.json"),
        "analysis_sha256": _sha256_file(artifact / "analysis.json"),
    }
    _write_json_exclusive(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--project-root", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    seal = subparsers.add_parser("seal-method")
    seal.add_argument("--project-root", type=Path, required=True)
    seal.add_argument("--protocol", type=Path, required=True)
    seal.add_argument("--expected-protocol-sha256", required=True)
    seal.add_argument("--output", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-secret")
    prepare.add_argument("--project-root", type=Path, required=True)
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--expected-protocol-sha256", required=True)
    prepare.add_argument("--method-seal", type=Path, required=True)
    prepare.add_argument("--expected-method-seal-sha256", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    custody = subparsers.add_parser("seal-custody")
    custody.add_argument("--project-root", type=Path, required=True)
    custody.add_argument("--protocol", type=Path, required=True)
    custody.add_argument("--expected-protocol-sha256", required=True)
    custody.add_argument("--method-seal", type=Path, required=True)
    custody.add_argument("--expected-method-seal-sha256", required=True)
    custody.add_argument("--secret", type=Path, required=True)
    custody.add_argument("--output", type=Path, required=True)
    generate = subparsers.add_parser("generate-suite")
    for command in (generate,):
        command.add_argument("--project-root", type=Path, required=True)
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--expected-protocol-sha256", required=True)
        command.add_argument("--method-seal", type=Path, required=True)
        command.add_argument("--expected-method-seal-sha256", required=True)
    generate.add_argument("--custody-seal", type=Path, required=True)
    generate.add_argument("--expected-custody-seal-sha256", required=True)
    generate.add_argument("--secret", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--project-root", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--expected-protocol-sha256", required=True)
    run.add_argument("--method-seal", type=Path, required=True)
    run.add_argument("--expected-method-seal-sha256", required=True)
    run.add_argument("--custody-seal", type=Path, required=True)
    run.add_argument("--expected-custody-seal-sha256", required=True)
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-unblind")
    verify.add_argument("--suite", type=Path, required=True)
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--secret", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        print(freeze_protocol(args.project_root.resolve(), args.protocol.resolve()))
    elif args.command == "seal-method":
        print(
            seal_method(
                args.project_root.resolve(),
                args.protocol.resolve(),
                args.expected_protocol_sha256,
                args.output.resolve(),
            )
        )
    elif args.command == "prepare-secret":
        print(
            prepare_secret(
                args.project_root.resolve(),
                args.protocol.resolve(),
                args.expected_protocol_sha256,
                args.method_seal.resolve(),
                args.expected_method_seal_sha256,
                args.output.resolve(),
            )
        )
    elif args.command == "seal-custody":
        print(
            seal_custody(
                args.project_root.resolve(),
                args.protocol.resolve(),
                args.expected_protocol_sha256,
                args.method_seal.resolve(),
                args.expected_method_seal_sha256,
                args.secret.resolve(),
                args.output.resolve(),
            )
        )
    elif args.command == "generate-suite":
        manifest = generate_suite(
            args.project_root.resolve(),
            args.protocol.resolve(),
            args.expected_protocol_sha256,
            args.method_seal.resolve(),
            args.expected_method_seal_sha256,
            args.custody_seal.resolve(),
            args.expected_custody_seal_sha256,
            args.secret.resolve(),
            args.output.resolve(),
        )
        print(_sha256_bytes(canonical_bytes(manifest) + b"\n"))
    elif args.command == "run":
        result = run_study(
            args.project_root.resolve(),
            args.protocol.resolve(),
            args.expected_protocol_sha256,
            args.method_seal.resolve(),
            args.expected_method_seal_sha256,
            args.custody_seal.resolve(),
            args.expected_custody_seal_sha256,
            args.suite.resolve(),
            args.output.resolve(),
        )
        print(json.dumps(result["primary_gate"], sort_keys=True))
    elif args.command == "validate":
        print(json.dumps(validate_artifact(args.output.resolve()), sort_keys=True))
    elif args.command == "verify-unblind":
        print(
            json.dumps(
                verify_unblind(
                    args.suite.resolve(),
                    args.artifact.resolve(),
                    args.secret.resolve(),
                    args.output.resolve(),
                ),
                sort_keys=True,
            )
        )
    else:
        raise AssertionError("unreachable command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

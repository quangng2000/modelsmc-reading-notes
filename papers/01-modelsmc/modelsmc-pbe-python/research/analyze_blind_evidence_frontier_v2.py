"""Fail-closed public analysis for the 12-task blind evidence-frontier study.

The analyzer intentionally has no private-reveal argument.  It verifies the
method freeze, the pre-provider task seal, every public task/run binding and
provider inventory before computing the preregistered trial-index-matched
randomization test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any, cast

SCHEMA = "blind-evidence-frontier-confirmation-analysis-v2"
STUDY_SCHEMA = "blind-evidence-frontier-confirmation-v2"
SEAL_SCHEMA = "blind-v2-provider-call-seal-v1"
MANIFEST_SCHEMA = "blinded-filter-map-suite-v2"
RESULT_SCHEMA = "iterative-typed-llm-beam-experiment-v2"
TASK_IDS = tuple(f"blind-v2-{index:02d}" for index in range(1, 13))
CHECKPOINTS = (21, 29)
TRIALS = 10_000
PRACTICAL_THRESHOLD = 6
MODEL_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "commitment_nonce",
        "mapper_family",
        "nonce",
        "private_seed",
        "seed_hex",
        "target",
        "target_mapper",
        "target_mapper_dsl",
        "target_predicate",
        "target_predicate_dsl",
    }
)


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


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


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


def _close(actual: float, expected: float, *, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _safe_repo_path(repo_root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"{name} must remain inside the repository")
    root = repo_root.resolve()
    result = (root / relative).resolve()
    if root not in result.parents:
        raise ValueError(f"{name} escaped the repository")
    return result


def _reject_private_keys(value: object, *, path: str = "public") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"private key {key!r} appears at {path}")
            _reject_private_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_private_keys(child, path=f"{path}[{index}]")


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("successes and trials are inconsistent")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _validate_source_bundle(
    repo_root: Path,
    raw_binding: object,
    *,
    name: str,
) -> str:
    binding = _object(raw_binding, name=name)
    _expect(
        binding.get("scheme"),
        "SHA256(domain || NUL || repeated(path || NUL || exact_file_bytes || NUL))",
        name=f"{name}.scheme",
    )
    domain = binding.get("domain")
    if not isinstance(domain, str) or not domain:
        raise ValueError(f"{name}.domain must be a nonempty string")
    paths = _array(binding.get("paths_in_order"), name=f"{name}.paths_in_order")
    if not paths or len(paths) != len(set(map(str, paths))):
        raise ValueError(f"{name}.paths_in_order must be nonempty and unique")
    source_hashes = _object(binding.get("source_sha256"), name=f"{name}.source_sha256")
    _expect(set(source_hashes), set(paths), name=f"{name}.source hash paths")
    material = bytearray(domain.encode() + b"\0")
    for index, relative in enumerate(paths):
        path = _safe_repo_path(repo_root, relative, name=f"{name}.paths_in_order[{index}]")
        payload = path.read_bytes()
        digest = sha256_bytes(payload)
        _expect(source_hashes.get(relative), digest, name=f"{name}.{relative}.sha256")
        material.extend(cast(str, relative).encode() + b"\0" + payload + b"\0")
    bundle_sha256 = sha256_bytes(bytes(material))
    _expect(binding.get("bundle_sha256"), bundle_sha256, name=f"{name}.bundle_sha256")
    return bundle_sha256


def _expected_seed_record(task_id: str, index: int) -> dict[str, int | str]:
    base = 301_000 + index * 1_000
    return {
        "task_id": task_id,
        "provider_seed": base,
        "tie_seed": base + 100,
        "matched_random_seed": base + 200,
    }


def _validate_study(repo_root: Path, study_path: Path) -> tuple[dict[str, Any], str]:
    study = _read_object(study_path)
    _expect(study.get("schema"), STUDY_SCHEMA, name="study schema")
    _expect(
        study.get("protocol_status"),
        "method-frozen-before-task-generation",
        name="study protocol status",
    )
    freeze = _object(study.get("freeze_requirements"), name="freeze_requirements")
    _expect(
        freeze.get("protocol_sha256_binding"),
        "external-provider-seal-and-run-protocol-records",
        name="protocol SHA-256 binding",
    )
    for field in ("generator", "harness", "analysis", "runner"):
        binding = _object(freeze.get(field), name=f"freeze.{field}")
        path = _safe_repo_path(repo_root, binding.get("path"), name=f"freeze.{field}.path")
        digest = sha256_file(path)
        _expect(binding.get("sha256"), digest, name=f"frozen {field} SHA-256")
        if field == "analysis":
            _expect(digest, sha256_file(Path(__file__)), name="executing analyzer SHA-256")
    prompt_sha256 = _validate_source_bundle(
        repo_root,
        freeze.get("prompt_template_binding"),
        name="prompt_template_binding",
    )
    _expect(
        freeze.get("prompt_template_sha256"),
        prompt_sha256,
        name="prompt_template_sha256",
    )
    _validate_source_bundle(repo_root, freeze.get("test_bundle"), name="test_bundle")
    _digest(
        freeze.get("task_secret_commitment_sha256"),
        name="task secret commitment",
    )
    seed_records = _array(freeze.get("run_seeds"), name="freeze.run_seeds")
    expected = [
        _expected_seed_record(task_id, index)
        for index, task_id in enumerate(TASK_IDS)
    ]
    _expect(seed_records, expected, name="frozen run order and seeds")
    model = _object(study.get("model"), name="study.model")
    for field, expected_value in {
        "served_name": "gpt-oss-120b",
        "revision": MODEL_REVISION,
        "reasoning_effort": "low",
        "temperature": 0,
        "max_tokens": 1600,
        "timeout_seconds": 420,
        "max_concurrency": 2,
        "provider_retry_policy": "none",
    }.items():
        _expect(model.get(field), expected_value, name=f"study.model.{field}")
    search = _object(study.get("search"), name="study.search")
    for field, expected_value in {
        "selection_policy": "evidence-frontier",
        "branching_factor": 4,
        "beam_width": 2,
        "maximum_rounds": 4,
        "start_seed": 17,
        "singleton_evidence": True,
        "stall_policy": "alternate-hole",
        "maximum_proposal_slots": 29,
        "maximum_provider_calls": 7,
    }.items():
        _expect(search.get(field), expected_value, name=f"study.search.{field}")
    return study, sha256_file(study_path)


def _index_records(
    records: object,
    *,
    name: str,
) -> dict[str, dict[str, Any]]:
    values = _array(records, name=name)
    if len(values) != len(TASK_IDS):
        raise ValueError(f"{name} must contain exactly 12 records")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(values):
        record = _object(raw, name=f"{name}[{index}]")
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or task_id in result:
            raise ValueError(f"{name} has an invalid or repeated task ID")
        result[task_id] = record
    _expect(tuple(result), TASK_IDS, name=f"{name} order")
    return result


def _validate_manifest_and_seal(
    repo_root: Path,
    study: Mapping[str, Any],
    study_sha256: str,
    provider_seal_path: Path,
    manifest_path: Path,
    method_seal_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str]:
    manifest = _read_object(manifest_path)
    _reject_private_keys(manifest)
    _expect(manifest.get("schema"), MANIFEST_SCHEMA, name="public manifest schema")
    _expect(manifest.get("task_count"), 12, name="public manifest task_count")
    manifest_records = _index_records(manifest.get("tasks"), name="public manifest tasks")
    secret = _digest(manifest.get("seed_commitment_sha256"), name="manifest secret commitment")

    seal = _read_object(provider_seal_path)
    _expect(seal.get("schema"), SEAL_SCHEMA, name="provider-call seal schema")
    _expect(seal.get("sealed_before_provider_calls"), True, name="pre-provider seal flag")
    _expect(seal.get("study_protocol_sha256"), study_sha256, name="sealed study SHA-256")
    _expect(seal.get("method_seal_sha256"), method_seal_sha256, name="sealed method SHA-256")
    endpoint = _object(seal.get("endpoint_health_record"), name="endpoint health binding")
    endpoint_path = _safe_repo_path(
        repo_root, endpoint.get("path"), name="endpoint health record path"
    )
    _expect(
        endpoint.get("sha256"),
        sha256_file(endpoint_path),
        name="endpoint health record SHA-256",
    )
    freeze = _object(study.get("freeze_requirements"), name="freeze requirements")
    frozen_secret = _digest(
        freeze.get("task_secret_commitment_sha256"), name="frozen secret commitment"
    )
    _expect(secret, frozen_secret, name="manifest/frozen secret commitment")
    _expect(seal.get("task_secret_commitment_sha256"), secret, name="sealed secret commitment")
    _digest(seal.get("hidden_target_manifest_sha256"), name="hidden target manifest SHA-256")
    manifest_binding = _object(seal.get("public_manifest"), name="sealed public manifest")
    bound_path = _safe_repo_path(
        repo_root, manifest_binding.get("path"), name="sealed public manifest path"
    )
    _expect(bound_path, manifest_path.resolve(), name="public manifest path argument")
    manifest_sha256 = sha256_file(manifest_path)
    _expect(manifest_binding.get("sha256"), manifest_sha256, name="public manifest SHA-256")

    seal_records = _index_records(seal.get("tasks"), name="provider-call seal tasks")
    for index, task_id in enumerate(TASK_IDS):
        manifest_record = manifest_records[task_id]
        seal_record = seal_records[task_id]
        for field in (
            "task_id",
            "task_file_sha256",
            "task_canonical_sha256",
            "target_commitment_sha256",
        ):
            _expect(seal_record.get(field), manifest_record.get(field), name=f"{task_id}.{field}")
        sealed_task_path = _safe_repo_path(
            repo_root, seal_record.get("path"), name=f"{task_id}.sealed task path"
        )
        expected_task_path = (manifest_path.parent / cast(str, manifest_record["path"])).resolve()
        _expect(sealed_task_path, expected_task_path, name=f"{task_id}.sealed task path")
        _expect(sealed_task_path.name, manifest_record.get("path"), name=f"{task_id}.task basename")
        expected_seeds = _expected_seed_record(task_id, index)
        for field in ("provider_seed", "tie_seed", "matched_random_seed"):
            _expect(seal_record.get(field), expected_seeds[field], name=f"{task_id}.{field}")
        task_path = manifest_path.parent / cast(str, manifest_record["path"])
        if task_path.parent.resolve() != manifest_path.parent.resolve():
            raise ValueError(f"unsafe public task path for {task_id}")
        task = _read_object(task_path)
        _reject_private_keys(task, path=f"public task {task_id}")
        file_digest = _digest(
            manifest_record.get("task_file_sha256"), name=f"{task_id}.task_file_sha256"
        )
        canonical_digest = _digest(
            manifest_record.get("task_canonical_sha256"),
            name=f"{task_id}.task_canonical_sha256",
        )
        _digest(
            manifest_record.get("target_commitment_sha256"),
            name=f"{task_id}.target_commitment_sha256",
        )
        _expect(sha256_file(task_path), file_digest, name=f"{task_id}.task file SHA-256")
        _expect(
            sha256_bytes(canonical_bytes(task)),
            canonical_digest,
            name=f"{task_id}.task canonical SHA-256",
        )
    return manifest_records, seal_records, manifest_sha256


def _validate_method_seal(
    repo_root: Path,
    method_seal_path: Path,
    expected_sha256: str,
    study_path: Path,
    study_sha256: str,
) -> str:
    actual_sha256 = sha256_file(method_seal_path)
    _expect(actual_sha256, expected_sha256, name="externally expected method-seal SHA-256")
    seal = _read_object(method_seal_path)
    _expect(seal.get("schema"), "blind-evidence-frontier-method-seal-v2", name="method seal schema")
    _expect(seal.get("sealed_before_task_generation"), True, name="method pre-generation seal")
    protocol = _object(seal.get("protocol"), name="method-seal protocol binding")
    bound_path = _safe_repo_path(
        repo_root, protocol.get("path"), name="method-seal protocol path"
    )
    _expect(bound_path, study_path.resolve(), name="method-seal protocol path")
    _expect(protocol.get("sha256"), study_sha256, name="method-seal protocol SHA-256")
    return actual_sha256


def _integer_atom(value: object, *, name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as error:
            raise ValueError(f"{name} must encode an integer") from error
    raise ValueError(f"{name} must encode an integer")


def _singleton_facts(
    task: Mapping[str, Any], *, task_id: str
) -> tuple[tuple[int, ...], dict[int, bool], dict[int, int]]:
    examples = _array(task.get("examples"), name=f"{task_id}.examples")
    observed: set[int] = set()
    predicates: dict[int, bool] = {}
    mappers: dict[int, int] = {}
    for index, raw in enumerate(examples):
        example = _object(raw, name=f"{task_id}.examples[{index}]")
        inputs = [
            _integer_atom(item, name=f"{task_id}.examples[{index}].input")
            for item in _array(example.get("input"), name=f"{task_id}.input")
        ]
        outputs = _array(example.get("output"), name=f"{task_id}.output")
        observed.update(inputs)
        if len(inputs) != 1 or len(outputs) > 1:
            continue
        item = inputs[0]
        keep = bool(outputs)
        if item in predicates and predicates[item] != keep:
            raise ValueError(f"{task_id} has conflicting singleton predicate facts")
        predicates[item] = keep
        if keep:
            expected = _integer_atom(outputs[0], name=f"{task_id}.singleton output")
            if item in mappers and mappers[item] != expected:
                raise ValueError(f"{task_id} has conflicting singleton mapper facts")
            mappers[item] = expected
    if set(predicates) != observed:
        raise ValueError(f"{task_id} lacks singleton evidence for its observed domain")
    return tuple(sorted(observed)), predicates, mappers


def _state_trace(
    raw_state: object,
    *,
    task_id: str,
    round_number: int,
    position: int,
    observed: tuple[int, ...],
    predicates: Mapping[int, bool],
    mappers: Mapping[int, int],
) -> dict[str, object]:
    state = _object(raw_state, name=f"{task_id}.round-{round_number}.state-{position}")
    semantics = _object(state.get("finite_component_semantics"), name="state semantics")
    keep_mask = _array(semantics.get("predicate_keep_mask"), name="predicate keep mask")
    mapper_values = _array(semantics.get("mapper_values"), name="mapper values")
    if len(keep_mask) != len(observed) or len(mapper_values) != len(observed):
        raise ValueError(f"{task_id} state signature length mismatch")
    index = {item: offset for offset, item in enumerate(observed)}
    predicate_violations = sum(
        keep_mask[index[item]] is not expected for item, expected in predicates.items()
    )
    mapper_violations = sum(
        _integer_atom(mapper_values[index[item]], name="mapper signature value") != expected
        for item, expected in mappers.items()
    )
    stored = _object(state.get("singleton_constraint_violations"), name="stored violations")
    _expect(stored.get("predicate"), predicate_violations, name="predicate violations")
    _expect(stored.get("mapper"), mapper_violations, name="mapper violations")
    score = _object(state.get("score"), name="state score")
    loss = _number(score.get("total_loss"), name="state total loss")
    exact = score.get("exact_program") is True
    if exact and (loss != 0.0 or predicate_violations or mapper_violations):
        raise ValueError(f"{task_id} exact state has inconsistent public evidence")
    return {
        "round": round_number,
        "selected_position": position,
        "selection_role": "loss-anchor" if position == 0 else "evidence-frontier-or-fill",
        "state_id": state.get("state_id"),
        "predicate": state.get("predicate"),
        "mapper": state.get("mapper"),
        "predicate_violations": predicate_violations,
        "mapper_violations": mapper_violations,
        "loss": loss,
        "exact": exact,
    }


def _validate_run_provider_seal(
    run_dir: Path, result: Mapping[str, Any], *, task_id: str
) -> tuple[str, dict[str, int]]:
    seal = _read_object(run_dir / "provider-seal.json")
    records = _array(seal.get("records"), name=f"{task_id}.provider records")
    normalized: list[dict[str, object]] = []
    listed: set[str] = set()
    previous = ""
    tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for index, raw in enumerate(records):
        record = _object(raw, name=f"{task_id}.provider.records[{index}]")
        relative_value = record.get("path")
        if not isinstance(relative_value, str):
            raise ValueError(f"{task_id} provider path must be a string")
        relative = Path(relative_value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "provider"
            or relative_value <= previous
        ):
            raise ValueError(f"{task_id} provider inventory path/order is invalid")
        previous = relative_value
        payload = (run_dir / relative).read_bytes()
        size = _integer(record.get("bytes"), name=f"{task_id}.{relative_value}.bytes")
        digest = _digest(record.get("sha256"), name=f"{task_id}.{relative_value}.sha256")
        _expect(len(payload), size, name=f"{task_id}.{relative_value}.bytes")
        _expect(sha256_bytes(payload), digest, name=f"{task_id}.{relative_value}.sha256")
        listed.add(relative.as_posix())
        normalized.append({"path": relative_value, "bytes": size, "sha256": digest})
        if relative.name == "response.json":
            response = json.loads(payload)
            if isinstance(response, dict) and isinstance(response.get("usage"), dict):
                usage = cast(dict[str, object], response["usage"])
                for field in tokens:
                    tokens[field] += _integer(usage.get(field), name=f"{task_id}.{field}")
    provider_root = run_dir / "provider"
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in provider_root.rglob("*")
        if path.is_file()
    }
    _expect(listed, actual, name=f"{task_id}.provider seal completeness")
    inventory_sha256 = sha256_bytes(canonical_bytes(normalized))
    _expect(seal.get("inventory_sha256"), inventory_sha256, name="provider inventory SHA-256")
    _expect(
        result.get("provider_inventory_sha256"),
        inventory_sha256,
        name="result provider inventory SHA-256",
    )
    return inventory_sha256, tokens


def _command_options(command: object, *, task_id: str) -> tuple[str, dict[str, str | bool]]:
    values = _array(command, name=f"{task_id}.preflight.command")
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"{task_id}.preflight.command must contain only strings")
    argv = cast(list[str], values)
    if len(argv) < 4 or not argv[0] or argv[1:3] != ["-m", "research.iterative_beam_experiment"]:
        raise ValueError(f"{task_id}.preflight.command has the wrong executable/module")
    options: dict[str, str | bool] = {}
    index = 3
    while index < len(argv):
        flag = argv[index]
        if not flag.startswith("--") or flag in options:
            raise ValueError(f"{task_id}.preflight.command has an invalid/repeated flag")
        if flag == "--singleton-evidence":
            options[flag] = True
            index += 1
        else:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"{task_id}.preflight.command lacks a value for {flag}")
            options[flag] = argv[index + 1]
            index += 2
    return argv[0], options


def _validate_preflight(
    *,
    run_dir: Path,
    task_id: str,
    task_path: Path,
    study_path: Path,
    study_sha256: str,
    method_seal_sha256: str,
    provider_call_seal_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    harness_sha256: str,
    seal_record: Mapping[str, Any],
) -> str:
    preflight_path = run_dir.with_name(run_dir.name + ".preflight.json")
    record = _read_object(preflight_path)
    _expect(record.get("schema"), "blind-v2-preflight-invocation-v1", name="preflight schema")
    _expect(
        record.get("status"),
        "validated-before-provider-call",
        name="preflight status",
    )
    expected_bindings = {
        "task_id": task_id,
        "study_protocol_sha256": study_sha256,
        "method_seal_sha256": method_seal_sha256,
        "provider_call_seal_sha256": provider_call_seal_sha256,
        "public_manifest_sha256": manifest_sha256,
        "task_file_sha256": sha256_file(task_path),
        "harness_sha256": harness_sha256,
        "preflight_record_path": str(preflight_path.resolve()),
    }
    for field, expected in expected_bindings.items():
        _expect(record.get(field), expected, name=f"{task_id}.preflight.{field}")
    _, options = _command_options(record.get("command"), task_id=task_id)
    if "--protocol-mode" in options:
        raise ValueError(f"{task_id}.preflight.command must not use --protocol-mode")
    expected_options: dict[str, str | bool] = {
        "--task": str(task_path.resolve()),
        "--output": str(run_dir.resolve()),
        "--blind-manifest": str(manifest_path.resolve()),
        "--study-protocol": str(study_path.resolve()),
        "--model": "gpt-oss-120b",
        "--reasoning-effort": "low",
        "--temperature": "0",
        "--max-tokens": "1600",
        "--timeout-seconds": "420",
        "--max-concurrency": "2",
        "--rounds": "4",
        "--beam-width": "2",
        "--branching-factor": "4",
        "--start-seed": "17",
        "--provider-seed": str(seal_record["provider_seed"]),
        "--tie-seed": str(seal_record["tie_seed"]),
        "--random-baseline-seed": str(seal_record["matched_random_seed"]),
        "--random-baseline-trials": "10000",
        "--selection-policy": "evidence-frontier",
        "--stall-policy": "alternate-hole",
        "--singleton-evidence": True,
        "--primary-checkpoint-round": "4",
    }
    if set(options) != {*expected_options, "--base-url"}:
        raise ValueError(f"{task_id}.preflight.command flags differ from the frozen launcher")
    for flag, expected in expected_options.items():
        _expect(options.get(flag), expected, name=f"{task_id}.preflight.command {flag}")
    base_url = options["--base-url"]
    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        raise ValueError(f"{task_id}.preflight.command has an invalid base URL")
    return sha256_file(preflight_path)


def _validate_run(
    *,
    repo_root: Path,
    study_path: Path,
    study_sha256: str,
    method_seal_sha256: str,
    provider_call_seal_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    task_id: str,
    task_record: Mapping[str, Any],
    seal_record: Mapping[str, Any],
    run_dir: Path,
) -> tuple[dict[str, object], dict[str, list[bool]]]:
    if (run_dir / "failure.json").exists():
        raise ValueError(f"{task_id} has an infrastructure failure; confirmatory attempt aborted")
    task_path = manifest_path.parent / cast(str, task_record["path"])
    task = _read_object(task_path)
    observed, predicates, mappers = _singleton_facts(task, task_id=task_id)
    protocol = _read_object(run_dir / "protocol.json")
    result = _read_object(run_dir / "result.json")
    _expect(result.get("schema"), RESULT_SCHEMA, name=f"{task_id}.result schema")
    _expect(result.get("protocol"), protocol, name=f"{task_id}.stored/result protocol")
    expected_protocol = {
        "protocol_mode": None,
        "blind_task_id": None,
        "task_sha256": task_record["task_file_sha256"],
        "study_protocol_sha256": study_sha256,
        "blind_manifest_sha256": manifest_sha256,
        "model": "gpt-oss-120b",
        "model_revision": MODEL_REVISION,
        "reasoning_effort": "low",
        "temperature": 0.0,
        "rounds": 4,
        "beam_width": 2,
        "branching_factor": 4,
        "maximum_proposal_slot_budget": 29,
        "selection_policy": "evidence-frontier",
        "singleton_evidence": True,
        "stall_policy": "alternate-hole",
        "start_seed": 17,
        "random_baseline_trials": TRIALS,
        "max_provider_concurrency": 2,
        "maximum_provider_calls": 7,
        "primary_checkpoint_round": 4,
        "primary_checkpoint_slot": 29,
        "provider_seed": seal_record["provider_seed"],
        "tie_seed": seal_record["tie_seed"],
        "random_baseline_seed": seal_record["matched_random_seed"],
    }
    for field, expected in expected_protocol.items():
        _expect(protocol.get(field), expected, name=f"{task_id}.protocol.{field}")
    _expect(
        protocol.get("proposal_slot_checkpoints"),
        [1, 5, 13, 21, 29],
        name=f"{task_id}.protocol.checkpoints",
    )
    harness = _safe_repo_path(repo_root, "research/iterative_beam_experiment.py", name="harness")
    harness_sha256 = sha256_file(harness)
    _expect(protocol.get("harness_sha256"), harness_sha256, name=f"{task_id}.harness")
    preflight_sha256 = _validate_preflight(
        run_dir=run_dir,
        task_id=task_id,
        task_path=task_path,
        study_path=study_path,
        study_sha256=study_sha256,
        method_seal_sha256=method_seal_sha256,
        provider_call_seal_sha256=provider_call_seal_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        harness_sha256=harness_sha256,
        seal_record=seal_record,
    )
    provider_inventory, tokens = _validate_run_provider_seal(run_dir, result, task_id=task_id)

    metrics = _object(result.get("llm_beam_metrics"), name=f"{task_id}.llm metrics")
    first_raw = metrics.get("first_exact_proposal_slot")
    first_exact = None if first_raw is None else _integer(first_raw, name=f"{task_id}.first exact")
    if first_exact is not None and not 1 <= first_exact <= 29:
        raise ValueError(f"{task_id} first exact slot is outside [1, 29]")
    success_map = _object(metrics.get("success_by_proposal_slot_checkpoint"), name="LLM success")
    loss_map = _object(metrics.get("best_loss_by_proposal_slot_checkpoint"), name="LLM losses")
    exact_by_slot: dict[str, bool] = {}
    best_loss_by_slot: dict[str, float] = {}
    for checkpoint in CHECKPOINTS:
        key = str(checkpoint)
        expected = first_exact is not None and first_exact <= checkpoint
        _expect(success_map.get(key), expected, name=f"{task_id}.llm success[{key}]")
        loss = _number(loss_map.get(key), name=f"{task_id}.llm loss[{key}]")
        if loss < 0 or (loss == 0.0) != expected:
            raise ValueError(f"{task_id}.llm loss/success inconsistent at {key}")
        exact_by_slot[key] = expected
        best_loss_by_slot[key] = loss
    _expect(metrics.get("success"), first_exact is not None, name=f"{task_id}.llm success")

    completed = _integer(metrics.get("rounds_completed"), name=f"{task_id}.rounds completed")
    if not 1 <= completed <= 4:
        raise ValueError(f"{task_id}.rounds_completed is outside [1, 4]")
    expected_rounds = [run_dir / f"round-{number:02d}.json" for number in range(1, completed + 1)]
    _expect(sorted(run_dir.glob("round-*.json")), expected_rounds, name=f"{task_id}.round files")
    traces: list[dict[str, object]] = []
    round_hashes: dict[str, str] = {}
    for round_number, round_path in enumerate(expected_rounds, start=1):
        round_hashes[str(round_number)] = sha256_file(round_path)
        round_record = _read_object(round_path)
        _expect(round_record.get("round"), round_number, name=f"{task_id}.round number")
        selected = _array(round_record.get("selected_beam"), name=f"{task_id}.selected beam")
        if not 1 <= len(selected) <= 2:
            raise ValueError(f"{task_id}.round-{round_number} beam width is invalid")
        traces.extend(
            _state_trace(
                state,
                task_id=task_id,
                round_number=round_number,
                position=position,
                observed=observed,
                predicates=predicates,
                mappers=mappers,
            )
            for position, state in enumerate(selected)
        )

    trials_path = run_dir / "matched-random-trials.json"
    raw_trials = json.loads(trials_path.read_text(encoding="utf-8"))
    trials = _array(raw_trials, name=f"{task_id}.matched random trials")
    if len(trials) != TRIALS:
        raise ValueError(f"{task_id} must contain exactly {TRIALS} matched-random trials")
    indicators = {str(checkpoint): [] for checkpoint in CHECKPOINTS}
    for index, raw_trial in enumerate(trials):
        trial = _object(raw_trial, name=f"{task_id}.random trial[{index}]")
        first = trial.get("first_exact_execution")
        first_slot = (
            None
            if first is None
            else _integer(first, name=f"{task_id}.random first exact")
        )
        if first_slot is not None and not 1 <= first_slot <= 29:
            raise ValueError(f"{task_id}.random trial first exact is outside [1, 29]")
        _expect(trial.get("success"), first_slot is not None, name=f"{task_id}.random success")
        for checkpoint in CHECKPOINTS:
            indicators[str(checkpoint)].append(first_slot is not None and first_slot <= checkpoint)
    random_metrics = _object(result.get("matched_random_beam"), name=f"{task_id}.random metrics")
    _expect(random_metrics.get("trials"), TRIALS, name=f"{task_id}.random trial count")
    stored_successes = _object(
        random_metrics.get("successes_by_proposal_slot_checkpoint"), name="random successes"
    )
    random_checkpoints: dict[str, dict[str, object]] = {}
    for checkpoint in CHECKPOINTS:
        key = str(checkpoint)
        successes = sum(indicators[key])
        _expect(stored_successes.get(key), successes, name=f"{task_id}.random successes[{key}]")
        random_checkpoints[key] = {
            "successes": successes,
            "trials": TRIALS,
            "success_rate": successes / TRIALS,
            "success_rate_wilson_95": wilson_interval(successes, TRIALS),
            "best_loss_summary": _object(
                random_metrics.get("best_loss_summary_by_proposal_slot_checkpoint"),
                name="random loss summaries",
            ).get(key),
        }
    task_row = {
        "task_id": task_id,
        "public_task_sha256": task_record["task_file_sha256"],
        "run_directory": str(run_dir),
        "run_protocol_sha256": sha256_file(run_dir / "protocol.json"),
        "result_sha256": sha256_file(run_dir / "result.json"),
        "matched_random_trials_sha256": sha256_file(trials_path),
        "round_file_sha256": round_hashes,
        "provider_inventory_sha256": provider_inventory,
        "preflight_invocation_sha256": preflight_sha256,
        "first_exact_proposal_slot": first_exact,
        "exact_by_slot": exact_by_slot,
        "best_loss_by_slot": best_loss_by_slot,
        "matched_random": random_checkpoints,
        "selected_state_violation_trajectory": traces,
        "random_evidence_violation_distributions": {
            "available": False,
            "reason": "frozen harness stores random losses and exact slots, not random states",
        },
        "effort": {
            field: metrics.get(field)
            for field in (
                "proposal_slots_consumed",
                "logical_candidate_evaluations",
                "physical_scorer_calls",
                "score_cache_hits",
                "unique_joint_semantic_cells_evaluated",
                "semantic_redundancy_rate",
                "provider_calls_attempted",
                "provider_calls_valid",
                "provider_calls_invalid",
            )
        }
        | tokens,
        "timing": result.get("timing"),
    }
    return task_row, indicators


def analyze_blind_evidence_frontier_v2(
    repo_root: Path,
    study_protocol_path: Path,
    method_seal_path: Path,
    provider_call_seal_path: Path,
    public_manifest_path: Path,
    run_dirs: Mapping[str, Path],
    *,
    expected_method_seal_sha256: str,
    expected_provider_call_seal_sha256: str,
) -> dict[str, object]:
    """Validate all public inputs and compute the preregistered primary test."""

    if set(run_dirs) != set(TASK_IDS):
        raise ValueError(f"run directories must be keyed by exactly {TASK_IDS}")
    repo_root = repo_root.resolve()
    study_path = study_protocol_path.resolve()
    manifest_path = public_manifest_path.resolve()
    study, study_sha256 = _validate_study(repo_root, study_path)
    method_seal_sha256 = _validate_method_seal(
        repo_root,
        method_seal_path.resolve(),
        _digest(expected_method_seal_sha256, name="expected method-seal SHA-256"),
        study_path,
        study_sha256,
    )
    provider_call_seal_path = provider_call_seal_path.resolve()
    provider_call_seal_sha256 = sha256_file(provider_call_seal_path)
    _expect(
        provider_call_seal_sha256,
        _digest(
            expected_provider_call_seal_sha256,
            name="expected provider-call-seal SHA-256",
        ),
        name="externally expected provider-call-seal SHA-256",
    )
    manifest_records, seal_records, manifest_sha256 = _validate_manifest_and_seal(
        repo_root,
        study,
        study_sha256,
        provider_call_seal_path,
        manifest_path,
        method_seal_sha256,
    )
    tasks: list[dict[str, object]] = []
    trial_indicators: dict[str, list[list[bool]]] = {str(value): [] for value in CHECKPOINTS}
    for task_id in TASK_IDS:
        run_dir = run_dirs[task_id].resolve()
        if not run_dir.is_dir():
            raise ValueError(f"run directory does not exist for {task_id}: {run_dir}")
        task, indicators = _validate_run(
            repo_root=repo_root,
            study_path=study_path,
            study_sha256=study_sha256,
            method_seal_sha256=method_seal_sha256,
            provider_call_seal_sha256=provider_call_seal_sha256,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            task_id=task_id,
            task_record=manifest_records[task_id],
            seal_record=seal_records[task_id],
            run_dir=run_dir,
        )
        tasks.append(task)
        for checkpoint in CHECKPOINTS:
            trial_indicators[str(checkpoint)].append(indicators[str(checkpoint)])

    endpoints: dict[str, dict[str, object]] = {}
    for checkpoint in CHECKPOINTS:
        key = str(checkpoint)
        llm_successes = sum(cast(dict[str, bool], task["exact_by_slot"])[key] for task in tasks)
        aggregate_random = [
            sum(task_trials[index] for task_trials in trial_indicators[key])
            for index in range(TRIALS)
        ]
        exceedances = sum(value >= llm_successes for value in aggregate_random)
        p_value = (1 + exceedances) / (TRIALS + 1)
        random_task_rates = [
            cast(dict[str, Any], task["matched_random"])[key]["success_rate"]
            for task in tasks
        ]
        endpoints[key] = {
            "role": "primary" if checkpoint == 29 else "secondary-descriptive",
            "llm_exact_successes": llm_successes,
            "task_count": 12,
            "llm_exact_rate": llm_successes / 12,
            "llm_exact_rate_wilson_95": wilson_interval(llm_successes, 12),
            "matched_random_aggregate_success_count_by_trial_index": aggregate_random,
            "matched_random_mean_task_success_rate": mean(random_task_rates),
            "mean_paired_success_advantage": mean(
                float(cast(dict[str, bool], task["exact_by_slot"])[key]) - rate
                for task, rate in zip(tasks, random_task_rates, strict=True)
            ),
            "randomization_exceedances": exceedances,
            "randomization_p_value": p_value,
            "randomization_p_value_formula": "(1 + # aggregate random trial-j >= X_LLM) / 10001",
        }

    primary = endpoints["29"]
    practical_pass = cast(int, primary["llm_exact_successes"]) >= PRACTICAL_THRESHOLD
    statistical_pass = cast(float, primary["randomization_p_value"]) <= 0.05
    return {
        "schema": SCHEMA,
        "label": "CONFIRMATORY BLIND-V2 — INTEGRITY VALIDATED",
        "blind": True,
        "confirmatory": True,
        "private_reveal_used": False,
        "integrity": {
            "passed": True,
            "basis": (
                "stage-1 freeze, stage-2 pre-call seal, 12 public task hashes, run settings, "
                "provider inventories, checkpoints, singleton counts, and 120000 random "
                "trials validated"
            ),
        },
        "bindings": {
            "study_protocol_sha256": study_sha256,
            "method_seal_sha256": method_seal_sha256,
            "provider_call_seal_sha256": provider_call_seal_sha256,
            "public_manifest_sha256": manifest_sha256,
            "hidden_target_manifest_sha256": _read_object(provider_call_seal_path)[
                "hidden_target_manifest_sha256"
            ],
        },
        "tasks": tasks,
        "endpoints": endpoints,
        "primary_decision": {
            "practical_threshold": PRACTICAL_THRESHOLD,
            "practical_threshold_passed": practical_pass,
            "alpha": 0.05,
            "randomization_test_passed": statistical_pass,
            "study_success": practical_pass and statistical_pass,
        },
        "claim_boundary": (
            "A pass supports bounded exact-discovery advantage over the matched random proposal "
            "mechanism on the frozen 12-task distribution; it is not wall-clock, "
            "exhaustive-search, "
            "importance-sampling, or billion-space evidence."
        ),
    }


def render_markdown(analysis: Mapping[str, Any]) -> str:
    primary = _object(
        _object(analysis.get("endpoints"), name="endpoints").get("29"),
        name="primary",
    )
    decision = _object(analysis.get("primary_decision"), name="primary decision")
    lines = [
        "# Confirmatory Blind-V2 Evidence-Frontier Study",
        "",
        "Integrity status: **VALIDATED**. This analysis used no private reveal.",
        "",
        "## Primary outcome",
        "",
        f"- GPT-OSS exact by slot 29: {primary['llm_exact_successes']}/12.",
        f"- Trial-index matched randomization p-value: {primary['randomization_p_value']:.6g}.",
        "- Practical threshold (at least 6/12): "
        f"{'PASS' if decision['practical_threshold_passed'] else 'FAIL'}.",
        f"- Confirmatory endpoint: **{'PASS' if decision['study_success'] else 'FAIL'}**.",
        "",
        "## Every preregistered task",
        "",
    ]
    for raw in _array(analysis.get("tasks"), name="tasks"):
        task = _object(raw, name="task")
        exact = _object(task.get("exact_by_slot"), name="task exact")
        losses = _object(task.get("best_loss_by_slot"), name="task losses")
        lines.append(
            f"- {task['task_id']}: slot 21 exact={str(exact['21']).lower()} "
            f"(loss {losses['21']:g}); slot 29 exact={str(exact['29']).lower()} "
            f"(loss {losses['29']:g}); first exact={task['first_exact_proposal_slot']}."
        )
    lines.extend(["", "## Claim boundary", "", str(analysis["claim_boundary"]), ""])
    return "\n".join(lines)


def write_analysis(output_dir: Path, analysis: Mapping[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "analysis.json"
    markdown_path = output_dir / "SUMMARY.md"
    json_path.write_bytes(canonical_bytes(analysis) + b"\n")
    markdown_path.write_text(render_markdown(analysis), encoding="utf-8")
    return json_path, markdown_path


def _parse_runs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        task_id, separator, raw_path = value.partition("=")
        if not separator or task_id in result:
            raise ValueError("--run must be unique TASK_ID=PATH values")
        result[task_id] = Path(raw_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--study-protocol", type=Path, required=True)
    parser.add_argument("--method-seal", type=Path, required=True)
    parser.add_argument("--provider-call-seal", type=Path, required=True)
    parser.add_argument("--expected-method-seal-sha256", required=True)
    parser.add_argument("--expected-provider-call-seal-sha256", required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    runs = parser.add_mutually_exclusive_group(required=True)
    runs.add_argument("--runs-root", type=Path)
    runs.add_argument("--run", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dirs = (
        {task_id: args.runs_root / task_id for task_id in TASK_IDS}
        if args.runs_root is not None
        else _parse_runs(args.run)
    )
    analysis = analyze_blind_evidence_frontier_v2(
        args.repo_root,
        args.study_protocol,
        args.method_seal,
        args.provider_call_seal,
        args.public_manifest,
        run_dirs,
        expected_method_seal_sha256=args.expected_method_seal_sha256,
        expected_provider_call_seal_sha256=args.expected_provider_call_seal_sha256,
    )
    write_analysis(args.output, analysis)


if __name__ == "__main__":
    main()

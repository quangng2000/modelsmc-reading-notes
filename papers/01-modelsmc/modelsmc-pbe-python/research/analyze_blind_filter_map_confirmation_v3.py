"""Reveal-free confirmatory analysis for the fresh blind-v3 matched study."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from research.blind_filter_map_confirmation_v3 import (
    ANALYSIS_SCHEMA,
    MAXIMUM_PROVIDER_CALLS,
    RESULT_SCHEMA,
    SLOT_CHECKPOINTS,
    TASK_IDS,
    canonical_bytes,
    expect,
    paired_sign_test_p_value,
    read_object,
    require_array,
    require_object,
    run_seed_for,
    sha256_file,
    write_json_exclusive,
)
from research.run_blind_filter_map_confirmation_v3 import (
    ARMS,
    _validate_method,
    _validate_provider_boundary,
)

FORBIDDEN_PROVIDER_SUBSTRINGS = (
    '"seed_hex"',
    '"target_predicate"',
    '"target_mapper"',
    '"commitment_nonce"',
    '"predicate_syntaxes"',
    '"mapper_syntaxes"',
    "complete_program_count",
    "complete program support count",
    "hidden target ast",
)


def _validate_provider_inventory(run_dir: Path, result: dict[str, Any], *, arm: str) -> None:
    seal_path = run_dir / "provider-seal.json"
    seal = read_object(seal_path)
    records = require_array(seal.get("records"), name=f"{seal_path}.records")
    normalized: list[dict[str, object]] = []
    listed: set[str] = set()
    previous: str | None = None
    for index, raw in enumerate(records):
        record = require_object(raw, name=f"{seal_path}.records[{index}]")
        if set(record) != {"path", "bytes", "sha256"}:
            raise ValueError(f"{seal_path}.records[{index}] has the wrong fields")
        relative_value = record.get("path")
        if not isinstance(relative_value, str):
            raise ValueError(f"{seal_path}.records[{index}].path must be a string")
        relative = Path(relative_value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 4
            or relative.parts[0] != "provider"
            or not relative.parts[1].startswith("round-")
            or not relative.parts[2].startswith("parent-")
            or relative.parts[3] not in {"request.json", "response.json", "result.json"}
        ):
            raise ValueError(f"unsafe or unexpected provider inventory path: {relative_value}")
        if previous is not None and relative_value <= previous:
            raise ValueError(f"{seal_path} records are not strictly sorted")
        previous = relative_value
        artifact = run_dir / relative
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError(f"missing/nonregular provider artifact: {artifact}")
        byte_count = record.get("bytes")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            raise ValueError(f"{seal_path}.records[{index}].bytes must be nonnegative")
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{seal_path}.records[{index}].sha256 is invalid")
        expect(artifact.stat().st_size, byte_count, name=f"{artifact} byte count")
        expect(sha256_file(artifact), digest, name=f"{artifact} SHA-256")
        listed.add(relative_value)
        normalized.append(
            {"path": relative_value, "bytes": byte_count, "sha256": digest}
        )
    provider_root = run_dir / "provider"
    if provider_root.exists() and (
        not provider_root.is_dir() or provider_root.is_symlink()
    ):
        raise ValueError(f"{provider_root} must be a regular directory")
    actual = (
        {
            path.relative_to(run_dir).as_posix()
            for path in provider_root.rglob("*")
            if path.is_file()
        }
        if provider_root.is_dir() and not provider_root.is_symlink()
        else set()
    )
    expect(listed, actual, name=f"{run_dir} provider inventory completeness")
    inventory_sha256 = hashlib.sha256(canonical_bytes(normalized)).hexdigest()
    expect(
        seal.get("inventory_sha256"),
        inventory_sha256,
        name=f"{run_dir} provider-seal digest",
    )
    expect(
        result.get("provider_inventory_sha256"),
        inventory_sha256,
        name=f"{run_dir} result provider digest",
    )
    request_paths = sorted(run_dir.glob("provider/**/request.json"))
    response_paths = sorted(run_dir.glob("provider/**/response.json"))
    result_paths = sorted(run_dir.glob("provider/**/result.json"))
    search = require_object(result.get("search"), name=f"{run_dir} search")
    provider_calls = search.get("provider_calls")
    expect(len(request_paths), provider_calls, name=f"{run_dir} request count")
    expect(len(result_paths), provider_calls, name=f"{run_dir} provider-result count")
    if len(response_paths) > len(request_paths):
        raise ValueError(f"{run_dir} has more responses than requests")
    if arm == "grammar-random":
        expect(provider_calls, 0, name=f"{run_dir} random provider calls")
        expect(response_paths, [], name=f"{run_dir} random provider responses")
    for request_path in request_paths:
        text = request_path.read_text(encoding="utf-8").lower()
        leaked = [value for value in FORBIDDEN_PROVIDER_SUBSTRINGS if value in text]
        if leaked:
            raise ValueError(f"provider-information-boundary leak in {request_path}: {leaked}")


def _validate_run(
    *,
    runs_root: Path,
    task_id: str,
    arm: str,
    task_path: Path,
    protocol_sha256: str,
    method_sha256: str,
    custody_sha256: str,
    provider_sha256: str,
) -> dict[str, object]:
    run_dir = runs_root / task_id / arm
    result_path = run_dir / "result.json"
    execution_path = run_dir / "executions.json"
    preflight_path = run_dir.with_name(run_dir.name + ".preflight.json")
    invocation_path = run_dir.with_name(run_dir.name + ".invocation.json")
    for path in (result_path, execution_path, preflight_path, invocation_path):
        if not path.is_file():
            raise ValueError(f"missing required matched-arm artifact: {path}")
    preflight = read_object(preflight_path)
    expect(
        preflight.get("schema"),
        "blind-filter-map-confirmation-v3-preflight-v1",
        name=f"{preflight_path} schema",
    )
    for field, expected in (
        ("task_id", task_id),
        ("arm", arm),
        ("study_protocol_sha256", protocol_sha256),
        ("method_seal_sha256", method_sha256),
        ("custody_seal_sha256", custody_sha256),
        ("provider_seal_sha256", provider_sha256),
        ("task_file_sha256", sha256_file(task_path)),
        ("derived_invocation_sha256", sha256_file(invocation_path)),
        ("output", str(run_dir.resolve())),
    ):
        expect(preflight.get(field), expected, name=f"{preflight_path}.{field}")
    invocation = read_object(invocation_path)
    expect(
        invocation.get("derived_from_study_protocol_sha256"),
        protocol_sha256,
        name=f"{invocation_path} method derivation",
    )
    expect(
        invocation.get("derived_from_provider_seal_sha256"),
        provider_sha256,
        name=f"{invocation_path} provider derivation",
    )
    runs = require_array(invocation.get("runs"), name=f"{invocation_path}.runs")
    expect(len(runs), 1, name=f"{invocation_path} run count")
    frozen_run = require_object(runs[0], name=f"{invocation_path}.runs[0]")
    expect(frozen_run.get("proposal_source"), arm, name=f"{invocation_path} arm")
    expect(frozen_run.get("logical_execution_cap"), SLOT_CHECKPOINTS[-1], name="slot cap")
    expect(
        frozen_run.get("provider_call_cap"),
        0 if arm == "grammar-random" else MAXIMUM_PROVIDER_CALLS,
        name="provider cap",
    )
    seeds = run_seed_for(task_id)
    for field in ("start_seed", "provider_seed", "sample_seed", "resample_seed"):
        expect(frozen_run.get(field), getattr(seeds, field), name=f"{task_id}.{arm}.{field}")
    expect(
        frozen_run.get("random_shortlist_seed"),
        seeds.random_shortlist_seed if arm == "grammar-random" else None,
        name=f"{task_id}.{arm}.random_shortlist_seed",
    )

    result = read_object(result_path)
    expect(result.get("schema"), RESULT_SCHEMA, name=f"{result_path} schema")
    run_protocol = require_object(result.get("protocol"), name=f"{result_path} protocol")
    expect(run_protocol.get("proposal_source"), arm, name=f"{result_path} arm")
    expect(
        run_protocol.get("proposal_slot_checkpoints"),
        list(SLOT_CHECKPOINTS),
        name=f"{result_path} checkpoints",
    )
    expect(
        run_protocol.get("logical_complete_program_execution_cap"),
        SLOT_CHECKPOINTS[-1],
        name=f"{result_path} slot cap",
    )
    expect(
        run_protocol.get("task_sha256"),
        sha256_file(task_path),
        name=f"{result_path} task SHA-256",
    )
    frozen_invocation = require_object(
        run_protocol.get("frozen_invocation"),
        name=f"{result_path} frozen invocation",
    )
    expect(
        frozen_invocation.get("study_protocol_sha256"),
        sha256_file(invocation_path),
        name=f"{result_path} invocation SHA-256",
    )
    search = require_object(result.get("search"), name=f"{result_path} search")
    expect(
        search.get("logical_complete_program_executions"),
        SLOT_CHECKPOINTS[-1],
        name=f"{result_path} logical slots",
    )
    provider_calls = search.get("provider_calls")
    if not isinstance(provider_calls, int) or isinstance(provider_calls, bool):
        raise ValueError(f"{result_path} provider_calls must be an integer")
    if arm == "grammar-random":
        expect(provider_calls, 0, name=f"{result_path} random provider calls")
    elif not 0 <= provider_calls <= MAXIMUM_PROVIDER_CALLS:
        raise ValueError(f"{result_path} exceeds the provider-call cap")
    executions_raw = json.loads(execution_path.read_text(encoding="utf-8"))
    executions = require_array(executions_raw, name=f"{execution_path} executions")
    expect(len(executions), SLOT_CHECKPOINTS[-1], name=f"{execution_path} count")
    slots = [
        require_object(record, name=f"{execution_path}[{index}]").get("slot")
        for index, record in enumerate(executions)
    ]
    expect(slots, list(range(1, SLOT_CHECKPOINTS[-1] + 1)), name=f"{execution_path} slots")
    _validate_provider_inventory(run_dir, result, arm=arm)
    first_exact = search.get("first_exact_slot")
    if first_exact is not None and (
        not isinstance(first_exact, int)
        or isinstance(first_exact, bool)
        or not 1 <= first_exact <= SLOT_CHECKPOINTS[-1]
    ):
        raise ValueError(f"{result_path} has invalid first_exact_slot")
    found_exact = search.get("found_exact")
    expect(found_exact, first_exact is not None, name=f"{result_path} exact consistency")
    best_score = require_object(search.get("best_score"), name=f"{result_path} best score")
    best_loss = best_score.get("total_loss")
    if not isinstance(best_loss, (int, float)) or isinstance(best_loss, bool):
        raise ValueError(f"{result_path} best loss must be numeric")
    return {
        "task_id": task_id,
        "arm": arm,
        "exact_by_21": first_exact is not None and first_exact <= 21,
        "exact_by_29": first_exact is not None,
        "first_exact_slot": first_exact,
        "best_loss": float(best_loss),
        "provider_calls": provider_calls,
        "result_file_sha256": sha256_file(result_path),
        "execution_file_sha256": sha256_file(execution_path),
        "provider_inventory_sha256": result.get("provider_inventory_sha256"),
    }


def analyze(
    *,
    repo_root: Path,
    protocol_path: Path,
    method_seal_path: Path,
    expected_method_seal_sha256: str,
    custody_seal_path: Path,
    expected_custody_seal_sha256: str,
    provider_seal_path: Path,
    expected_provider_seal_sha256: str,
    runs_root: Path,
) -> dict[str, object]:
    root = repo_root.resolve()
    protocol, _, protocol_sha256, method_sha256, _ = _validate_method(
        repo_root=root,
        protocol_path=protocol_path,
        method_seal_path=method_seal_path,
        expected_method_seal_sha256=expected_method_seal_sha256,
    )
    _, task_paths, custody_sha256, provider_sha256 = _validate_provider_boundary(
        repo_root=root,
        protocol_sha256=protocol_sha256,
        method_sha256=method_sha256,
        custody_seal_path=custody_seal_path,
        expected_custody_seal_sha256=expected_custody_seal_sha256,
        provider_seal_path=provider_seal_path,
        expected_provider_seal_sha256=expected_provider_seal_sha256,
    )
    rows: list[dict[str, object]] = []
    by_task: dict[str, dict[str, dict[str, object]]] = {}
    for task_id in TASK_IDS:
        arms: dict[str, dict[str, object]] = {}
        for arm in ARMS:
            row = _validate_run(
                runs_root=runs_root.resolve(),
                task_id=task_id,
                arm=arm,
                task_path=task_paths[task_id],
                protocol_sha256=protocol_sha256,
                method_sha256=method_sha256,
                custody_sha256=custody_sha256,
                provider_sha256=provider_sha256,
            )
            rows.append(row)
            arms[arm] = row
        by_task[task_id] = arms
    llm_successes = sum(
        cast(bool, arms["llm"]["exact_by_29"]) for arms in by_task.values()
    )
    random_successes = sum(
        cast(bool, arms["grammar-random"]["exact_by_29"])
        for arms in by_task.values()
    )
    llm_wins = sum(
        cast(bool, arms["llm"]["exact_by_29"])
        and not cast(bool, arms["grammar-random"]["exact_by_29"])
        for arms in by_task.values()
    )
    random_wins = sum(
        cast(bool, arms["grammar-random"]["exact_by_29"])
        and not cast(bool, arms["llm"]["exact_by_29"])
        for arms in by_task.values()
    )
    p_value = paired_sign_test_p_value(llm_wins, random_wins)
    advantage = llm_successes - random_successes
    practical = llm_successes >= 6 and advantage >= 4
    success = p_value <= 0.05 and practical
    return {
        "schema": ANALYSIS_SCHEMA,
        "status": "complete-reveal-free-confirmatory-analysis",
        "method_integrity": "valid",
        "study_protocol_sha256": protocol_sha256,
        "method_seal_sha256": method_sha256,
        "custody_seal_sha256": custody_sha256,
        "provider_seal_sha256": provider_sha256,
        "primary": {
            "endpoint": "semantic exact discovery at or before logical slot 29",
            "llm_successes": llm_successes,
            "grammar_random_successes": random_successes,
            "paired_advantage": advantage,
            "llm_only_wins": llm_wins,
            "grammar_random_only_wins": random_wins,
            "discordant_pairs": llm_wins + random_wins,
            "one_sided_exact_sign_test_p_value": p_value,
            "alpha": 0.05,
            "practical_threshold_met": practical,
            "study_success": success,
        },
        "secondary": {
            "slot_21_llm_successes": sum(
                cast(bool, arms["llm"]["exact_by_21"]) for arms in by_task.values()
            ),
            "slot_21_grammar_random_successes": sum(
                cast(bool, arms["grammar-random"]["exact_by_21"])
                for arms in by_task.values()
            ),
        },
        "runs": rows,
        "claim_scope": require_object(protocol.get("reporting"), name="reporting").get(
            "required_claim_limits"
        ),
        "private_reveal_read": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--study-protocol", type=Path, required=True)
    parser.add_argument("--method-seal", type=Path, required=True)
    parser.add_argument("--expected-method-seal-sha256", required=True)
    parser.add_argument("--custody-seal", type=Path, required=True)
    parser.add_argument("--expected-custody-seal-sha256", required=True)
    parser.add_argument("--provider-seal", type=Path, required=True)
    parser.add_argument("--expected-provider-seal-sha256", required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = analyze(
            repo_root=args.repo_root,
            protocol_path=args.study_protocol,
            method_seal_path=args.method_seal,
            expected_method_seal_sha256=args.expected_method_seal_sha256,
            custody_seal_path=args.custody_seal,
            expected_custody_seal_sha256=args.expected_custody_seal_sha256,
            provider_seal_path=args.provider_seal,
            expected_provider_seal_sha256=args.expected_provider_seal_sha256,
            runs_root=args.runs_root,
        )
    except (OSError, ValueError) as error:
        aborted = {
            "schema": ANALYSIS_SCHEMA,
            "status": "aborted-method-integrity-or-infrastructure-failure",
            "error_type": type(error).__name__,
            "detail": str(error),
            "private_reveal_read": False,
        }
        write_json_exclusive(args.output.resolve(), aborted)
        print(json.dumps(aborted, indent=2, sort_keys=True))
        return 2
    write_json_exclusive(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

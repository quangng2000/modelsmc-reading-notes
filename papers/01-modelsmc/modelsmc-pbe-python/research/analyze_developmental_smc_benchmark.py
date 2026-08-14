"""Fail-closed descriptive analysis for the developmental SMC benchmark."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from research.run_developmental_smc_benchmark import (
    ARMS,
    TASK_IDS,
    _index_runs,
    _object,
    canonical_bytes,
    sha256_file,
    validate_protocol,
    validate_public_suite,
)

RESULT_SCHEMA = "evidence-shortlist-product-path-smc-result-v1"
ANALYSIS_SCHEMA = "developmental-evidence-shortlist-smc-analysis-v1"
CHECKPOINTS = (1, 5, 13, 21, 29)
ROUND_PROPOSALS = (4, 8, 8, 8)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _expect(actual: object, expected: object, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _validate_result(
    *,
    run_dir: Path,
    task_id: str,
    arm: str,
    run: Mapping[str, Any],
    protocol_sha256: str,
    task_sha256: str,
) -> dict[str, object]:
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ValueError(f"missing regular run directory: {run_dir}")
    preflight_path = run_dir.with_name(run_dir.name + ".preflight.json")
    preflight = _read_object(preflight_path)
    _expect(preflight.get("status"), "validated-before-provider-call", name="preflight status")
    _expect(preflight.get("protocol_sha256"), protocol_sha256, name="preflight protocol")
    _expect(preflight.get("task_sha256"), task_sha256, name="preflight task")
    _expect(preflight.get("task_id"), task_id, name="preflight task_id")
    _expect(preflight.get("arm"), arm, name="preflight arm")
    result_path = run_dir / "result.json"
    result = _read_object(result_path)
    _expect(result.get("schema"), RESULT_SCHEMA, name=f"{task_id}/{arm} result schema")
    embedded = _object(result.get("protocol"), name="result.protocol")
    _expect(embedded.get("task_sha256"), task_sha256, name="embedded task SHA-256")
    _expect(embedded.get("proposal_source"), run.get("proposal_source"), name="proposal source")
    _expect(embedded.get("epsilon"), run.get("epsilon"), name="epsilon")
    _expect(embedded.get("early_stop"), False, name="early stop")
    _expect(
        tuple(embedded.get("proposal_slot_checkpoints", ())),
        CHECKPOINTS,
        name="checkpoints",
    )
    _expect(embedded.get("logical_complete_program_execution_cap"), 29, name="budget")
    _expect(
        embedded.get("maximum_provider_calls"),
        run.get("provider_call_cap"),
        name="provider cap",
    )
    frozen = _object(embedded.get("frozen_invocation"), name="frozen_invocation")
    _expect(frozen.get("run_id"), run.get("id"), name="frozen run ID")
    _expect(frozen.get("study_protocol_sha256"), protocol_sha256, name="frozen protocol")
    search = _object(result.get("search"), name="result.search")
    _expect(search.get("logical_complete_program_executions"), 29, name="executions")
    provider_calls = search.get("provider_calls")
    if not isinstance(provider_calls, int) or isinstance(provider_calls, bool):
        raise ValueError("provider_calls must be an integer")
    if not 0 <= provider_calls <= cast(int, run["provider_call_cap"]):
        raise ValueError("provider call count exceeded its frozen cap")
    if arm != "llm-smc" and provider_calls != 0:
        raise ValueError("non-LLM control made a provider call")
    rounds = result.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 4:
        raise ValueError("result must contain all four rounds")
    for index, round_record in enumerate(rounds):
        record = _object(round_record, name=f"rounds[{index}]")
        _expect(record.get("round"), index + 1, name="round number")
        _expect(record.get("proposal_count"), ROUND_PROPOSALS[index], name="round proposals")
        _expect(record.get("parent_count"), 1 if index == 0 else 2, name="round parents")
        if index < 3:
            ancestors = record.get("systematic_resample_ancestors")
            if not isinstance(ancestors, list) or len(ancestors) != 2:
                raise ValueError("nonterminal round must resample exactly two parents")
        elif "systematic_resample_ancestors" in record:
            raise ValueError("terminal round must remain weighted")
    inference = _object(result.get("inference"), name="result.inference")
    particles = inference.get("final_particles")
    if not isinstance(particles, list) or len(particles) != 8:
        raise ValueError("terminal cloud must contain eight weighted slots")
    provider_seal = _read_object(run_dir / "provider-seal.json")
    _expect(
        result.get("provider_inventory_sha256"),
        provider_seal.get("inventory_sha256"),
        name="provider inventory",
    )
    found_exact = search.get("found_exact") is True
    first_exact = search.get("first_exact_slot")
    if found_exact:
        if not isinstance(first_exact, int) or isinstance(first_exact, bool) or not 1 <= first_exact <= 29:
            raise ValueError("successful run has invalid first_exact_slot")
    elif first_exact is not None:
        raise ValueError("unsuccessful run must have null first_exact_slot")
    best_score = _object(search.get("best_score"), name="search.best_score")
    _expect(best_score.get("exact_program"), found_exact, name="best exact flag")
    return {
        "task_id": task_id,
        "arm": arm,
        "found_public_example_exact": found_exact,
        "first_exact_slot": first_exact,
        "best_loss": _number(best_score.get("total_loss"), name="best loss"),
        "provider_calls": provider_calls,
        "physical_scorer_calls": search.get("physical_scorer_calls_before_reference"),
        "final_ess": _number(inference.get("final_ess"), name="final ESS"),
        "final_relative_ess": _number(
            inference.get("final_relative_ess"), name="final relative ESS"
        ),
        "result_sha256": sha256_file(result_path),
        "preflight_sha256": sha256_file(preflight_path),
    }


def analyze(
    *,
    repo_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    suite_dir: Path,
    runs_root: Path,
) -> dict[str, object]:
    protocol_file = protocol_path.resolve()
    protocol = validate_protocol(repo_root.resolve(), protocol_file, expected_protocol_sha256)
    task_paths = validate_public_suite(
        suite_dir,
        _object(protocol.get("task_suite"), name="task_suite"),
    )
    runs = _index_runs(protocol)
    records: list[dict[str, object]] = []
    for arm in ARMS:
        for task_id in TASK_IDS:
            run = runs[f"{task_id}--{arm}"]
            records.append(
                _validate_result(
                    run_dir=runs_root.resolve() / arm / task_id,
                    task_id=task_id,
                    arm=arm,
                    run=run,
                    protocol_sha256=expected_protocol_sha256,
                    task_sha256=sha256_file(task_paths[task_id]),
                )
            )
    summaries: dict[str, object] = {}
    for arm in ARMS:
        arm_records = [record for record in records if record["arm"] == arm]
        successes = sum(record["found_public_example_exact"] is True for record in arm_records)
        summaries[arm] = {
            "successes": successes,
            "tasks": len(arm_records),
            "success_fraction": successes / len(arm_records),
            "wilson_95": wilson_interval(successes, len(arm_records)),
            "mean_best_loss": sum(cast(float, record["best_loss"]) for record in arm_records)
            / len(arm_records),
            "mean_provider_calls": sum(
                cast(int, record["provider_calls"]) for record in arm_records
            )
            / len(arm_records),
        }
    return {
        "schema": ANALYSIS_SCHEMA,
        "integrity": {"passed": True, "complete_task_arm_runs": len(records)},
        "classification": {
            "developmental": True,
            "blind": False,
            "confirmatory": False,
            "public_examples_only": True,
            "private_reveal_used": False,
        },
        "bindings": {
            "protocol_sha256": expected_protocol_sha256,
            "manifest_sha256": sha256_file(suite_dir.resolve() / "manifest.json"),
        },
        "endpoint": (
            "zero loss on the reused public examples by logical slot 29; this is not "
            "fresh hidden-test semantic success"
        ),
        "arm_summaries": summaries,
        "runs": records,
        "inference_note": (
            "One frozen seed per reused task-arm is descriptive; Wilson intervals summarize "
            "task variation and are not confirmatory efficacy intervals."
        ),
    }


def _write_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value) + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {args.output}")
    result = analyze(
        repo_root=args.repo_root,
        protocol_path=args.protocol,
        expected_protocol_sha256=args.expected_protocol_sha256,
        suite_dir=args.suite_dir,
        runs_root=args.runs_root,
    )
    _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result["arm_summaries"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fail-closed hidden-semantic analysis for the paired ExeDec debug benchmark V2.

All 64 public-task SMC runs and their frozen bindings are validated before this
module opens any provider-private debug oracle.  It then evaluates every one of
the 29 executed complete programs per run on the stored semantic probes.  The
tasks and targets are released public data, so every result remains debug-only,
non-blind, non-confirmatory, and potentially contaminated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.grammar import arithmetic_expressions, filter_predicates
from research.execution_guided_repair import _assemble_program, _dsl_catalog
from research.exedec_deepcoder_ho_adapter_v1 import ORACLE_SCHEMA, evaluate_candidate
from research.run_exedec_deepcoder_ho_debug_benchmark_v2 import (
    ARMS,
    BUNDLE_SCHEMA,
    LLM_PROVIDER_CALL_CAP,
    LOGICAL_EXECUTION_CAP,
    OFFSPRING_PER_PARENT,
    PARENT_COUNT,
    PREFLIGHT_SCHEMA,
    SEED_IDS,
    TASK_IDS,
    _array,
    _digest,
    _expect,
    _index_runs,
    _object,
    _read_object,
    _workspace_root,
    canonical_bytes,
    sha256_file,
    validate_protocol,
    validate_public_bundle,
)

RESULT_SCHEMA = "evidence-shortlist-product-path-smc-result-v1"
ANALYSIS_SCHEMA = "exedec-deepcoder-ho-paired-smc-debug-analysis-v2"
CHECKPOINTS = (1, 5, 13, 21, 29)
ROUND_PROPOSALS = (4, 8, 8, 8)


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


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


def exact_two_sided_sign_p(llm_wins: int, grammar_wins: int) -> float:
    """Return the exact two-sided conditional sign-test p-value."""

    wins = _nonnegative_integer(llm_wins, name="LLM wins")
    losses = _nonnegative_integer(grammar_wins, name="grammar wins")
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _run_dir(runs_root: Path, task_id: str, seed_id: int, arm: str) -> Path:
    return runs_root.resolve() / arm / task_id / f"seed-{seed_id:02d}"


def _program_key(record: Mapping[str, Any], *, slot: int) -> tuple[str, str]:
    raw = record.get("program") if slot == 1 else record.get("sampled_program")
    if not isinstance(raw, dict) or set(raw) != {"predicate", "mapper"}:
        raise ValueError(f"execution slot {slot} has no complete program key")
    predicate = raw.get("predicate")
    mapper = raw.get("mapper")
    if not isinstance(predicate, str) or not isinstance(mapper, str):
        raise ValueError(f"execution slot {slot} has non-string program syntax")
    return predicate, mapper


def _validate_result(
    *,
    run_dir: Path,
    task_id: str,
    seed_id: int,
    arm: str,
    run: Mapping[str, Any],
    protocol_sha256: str,
    bundle_manifest_sha256: str,
    task_sha256: str,
) -> tuple[dict[str, object], tuple[tuple[str, str], ...]]:
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ValueError(f"missing regular run directory: {run_dir}")
    if (run_dir / "failure.json").exists():
        raise ValueError(f"run has a failure artifact: {run_dir}")
    preflight_path = run_dir.with_name(run_dir.name + ".preflight.json")
    preflight = _read_object(preflight_path)
    _expect(preflight.get("schema"), PREFLIGHT_SCHEMA, name="preflight schema")
    _expect(preflight.get("status"), "validated-before-debug-run", name="preflight status")
    _expect(
        preflight.get("classification"),
        "debug-only-public-released-task-not-confirmatory",
        name="preflight classification",
    )
    _expect(preflight.get("protocol_sha256"), protocol_sha256, name="preflight protocol")
    _expect(
        preflight.get("bundle_manifest_sha256"),
        bundle_manifest_sha256,
        name="preflight bundle",
    )
    _expect(preflight.get("public_task_sha256"), task_sha256, name="preflight task")
    _expect(preflight.get("private_oracle_opened"), False, name="preflight custody")
    _expect(preflight.get("task_id"), task_id, name="preflight task ID")
    _expect(preflight.get("seed_id"), seed_id, name="preflight seed ID")
    _expect(preflight.get("arm"), arm, name="preflight arm")
    command = preflight.get("command")
    if not isinstance(command, list) or any(not isinstance(value, str) for value in command):
        raise ValueError("preflight command is malformed")
    if any(".oracle.json" in value or "/private/" in value for value in command):
        raise ValueError("preflight command exposes a private debug oracle")

    result_path = run_dir / "result.json"
    result = _read_object(result_path)
    _expect(result.get("schema"), RESULT_SCHEMA, name="result schema")
    embedded = _object(result.get("protocol"), name="result.protocol")
    _expect(embedded.get("task_sha256"), task_sha256, name="embedded task SHA-256")
    _expect(embedded.get("proposal_source"), run.get("proposal_source"), name="proposal source")
    _expect(embedded.get("epsilon"), run.get("epsilon"), name="epsilon")
    _expect(embedded.get("evidence_scale"), run.get("evidence_scale"), name="evidence scale")
    _expect(embedded.get("early_stop"), False, name="early stop")
    _expect(
        tuple(embedded.get("proposal_slot_checkpoints", ())),
        CHECKPOINTS,
        name="checkpoints",
    )
    _expect(
        embedded.get("logical_complete_program_execution_cap"),
        LOGICAL_EXECUTION_CAP,
        name="execution budget",
    )
    _expect(
        embedded.get("maximum_provider_calls"),
        run.get("provider_call_cap"),
        name="provider cap",
    )
    frozen = _object(embedded.get("frozen_invocation"), name="frozen_invocation")
    _expect(frozen.get("run_id"), run.get("id"), name="frozen run ID")
    _expect(frozen.get("study_protocol_sha256"), protocol_sha256, name="frozen protocol")

    search = _object(result.get("search"), name="result.search")
    _expect(
        search.get("logical_complete_program_executions"),
        LOGICAL_EXECUTION_CAP,
        name="executions",
    )
    provider_calls = _nonnegative_integer(search.get("provider_calls"), name="provider calls")
    if provider_calls > cast(int, run["provider_call_cap"]):
        raise ValueError("provider call count exceeded its frozen cap")
    if arm == "grammar-random" and provider_calls != 0:
        raise ValueError("grammar-random control made a provider call")
    found_public = search.get("found_exact") is True
    first_public = search.get("first_exact_slot")
    if found_public:
        if (
            not isinstance(first_public, int)
            or isinstance(first_public, bool)
            or not 1 <= first_public <= LOGICAL_EXECUTION_CAP
        ):
            raise ValueError("public-exact run has an invalid first exact slot")
    elif first_public is not None:
        raise ValueError("public-inexact run must have null first exact slot")
    best_score = _object(search.get("best_score"), name="search.best_score")
    _expect(best_score.get("exact_program"), found_public, name="best public exact flag")

    rounds = _array(result.get("rounds"), name="rounds")
    if len(rounds) != 4:
        raise ValueError("result must contain four rounds")
    for index, raw_round in enumerate(rounds):
        record = _object(raw_round, name=f"rounds[{index}]")
        _expect(record.get("round"), index + 1, name="round number")
        _expect(record.get("proposal_count"), ROUND_PROPOSALS[index], name="round proposals")
        _expect(record.get("parent_count"), 1 if index == 0 else 2, name="round parents")
        if index < 3:
            ancestors = record.get("systematic_resample_ancestors")
            if not isinstance(ancestors, list) or len(ancestors) != PARENT_COUNT:
                raise ValueError("nonterminal round must resample exactly two parents")
        elif "systematic_resample_ancestors" in record:
            raise ValueError("terminal round must remain weighted")

    inference = _object(result.get("inference"), name="result.inference")
    final_particles = _array(inference.get("final_particles"), name="final particles")
    if len(final_particles) != PARENT_COUNT * OFFSPRING_PER_PARENT:
        raise ValueError("terminal particle cloud has the wrong size")
    final_ess = _number(inference.get("final_ess"), name="final ESS")
    if not 1.0 <= final_ess <= len(final_particles) + 1e-9:
        raise ValueError("final ESS is outside its particle bounds")
    reference = _object(result.get("exact_reference"), name="exact reference")
    if _nonnegative_integer(
        reference.get("post_provider_exhaustive_programs"),
        name="reference program count",
    ) > cast(int, run["exact_reference_limit"]):
        raise ValueError("reference enumeration exceeded the frozen limit")

    executions_path = run_dir / "executions.json"
    try:
        executions = json.loads(executions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid executions artifact: {error}") from error
    if not isinstance(executions, list) or len(executions) != LOGICAL_EXECUTION_CAP:
        raise ValueError("executions artifact must contain exactly 29 slots")
    program_keys: list[tuple[str, str]] = []
    for slot, raw_execution in enumerate(executions, start=1):
        execution = _object(raw_execution, name=f"executions[{slot - 1}]")
        _expect(execution.get("slot"), slot, name="execution slot")
        program_keys.append(_program_key(execution, slot=slot))

    provider_seal = _read_object(run_dir / "provider-seal.json")
    _expect(
        result.get("provider_inventory_sha256"),
        provider_seal.get("inventory_sha256"),
        name="provider inventory",
    )
    provider_records = provider_seal.get("records")
    if not isinstance(provider_records, list):
        raise ValueError("provider seal has no records")
    provider_results = sum(
        isinstance(record, dict) and str(record.get("path", "")).endswith("/result.json")
        for record in provider_records
    )
    _expect(provider_results, provider_calls, name="sealed provider result count")

    record = {
        "run_id": run["id"],
        "task_id": task_id,
        "seed_id": seed_id,
        "arm": arm,
        "found_public_example_exact": found_public,
        "first_public_example_exact_slot": first_public,
        "best_public_loss": _number(best_score.get("total_loss"), name="best public loss"),
        "provider_calls": provider_calls,
        "physical_scorer_calls_before_reference": _nonnegative_integer(
            search.get("physical_scorer_calls_before_reference"),
            name="physical scorer calls",
        ),
        "final_ess": final_ess,
        "final_relative_ess": _number(
            inference.get("final_relative_ess"),
            name="final relative ESS",
        ),
        "self_normalized_public_exact_mass": _number(
            inference.get("self_normalized_exact_mass"),
            name="self-normalized exact mass",
        ),
        "exact_public_target_mass": _number(
            reference.get("exact_target_mass"),
            name="exact target mass",
        ),
        "self_normalized_public_mean_loss": _number(
            inference.get("self_normalized_target_mean_loss"),
            name="self-normalized target mean loss",
        ),
        "exact_public_target_mean_loss": _number(
            reference.get("target_mean_loss"),
            name="exact target mean loss",
        ),
        "preflight_sha256": sha256_file(preflight_path),
        "result_sha256": sha256_file(result_path),
        "executions_sha256": sha256_file(executions_path),
        "provider_seal_sha256": sha256_file(run_dir / "provider-seal.json"),
    }
    return record, tuple(program_keys)


def _validate_private_oracles(
    *,
    repo_root: Path,
    bundle_dir: Path,
    bundle_binding: Mapping[str, Any],
) -> dict[str, tuple[dict[str, Any], str]]:
    """Open private debug oracles only after public result validation succeeds."""

    root = bundle_dir.resolve()
    workspace = _workspace_root(repo_root)
    if workspace not in root.parents:
        raise ValueError("private oracle root escaped the workspace")
    manifest = _read_object(root / "manifest.json")
    _expect(manifest.get("schema"), BUNDLE_SCHEMA, name="private manifest schema")
    records = _array(manifest.get("tasks"), name="manifest.tasks")
    _expect(records, bundle_binding.get("tasks"), name="private frozen manifest records")
    oracles: dict[str, tuple[dict[str, Any], str]] = {}
    for task_id, raw in zip(TASK_IDS, records, strict=True):
        record = _object(raw, name=f"private binding {task_id}")
        relative = record.get("private_oracle_path")
        if not isinstance(relative, str) or relative != f"private/{task_id}.oracle.json":
            raise ValueError(f"invalid private oracle path for {task_id}")
        path = (root / relative).resolve()
        if path.parent != (root / "private").resolve() or path.is_symlink():
            raise ValueError(f"unsafe private oracle path for {task_id}")
        digest = sha256_file(path)
        _expect(
            digest,
            _digest(record.get("private_oracle_sha256"), name="private oracle SHA-256"),
            name=f"{task_id} private oracle SHA-256",
        )
        oracle = _read_object(path)
        _expect(oracle.get("schema"), ORACLE_SCHEMA, name="oracle schema")
        _expect(
            oracle.get("classification"),
            "debug-only-public-released-target-not-confirmatory",
            name="oracle classification",
        )
        examples = _array(oracle.get("examples"), name="oracle.examples")
        if not examples:
            raise ValueError(f"{task_id} private oracle has no probes")
        oracles[task_id] = oracle, digest
    return oracles


def _semantic_evaluation(
    *,
    public_task: Path,
    oracle: Mapping[str, object],
    program_keys: Sequence[tuple[str, str]],
) -> dict[str, object]:
    config = load_experiment_config(public_task)
    constants = tuple(config.spec.integer_constants)
    predicates = _dsl_catalog(filter_predicates(constants))
    mappers = _dsl_catalog(arithmetic_expressions("Item", constants))
    cache: dict[tuple[str, str], dict[str, object]] = {}
    slot_evaluations: list[dict[str, object]] = []
    for slot, key in enumerate(program_keys, start=1):
        if key not in cache:
            predicate, mapper = key
            if predicate not in predicates or mapper not in mappers:
                raise ValueError(f"executed program at slot {slot} is outside the task grammar")
            candidate = _assemble_program(predicates[predicate], mappers[mapper])
            cache[key] = evaluate_candidate(candidate, oracle)
        evaluation = cache[key]
        exact = evaluation.get("exact") is True
        loss = (
            _number(evaluation.get("total_loss"), name="hidden semantic loss")
            if evaluation.get("kind") == "Scored"
            else math.inf
        )
        slot_evaluations.append({"slot": slot, "exact": exact, "loss": loss})
    exact_slots = [
        cast(int, record["slot"])
        for record in slot_evaluations
        if record["exact"] is True
    ]
    best_loss = min(cast(float, record["loss"]) for record in slot_evaluations)
    if not math.isfinite(best_loss):
        raise ValueError("every hidden semantic candidate evaluation was rejected")
    return {
        "found_hidden_semantic_exact": bool(exact_slots),
        "first_hidden_semantic_exact_slot": min(exact_slots) if exact_slots else None,
        "best_hidden_semantic_loss": best_loss,
        "unique_executed_programs": len(cache),
        "hidden_probe_count": len(cast(list[object], oracle["examples"])),
    }


def _arm_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    successes = sum(record["found_hidden_semantic_exact"] is True for record in records)
    public_successes = sum(record["found_public_example_exact"] is True for record in records)
    first_slots = [
        cast(int, record["first_hidden_semantic_exact_slot"])
        for record in records
        if record["first_hidden_semantic_exact_slot"] is not None
    ]
    mass_errors = [
        cast(float, record["self_normalized_public_exact_mass"])
        - cast(float, record["exact_public_target_mass"])
        for record in records
    ]
    return {
        "hidden_semantic_successes": successes,
        "runs": len(records),
        "hidden_semantic_success_fraction": successes / len(records),
        "hidden_semantic_success_wilson_95": wilson_interval(successes, len(records)),
        "public_example_exact_successes": public_successes,
        "public_example_exact_success_fraction": public_successes / len(records),
        "mean_first_hidden_success_slot_conditional": (
            statistics.fmean(first_slots) if first_slots else None
        ),
        "mean_best_hidden_semantic_loss": statistics.fmean(
            cast(float, record["best_hidden_semantic_loss"]) for record in records
        ),
        "mean_provider_calls": statistics.fmean(
            cast(int, record["provider_calls"]) for record in records
        ),
        "total_provider_calls": sum(cast(int, record["provider_calls"]) for record in records),
        "mean_final_ess": statistics.fmean(
            cast(float, record["final_ess"]) for record in records
        ),
        "public_exact_mass_bias": statistics.fmean(mass_errors),
        "public_exact_mass_rmse": math.sqrt(
            statistics.fmean(error * error for error in mass_errors)
        ),
    }


def analyze(
    *,
    repo_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    bundle_dir: Path,
    runs_root: Path,
) -> dict[str, object]:
    root = repo_root.resolve()
    protocol_file = protocol_path.resolve()
    protocol = validate_protocol(root, protocol_file, expected_protocol_sha256)
    bundle_binding = _object(protocol.get("debug_bundle"), name="debug_bundle")
    public_tasks = validate_public_bundle(root, bundle_dir, bundle_binding)
    runs = _index_runs(protocol)
    protocol_sha256 = sha256_file(protocol_file)
    manifest_sha256 = sha256_file(bundle_dir.resolve() / "manifest.json")

    # Phase 1 validates all public-side run artifacts before opening any oracle.
    public_records: list[dict[str, object]] = []
    executed: dict[str, tuple[tuple[str, str], ...]] = {}
    for arm in ARMS:
        for task_id in TASK_IDS:
            for seed_id in SEED_IDS:
                run_id = f"{task_id}--seed-{seed_id:02d}--{arm}"
                record, keys = _validate_result(
                    run_dir=_run_dir(runs_root, task_id, seed_id, arm),
                    task_id=task_id,
                    seed_id=seed_id,
                    arm=arm,
                    run=runs[run_id],
                    protocol_sha256=protocol_sha256,
                    bundle_manifest_sha256=manifest_sha256,
                    task_sha256=sha256_file(public_tasks[task_id]),
                )
                public_records.append(record)
                executed[run_id] = keys
    if len(public_records) != len(ARMS) * len(TASK_IDS) * len(SEED_IDS):
        raise ValueError("public run matrix is incomplete")

    # Phase 2 opens bound debug oracles and evaluates every executed candidate.
    oracles = _validate_private_oracles(
        repo_root=root,
        bundle_dir=bundle_dir,
        bundle_binding=bundle_binding,
    )
    records: list[dict[str, object]] = []
    for record in public_records:
        task_id = cast(str, record["task_id"])
        run_id = cast(str, record["run_id"])
        oracle, oracle_sha256 = oracles[task_id]
        semantic = _semantic_evaluation(
            public_task=public_tasks[task_id],
            oracle=oracle,
            program_keys=executed[run_id],
        )
        records.append(record | semantic | {"private_oracle_sha256": oracle_sha256})

    by_key = {
        (
            cast(str, record["task_id"]),
            cast(int, record["seed_id"]),
            cast(str, record["arm"]),
        ): record
        for record in records
    }
    llm_wins = 0
    grammar_wins = 0
    both_success = 0
    neither_success = 0
    paired_records: list[dict[str, object]] = []
    for task_id in TASK_IDS:
        for seed_id in SEED_IDS:
            llm = by_key[(task_id, seed_id, "llm-smc")]
            grammar = by_key[(task_id, seed_id, "grammar-random")]
            llm_success = llm["found_hidden_semantic_exact"] is True
            grammar_success = grammar["found_hidden_semantic_exact"] is True
            llm_wins += int(llm_success and not grammar_success)
            grammar_wins += int(grammar_success and not llm_success)
            both_success += int(llm_success and grammar_success)
            neither_success += int(not llm_success and not grammar_success)
            paired_records.append(
                {
                    "task_id": task_id,
                    "seed_id": seed_id,
                    "llm_hidden_semantic_success": llm_success,
                    "grammar_random_hidden_semantic_success": grammar_success,
                    "llm_first_success_slot": llm["first_hidden_semantic_exact_slot"],
                    "grammar_random_first_success_slot": grammar[
                        "first_hidden_semantic_exact_slot"
                    ],
                }
            )
    arm_summaries = {
        arm: _arm_summary([record for record in records if record["arm"] == arm])
        for arm in ARMS
    }
    llm_fraction = cast(float, arm_summaries["llm-smc"]["hidden_semantic_success_fraction"])
    grammar_fraction = cast(
        float,
        arm_summaries["grammar-random"]["hidden_semantic_success_fraction"],
    )
    per_task: dict[str, object] = {}
    for task_id in TASK_IDS:
        task_records = [record for record in records if record["task_id"] == task_id]
        per_task[task_id] = {
            arm: _arm_summary([record for record in task_records if record["arm"] == arm])
            for arm in ARMS
        }
    provider_calls = sum(cast(int, record["provider_calls"]) for record in records)
    provider_cap = len(TASK_IDS) * len(SEED_IDS) * LLM_PROVIDER_CALL_CAP
    if provider_calls > provider_cap:
        raise ValueError("aggregate provider calls exceeded the frozen matrix cap")
    return {
        "schema": ANALYSIS_SCHEMA,
        "integrity": {
            "passed": True,
            "complete_runs": len(records),
            "complete_paired_blocks": len(paired_records),
            "public_runs_validated_before_private_oracles_opened": True,
            "logical_slots_per_run": LOGICAL_EXECUTION_CAP,
            "aggregate_logical_slots": len(records) * LOGICAL_EXECUTION_CAP,
            "provider_calls": provider_calls,
            "provider_call_cap": provider_cap,
        },
        "classification": {
            "debug_only": True,
            "public_released_tasks": True,
            "potential_model_training_contamination": True,
            "blind": False,
            "confirmatory": False,
            "official_exedec_split_result": False,
            "external_benchmark_generalization_claim": False,
        },
        "bindings": {
            "protocol_sha256": protocol_sha256,
            "bundle_manifest_sha256": manifest_sha256,
        },
        "primary_endpoint": (
            "candidate exact on every stored provider-private debug semantic probe by "
            "logical complete-program slot 29"
        ),
        "paired_comparison": {
            "paired_blocks": len(paired_records),
            "llm_wins": llm_wins,
            "grammar_random_wins": grammar_wins,
            "both_success": both_success,
            "neither_success": neither_success,
            "discordant_blocks": llm_wins + grammar_wins,
            "llm_minus_grammar_success_fraction": llm_fraction - grammar_fraction,
            "exact_two_sided_sign_test_p": exact_two_sided_sign_p(llm_wins, grammar_wins),
            "test_note": "ties are excluded from the exact conditional sign test",
        },
        "arm_summaries": arm_summaries,
        "per_task_descriptive": per_task,
        "paired_blocks": paired_records,
        "runs": records,
        "inference_boundary": (
            "descriptive debug comparison on four deduplicated public released targets; "
            "the exact sign p-value is not confirmatory evidence and has low power"
        ),
    }


def _summary_markdown(analysis: Mapping[str, object]) -> str:
    paired = cast(Mapping[str, object], analysis["paired_comparison"])
    arms = cast(Mapping[str, object], analysis["arm_summaries"])
    llm = cast(Mapping[str, object], arms["llm-smc"])
    grammar = cast(Mapping[str, object], arms["grammar-random"])
    return "\n".join(
        [
            "# ExeDec DeepCoder-HO Paired SMC Debug Benchmark V1",
            "",
            "This is a debug-only result on public released targets. It is not blind, ",
            "confirmatory, contamination-free, or an official ExeDec benchmark result.",
            "",
            "## Hidden semantic endpoint",
            "",
            f"- LLM-SMC: {llm['hidden_semantic_successes']}/{llm['runs']}",
            f"- Grammar-random: {grammar['hidden_semantic_successes']}/{grammar['runs']}",
            f"- Paired difference: {cast(float, paired['llm_minus_grammar_success_fraction']):.6g}",
            f"- Discordant blocks: {paired['discordant_blocks']}",
            "- Exact two-sided sign-test p: "
            f"{cast(float, paired['exact_two_sided_sign_test_p']):.6g}",
            "",
            "Every run used 29 logical complete-program slots. Private debug oracles were ",
            "opened only by the analyzer after all 64 public-side run artifacts validated.",
            "",
        ]
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def _seal_inventory(output: Path) -> dict[str, object]:
    records = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "inventory.json"
    ]
    inventory = {
        "schema": f"{ANALYSIS_SCHEMA}-inventory-v1",
        "file_count": len(records),
        "records": records,
        "records_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
    }
    _write_json(output / "inventory.json", inventory)
    return inventory


def publish_analysis(
    *,
    output: Path,
    repo_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    bundle_dir: Path,
    runs_root: Path,
) -> dict[str, object]:
    target = output.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite analysis artifact: {target}")
    staging = target.with_name(f".{target.name}.incomplete")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite incomplete analysis: {staging}")
    staging.mkdir(parents=True)
    try:
        result = analyze(
            repo_root=repo_root,
            protocol_path=protocol_path,
            expected_protocol_sha256=expected_protocol_sha256,
            bundle_dir=bundle_dir,
            runs_root=runs_root,
        )
        _write_json(staging / "analysis.json", result)
        (staging / "SUMMARY.md").write_text(_summary_markdown(result), encoding="utf-8")
        inventory = _seal_inventory(staging)
        staging.rename(target)
        return {
            "status": "complete",
            "output": target.as_posix(),
            "inventory_sha256": inventory["records_sha256"],
        }
    except Exception as error:
        _write_json(
            staging / "failure.json",
            {
                "schema": f"{ANALYSIS_SCHEMA}-failure-v1",
                "status": "failed-closed",
                "error_type": type(error).__name__,
                "detail": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        _seal_inventory(staging)
        staging.rename(target)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = publish_analysis(
        output=args.output,
        repo_root=args.repo_root,
        protocol_path=args.protocol,
        expected_protocol_sha256=args.expected_protocol_sha256,
        bundle_dir=args.bundle_dir,
        runs_root=args.runs_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

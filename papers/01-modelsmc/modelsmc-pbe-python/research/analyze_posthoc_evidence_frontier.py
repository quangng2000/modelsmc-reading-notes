"""Audit the two-task developmental evidence-frontier follow-up.

This module is intentionally narrow.  It accepts exactly the previously seen
``blind-02`` and ``blind-03`` tasks and labels every emitted report as post-hoc
and non-blind.  A passing result is a mechanism-development gate, not a blind
replication or a search-speedup result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

LABEL = "POST-HOC NON-BLIND — DEVELOPMENTAL, NON-CONFIRMATORY"
SCHEMA = "post-hoc-non-blind-evidence-frontier-analysis-v1"
RESULT_SCHEMA = "iterative-typed-llm-beam-experiment-v2"
TASK_IDS = ("blind-02", "blind-03")
CHECKPOINTS = (21, 29)
EXPECTED_SEARCH = {
    "selection_policy": "evidence-frontier",
    "branching_factor": 4,
    "beam_width": 2,
    "rounds": 4,
    "maximum_proposal_slot_budget": 29,
    "maximum_provider_calls": 7,
    "proposal_slot_checkpoints": [1, 5, 13, 21, 29],
    "start_seed": 17,
    "random_baseline_trials": 10000,
    "singleton_evidence": True,
    "stall_policy": "alternate-hole",
    "primary_checkpoint_round": 4,
}


def canonical_bytes(value: object) -> bytes:
    """Return stable JSON bytes without a trailing newline."""

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


def _safe_repo_path(repo_root: Path, relative_value: object, *, name: str) -> Path:
    if not isinstance(relative_value, str):
        raise ValueError(f"{name} must be a relative path")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must remain inside the repository")
    root = repo_root.resolve()
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{name} escaped the repository")
    return resolved


def _validate_provider_seal(run_dir: Path, result: Mapping[str, Any], task_id: str) -> str:
    seal = _read_object(run_dir / "provider-seal.json")
    records = _array(seal.get("records"), name=f"{task_id}.provider-seal.records")
    normalized: list[dict[str, object]] = []
    listed: set[str] = set()
    previous = ""
    for index, raw in enumerate(records):
        record = _object(raw, name=f"{task_id}.provider-seal.records[{index}]")
        relative_value = record.get("path")
        if not isinstance(relative_value, str):
            raise ValueError(f"{task_id} provider path must be a string")
        relative = Path(relative_value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "provider"
        ):
            raise ValueError(f"{task_id} unsafe provider path: {relative_value}")
        if relative_value <= previous:
            raise ValueError(f"{task_id} provider inventory is not strictly sorted")
        previous = relative_value
        artifact = run_dir / relative
        payload = artifact.read_bytes()
        size = _integer(record.get("bytes"), name=f"{task_id}.{relative_value}.bytes")
        digest = _digest(record.get("sha256"), name=f"{task_id}.{relative_value}.sha256")
        _expect(len(payload), size, name=f"{task_id}.{relative_value}.bytes")
        _expect(sha256_bytes(payload), digest, name=f"{task_id}.{relative_value}.sha256")
        listed.add(relative.as_posix())
        normalized.append({"path": relative_value, "bytes": size, "sha256": digest})

    provider_root = run_dir / "provider"
    actual = (
        {
            path.relative_to(run_dir).as_posix()
            for path in provider_root.rglob("*")
            if path.is_file()
        }
        if provider_root.is_dir()
        else set()
    )
    _expect(listed, actual, name=f"{task_id}.provider-seal completeness")
    inventory_digest = sha256_bytes(canonical_bytes(normalized))
    _expect(
        seal.get("inventory_sha256"),
        inventory_digest,
        name=f"{task_id}.provider-seal.inventory_sha256",
    )
    _expect(
        result.get("provider_inventory_sha256"),
        inventory_digest,
        name=f"{task_id}.result.provider_inventory_sha256",
    )
    return inventory_digest


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
    task: Mapping[str, Any],
    *,
    task_id: str,
) -> tuple[tuple[int, ...], dict[int, bool], dict[int, int]]:
    examples = _array(task.get("examples"), name=f"{task_id}.examples")
    observed: set[int] = set()
    predicate: dict[int, bool] = {}
    mapper: dict[int, int] = {}
    for example_index, raw_example in enumerate(examples):
        example = _object(raw_example, name=f"{task_id}.examples[{example_index}]")
        inputs = _array(example.get("input"), name=f"{task_id}.examples[{example_index}].input")
        outputs = _array(example.get("output"), name=f"{task_id}.examples[{example_index}].output")
        items = [
            _integer_atom(item, name=f"{task_id}.examples[{example_index}].input item")
            for item in inputs
        ]
        observed.update(items)
        if len(items) != 1 or len(outputs) > 1:
            continue
        item = items[0]
        keep = len(outputs) == 1
        if item in predicate and predicate[item] != keep:
            raise ValueError(f"{task_id} has conflicting singleton predicate facts")
        predicate[item] = keep
        if keep:
            expected = _integer_atom(
                outputs[0], name=f"{task_id}.examples[{example_index}].output item"
            )
            if item in mapper and mapper[item] != expected:
                raise ValueError(f"{task_id} has conflicting singleton mapper facts")
            mapper[item] = expected
    return tuple(sorted(observed)), predicate, mapper


def _state_trace(
    state_value: object,
    *,
    task_id: str,
    round_number: int,
    index: int,
    observed_items: tuple[int, ...],
    predicate_facts: Mapping[int, bool],
    mapper_facts: Mapping[int, int],
) -> dict[str, object]:
    state = _object(state_value, name=f"{task_id}.round-{round_number}.selected[{index}]")
    violations = _object(
        state.get("singleton_constraint_violations"),
        name=f"{task_id}.round-{round_number}.selected[{index}].violations",
    )
    predicate = _integer(violations.get("predicate"), name="predicate violations")
    mapper = _integer(violations.get("mapper"), name="mapper violations")
    if predicate < 0 or mapper < 0:
        raise ValueError(f"{task_id} has a negative singleton-violation count")
    semantics = _object(
        state.get("finite_component_semantics"),
        name=f"{task_id}.round-{round_number}.selected[{index}].semantics",
    )
    keep_mask = _array(semantics.get("predicate_keep_mask"), name="predicate keep mask")
    mapper_values = _array(semantics.get("mapper_values"), name="mapper values")
    if len(keep_mask) != len(observed_items) or len(mapper_values) != len(observed_items):
        raise ValueError(f"{task_id} component signatures do not match the observed domain")
    observed_index = {item: position for position, item in enumerate(observed_items)}
    recomputed_predicate = sum(
        keep_mask[observed_index[item]] is not expected
        for item, expected in predicate_facts.items()
    )
    recomputed_mapper = sum(
        _integer_atom(mapper_values[observed_index[item]], name="mapper signature value")
        != expected
        for item, expected in mapper_facts.items()
    )
    _expect(predicate, recomputed_predicate, name=f"{task_id}.predicate violation count")
    _expect(mapper, recomputed_mapper, name=f"{task_id}.mapper violation count")
    score = _object(state.get("score"), name=f"{task_id}.round-{round_number}.score")
    loss = _number(score.get("total_loss"), name=f"{task_id}.round-{round_number}.loss")
    exact = score.get("exact_program") is True
    if exact and (loss != 0.0 or predicate != 0 or mapper != 0):
        raise ValueError(f"{task_id} exact selected state has inconsistent loss/violations")
    history = _array(state.get("history"), name=f"{task_id}.round-{round_number}.history")
    repaired_hole = None
    if history:
        last = _object(history[-1], name=f"{task_id}.round-{round_number}.history[-1]")
        repaired_hole = last.get("hole")
        if repaired_hole not in {"predicate", "mapper"}:
            raise ValueError(f"{task_id} selected state has invalid repaired hole")
    return {
        "label": LABEL,
        "round": round_number,
        "selected_position": index,
        "selection_role": "loss-anchor" if index == 0 else "evidence-frontier-or-fill",
        "state_id": state.get("state_id"),
        "predicate": state.get("predicate"),
        "mapper": state.get("mapper"),
        "predicate_violations": predicate,
        "mapper_violations": mapper,
        "loss": loss,
        "exact": exact,
        "last_repaired_hole": repaired_hole,
    }


def _validate_rounds(
    run_dir: Path,
    metrics: Mapping[str, Any],
    task_id: str,
    *,
    observed_items: tuple[int, ...],
    predicate_facts: Mapping[int, bool],
    mapper_facts: Mapping[int, int],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    completed = _integer(metrics.get("rounds_completed"), name=f"{task_id}.rounds_completed")
    if not 1 <= completed <= 4:
        raise ValueError(f"{task_id}.rounds_completed must be in [1, 4]")
    expected_files = [
        run_dir / f"round-{round_number:02d}.json"
        for round_number in range(1, completed + 1)
    ]
    actual_files = sorted(run_dir.glob("round-*.json"))
    _expect(actual_files, expected_files, name=f"{task_id}.round files")
    traces: list[dict[str, object]] = []
    round_sha256: dict[str, str] = {}
    for round_number, path in enumerate(expected_files, start=1):
        round_sha256[str(round_number)] = sha256_file(path)
        document = _read_object(path)
        _expect(document.get("round"), round_number, name=f"{task_id}.round number")
        selected = _array(
            document.get("selected_beam"),
            name=f"{task_id}.round-{round_number}.selected_beam",
        )
        if not 1 <= len(selected) <= 2:
            raise ValueError(f"{task_id}.round-{round_number} selected beam width is invalid")
        for index, state in enumerate(selected):
            traces.append(
                _state_trace(
                    state,
                    task_id=task_id,
                    round_number=round_number,
                    index=index,
                    observed_items=observed_items,
                    predicate_facts=predicate_facts,
                    mapper_facts=mapper_facts,
                )
            )
    return traces, round_sha256


def _validate_exact_metrics(metrics: Mapping[str, Any], task_id: str) -> dict[str, object]:
    first_value = metrics.get("first_exact_proposal_slot")
    first_exact = (
        None
        if first_value is None
        else _integer(first_value, name=f"{task_id}.first_exact")
    )
    if first_exact is not None and not 1 <= first_exact <= 29:
        raise ValueError(f"{task_id}.first_exact_proposal_slot is outside the frozen budget")
    success_map = _object(
        metrics.get("success_by_proposal_slot_checkpoint"),
        name=f"{task_id}.success_by_checkpoint",
    )
    loss_map = _object(
        metrics.get("best_loss_by_proposal_slot_checkpoint"),
        name=f"{task_id}.best_loss_by_checkpoint",
    )
    success: dict[str, bool] = {}
    losses: dict[str, float] = {}
    for checkpoint in CHECKPOINTS:
        key = str(checkpoint)
        expected = first_exact is not None and first_exact <= checkpoint
        _expect(success_map.get(key), expected, name=f"{task_id}.success[{key}]")
        loss = _number(loss_map.get(key), name=f"{task_id}.best_loss[{key}]")
        if expected and loss != 0.0:
            raise ValueError(f"{task_id}.best_loss[{key}] must be zero after exact discovery")
        success[key] = expected
        losses[key] = loss
    _expect(metrics.get("success"), first_exact is not None, name=f"{task_id}.success")
    return {
        "first_exact_proposal_slot": first_exact,
        "exact_by_slot": success,
        "best_loss_by_slot": losses,
    }


def analyze_posthoc_evidence_frontier(
    repo_root: Path,
    study_protocol_path: Path,
    run_dirs: Mapping[str, Path],
) -> dict[str, object]:
    """Validate and summarize exactly the blind-02/03 developmental reruns."""

    if set(run_dirs) != set(TASK_IDS):
        raise ValueError(f"run directories must be exactly {TASK_IDS}")
    repo_root = repo_root.resolve()
    study_protocol_path = study_protocol_path.resolve()
    study = _read_object(study_protocol_path)
    _expect(study.get("schema"), "post-hoc-developmental-evidence-frontier-v1", name="study schema")
    _expect(
        study.get("protocol_status"),
        "developmental-post-hoc-frozen",
        name="study protocol status",
    )
    classification = _object(study.get("study_classification"), name="study classification")
    _expect(classification.get("blind"), False, name="study blind flag")
    _expect(classification.get("confirmatory"), False, name="study confirmatory flag")
    study_sha256 = sha256_file(study_protocol_path)

    freeze = _object(study.get("freeze_before_followup_calls"), name="freeze bindings")
    harness_path = _safe_repo_path(repo_root, freeze.get("harness_path"), name="harness path")
    harness_sha256 = sha256_file(harness_path)
    _expect(freeze.get("harness_sha256"), harness_sha256, name="frozen harness SHA-256")
    analysis_sha256 = sha256_file(Path(__file__))
    _expect(freeze.get("analysis_code_sha256"), analysis_sha256, name="frozen analysis SHA-256")
    _digest(freeze.get("prompt_template_sha256"), name="frozen prompt-template SHA-256")

    bindings = _object(study.get("bindings_to_original_study"), name="original bindings")
    task_records = _array(bindings.get("tasks_in_fixed_run_order"), name="fixed task records")
    task_order = [
        record.get("task_id") for record in task_records if isinstance(record, dict)
    ]
    _expect(task_order, list(TASK_IDS), name="fixed task order")

    tasks: list[dict[str, object]] = []
    for raw_record in task_records:
        record = _object(raw_record, name="task record")
        task_id = cast(str, record["task_id"])
        task_path = _safe_repo_path(repo_root, record.get("path"), name=f"{task_id}.task path")
        task_sha256 = sha256_file(task_path)
        _expect(record.get("sha256"), task_sha256, name=f"{task_id}.task SHA-256")
        public_task = _read_object(task_path)
        observed_items, predicate_facts, mapper_facts = _singleton_facts(
            public_task, task_id=task_id
        )

        run_dir = run_dirs[task_id].resolve()
        stored_protocol = _read_object(run_dir / "protocol.json")
        result = _read_object(run_dir / "result.json")
        _expect(result.get("schema"), RESULT_SCHEMA, name=f"{task_id}.result schema")
        _expect(result.get("protocol"), stored_protocol, name=f"{task_id}.stored/result protocol")
        _expect(stored_protocol.get("task_sha256"), task_sha256, name=f"{task_id}.run task SHA-256")
        _expect(
            stored_protocol.get("study_protocol_sha256"),
            study_sha256,
            name=f"{task_id}.study protocol SHA-256",
        )
        _expect(
            stored_protocol.get("harness_sha256"),
            harness_sha256,
            name=f"{task_id}.harness SHA-256",
        )
        for key, expected in EXPECTED_SEARCH.items():
            _expect(stored_protocol.get(key), expected, name=f"{task_id}.protocol.{key}")
        expected_seeds = {
            "provider_seed": record.get("provider_seed"),
            "tie_seed": record.get("tie_seed"),
            "random_baseline_seed": record.get("matched_random_seed"),
        }
        for key, expected in expected_seeds.items():
            _expect(stored_protocol.get(key), expected, name=f"{task_id}.{key}")

        inventory_sha256 = _validate_provider_seal(run_dir, result, task_id)
        metrics = _object(result.get("llm_beam_metrics"), name=f"{task_id}.llm metrics")
        exact = _validate_exact_metrics(metrics, task_id)
        traces, round_sha256 = _validate_rounds(
            run_dir,
            metrics,
            task_id,
            observed_items=observed_items,
            predicate_facts=predicate_facts,
            mapper_facts=mapper_facts,
        )
        result_sha256 = sha256_file(run_dir / "result.json")
        tasks.append(
            {
                "label": LABEL,
                "task_id": task_id,
                "task_sha256": task_sha256,
                "run_protocol_sha256": sha256_file(run_dir / "protocol.json"),
                "result_sha256": result_sha256,
                "round_file_sha256": round_sha256,
                "provider_inventory_sha256": inventory_sha256,
                "seeds": expected_seeds,
                **exact,
                "selected_state_violation_trajectory": traces,
            }
        )

    pass_by_29 = all(
        cast(dict[str, bool], task["exact_by_slot"])["29"] for task in tasks
    )
    return {
        "schema": SCHEMA,
        "label": LABEL,
        "study_classification": "post-hoc developmental mechanism check on two seen failures",
        "blind": False,
        "confirmatory": False,
        "study_protocol_sha256": study_sha256,
        "harness_sha256": harness_sha256,
        "analysis_code_sha256": analysis_sha256,
        "search_configuration": EXPECTED_SEARCH,
        "tasks": tasks,
        "developmental_gate": {
            "label": LABEL,
            "criterion": "both seen tasks exact by proposal slot 29",
            "passed": pass_by_29,
            "claim_boundary": (
                "A pass supports only proceeding to a separately frozen fresh blind study; "
                "it is not confirmatory evidence or a general search-speedup result."
            ),
        },
    }


def render_markdown(analysis: Mapping[str, Any]) -> str:
    """Render a visibly non-blind developmental report."""

    gate = _object(analysis.get("developmental_gate"), name="developmental gate")
    lines = [
        f"# {LABEL}",
        "",
        "This report reuses two previously inspected failures. It is not a blind or "
        "confirmatory experiment.",
        "",
        f"Developmental gate: **{'PASS' if gate.get('passed') is True else 'FAIL'}**.",
        "",
        "## Seen-task outcomes",
        "",
    ]
    for raw_task in _array(analysis.get("tasks"), name="analysis tasks"):
        task = _object(raw_task, name="analysis task")
        exact = _object(task.get("exact_by_slot"), name="exact checkpoints")
        lines.append(
            f"- **{LABEL} — {task['task_id']}**: exact by slot 21 = "
            f"{str(exact['21']).lower()}; exact by slot 29 = {str(exact['29']).lower()}; "
            f"first exact slot = {task['first_exact_proposal_slot']}."
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            str(gate["claim_boundary"]),
            "",
            f"Study protocol SHA-256: `{analysis['study_protocol_sha256']}`  ",
            f"Harness SHA-256: `{analysis['harness_sha256']}`  ",
            f"Analyzer SHA-256: `{analysis['analysis_code_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_analysis(output_dir: Path, analysis: Mapping[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "POST_HOC_NON_BLIND_ANALYSIS.json"
    markdown_path = output_dir / "POST_HOC_NON_BLIND_SUMMARY.md"
    json_path.write_bytes(canonical_bytes(analysis) + b"\n")
    markdown_path.write_text(render_markdown(analysis), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--study-protocol", type=Path, required=True)
    parser.add_argument("--blind-02-run", type=Path, required=True)
    parser.add_argument("--blind-03-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = analyze_posthoc_evidence_frontier(
        args.repo_root,
        args.study_protocol,
        {"blind-02": args.blind_02_run, "blind-03": args.blind_03_run},
    )
    write_analysis(args.output, analysis)


if __name__ == "__main__":
    main()

"""Run an interpreter-grounded, alternating LLM repair loop."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import cast

import httpx

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.core.evaluate import evaluate_expression, evaluate_program
from modelsmc_pbe.core.results import ScoredProgram
from modelsmc_pbe.core.scorer import ProgramScorer
from modelsmc_pbe.core.types import Node
from modelsmc_pbe.domain.ast import AstNode, canonical_key, clone_program
from modelsmc_pbe.grammar import arithmetic_expressions, filter_predicates
from research.automatic_repair_feedback import (
    AutomaticRepairFeedback,
    derive_automatic_feedback,
)

MODEL_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
_DSL_OPERATORS = {
    "Add": "add",
    "Subtract": "sub",
    "Multiply": "mul",
    "LessThan": "lt",
    "EqualInt": "eq",
    "And": "and",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def render_expression_dsl(expression: Mapping[str, object]) -> str:
    """Render the finite repair catalogs to a lossless prefix DSL."""

    kind = cast(str, expression["kind"])
    if kind == "Item":
        return "item"
    if kind == "IntLiteral":
        return cast(str, expression["intValue"])
    operator = _DSL_OPERATORS.get(kind)
    if operator is None:
        raise ValueError(f"unsupported repair-catalog expression kind: {kind}")
    left = cast(Mapping[str, object], expression["left"])
    right = cast(Mapping[str, object], expression["right"])
    return f"{operator}({render_expression_dsl(left)},{render_expression_dsl(right)})"


def _dsl_catalog(expressions: tuple[AstNode, ...]) -> dict[str, AstNode]:
    catalog = {render_expression_dsl(expression): expression for expression in expressions}
    if len(catalog) != len(expressions):
        raise ValueError("repair DSL is not bijective over the grammar catalog")
    return catalog


def _behavior_mask(predicate: Node, domain: tuple[int, ...]) -> str:
    decisions = []
    for item in domain:
        result = evaluate_expression(predicate, list(domain), item=item)
        if not isinstance(result, bool):
            raise TypeError("predicate catalog member did not evaluate to Bool")
        decisions.append("1" if result else "0")
    return "".join(decisions)


def _assemble_program(predicate: AstNode, mapper: AstNode) -> AstNode:
    return {
        "kind": "FoldRightProgram",
        "initial": {"kind": "EmptyIntList"},
        "reducer": {
            "kind": "IfThenElse",
            "condition": clone_program(predicate),
            "thenExpr": {
                "kind": "PrependInt",
                "head": clone_program(mapper),
                "tail": {"kind": "Accumulator"},
            },
            "elseExpr": {"kind": "Accumulator"},
        },
    }


def _mapper_violations(feedback: AutomaticRepairFeedback, mapper: Node) -> list[dict[str, object]]:
    violations = []
    for constraint in feedback.mapper_constraints:
        actual = evaluate_expression(mapper, [], item=constraint.item)
        if actual != constraint.expected_value:
            violations.append(
                {
                    "item": constraint.item,
                    "expected_value": constraint.expected_value,
                    "actual_value": actual,
                    "source_examples": constraint.source_examples,
                }
            )
    return violations


def _compatible_mapper_candidates(
    feedback: AutomaticRepairFeedback,
    catalog: Mapping[str, AstNode],
) -> tuple[str, ...]:
    required: dict[int, int] = {}
    for constraint in feedback.mapper_constraints:
        previous = required.get(constraint.item)
        if previous is not None and previous != constraint.expected_value:
            return ()
        required[constraint.item] = constraint.expected_value
    return tuple(
        dsl
        for dsl, mapper in sorted(catalog.items())
        if all(
            evaluate_expression(cast(Node, mapper), [], item=item) == expected
            for item, expected in required.items()
        )
    )


def _compatible_predicate_candidates(
    feedback: AutomaticRepairFeedback,
    catalog: Mapping[str, AstNode],
    current: Node,
    domain: tuple[int, ...],
) -> tuple[str, ...]:
    required: dict[int, bool] = {}
    for toggle in feedback.predicate_toggles:
        previous = required.get(toggle.item)
        if previous is not None and previous is not toggle.keep:
            return ()
        required[toggle.item] = toggle.keep
    if not required:
        return ()
    current_mask = _behavior_mask(current, domain)
    candidates: list[tuple[int, str]] = []
    for dsl, predicate in sorted(catalog.items()):
        if any(
            evaluate_expression(cast(Node, predicate), list(domain), item=item) is not keep
            for item, keep in required.items()
        ):
            continue
        mask = _behavior_mask(cast(Node, predicate), domain)
        distance = sum(left != right for left, right in zip(mask, current_mask, strict=True))
        candidates.append((distance, dsl))
    if not candidates:
        return ()
    minimum = min(distance for distance, _ in candidates)
    return tuple(dsl for distance, dsl in candidates if distance == minimum)


def _repair_request(
    *,
    model: str,
    reasoning_effort: str,
    include_reasoning: bool,
    max_tokens: int,
    seed: int,
    hole: str,
    domain: tuple[int, ...],
    predicate: AstNode,
    mapper: AstNode,
    feedback: AutomaticRepairFeedback,
    violations: list[dict[str, object]],
    allowed_candidates: tuple[str, ...],
) -> dict[str, object]:
    if hole == "mapper":
        field = "mapper"
        system = (
            "Repair only the mapper of a deterministic filter-map program. All evidence "
            "was produced mechanically by the interpreter from training examples. Keep "
            "the predicate frozen. Return exactly one grammar expression in the required "
            "JSON field and no prose in the final answer."
        )
        instructions = (
            "Mapper DSL: item, a declared integer constant, add(a,b), sub(a,b), or "
            "mul(a,b). The finite grammar permits only item/item or item/constant operand "
            "pairs, in either order when a constant is used. Satisfy the mapper constraints. "
            "A mapper cannot repair output-length mismatches."
        )
    else:
        field = "predicate"
        system = (
            "Repair only the predicate of a deterministic filter-map program. All evidence "
            "was produced mechanically by concrete interpreter counterfactuals on training "
            "examples. Keep the mapper frozen. Return exactly one grammar expression in the "
            "required JSON field and no prose in the final answer."
        )
        instructions = (
            "Predicate DSL atoms are lt(item,c), lt(c,item), or eq(item,c), where c is a "
            "declared integer constant. A predicate is either one atom or and(atom,atom). "
            "Apply the supported predicate toggles while changing as little behavior as "
            "possible."
        )
    user_payload = {
        "domain_order": domain,
        "current_predicate": render_expression_dsl(predicate),
        "current_predicate_mask": _behavior_mask(cast(Node, predicate), domain),
        "current_mapper": render_expression_dsl(mapper),
        "interpreter_feedback": feedback.to_dict(),
        "mapper_violations": violations,
        "repair_hole": hole,
        "allowed_candidates": allowed_candidates,
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [field],
        "properties": {field: {"type": "string", "enum": list(allowed_candidates)}},
    }
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": instructions
                + "\nMachineEvidence="
                + _canonical_bytes(user_payload).decode(),
            },
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "seed": seed,
        "reasoning_effort": reasoning_effort,
        "include_reasoning": include_reasoning,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"execution_guided_{hole}_repair",
                "strict": True,
                "schema": schema,
            },
        },
    }


async def _call_model(
    *,
    base_url: str,
    timeout_seconds: float,
    payload: dict[str, object],
    stage: Path,
    field: str,
) -> str:
    request_bytes = _canonical_bytes(payload)
    archived_request = request_bytes + b"\n"
    (stage / "request.json").write_bytes(archived_request)
    (stage / "request.sha256").write_text(
        hashlib.sha256(archived_request).hexdigest() + "\n", encoding="utf-8"
    )
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            base_url.rstrip("/") + "/chat/completions",
            content=request_bytes,
            headers={"Content-Type": "application/json"},
        )
    elapsed = time.perf_counter() - started
    response_bytes = response.content
    (stage / "response.json").write_bytes(response_bytes)
    (stage / "response.sha256").write_text(
        hashlib.sha256(response_bytes).hexdigest() + "\n", encoding="utf-8"
    )
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    _write_json(
        stage / "transport.json",
        {
            "elapsed_seconds": elapsed,
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage"),
        },
    )
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"model did not stop cleanly: {choice.get('finish_reason')}")
    content = choice["message"].get("content")
    if not isinstance(content, str):
        raise ValueError("model returned no final JSON content")
    decoded = json.loads(content)
    if not isinstance(decoded, dict) or set(decoded) != {field}:
        raise ValueError(f"model final JSON must contain exactly {field!r}")
    value = decoded[field]
    if not isinstance(value, str) or not value:
        raise ValueError(f"model field {field!r} must be a nonempty string")
    return value


async def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    config = load_experiment_config(args.task)
    constants = tuple(sorted(set(config.spec.integer_constants)))
    domain = tuple(
        sorted(
            {
                item
                for example in config.spec.examples
                for item in cast(list[int], example.input_value)
            }
        )
    )
    mapper_catalog = _dsl_catalog(arithmetic_expressions("Item", constants))
    predicate_catalog = _dsl_catalog(filter_predicates(constants))
    try:
        mapper = clone_program(mapper_catalog[args.initial_mapper])
        predicate = clone_program(predicate_catalog[args.initial_predicate])
    except KeyError as error:
        raise ValueError(f"initial expression is outside the grammar: {error.args[0]}") from error

    task_bytes = args.task.resolve().read_bytes()
    _write_json(
        output / "manifest.json",
        {
            "schema": "execution-guided-repair-v1",
            "task": str(args.task.resolve()),
            "task_sha256": hashlib.sha256(task_bytes).hexdigest(),
            "model": args.model,
            "model_revision": MODEL_REVISION,
            "method_interpretation": (
                "heuristic alternating local repair; mapper constraints are conditional "
                "on the frozen predicate, and predicate toggles are concrete per-example "
                "counterfactual repairs; no normalized proposal probability is claimed"
            ),
            "reasoning_effort": args.reasoning_effort,
            "include_reasoning": args.include_reasoning,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "initial_predicate": args.initial_predicate,
            "initial_mapper": args.initial_mapper,
            "predicate_catalog_size": len(predicate_catalog),
            "mapper_catalog_size": len(mapper_catalog),
            "gold_visible": False,
        },
    )

    scorer = ProgramScorer(config)
    history: list[dict[str, object]] = []
    seen_states: set[tuple[str, str]] = set()
    status = "iteration-budget-exhausted"
    for iteration in range(args.max_iterations + 1):
        state = (canonical_key(predicate), canonical_key(mapper))
        if state in seen_states:
            status = "repeated-state"
            break
        seen_states.add(state)
        feedback = derive_automatic_feedback(config.spec, cast(Node, predicate), cast(Node, mapper))
        current_score = scorer.score(_assemble_program(predicate, mapper))
        if not isinstance(current_score, ScoredProgram):
            raise ValueError(f"current program was rejected: {current_score.reason}")
        stage = output / f"iteration-{iteration:02d}"
        stage.mkdir()
        _write_json(stage / "evidence.json", feedback.to_dict())
        record: dict[str, object] = {
            "iteration": iteration,
            "predicate": render_expression_dsl(predicate),
            "predicate_mask": _behavior_mask(cast(Node, predicate), domain),
            "mapper": render_expression_dsl(mapper),
            "zero_loss": feedback.zero_loss,
            "score": asdict(current_score),
        }
        history.append(record)
        if feedback.zero_loss:
            status = "zero-loss"
            break
        if iteration == args.max_iterations:
            break

        violations = _mapper_violations(feedback, cast(Node, mapper))
        if violations:
            hole = "mapper"
            catalog = mapper_catalog
            allowed_candidates = _compatible_mapper_candidates(feedback, mapper_catalog)
        elif feedback.predicate_toggles:
            hole = "predicate"
            catalog = predicate_catalog
            allowed_candidates = _compatible_predicate_candidates(
                feedback,
                predicate_catalog,
                cast(Node, predicate),
                domain,
            )
        else:
            status = "no-supported-one-step-repair"
            break
        if not allowed_candidates:
            status = f"no-consistent-{hole}-candidate"
            break
        record["repair_hole"] = hole
        payload = _repair_request(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            include_reasoning=args.include_reasoning,
            max_tokens=args.max_tokens,
            seed=args.seed + iteration,
            hole=hole,
            domain=domain,
            predicate=predicate,
            mapper=mapper,
            feedback=feedback,
            violations=violations,
            allowed_candidates=allowed_candidates,
        )
        try:
            proposal_dsl = await _call_model(
                base_url=args.base_url,
                timeout_seconds=args.timeout_seconds,
                payload=payload,
                stage=stage,
                field=hole,
            )
            proposal = catalog.get(proposal_dsl)
            if proposal is None:
                raise ValueError(f"model proposal is outside the {hole} catalog: {proposal_dsl}")
            if proposal_dsl not in allowed_candidates:
                raise ValueError(f"model proposal violates the supported {hole} constraints")
        except Exception as error:
            _write_json(
                stage / "failure.json", {"type": type(error).__name__, "detail": str(error)}
            )
            status = "invalid-model-repair"
            break
        record["proposal"] = proposal_dsl
        proposed_predicate = predicate if hole == "mapper" else proposal
        proposed_mapper = proposal if hole == "mapper" else mapper
        proposed_score = scorer.score(_assemble_program(proposed_predicate, proposed_mapper))
        if not isinstance(proposed_score, ScoredProgram):
            raise ValueError(f"grammar-valid proposal was rejected: {proposed_score.reason}")
        improving = proposed_score.total_loss < current_score.total_loss
        _write_json(
            stage / "model-proposal.json",
            {
                "hole": hole,
                "dsl": proposal_dsl,
                "canonical_ast": canonical_key(proposal),
                "score": asdict(proposed_score),
                "accepted": improving,
            },
        )
        if not improving:
            status = "non-improving-model-proposal"
            break
        predicate = clone_program(proposed_predicate)
        mapper = clone_program(proposed_mapper)

    program = _assemble_program(predicate, mapper)
    final_checks = []
    for index, example in enumerate(config.spec.examples, start=1):
        actual = evaluate_program(cast(Node, program), example.input_value)
        final_checks.append(
            {
                "source_example": index,
                "expected": example.output_value,
                "actual": actual,
                "exact": actual == example.output_value,
            }
        )
    summary = {
        "status": status,
        "zero_loss": all(cast(bool, check["exact"]) for check in final_checks),
        "iterations": history,
        "final_predicate": render_expression_dsl(predicate),
        "final_predicate_mask": _behavior_mask(cast(Node, predicate), domain),
        "final_mapper": render_expression_dsl(mapper),
        "final_program": program,
        "final_checks": final_checks,
        "llm_calls_attempted": sum("repair_hole" in record for record in history),
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--timeout-seconds", type=float, default=420.0)
    parser.add_argument("--seed", type=int, default=90173)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--initial-mapper", default="item")
    parser.add_argument("--initial-predicate", default="and(lt(-1,item),lt(item,3))")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

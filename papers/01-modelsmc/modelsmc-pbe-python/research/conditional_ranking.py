"""Replayable staged-reasoning diagnostic for finite filter/map catalogs.

This is an experiment harness, not a search implementation.  It deliberately
keeps model-visible scoring evidence separate from the retrospective oracle
join used to calculate ranks and proposal mass.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import hmac
import json
import math
import os
import random
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.core.types import child
from modelsmc_pbe.domain import canonical_key
from modelsmc_pbe.grammar import bounded_square_target
from modelsmc_pbe.induction import SkeletonKind
from modelsmc_pbe.proposals import (
    CandidateKind,
    CandidateScoreRequest,
    CandidateSequenceScore,
)
from modelsmc_pbe.proposals.labels import (
    CompatibilityMapping,
    CompatibilityProgram,
    SemanticPromptProtocol,
    build_compatibility_prompt,
    build_program_paths,
    compatibility_template_version,
    extract_label_pair,
)
from modelsmc_pbe.proposals.vllm_response import decode_candidate_scores
from modelsmc_pbe.search.importance import FactorizedSupportBuilder, ImportanceSMCOptions

PROTOCOL_FILENAME = "protocol-conditional-ranking-gptoss120b-v1.json"
HARNESS_SCHEMA = "conditional-ranking-diagnostic-v1"
TIE_BREAK_DOMAIN = "modelsmc-pbe-conditional-ranking-v1"
NULL_DIGEST = hashlib.sha256(b"").hexdigest()
LABEL_PROTOCOL = SemanticPromptProtocol.HARMONY_GPT_OSS_V1


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _server_root(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized[:-3] if normalized.endswith("/v1") else normalized


def _git_metadata(project_root: Path) -> dict[str, object]:
    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
    }


def _inventory_bytes(root: Path) -> bytes:
    excluded = {"SHA256SUMS", "BUNDLE_SHA256"}
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        lines.append(f"{_sha256_file(path)}  {relative}\n")
    return "".join(lines).encode()


def seal_bundle(root: Path) -> str:
    """Hash every existing file in ``root`` and return the inventory digest."""

    inventory = _inventory_bytes(root)
    digest = _sha256_bytes(inventory)
    (root / "SHA256SUMS").write_bytes(inventory)
    (root / "BUNDLE_SHA256").write_text(digest + "\n", encoding="utf-8")
    return digest


def verify_bundle(root: Path, expected: str) -> None:
    recorded = (root / "BUNDLE_SHA256").read_text(encoding="utf-8").strip()
    if recorded != expected:
        raise ValueError(f"bundle argument {expected} does not match recorded digest {recorded}")
    actual_inventory = _inventory_bytes(root)
    stored_inventory = (root / "SHA256SUMS").read_bytes()
    if actual_inventory != stored_inventory:
        raise ValueError(f"bundle inventory changed after sealing: {root}")
    actual = _sha256_bytes(actual_inventory)
    if actual != expected:
        raise ValueError(f"bundle digest changed after sealing: {root}")


def _fresh_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostic path: {path}")
    path.mkdir(parents=True)


def _permutation(count: int, seed: int, label: str) -> tuple[int, ...]:
    generator = random.Random(f"{HARNESS_SCHEMA}:{seed}:{label}")
    values = list(range(count))
    generator.shuffle(values)
    return tuple(values)


def _examples_context(task: Mapping[str, object], order: Sequence[int]) -> str:
    examples = cast(list[object], task["examples"])
    permuted = [examples[index] for index in order]
    return json.dumps(permuted, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fixed_skeleton() -> str:
    return (
        "foldr((item, acc) => if ?predicate(item) then ?mapped_value(item) :: acc else acc, [], xs)"
    )


def _reasoning_payload(
    *,
    protocol: Mapping[str, object],
    task: Mapping[str, object],
    example_order: Sequence[int],
    seed: int,
) -> dict[str, object]:
    model = cast(Mapping[str, object], protocol["model"])
    generation = cast(Mapping[str, object], model["reasoning_generation"])
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "observations",
            "filter_hypothesis",
            "mapping_hypothesis",
            "uncertainties",
        ],
        "properties": {
            "observations": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 8,
            },
            "filter_hypothesis": {"type": "string"},
            "mapping_hypothesis": {"type": "string"},
            "uncertainties": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6,
            },
        },
    }
    system = (
        "Analyze a programming-by-example task. Infer what the two holes in the fixed "
        "skeleton must do using only the supplied input/output examples. Do not invent "
        "extra observations and do not assume access to a candidate catalog, oracle, or "
        "execution loss. Return the required compact JSON summary."
    )
    user = (
        f"Signature={json.dumps(task['signature'], sort_keys=True, separators=(',', ':'))}\n"
        f"FixedSkeleton={_fixed_skeleton()}\n"
        f"Examples={_examples_context(task, example_order)}\n"
        "Explain which input elements appear to survive in order and how surviving "
        "values appear to be transformed. State uncertainty where the sparse examples "
        "do not uniquely determine behavior."
    )
    return {
        "model": model["served_name"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "max_tokens": generation["max_completion_tokens"],
        "seed": seed,
        "reasoning_effort": generation["reasoning_effort"],
        "include_reasoning": True,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pbe_hole_reasoning",
                "strict": True,
                "schema": schema,
            },
        },
    }


def _validate_reasoning_summary(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "observations",
        "filter_hypothesis",
        "mapping_hypothesis",
        "uncertainties",
    }:
        raise ValueError("reasoning final JSON has the wrong object fields")
    for field in ("observations", "uncertainties"):
        items = value[field]
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            raise ValueError(f"reasoning field {field} must be a string array")
    if not 1 <= len(cast(list[object], value["observations"])) <= 8:
        raise ValueError("reasoning observations must contain one to eight items")
    if len(cast(list[object], value["uncertainties"])) > 6:
        raise ValueError("reasoning uncertainties must contain at most six items")
    for field in ("filter_hypothesis", "mapping_hypothesis"):
        if not isinstance(value[field], str) or not cast(str, value[field]).strip():
            raise ValueError(f"reasoning field {field} must be a nonempty string")
    return cast(dict[str, object], value)


def _support_snapshot(project_root: Path, task_path: Path) -> dict[str, object]:
    config = load_experiment_config(task_path)
    support = FactorizedSupportBuilder(
        spec=config.spec,
        smc=config.smc,
        options=ImportanceSMCOptions(
            multi_family=True,
            support_limit=250_000,
            hole_state_limit=250_000,
            hole_max_cost=3,
        ),
    ).build()
    family = next(
        item
        for item in support.families
        if item.hypothesis.kind is SkeletonKind.FOLD_RIGHT_FILTER_MAP
    )
    catalogs = {catalog.hole.name: catalog for catalog in family.catalogs}
    predicate = catalogs["predicate"]
    mapper = catalogs["mapped_value"]
    if len(predicate.fillings) != 600 or len(mapper.fillings) != 60:
        raise RuntimeError("diagnostic catalog sizes changed")
    if family.support_count != 36_000 or support.support_states != 36_198:
        raise RuntimeError("diagnostic complete support counts changed")
    if family.deduction.examples_for("predicate") or family.deduction.examples_for("mapped_value"):
        raise RuntimeError("sparse-task hole deduction is no longer empty")
    return {
        "project_root": str(project_root),
        "support_states": support.support_states,
        "family_support_states": family.support_count,
        "deduction_examples": {"predicate": 0, "mapped_value": 0},
        "predicates": [filling.key for filling in predicate.fillings],
        "mappers": [filling.key for filling in mapper.fillings],
    }


def prepare(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    output = args.output.resolve()
    _fresh_directory(output)
    prepared = output / "prepared"
    prepared.mkdir()

    protocol_source = args.protocol.resolve()
    protocol = cast(dict[str, object], _read_json(protocol_source))
    if protocol.get("status") != "frozen-before-provider-execution":
        raise ValueError("diagnostic protocol is not frozen before provider execution")
    task_record = cast(Mapping[str, object], protocol["task"])
    task_source = project_root / cast(str, task_record["path"])
    task = cast(dict[str, object], _read_json(task_source))
    support = _support_snapshot(project_root, task_source)
    predicates = cast(list[str], support.pop("predicates"))
    mappers = cast(list[str], support.pop("mappers"))

    oracle = cast(Mapping[str, object], protocol["oracle_reference"])
    exact_predicate_indices = tuple(
        int(value) for value in cast(list[object], oracle["predicate_indices_zero_based"])
    )
    if exact_predicate_indices != (238, 537):
        raise ValueError("counterfactual predicate-prefix indices changed")

    shutil.copyfile(protocol_source, prepared / "protocol.json")
    shutil.copyfile(task_source, prepared / "task.json")
    _write_json(prepared / "support.json", support)
    _write_json(
        prepared / "catalogs" / "predicates.json",
        [{"key": key, "sha256": _sha256_bytes(key.encode())} for key in predicates],
    )
    _write_json(
        prepared / "catalogs" / "mappers.json",
        [{"key": key, "sha256": _sha256_bytes(key.encode())} for key in mappers],
    )
    _write_json(
        prepared / "counterfactual_mapper_prefixes.json",
        [
            {
                "catalog_index_zero_based": index,
                "predicate_key": predicates[index],
                "predicate_sha256": _sha256_bytes(predicates[index].encode()),
            }
            for index in exact_predicate_indices
        ],
    )

    design = cast(Mapping[str, object], protocol["design"])
    seeds = tuple(int(value) for value in cast(list[object], design["trial_seeds"]))
    trials = []
    for seed in seeds:
        example_order = _permutation(len(cast(list[object], task["examples"])), seed, "examples")
        predicate_order = _permutation(len(predicates), seed, "predicate-provider-order")
        mapper_orders = {
            str(index): _permutation(len(mappers), seed, f"mapper-provider-order:prefix-{index}")
            for index in exact_predicate_indices
        }
        request = _reasoning_payload(
            protocol=protocol,
            task=task,
            example_order=example_order,
            seed=seed,
        )
        request_path = prepared / "reasoning_requests" / f"trial-{seed}.json"
        _write_json(request_path, request)
        trials.append(
            {
                "seed": seed,
                "example_order": example_order,
                "predicate_provider_order": predicate_order,
                "mapper_provider_orders": mapper_orders,
                "reasoning_request": request_path.relative_to(prepared).as_posix(),
                "reasoning_request_sha256": _sha256_file(request_path),
            }
        )
    _write_json(prepared / "trials.json", trials)
    _write_json(
        prepared / "pointwise_prompt_contract.json",
        {
            "label_protocol": LABEL_PROTOCOL.value,
            "template_version": compatibility_template_version(LABEL_PROTOCOL),
            "candidate_text": {
                "predicate": "Predicate candidate for ?predicate:\\n{canonical_ast}",
                "mapper": "Mapper candidate for ?mapped_value:\\n{canonical_ast}",
            },
            "reasoning_prefix_source": (
                "validated final JSON only; hidden analysis is archived but not injected"
            ),
            "gold_labels_visible": False,
        },
    )
    _write_json(
        prepared / "manifest.json",
        {
            "schema": HARNESS_SCHEMA,
            "created_at": _now(),
            "protocol_source": str(protocol_source),
            "protocol_sha256": _sha256_file(protocol_source),
            "task_source": str(task_source),
            "task_sha256": _sha256_file(task_source),
            "harness_sha256": _sha256_file(Path(__file__)),
            "git": _git_metadata(project_root),
            "environment": {
                "hostname": os.uname().nodename,
                "python": os.sys.version,
            },
        },
    )
    digest = seal_bundle(prepared)
    print(digest, flush=True)


async def _tokenize_chat(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    messages: object,
) -> int:
    response = await client.post(
        f"{_server_root(base_url)}/tokenize",
        content=_canonical_bytes({"model": model, "messages": messages}),
        headers={"content-type": "application/json"},
    )
    response.raise_for_status()
    body = response.json()
    count = body.get("count") if isinstance(body, dict) else None
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("tokenize response has no integer count")
    return count


async def _post_with_retries(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    request_bytes: bytes,
    attempts_dir: Path,
    timeout_seconds: float,
) -> bytes:
    attempts_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        started = time.perf_counter()
        error: str | None = None
        status: int | None = None
        response_bytes = b""
        try:
            response = await client.post(
                endpoint,
                content=request_bytes,
                headers={"content-type": "application/json"},
                timeout=timeout_seconds,
            )
            status = response.status_code
            response_bytes = response.content
        except (httpx.TimeoutException, httpx.TransportError) as exception:
            error = f"{type(exception).__name__}: {exception}"
        elapsed = time.perf_counter() - started
        if response_bytes:
            with gzip.open(attempts_dir / f"attempt-{attempt}.body.gz", "wb") as stream:
                stream.write(response_bytes)
        _write_json(
            attempts_dir / f"attempt-{attempt}.meta.json",
            {
                "attempt": attempt,
                "elapsed_seconds": elapsed,
                "error": error,
                "request_sha256": _sha256_bytes(request_bytes),
                "response_bytes": len(response_bytes),
                "response_sha256": _sha256_bytes(response_bytes),
                "status": status,
            },
        )
        if error is None and status is not None and status < 500:
            if status >= 400:
                raise RuntimeError(f"provider returned non-retriable HTTP {status}")
            return response_bytes
        if attempt == 3:
            detail = error if error is not None else f"HTTP {status}"
            raise RuntimeError(f"provider failed after three identical attempts: {detail}")
    raise AssertionError("unreachable retry loop")


async def _run_reasoning(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    prepared = output / "prepared"
    verify_bundle(prepared, args.prepared_sha256)
    reasoning = output / "reasoning"
    _fresh_directory(reasoning)
    protocol = cast(Mapping[str, object], _read_json(prepared / "protocol.json"))
    model_record = cast(Mapping[str, object], protocol["model"])
    generation = cast(Mapping[str, object], model_record["reasoning_generation"])
    trials = cast(list[Mapping[str, object]], _read_json(prepared / "trials.json"))
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=4)) as client:
        for trial in trials:
            seed = int(trial["seed"])
            request_path = prepared / cast(str, trial["reasoning_request"])
            request_bytes = request_path.read_bytes().rstrip(b"\n")
            request = cast(dict[str, object], json.loads(request_bytes))
            token_count = await _tokenize_chat(
                client,
                base_url=args.base_url,
                model=cast(str, model_record["served_name"]),
                messages=request["messages"],
            )
            maximum = int(model_record["max_model_len"])
            completion = int(generation["max_completion_tokens"])
            if token_count + completion > maximum:
                raise ValueError(
                    f"trial {seed} chat path exceeds context: {token_count}+{completion}>{maximum}"
                )
            trial_dir = reasoning / f"trial-{seed}"
            trial_dir.mkdir()
            _write_json(
                trial_dir / "token_preflight.json",
                {
                    "input_tokens": token_count,
                    "max_completion_tokens": completion,
                    "max_model_len": maximum,
                },
            )
            response_bytes = await _post_with_retries(
                client,
                endpoint=endpoint,
                request_bytes=request_bytes,
                attempts_dir=trial_dir / "attempts",
                timeout_seconds=args.timeout_seconds,
            )
            body = json.loads(response_bytes)
            choices = body.get("choices") if isinstance(body, dict) else None
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("reasoning response must contain exactly one choice")
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
                raise ValueError("reasoning response did not finish with stop")
            message = choice.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError("reasoning response has no final string content")
            summary = _validate_reasoning_summary(json.loads(message["content"]))
            hidden = message.get("reasoning")
            if hidden is not None and not isinstance(hidden, str):
                raise ValueError("reasoning response has a non-string reasoning field")
            _write_json(trial_dir / "final.json", summary)
            (trial_dir / "hidden_reasoning.txt").write_text(
                "" if hidden is None else hidden,
                encoding="utf-8",
            )
            _write_json(
                trial_dir / "selected.json",
                {
                    "finish_reason": choice["finish_reason"],
                    "final_sha256": _sha256_file(trial_dir / "final.json"),
                    "hidden_reasoning_sha256": _sha256_file(trial_dir / "hidden_reasoning.txt"),
                    "response_sha256": _sha256_bytes(response_bytes),
                    "usage": body.get("usage"),
                },
            )
            print(f"reasoning trial {seed} complete", flush=True)
    _write_json(
        reasoning / "manifest.json",
        {
            "schema": HARNESS_SCHEMA,
            "completed_at": _now(),
            "prepared_bundle_sha256": args.prepared_sha256,
            "base_url": args.base_url,
        },
    )
    print(seal_bundle(reasoning), flush=True)


def run_reasoning(args: argparse.Namespace) -> None:
    asyncio.run(_run_reasoning(args))


def _base_context(
    *,
    task: Mapping[str, object],
    example_order: Sequence[int],
    reasoning: Mapping[str, object] | None,
) -> str:
    text = (
        f"Signature: {json.dumps(task['signature'], sort_keys=True, separators=(',', ':'))}\n"
        f"Fixed skeleton: {_fixed_skeleton()}\n"
        f"Observed examples: {_examples_context(task, example_order)}\n"
        "The candidate below fills exactly one named hole. Use only the observed examples."
    )
    if reasoning is not None:
        text += (
            "\nFrozen analysis previously produced by this same model from only the raw "
            "task and skeleton:\n"
            + json.dumps(reasoning, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    return text


def _predicate_context(base: str) -> str:
    return (
        base + "\nCurrent slate: predicate candidates for ?predicate. A candidate is compatible "
        "when its keep/drop behavior can account for the order and cardinality of the "
        "observed outputs while allowing one deterministic mapped-value expression."
    )


def _mapper_context(base: str, predicate_key: str) -> str:
    return (
        base + "\nCurrent slate: mapper candidates for ?mapped_value under the following predicate "
        "under evaluation:\n"
        + predicate_key
        + "\nA mapper candidate is compatible exactly when the resulting fixed skeleton can "
        "match all observed input/output examples."
    )


def _program_text(phase: str, key: str) -> str:
    label = (
        "Predicate candidate for ?predicate"
        if phase == "predicate"
        else ("Mapper candidate for ?mapped_value")
    )
    return f"{label}:\n{key}"


async def _tokenize_text(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    endpoint: str,
    model: str,
    text: str,
) -> int:
    async with semaphore:
        response = await client.post(
            endpoint,
            content=_canonical_bytes({"model": model, "prompt": text, "add_special_tokens": False}),
            headers={"content-type": "application/json"},
            timeout=60.0,
        )
    response.raise_for_status()
    body = response.json()
    count = body.get("count") if isinstance(body, dict) else None
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("tokenize response has no integer count")
    return count


async def _preflight_score_paths(
    *,
    scoring_plan: Sequence[Mapping[str, object]],
    prepared: Path,
    scoring_dir: Path,
    base_url: str,
    model: str,
    max_model_len: int,
    concurrency: int,
) -> None:
    predicate_catalog = cast(
        list[Mapping[str, str]], _read_json(prepared / "catalogs" / "predicates.json")
    )
    mapper_catalog = cast(
        list[Mapping[str, str]], _read_json(prepared / "catalogs" / "mappers.json")
    )
    candidates = {
        "predicate": tuple(item["key"] for item in predicate_catalog),
        "mapper": tuple(item["key"] for item in mapper_catalog),
    }
    endpoint = f"{_server_root(base_url)}/tokenize"
    semaphore = asyncio.Semaphore(concurrency)
    records: list[dict[str, object]] = []
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency)) as client:
        for slate in scoring_plan:
            context = (scoring_dir / cast(str, slate["context_path"])).read_text(encoding="utf-8")
            phase = cast(str, slate["phase"])
            prompt = build_compatibility_prompt(context, LABEL_PROTOCOL)
            paths = []
            identities = []
            for key in candidates[phase]:
                program = CompatibilityProgram(
                    program_key=key,
                    program_text=_program_text(phase, key),
                )
                for path in build_program_paths(program, LABEL_PROTOCOL):
                    text = prompt + path.candidate
                    paths.append(text)
                    identities.append(
                        {
                            "candidate_sha256": _sha256_bytes(key.encode()),
                            "mapping": path.mapping.value,
                            "label": path.label.value,
                            "text_sha256": _sha256_bytes(text.encode()),
                        }
                    )
            counts = await asyncio.gather(
                *(
                    _tokenize_text(
                        client,
                        semaphore,
                        endpoint=endpoint,
                        model=model,
                        text=text,
                    )
                    for text in paths
                )
            )
            maximum = max(counts)
            if maximum > max_model_len:
                raise ValueError(
                    f"score slate {slate['slate_id']} exceeds context: {maximum}>{max_model_len}"
                )
            for identity, count in zip(identities, counts, strict=True):
                identity["tokens"] = count
                identity["slate_id"] = slate["slate_id"]
                records.append(identity)
            print(
                f"token preflight {slate['slate_id']}: {len(counts)} paths, max {maximum}",
                flush=True,
            )
    preflight_path = scoring_dir / "token_preflight.jsonl.gz"
    with gzip.open(preflight_path, "wt", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(_canonical_bytes(record).decode() + "\n")
    _write_json(
        scoring_dir / "token_preflight_summary.json",
        {
            "paths": len(records),
            "maximum_tokens": max(int(record["tokens"]) for record in records),
            "max_model_len": max_model_len,
            "tokenizer_endpoint": endpoint,
        },
    )


async def _prepare_scoring(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    prepared = output / "prepared"
    reasoning = output / "reasoning"
    verify_bundle(prepared, args.prepared_sha256)
    verify_bundle(reasoning, args.reasoning_sha256)
    scoring_dir = output / "scoring_plan"
    _fresh_directory(scoring_dir)
    protocol = cast(Mapping[str, object], _read_json(prepared / "protocol.json"))
    task = cast(Mapping[str, object], _read_json(prepared / "task.json"))
    trials = cast(list[Mapping[str, object]], _read_json(prepared / "trials.json"))
    prefixes = cast(
        list[Mapping[str, object]],
        _read_json(prepared / "counterfactual_mapper_prefixes.json"),
    )
    plan = []
    for trial in trials:
        seed = int(trial["seed"])
        example_order = tuple(int(value) for value in cast(list[object], trial["example_order"]))
        final = cast(Mapping[str, object], _read_json(reasoning / f"trial-{seed}" / "final.json"))
        for arm in ("N", "R"):
            model_reasoning = None if arm == "N" else final
            base = _base_context(
                task=task,
                example_order=example_order,
                reasoning=model_reasoning,
            )
            predicate_context = _predicate_context(base)
            predicate_path = Path("contexts") / f"trial-{seed}" / arm / "predicate.txt"
            (scoring_dir / predicate_path).parent.mkdir(parents=True, exist_ok=True)
            (scoring_dir / predicate_path).write_text(predicate_context, encoding="utf-8")
            plan.append(
                {
                    "slate_id": f"trial-{seed}-{arm}-predicate",
                    "trial_seed": seed,
                    "arm": arm,
                    "phase": "predicate",
                    "prefix_catalog_index_zero_based": None,
                    "prefix_sha256": NULL_DIGEST,
                    "context_path": predicate_path.as_posix(),
                    "context_sha256": _sha256_file(scoring_dir / predicate_path),
                    "provider_order": trial["predicate_provider_order"],
                }
            )
            mapper_orders = cast(Mapping[str, object], trial["mapper_provider_orders"])
            for prefix in prefixes:
                prefix_index = int(prefix["catalog_index_zero_based"])
                predicate_key = cast(str, prefix["predicate_key"])
                mapper_context = _mapper_context(base, predicate_key)
                mapper_path = (
                    Path("contexts") / f"trial-{seed}" / arm / f"mapper-prefix-{prefix_index}.txt"
                )
                (scoring_dir / mapper_path).write_text(mapper_context, encoding="utf-8")
                plan.append(
                    {
                        "slate_id": f"trial-{seed}-{arm}-mapper-prefix-{prefix_index}",
                        "trial_seed": seed,
                        "arm": arm,
                        "phase": "mapper",
                        "prefix_catalog_index_zero_based": prefix_index,
                        "prefix_sha256": prefix["predicate_sha256"],
                        "context_path": mapper_path.as_posix(),
                        "context_sha256": _sha256_file(scoring_dir / mapper_path),
                        "provider_order": mapper_orders[str(prefix_index)],
                    }
                )
    _write_json(scoring_dir / "plan.json", plan)
    model = cast(Mapping[str, object], protocol["model"])
    await _preflight_score_paths(
        scoring_plan=plan,
        prepared=prepared,
        scoring_dir=scoring_dir,
        base_url=args.base_url,
        model=cast(str, model["served_name"]),
        max_model_len=int(model["max_model_len"]),
        concurrency=args.tokenize_concurrency,
    )
    _write_json(
        scoring_dir / "manifest.json",
        {
            "schema": HARNESS_SCHEMA,
            "created_at": _now(),
            "prepared_bundle_sha256": args.prepared_sha256,
            "reasoning_bundle_sha256": args.reasoning_sha256,
            "base_url": args.base_url,
            "slates": len(plan),
        },
    )
    print(seal_bundle(scoring_dir), flush=True)


def prepare_scoring(args: argparse.Namespace) -> None:
    asyncio.run(_prepare_scoring(args))


def _score_record(
    *,
    program: CompatibilityProgram,
    path_specs: Sequence[object],
    raw_scores: Sequence[CandidateSequenceScore],
    prompt_sha256: str,
) -> dict[str, object]:
    specs = cast(Sequence[Any], path_specs)
    plus_a, plus_b, plus_proof = extract_label_pair(
        CompatibilityMapping.A_IS_COMPATIBLE,
        specs[0],
        specs[1],
        raw_scores[0],
        raw_scores[1],
    )
    minus_a, minus_b, minus_proof = extract_label_pair(
        CompatibilityMapping.B_IS_COMPATIBLE,
        specs[2],
        specs[3],
        raw_scores[2],
        raw_scores[3],
    )
    score = 0.5 * (
        plus_proof.label_a_logprob
        - plus_proof.label_b_logprob
        + minus_proof.label_b_logprob
        - minus_proof.label_a_logprob
    )
    return {
        "candidate_key": program.program_key,
        "candidate_sha256": program.sha256,
        "compatibility_log_score": score,
        "paths": [asdict(item) for item in (plus_a, plus_b, minus_a, minus_b)],
        "boundary_proofs": [asdict(plus_proof), asdict(minus_proof)],
        "prompt_sha256": prompt_sha256,
        "template_version": compatibility_template_version(LABEL_PROTOCOL),
    }


async def _score_http_chunk(
    *,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    endpoint: str,
    model: str,
    prompt: str,
    programs: Sequence[CompatibilityProgram],
    raw_dir: Path,
    chunk_index: int,
    timeout_seconds: float,
) -> list[dict[str, object]]:
    specs = tuple(
        path for program in programs for path in build_program_paths(program, LABEL_PROTOCOL)
    )
    candidates = tuple(spec.candidate for spec in specs)
    prompts = [prompt + candidate for candidate in candidates]
    payload = {
        "model": model,
        "prompt": prompts,
        "max_tokens": 0,
        "echo": True,
        "prompt_logprobs": 0,
        "stream": False,
        "add_special_tokens": False,
    }
    request_bytes = _canonical_bytes(payload)
    chunk_dir = raw_dir / f"request-{chunk_index:04d}"
    chunk_dir.mkdir(parents=True)
    with gzip.open(chunk_dir / "request.json.gz", "wb") as stream:
        stream.write(request_bytes)
    async with semaphore:
        response_bytes = await _post_with_retries(
            client,
            endpoint=endpoint,
            request_bytes=request_bytes,
            attempts_dir=chunk_dir / "attempts",
            timeout_seconds=timeout_seconds,
        )
    with gzip.open(chunk_dir / "selected_response.json.gz", "wb") as stream:
        stream.write(response_bytes)
    response = httpx.Response(
        200,
        content=response_bytes,
        request=httpx.Request("POST", endpoint),
    )
    raw_request = CandidateScoreRequest(
        prompt_prefix=prompt,
        candidates=candidates,
        hole=None,
        integer_constants=(),
        candidate_kind=CandidateKind.LABEL,
    )
    decoded = decode_candidate_scores(
        response,
        prompts,
        candidates,
        tuple(None for _ in candidates),
    )
    if tuple(score.candidate for score in decoded) != raw_request.candidates:
        raise ValueError("provider changed raw label-path order")
    prompt_sha256 = _sha256_bytes(prompt.encode())
    return [
        _score_record(
            program=program,
            path_specs=specs[4 * index : 4 * index + 4],
            raw_scores=decoded[4 * index : 4 * index + 4],
            prompt_sha256=prompt_sha256,
        )
        for index, program in enumerate(programs)
    ]


async def _run_scoring(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    prepared = output / "prepared"
    reasoning = output / "reasoning"
    scoring_plan = output / "scoring_plan"
    verify_bundle(prepared, args.prepared_sha256)
    verify_bundle(reasoning, args.reasoning_sha256)
    verify_bundle(scoring_plan, args.scoring_plan_sha256)
    scores_dir = output / "scores"
    _fresh_directory(scores_dir)
    protocol = cast(Mapping[str, object], _read_json(prepared / "protocol.json"))
    model_record = cast(Mapping[str, object], protocol["model"])
    scoring_record = cast(Mapping[str, object], model_record["candidate_scoring"])
    if int(scoring_record["candidate_batch_size"]) % 4 != 0:
        raise ValueError("raw candidate batch size must be divisible by four")
    programs_per_chunk = int(scoring_record["candidate_batch_size"]) // 4
    predicate_catalog = cast(
        list[Mapping[str, str]], _read_json(prepared / "catalogs" / "predicates.json")
    )
    mapper_catalog = cast(
        list[Mapping[str, str]], _read_json(prepared / "catalogs" / "mappers.json")
    )
    catalog = {
        "predicate": tuple(item["key"] for item in predicate_catalog),
        "mapper": tuple(item["key"] for item in mapper_catalog),
    }
    plan = cast(list[Mapping[str, object]], _read_json(scoring_plan / "plan.json"))
    endpoint = f"{args.base_url.rstrip('/')}/completions"
    semaphore = asyncio.Semaphore(args.max_concurrency)
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=args.max_concurrency),
        timeout=None,
    ) as client:
        for slate_index, slate in enumerate(plan, start=1):
            slate_id = cast(str, slate["slate_id"])
            phase = cast(str, slate["phase"])
            context = (scoring_plan / cast(str, slate["context_path"])).read_text(encoding="utf-8")
            prompt = build_compatibility_prompt(context, LABEL_PROTOCOL)
            order = tuple(int(value) for value in cast(list[object], slate["provider_order"]))
            keys = catalog[phase]
            if sorted(order) != list(range(len(keys))):
                raise ValueError(f"slate {slate_id} provider order is not a permutation")
            programs = tuple(
                CompatibilityProgram(
                    program_key=keys[index],
                    program_text=_program_text(phase, keys[index]),
                )
                for index in order
            )
            chunks = tuple(
                programs[start : start + programs_per_chunk]
                for start in range(0, len(programs), programs_per_chunk)
            )
            slate_dir = scores_dir / "slates" / slate_id
            slate_dir.mkdir(parents=True)
            raw_dir = slate_dir / "raw"
            raw_dir.mkdir()
            started = time.perf_counter()
            chunk_records = await asyncio.gather(
                *(
                    _score_http_chunk(
                        client=client,
                        semaphore=semaphore,
                        endpoint=endpoint,
                        model=cast(str, model_record["served_name"]),
                        prompt=prompt,
                        programs=chunk,
                        raw_dir=raw_dir,
                        chunk_index=index,
                        timeout_seconds=args.timeout_seconds,
                    )
                    for index, chunk in enumerate(chunks)
                )
            )
            records = [record for chunk in chunk_records for record in chunk]
            if len(records) != len(keys) or {record["candidate_key"] for record in records} != set(
                keys
            ):
                raise ValueError(f"slate {slate_id} did not return one score per candidate")
            with gzip.open(slate_dir / "scores.jsonl.gz", "wt", encoding="utf-8") as stream:
                for record in records:
                    stream.write(_canonical_bytes(record).decode() + "\n")
            _write_json(
                slate_dir / "manifest.json",
                {
                    **dict(slate),
                    "candidate_count": len(records),
                    "raw_path_count": 4 * len(records),
                    "prompt_sha256": _sha256_bytes(prompt.encode()),
                    "elapsed_seconds": time.perf_counter() - started,
                    "completed_at": _now(),
                },
            )
            print(
                f"score slate {slate_index}/{len(plan)} {slate_id} complete "
                f"({len(records)} candidates)",
                flush=True,
            )
    _write_json(
        scores_dir / "manifest.json",
        {
            "schema": HARNESS_SCHEMA,
            "completed_at": _now(),
            "prepared_bundle_sha256": args.prepared_sha256,
            "reasoning_bundle_sha256": args.reasoning_sha256,
            "scoring_plan_bundle_sha256": args.scoring_plan_sha256,
            "base_url": args.base_url,
            "slates": len(plan),
        },
    )
    print(seal_bundle(scores_dir), flush=True)


def run_scoring(args: argparse.Namespace) -> None:
    asyncio.run(_run_scoring(args))


def _read_score_map(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            key = record.get("candidate_key")
            value = record.get("compatibility_log_score")
            if (
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError(f"invalid score record in {path}")
            score = float(value)
            if not math.isfinite(score) or key in result:
                raise ValueError(f"non-finite or duplicate score in {path}")
            result[key] = score
    return result


def competition_rank(scores: Mapping[str, float], key: str) -> tuple[int, tuple[int, int]]:
    target = scores[key]
    greater = sum(value > target for value in scores.values())
    greater_or_equal = sum(value >= target for value in scores.values())
    return 1 + greater, (1 + greater, greater_or_equal)


def _tie_digest(
    *,
    seed: int,
    phase: str,
    prefix_sha256: str,
    candidate_key: str,
) -> str:
    key = f"{TIE_BREAK_DOMAIN}:{seed}".encode()
    message = (
        phase.encode()
        + b"\0"
        + prefix_sha256.encode()
        + b"\0"
        + _sha256_bytes(candidate_key.encode()).encode()
    )
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def ranked_keys(
    scores: Mapping[str, float],
    *,
    seed: int,
    phase: str,
    prefix_sha256: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            scores,
            key=lambda candidate: (
                -scores[candidate],
                _tie_digest(
                    seed=seed,
                    phase=phase,
                    prefix_sha256=prefix_sha256,
                    candidate_key=candidate,
                ),
            ),
        )
    )


def smoothed_top_k_probability(
    key: str,
    ranking: Sequence[str],
    *,
    catalog_size: int,
    top_k: int,
    epsilon: float,
) -> float:
    floor = epsilon / catalog_size
    return floor + ((1.0 - epsilon) / top_k if key in ranking[:top_k] else 0.0)


def independent_n50(probability: float) -> int:
    if not 0.0 < probability < 1.0:
        raise ValueError("N50 requires probability strictly between zero and one")
    return math.ceil(math.log(0.5) / math.log1p(-probability))


def _target_keys() -> tuple[tuple[str, str], str]:
    target = bounded_square_target()
    reducer = child(target, "reducer")
    predicate = cast(dict[str, object], child(reducer, "condition"))
    mapper = cast(dict[str, object], child(child(reducer, "thenExpr"), "head"))
    swapped = {
        "kind": "And",
        "left": predicate["right"],
        "right": predicate["left"],
    }
    return (canonical_key(predicate), canonical_key(swapped)), canonical_key(mapper)


def evaluate(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    prepared = output / "prepared"
    reasoning = output / "reasoning"
    scoring_plan = output / "scoring_plan"
    scores_dir = output / "scores"
    verify_bundle(prepared, args.prepared_sha256)
    verify_bundle(reasoning, args.reasoning_sha256)
    verify_bundle(scoring_plan, args.scoring_plan_sha256)
    verify_bundle(scores_dir, args.scores_sha256)
    evaluation = output / "evaluation"
    _fresh_directory(evaluation)
    protocol = cast(Mapping[str, object], _read_json(prepared / "protocol.json"))
    proposal = cast(Mapping[str, object], protocol["proposal"])
    epsilon = float(proposal["epsilon"])
    predicate_k = int(proposal["predicate_k"])
    mapper_k = int(proposal["mapper_k"])
    exact_predicates, exact_mapper = _target_keys()
    predicate_catalog = tuple(
        item["key"]
        for item in cast(
            list[Mapping[str, str]], _read_json(prepared / "catalogs" / "predicates.json")
        )
    )
    mapper_catalog = tuple(
        item["key"]
        for item in cast(
            list[Mapping[str, str]], _read_json(prepared / "catalogs" / "mappers.json")
        )
    )
    if any(key not in predicate_catalog for key in exact_predicates) or exact_mapper not in (
        mapper_catalog
    ):
        raise RuntimeError("retrospective target keys are outside the frozen catalogs")
    plan = cast(list[Mapping[str, object]], _read_json(scoring_plan / "plan.json"))
    by_identity = {
        (
            int(item["trial_seed"]),
            cast(str, item["arm"]),
            cast(str, item["phase"]),
            item["prefix_catalog_index_zero_based"],
        ): item
        for item in plan
    }
    seeds = sorted({int(item["trial_seed"]) for item in plan})
    arms: dict[str, object] = {}
    for arm in ("N", "R"):
        trials = []
        trial_masses = []
        for seed in seeds:
            predicate_slate = by_identity[(seed, arm, "predicate", None)]
            predicate_id = cast(str, predicate_slate["slate_id"])
            predicate_scores = _read_score_map(
                scores_dir / "slates" / predicate_id / "scores.jsonl.gz"
            )
            predicate_ranking = ranked_keys(
                predicate_scores,
                seed=seed,
                phase="predicate",
                prefix_sha256=NULL_DIGEST,
            )
            predicate_results = []
            mapper_results = []
            mass = 0.0
            for exact_predicate in exact_predicates:
                index = predicate_catalog.index(exact_predicate)
                rank, interval = competition_rank(predicate_scores, exact_predicate)
                operational_rank = predicate_ranking.index(exact_predicate) + 1
                q_predicate = smoothed_top_k_probability(
                    exact_predicate,
                    predicate_ranking,
                    catalog_size=len(predicate_catalog),
                    top_k=predicate_k,
                    epsilon=epsilon,
                )
                predicate_results.append(
                    {
                        "catalog_index_zero_based": index,
                        "candidate_sha256": _sha256_bytes(exact_predicate.encode()),
                        "score": predicate_scores[exact_predicate],
                        "competition_rank": rank,
                        "tie_interval": interval,
                        "operational_rank": operational_rank,
                        "recall": {
                            str(cutoff): operational_rank <= cutoff for cutoff in (1, 5, 10, 20)
                        },
                        "q_predicate": q_predicate,
                    }
                )
                mapper_slate = by_identity[(seed, arm, "mapper", index)]
                mapper_id = cast(str, mapper_slate["slate_id"])
                mapper_scores = _read_score_map(
                    scores_dir / "slates" / mapper_id / "scores.jsonl.gz"
                )
                mapper_ranking = ranked_keys(
                    mapper_scores,
                    seed=seed,
                    phase="mapper",
                    prefix_sha256=cast(str, mapper_slate["prefix_sha256"]),
                )
                mapper_rank, mapper_interval = competition_rank(mapper_scores, exact_mapper)
                mapper_operational_rank = mapper_ranking.index(exact_mapper) + 1
                q_mapper = smoothed_top_k_probability(
                    exact_mapper,
                    mapper_ranking,
                    catalog_size=len(mapper_catalog),
                    top_k=mapper_k,
                    epsilon=epsilon,
                )
                path_mass = q_predicate * q_mapper
                mass += path_mass
                mapper_results.append(
                    {
                        "predicate_catalog_index_zero_based": index,
                        "candidate_sha256": _sha256_bytes(exact_mapper.encode()),
                        "score": mapper_scores[exact_mapper],
                        "competition_rank": mapper_rank,
                        "tie_interval": mapper_interval,
                        "operational_rank": mapper_operational_rank,
                        "recall": {
                            str(cutoff): mapper_operational_rank <= cutoff
                            for cutoff in (1, 5, 10, 20)
                        },
                        "q_mapper_given_predicate": q_mapper,
                        "exact_path_mass": path_mass,
                    }
                )
            trial_masses.append(mass)
            trials.append(
                {
                    "seed": seed,
                    "exact_predicates": predicate_results,
                    "conditional_exact_mappers": mapper_results,
                    "exact_program_mass": mass,
                    "iid_n50": independent_n50(mass),
                }
            )
        ensemble_mass = math.fsum(trial_masses) / len(trial_masses)
        arms[arm] = {
            "trials": trials,
            "equal_trial_mixture_exact_program_mass": ensemble_mass,
            "conditional_iid_n50": independent_n50(ensemble_mass),
        }
    summary = {
        "schema": HARNESS_SCHEMA,
        "evaluated_at": _now(),
        "gold_join_after_scores_bundle_sha256": args.scores_sha256,
        "interpretation": (
            "Fixed-task counterfactual conditional-ranking diagnostic. N50 applies only "
            "to IID draws from the declared frozen equal-trial-mixture proposal."
        ),
        "arms": arms,
    }
    _write_json(evaluation / "summary.json", summary)
    n_arm = cast(Mapping[str, object], arms["N"])
    r_arm = cast(Mapping[str, object], arms["R"])
    report = (
        "# GPT-OSS-120B conditional-ranking diagnostic\n\n"
        "This is a fixed-task, counterfactual prefix diagnostic—not an unconditional "
        "SMC success-rate estimate.\n\n"
        "| Arm | Exact-program proposal mass | Conditional IID N50 |\n"
        "| --- | ---: | ---: |\n"
        f"| N: no reasoning prefix | {float(n_arm['equal_trial_mixture_exact_program_mass']):.10g} "
        f"| {int(n_arm['conditional_iid_n50'])} |\n"
        f"| R: frozen model reasoning prefix | "
        f"{float(r_arm['equal_trial_mixture_exact_program_mass']):.10g} "
        f"| {int(r_arm['conditional_iid_n50'])} |\n\n"
        "Full per-trial competition ranks, tie intervals, operational ranks, and "
        "prefix-consistent path masses are in `summary.json`.\n"
    )
    (evaluation / "REPORT.md").write_text(report, encoding="utf-8")
    _write_json(
        evaluation / "manifest.json",
        {
            "schema": HARNESS_SCHEMA,
            "prepared_bundle_sha256": args.prepared_sha256,
            "reasoning_bundle_sha256": args.reasoning_sha256,
            "scoring_plan_bundle_sha256": args.scoring_plan_sha256,
            "scores_bundle_sha256": args.scores_sha256,
        },
    )
    print(seal_bundle(evaluation), flush=True)
    print(report, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name(PROTOCOL_FILENAME),
    )
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.set_defaults(function=prepare)

    reasoning_parser = subparsers.add_parser("run-reasoning")
    reasoning_parser.add_argument("--output", type=Path, required=True)
    reasoning_parser.add_argument("--prepared-sha256", required=True)
    reasoning_parser.add_argument("--base-url", required=True)
    reasoning_parser.add_argument("--timeout-seconds", type=float, default=300.0)
    reasoning_parser.set_defaults(function=run_reasoning)

    scoring_prepare_parser = subparsers.add_parser("prepare-scoring")
    scoring_prepare_parser.add_argument("--output", type=Path, required=True)
    scoring_prepare_parser.add_argument("--prepared-sha256", required=True)
    scoring_prepare_parser.add_argument("--reasoning-sha256", required=True)
    scoring_prepare_parser.add_argument("--base-url", required=True)
    scoring_prepare_parser.add_argument("--tokenize-concurrency", type=int, default=64)
    scoring_prepare_parser.set_defaults(function=prepare_scoring)

    scoring_parser = subparsers.add_parser("run-scoring")
    scoring_parser.add_argument("--output", type=Path, required=True)
    scoring_parser.add_argument("--prepared-sha256", required=True)
    scoring_parser.add_argument("--reasoning-sha256", required=True)
    scoring_parser.add_argument("--scoring-plan-sha256", required=True)
    scoring_parser.add_argument("--base-url", required=True)
    scoring_parser.add_argument("--max-concurrency", type=int, default=8)
    scoring_parser.add_argument("--timeout-seconds", type=float, default=600.0)
    scoring_parser.set_defaults(function=run_scoring)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--prepared-sha256", required=True)
    evaluate_parser.add_argument("--reasoning-sha256", required=True)
    evaluate_parser.add_argument("--scoring-plan-sha256", required=True)
    evaluate_parser.add_argument("--scores-sha256", required=True)
    evaluate_parser.set_defaults(function=evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()

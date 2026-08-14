"""Shared, target-free definitions for the fresh blind filter-map confirmation.

This module deliberately contains no provider client.  It defines the public
task law, deterministic run schedule, commitment encodings, and small
validation helpers shared by the custodial generator, launcher, and analyzer.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from modelsmc_pbe.core.evaluate import evaluate_expression, evaluate_program
from modelsmc_pbe.core.types import Node
from modelsmc_pbe.domain.ast import AstNode, clone_program
from modelsmc_pbe.grammar import arithmetic_expressions, filter_predicates

METHOD_SCHEMA = "blind-filter-map-confirmation-v3"
METHOD_STATUS = "method-frozen-before-secret-preparation"
METHOD_SEAL_SCHEMA = "blind-filter-map-confirmation-v3-method-seal-v1"
CUSTODY_SEAL_SCHEMA = "blind-filter-map-confirmation-v3-custody-seal-v1"
PROVIDER_SEAL_SCHEMA = "blind-filter-map-confirmation-v3-provider-seal-v1"
MANIFEST_SCHEMA = "blinded-filter-map-suite-v3"
TARGET_SCHEMA = "blinded-filter-map-target-v3"
RESULT_SCHEMA = "evidence-shortlist-product-path-smc-result-v1"
ANALYSIS_SCHEMA = "blind-filter-map-confirmation-v3-analysis-v1"

TASK_IDS = tuple(f"blind-v3-{index:02d}" for index in range(1, 13))
CONSTANTS = tuple(range(-3, 5))
DOMAIN = CONSTANTS
ROUNDS = 4
SHORTLIST_SIZE = 4
PARENT_COUNT = 2
OFFSPRING_PER_PARENT = 4
FIRST_ROUND_OFFSPRING = 4
SLOT_CHECKPOINTS = (1, 5, 13, 21, 29)
MAXIMUM_PROVIDER_CALLS = 7
MODEL_NAME = "gpt-oss-120b"
MODEL_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
REASONING_EFFORT = "low"
TEMPERATURE = 0.0
MAX_TOKENS = 1200
TIMEOUT_SECONDS = 420.0
MAX_CONCURRENCY = 2
EPSILON = 0.05
EVIDENCE_SCALE = 2.0
EXACT_REFERENCE_LIMIT = 0

SEED_COMMITMENT_DOMAIN = b"blind-filter-map-confirmation-v3-secret\0"
TARGET_DRAW_DOMAIN = b"blind-filter-map-confirmation-v3-target-draw\0"
RUN_SEED_DOMAIN = b"blind-filter-map-confirmation-v3-run-seed\0"
TARGET_COMMITMENT_DOMAIN = "blind-filter-map-target-v3"
BUNDLE_SCHEME = "SHA256(domain || NUL || repeated(path || NUL || exact_file_bytes || NUL))"

FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "commitment_nonce",
        "nonce",
        "private_seed",
        "rejection_log",
        "seed_hex",
        "target",
        "target_mapper",
        "target_mapper_dsl",
        "target_predicate",
        "target_predicate_dsl",
    }
)


@dataclass(frozen=True)
class RunSeeds:
    """Frozen common-random-number schedule for one matched task pair."""

    task_id: str
    start_seed: int
    provider_seed: int
    sample_seed: int
    resample_seed: int
    random_shortlist_seed: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "task_id": self.task_id,
            "start_seed": self.start_seed,
            "provider_seed": self.provider_seed,
            "sample_seed": self.sample_seed,
            "resample_seed": self.resample_seed,
            "random_shortlist_seed": self.random_shortlist_seed,
        }


@dataclass(frozen=True)
class TargetDraw:
    """The first accepted draw and its private deterministic audit trail."""

    predicate: AstNode
    mapper: AstNode
    predicate_dsl: str
    mapper_dsl: str
    accepted_attempt: int
    rejected: tuple[dict[str, object], ...]


def canonical_bytes(value: object) -> bytes:
    """Serialize a commitment value independently of whitespace and key order."""

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


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def write_json_exclusive(path: Path, value: object, *, mode: int = 0o644) -> None:
    """Write canonical JSON once, never silently replacing sealed evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")


def require_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def require_object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def require_array(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return cast(list[Any], value)


def expect(actual: object, expected: object, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def safe_repo_path(repo_root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{name} must be a safe repository-relative path")
    root = repo_root.resolve()
    result = (root / relative).resolve()
    if root not in result.parents:
        raise ValueError(f"{name} escaped the repository")
    return result


def reject_private_keys(value: object, *, path: str = "public") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"private key {key!r} appears at {path}")
            reject_private_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_private_keys(child, path=f"{path}[{index}]")


def seed_commitment(seed: bytes) -> str:
    if len(seed) != 32:
        raise ValueError("suite secret must contain exactly 32 bytes")
    return sha256_bytes(SEED_COMMITMENT_DOMAIN + seed)


def _hmac(seed: bytes, label: str) -> bytes:
    return hmac.new(seed, TARGET_DRAW_DOMAIN + label.encode(), hashlib.sha256).digest()


def _uniform_index(seed: bytes, label: str, size: int) -> int:
    """Draw exactly uniformly from ``range(size)`` using rejection on HMAC words."""

    if size < 1:
        raise ValueError("uniform choice must have nonempty support")
    modulus = 1 << 256
    limit = modulus - modulus % size
    counter = 0
    while True:
        value = int.from_bytes(_hmac(seed, f"{label}\0word-{counter}"), "big")
        if value < limit:
            return value % size
        counter += 1


def _render_expression_dsl(expression: Mapping[str, object]) -> str:
    kind = expression.get("kind")
    if kind == "Item":
        return "item"
    if kind == "IntLiteral":
        return cast(str, expression["intValue"])
    operators = {
        "Add": "add",
        "Subtract": "sub",
        "Multiply": "mul",
        "LessThan": "lt",
        "EqualInt": "eq",
        "And": "and",
    }
    if kind not in operators:
        raise ValueError(f"unsupported catalog expression kind: {kind}")
    left = cast(Mapping[str, object], expression["left"])
    right = cast(Mapping[str, object], expression["right"])
    rendered_left = _render_expression_dsl(left)
    rendered_right = _render_expression_dsl(right)
    return f"{operators[cast(str, kind)]}({rendered_left},{rendered_right})"


def _catalogs() -> tuple[dict[str, AstNode], dict[str, AstNode]]:
    predicates = {
        _render_expression_dsl(expression): expression
        for expression in filter_predicates(CONSTANTS)
    }
    mappers = {
        _render_expression_dsl(expression): expression
        for expression in arithmetic_expressions("Item", CONSTANTS)
    }
    return predicates, mappers


def _sample_atom_dsl(seed: bytes, label: str) -> str:
    constant = CONSTANTS[_uniform_index(seed, f"{label}\0constant", len(CONSTANTS))]
    templates = ("lt(item,c)", "lt(c,item)", "eq(item,c)")
    template = templates[_uniform_index(seed, f"{label}\0template", len(templates))]
    return template.replace("c", str(constant))


def _sample_predicate_dsl(seed: bytes, label: str) -> str:
    form = _uniform_index(seed, f"{label}\0form", 2)
    left = _sample_atom_dsl(seed, f"{label}\0left")
    if form == 0:
        return left
    right = _sample_atom_dsl(seed, f"{label}\0right")
    return f"and({left},{right})"


def _sample_mapper_dsl(seed: bytes, label: str) -> str:
    form = _uniform_index(seed, f"{label}\0form", 5)
    if form == 0:
        return "item"
    constant = CONSTANTS[_uniform_index(seed, f"{label}\0constant", len(CONSTANTS))]
    if form == 1:
        return str(constant)
    operator = ("add", "sub", "mul")[
        _uniform_index(seed, f"{label}\0operator", 3)
    ]
    if form == 2:
        return f"{operator}(item,item)"
    if form == 3:
        return f"{operator}(item,{constant})"
    return f"{operator}({constant},item)"


def _evaluate_bool(expression: AstNode, item: int) -> bool:
    value = evaluate_expression(cast(Node, expression), [], item=item)
    if not isinstance(value, bool):
        raise TypeError("predicate did not evaluate to Boolean")
    return value


def _evaluate_int(expression: AstNode, item: int) -> int:
    value = evaluate_expression(cast(Node, expression), [], item=item)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("mapper did not evaluate to integer")
    return value


def sample_target(seed: bytes, task_id: str) -> TargetDraw:
    """Return the first informative draw from the public, target-independent law.

    Each attempt draws predicate and mapper independently from the same
    normalized recursive component grammar used by the SMC restart.  Rejection
    depends only on the sampled program's observable behavior: retain 3--5
    domain items and produce at least three distinct retained values.  It never
    consults a model, prompt, search seed, initial program, or another task.
    """

    if len(seed) != 32:
        raise ValueError("suite secret must contain exactly 32 bytes")
    if task_id not in TASK_IDS:
        raise ValueError(f"unknown task ID: {task_id}")
    predicates, mappers = _catalogs()
    rejected: list[dict[str, object]] = []
    for attempt in range(1, 1_000_001):
        label = f"{task_id}\0attempt-{attempt}"
        predicate_dsl = _sample_predicate_dsl(seed, f"{label}\0predicate")
        mapper_dsl = _sample_mapper_dsl(seed, f"{label}\0mapper")
        predicate = predicates[predicate_dsl]
        mapper = mappers[mapper_dsl]
        keep = tuple(_evaluate_bool(predicate, item) for item in DOMAIN)
        support = tuple(item for item, retained in zip(DOMAIN, keep, strict=True) if retained)
        outputs = tuple(_evaluate_int(mapper, item) for item in support)
        reasons: list[str] = []
        if not 3 <= len(support) <= 5:
            reasons.append("retained-domain-cardinality-outside-3-to-5")
        if len(set(outputs)) < 3:
            reasons.append("fewer-than-three-distinct-retained-outputs")
        if not reasons:
            return TargetDraw(
                predicate=predicate,
                mapper=mapper,
                predicate_dsl=predicate_dsl,
                mapper_dsl=mapper_dsl,
                accepted_attempt=attempt,
                rejected=tuple(rejected),
            )
        rejected.append(
            {
                "attempt": attempt,
                "predicate_dsl": predicate_dsl,
                "mapper_dsl": mapper_dsl,
                "reasons": reasons,
            }
        )
    raise RuntimeError(f"target rejection sampler exceeded its frozen cap for {task_id}")


def assemble_program(predicate: AstNode, mapper: AstNode) -> AstNode:
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


def _encoded_list(values: Sequence[int]) -> list[str]:
    return [str(value) for value in values]


def _seeded_permutation(seed: bytes, label: str) -> tuple[int, ...]:
    decorated = [
        (_hmac(seed, f"{label}\0position-{index}\0value-{item}"), index, item)
        for index, item in enumerate(DOMAIN)
    ]
    return tuple(item for _, _, item in sorted(decorated))


def task_document(task_id: str, target: TargetDraw, *, seed: bytes) -> dict[str, object]:
    first = _seeded_permutation(seed, f"{task_id}\0examples\0permutation-1")
    second = _seeded_permutation(seed, f"{task_id}\0examples\0permutation-2")
    inputs = (
        (),
        *((item,) for item in DOMAIN),
        DOMAIN,
        first,
        (*first, *second),
    )
    program = assemble_program(target.predicate, target.mapper)
    examples = []
    for input_value in inputs:
        output = evaluate_program(cast(Node, program), list(input_value))
        if not isinstance(output, list) or any(
            not isinstance(item, int) or isinstance(item, bool) for item in output
        ):
            raise TypeError("target program returned a non-integer list")
        examples.append(
            {"input": _encoded_list(input_value), "output": _encoded_list(output)}
        )
    return {
        "name": f"opaque ordered-list transformation {task_id}",
        "signature": {"input": "List<Int>", "output": "List<Int>"},
        "examples": examples,
        "integerConstants": _encoded_list(CONSTANTS),
        "particles": 4,
        "iterations": ROUNDS,
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


def frozen_run_seeds() -> tuple[RunSeeds, ...]:
    """Return the explicit collision-free seed schedule frozen in Stage 1."""

    records = []
    for index, task_id in enumerate(TASK_IDS, start=1):
        base = 401_000 + 1_000 * (index - 1)
        records.append(
            RunSeeds(
                task_id=task_id,
                start_seed=base + 10,
                provider_seed=base + 20,
                sample_seed=base + 30,
                resample_seed=base + 40,
                random_shortlist_seed=base + 50,
            )
        )
    return tuple(records)


def run_seed_for(task_id: str) -> RunSeeds:
    matches = [record for record in frozen_run_seeds() if record.task_id == task_id]
    if len(matches) != 1:
        raise ValueError(f"unknown task ID: {task_id}")
    return matches[0]


def paired_sign_test_p_value(llm_wins: int, random_wins: int) -> float:
    """Exact one-sided sign-test tail conditional on discordant matched pairs."""

    import math

    if llm_wins < 0 or random_wins < 0:
        raise ValueError("discordant win counts must be nonnegative")
    discordant = llm_wins + random_wins
    if discordant == 0:
        return 1.0
    numerator = sum(math.comb(discordant, value) for value in range(llm_wins, discordant + 1))
    return numerator / (2**discordant)

"""Generate and verify committed blind filter-map suites.

The public directory is the complete provider-side input boundary.  It contains
only opaque PBE specifications and binding commitments.  The target ASTs,
mapper-family assignments, commitment nonces, and suite seed are written only
to ``private/reveal.json`` and are not needed by the beam-search harness.

Usage::

    python -m research.generate_blinded_filter_map_tasks prepare-secret \
        --output blind-v2.secret --suite-version v2
    python -m research.generate_blinded_filter_map_tasks generate --output blind-suite \
        --suite-version v2 --seed-file blind-v2.secret
    python -m research.generate_blinded_filter_map_tasks verify \
        --public blind-suite/public --reveal blind-suite/private/reveal.json
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import random
import secrets
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.core.evaluate import evaluate_expression, evaluate_program
from modelsmc_pbe.core.types import Node
from modelsmc_pbe.domain.ast import AstNode, canonical_key, clone_program
from modelsmc_pbe.grammar import arithmetic_expressions, filter_predicates

SCHEMA = "blinded-filter-map-suite-v1"
TARGET_SCHEMA = "blinded-filter-map-target-v1"
V2_SCHEMA = "blinded-filter-map-suite-v2"
V2_TARGET_SCHEMA = "blinded-filter-map-target-v2"
CONSTANTS = tuple(range(-3, 5))
TASK_COUNT = 4
MAPPER_FAMILIES = ("square", "scale", "shift", "reverse")
SEED_COMMITMENT_DOMAIN = b"blind-beam-four-task-v1\0"
V2_SEED_COMMITMENT_DOMAIN = b"blind-evidence-frontier-v2\0"
V2_PROTOCOL_TASK_SCHEDULE = (
    ("blind-v2-01", "bounded-open-interval", "quadratic-item-times-item"),
    ("blind-v2-02", "lower-half-line", "nontrivial-integer-scale"),
    ("blind-v2-03", "upper-half-line", "nonzero-integer-shift"),
    ("blind-v2-04", "bounded-open-interval", "reverse-affine-constant-minus-item"),
    ("blind-v2-05", "lower-half-line", "quadratic-item-times-item"),
    ("blind-v2-06", "upper-half-line", "nontrivial-integer-scale"),
    ("blind-v2-07", "bounded-open-interval", "nonzero-integer-shift"),
    ("blind-v2-08", "lower-half-line", "reverse-affine-constant-minus-item"),
    ("blind-v2-09", "upper-half-line", "quadratic-item-times-item"),
    ("blind-v2-10", "bounded-open-interval", "nontrivial-integer-scale"),
    ("blind-v2-11", "lower-half-line", "nonzero-integer-shift"),
    ("blind-v2-12", "upper-half-line", "reverse-affine-constant-minus-item"),
)
_PROTOCOL_PREDICATE_SHAPES = {
    "bounded-open-interval": "bounded",
    "lower-half-line": "lower",
    "upper-half-line": "upper",
}
_PROTOCOL_MAPPER_FAMILIES = {
    "quadratic-item-times-item": "square",
    "nontrivial-integer-scale": "scale",
    "nonzero-integer-shift": "shift",
    "reverse-affine-constant-minus-item": "reverse",
}


@dataclass(frozen=True)
class SuiteDefinition:
    version: str
    schema: str
    target_schema: str
    seed_commitment_domain: bytes
    schedule: tuple[tuple[str, str, str], ...]
    require_unique_bounded_predicates: bool


V1_DEFINITION = SuiteDefinition(
    version="v1",
    schema=SCHEMA,
    target_schema=TARGET_SCHEMA,
    seed_commitment_domain=SEED_COMMITMENT_DOMAIN,
    schedule=tuple(
        (f"blind-{index:02d}", shape, family)
        for index, (shape, family) in enumerate(
            zip(("bounded", "lower", "upper", "bounded"), MAPPER_FAMILIES, strict=True),
            start=1,
        )
    ),
    require_unique_bounded_predicates=True,
)
V2_DEFINITION = SuiteDefinition(
    version="v2",
    schema=V2_SCHEMA,
    target_schema=V2_TARGET_SCHEMA,
    seed_commitment_domain=V2_SEED_COMMITMENT_DOMAIN,
    schedule=tuple(
        (
            task_id,
            _PROTOCOL_PREDICATE_SHAPES[predicate_family],
            _PROTOCOL_MAPPER_FAMILIES[mapper_family],
        )
        for task_id, predicate_family, mapper_family in V2_PROTOCOL_TASK_SCHEDULE
    ),
    require_unique_bounded_predicates=False,
)
_DEFINITIONS_BY_VERSION = {item.version: item for item in (V1_DEFINITION, V2_DEFINITION)}
_DEFINITIONS_BY_SCHEMA = {item.schema: item for item in (V1_DEFINITION, V2_DEFINITION)}
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "commitment_nonce",
        "mapper_family",
        "nonce",
        "private_seed",
        "seed_hex",
        "target",
        "target_mapper",
        "target_predicate",
    }
)


def canonical_bytes(value: object) -> bytes:
    """Serialize one commitment value independently of whitespace and key order."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def sha256(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""

    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_bytes(value) + b"\n")


def _write_private_new(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _parse_seed(seed_hex: str | None) -> tuple[bytes, str]:
    if seed_hex is None:
        seed = secrets.token_bytes(32)
        return seed, seed.hex()
    try:
        seed = bytes.fromhex(seed_hex)
    except ValueError as error:
        raise ValueError("--seed-hex must contain exactly 64 hexadecimal characters") from error
    if len(seed) != 32 or seed.hex() != seed_hex.lower():
        raise ValueError("--seed-hex must contain exactly 64 hexadecimal characters")
    return seed, seed.hex()


def _definition_for_version(suite_version: str) -> SuiteDefinition:
    try:
        return _DEFINITIONS_BY_VERSION[suite_version]
    except KeyError as error:
        raise ValueError(f"unknown suite version: {suite_version}") from error


def _definition_for_schema(schema: object) -> SuiteDefinition:
    try:
        return _DEFINITIONS_BY_SCHEMA[cast(str, schema)]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown suite schema: {schema!r}") from error


def seed_commitment(seed: bytes, *, suite_version: str = "v2") -> str:
    """Commit to an exact 32-byte suite secret under a versioned domain."""

    if len(seed) != 32:
        raise ValueError("suite secret must contain exactly 32 bytes")
    definition = _definition_for_version(suite_version)
    return sha256(definition.seed_commitment_domain + seed)


def prepare_secret(path: Path, *, suite_version: str = "v2") -> str:
    """Create a mode-0600 raw secret file and return only its commitment."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    seed = secrets.token_bytes(32)
    _write_private_new(destination, seed)
    return seed_commitment(seed, suite_version=suite_version)


def _generation_seed(
    *, seed_hex: str | None, seed_file: Path | None
) -> tuple[bytes, str]:
    if seed_hex is not None and seed_file is not None:
        raise ValueError("--seed-hex and --seed-file are mutually exclusive")
    if seed_file is None:
        return _parse_seed(seed_hex)
    seed = seed_file.expanduser().resolve().read_bytes()
    if len(seed) != 32:
        raise ValueError("--seed-file must contain exactly 32 raw bytes")
    return seed, seed.hex()


def _digest(seed: bytes, label: str, *, schema: str = SCHEMA) -> bytes:
    return hmac.new(seed, f"{schema}\0{label}".encode(), hashlib.sha256).digest()


def _seeded_order[Item](
    items: Sequence[Item], seed: bytes, label: str, *, schema: str = SCHEMA
) -> tuple[Item, ...]:
    decorated = [
        (
            _digest(
                seed,
                f"{label}\0{index}\0{canonical_key_for_value(item)}",
                schema=schema,
            ),
            index,
            item,
        )
        for index, item in enumerate(items)
    ]
    return tuple(item for _, _, item in sorted(decorated, key=lambda record: record[:2]))


def canonical_key_for_value(value: object) -> str:
    """Return a stable key for ASTs, tuples, strings, and tagged input records."""

    if isinstance(value, Mapping):
        return canonical_key(value)
    return canonical_bytes(value).decode()


def _evaluate_bool(expression: AstNode, item: int) -> bool:
    value = evaluate_expression(cast(Node, expression), [], item=item)
    if not isinstance(value, bool):
        raise TypeError("predicate catalog member did not evaluate to Bool")
    return value


def _evaluate_int(expression: AstNode, item: int) -> int:
    value = evaluate_expression(cast(Node, expression), [], item=item)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("mapper catalog member did not evaluate to Int")
    return value


def predicate_probe_domain(constants: Sequence[int] = CONSTANTS) -> tuple[int, ...]:
    """Return the frozen observed integer domain."""

    stable = tuple(sorted(set(constants)))
    if not stable or stable != tuple(range(stable[0], stable[-1] + 1)):
        raise ValueError("the blind-suite proof requires nonempty consecutive constants")
    return stable


def _predicate_classes() -> dict[str, tuple[tuple[tuple[bool, ...], AstNode], ...]]:
    domain = predicate_probe_domain()
    grouped: dict[tuple[bool, ...], list[AstNode]] = defaultdict(list)
    for predicate in filter_predicates(CONSTANTS):
        signature = tuple(_evaluate_bool(predicate, item) for item in domain)
        grouped[signature].append(predicate)
    eligible: dict[str, list[tuple[tuple[bool, ...], AstNode]]] = {
        "bounded": [],
        "lower": [],
        "upper": [],
    }
    for signature, expressions in grouped.items():
        if not 3 <= sum(signature) <= 5:
            continue
        representative = min(expressions, key=canonical_key)
        if not signature[0] and not signature[-1]:
            shape = "bounded"
        elif signature[-1] and not signature[0]:
            shape = "lower"
        elif signature[0] and not signature[-1]:
            shape = "upper"
        else:
            continue
        eligible[shape].append((signature, representative))
    result = {
        shape: tuple(sorted(records, key=lambda record: canonical_key(record[1])))
        for shape, records in eligible.items()
    }
    if any(not result[shape] for shape in result):
        raise ValueError("one or more declared predicate shapes are empty")
    return result


def _mapper_signature(expression: AstNode) -> tuple[int, ...]:
    return tuple(_evaluate_int(expression, item) for item in (-2, -1, 0, 1, 2))


def _mapper_family(signature: tuple[int, ...]) -> str | None:
    minus_two, minus_one, zero, one, two = signature
    if signature == (4, 1, 0, 1, 4):
        return "square"
    if one == zero + 1 and minus_one == zero - 1 and two == zero + 2 and minus_two == zero - 2:
        return "shift" if zero != 0 else None
    if one == zero - 1 and minus_one == zero + 1 and two == zero - 2 and minus_two == zero + 2:
        return "reverse"
    if zero == 0 and minus_one == -one and two == 2 * one and minus_two == -2 * one:
        return "scale" if one not in {-1, 0, 1} else None
    return None


def _mapper_classes() -> dict[str, tuple[AstNode, ...]]:
    grouped: dict[tuple[int, ...], list[AstNode]] = defaultdict(list)
    for mapper in arithmetic_expressions("Item", CONSTANTS):
        grouped[_mapper_signature(mapper)].append(mapper)
    by_family: dict[str, list[AstNode]] = {family: [] for family in MAPPER_FAMILIES}
    for signature, expressions in grouped.items():
        family = _mapper_family(signature)
        if family is not None:
            by_family[family].append(min(expressions, key=canonical_key))
    result = {
        family: tuple(sorted(expressions, key=canonical_key))
        for family, expressions in by_family.items()
    }
    if any(not result[family] for family in MAPPER_FAMILIES):
        raise ValueError("one or more declared mapper families are empty")
    return result


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
    operator = operators[cast(str, kind)]
    return f"{operator}({_render_expression_dsl(left)},{_render_expression_dsl(right)})"


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


def _encoded_list(values: Sequence[int]) -> list[str]:
    return [str(value) for value in values]


def _target_output(predicate: AstNode, mapper: AstNode, inputs: Sequence[int]) -> list[int]:
    return [
        _evaluate_int(mapper, item)
        for item in inputs
        if _evaluate_bool(predicate, item)
    ]


def _task_document(
    task_id: str,
    predicate: AstNode,
    mapper: AstNode,
    *,
    seed: bytes,
    schema: str = SCHEMA,
    iterations: int = 5,
) -> dict[str, object]:
    domain = predicate_probe_domain()
    first_permutation = _seeded_order(
        domain, seed, f"{task_id}\0permutation-1", schema=schema
    )
    second_permutation = _seeded_order(
        domain, seed, f"{task_id}\0permutation-2", schema=schema
    )
    inputs = (
        (),
        *((item,) for item in domain),
        domain,
        first_permutation,
        (*first_permutation, *second_permutation),
    )
    examples = [
        {
            "input": _encoded_list(input_value),
            "output": _encoded_list(_target_output(predicate, mapper, input_value)),
        }
        for input_value in inputs
    ]
    return {
        "name": f"opaque ordered-list transformation {task_id}",
        "signature": {"input": "List<Int>", "output": "List<Int>"},
        "examples": examples,
        "integerConstants": _encoded_list(CONSTANTS),
        "particles": 4,
        "iterations": iterations,
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


def _initial_components() -> tuple[AstNode, AstNode]:
    """Reproduce the harness's frozen grammar-order start at seed 17."""

    mapper_nodes = arithmetic_expressions("Item", CONSTANTS)
    predicate_nodes = filter_predicates(CONSTANTS)
    rng = random.Random(17)
    mapper = mapper_nodes[rng.randrange(len(mapper_nodes))]
    predicate = predicate_nodes[rng.randrange(len(predicate_nodes))]
    expected = (
        "and(lt(item,-2),lt(item,-1))",
        "mul(4,item)",
    )
    actual = (_render_expression_dsl(predicate), _render_expression_dsl(mapper))
    if actual != expected:
        raise ValueError(f"frozen start changed from {expected} to {actual}")
    return predicate, mapper


def _select_target(
    *,
    task_id: str,
    shape: str,
    family: str,
    seed: bytes,
    used_predicate_signatures: set[tuple[bool, ...]],
    used_joint_behaviors: set[tuple[tuple[bool, ...], tuple[int, ...]]],
    schema: str = SCHEMA,
    require_unique_bounded_predicates: bool = True,
) -> tuple[tuple[bool, ...], AstNode, AstNode, list[dict[str, object]]]:
    """Select the first structurally valid pair in a seed-derived order."""

    domain = predicate_probe_domain()
    initial_predicate, initial_mapper = _initial_components()
    initial_predicate_signature = tuple(
        _evaluate_bool(initial_predicate, item) for item in domain
    )
    predicate_candidates = _seeded_order(
        _predicate_classes()[shape],
        seed,
        f"{task_id}\0{shape}\0predicate",
        schema=schema,
    )
    mapper_candidates = _seeded_order(
        _mapper_classes()[family],
        seed,
        f"{task_id}\0{family}\0mapper",
        schema=schema,
    )
    rejections: list[dict[str, object]] = []
    for predicate_signature, predicate in predicate_candidates:
        support = tuple(
            item for item, keep in zip(domain, predicate_signature, strict=True) if keep
        )
        predicate_distance = sum(
            left != right
            for left, right in zip(
                predicate_signature,
                initial_predicate_signature,
                strict=True,
            )
        )
        predicate_reasons = []
        if predicate_distance < 2:
            predicate_reasons.append("predicate-too-close-to-frozen-start")
        if (
            require_unique_bounded_predicates
            and shape == "bounded"
            and predicate_signature in used_predicate_signatures
        ):
            predicate_reasons.append("duplicate-bounded-predicate-behavior")
        if predicate_reasons:
            rejections.append(
                {
                    "predicate": _render_expression_dsl(predicate),
                    "reasons": predicate_reasons,
                }
            )
            continue
        for mapper in mapper_candidates:
            target_outputs = tuple(_evaluate_int(mapper, item) for item in support)
            initial_outputs = tuple(_evaluate_int(initial_mapper, item) for item in support)
            mapper_reasons = []
            if len(set(target_outputs)) < 3:
                mapper_reasons.append("fewer-than-three-distinct-retained-outputs")
            if sum(a != b for a, b in zip(target_outputs, initial_outputs, strict=True)) < 2:
                mapper_reasons.append("mapper-too-close-to-frozen-start")
            joint_behavior = (
                predicate_signature,
                tuple(
                    _evaluate_int(mapper, item) if keep else 0
                    for item, keep in zip(domain, predicate_signature, strict=True)
                ),
            )
            if joint_behavior in used_joint_behaviors:
                mapper_reasons.append("duplicate-joint-target-behavior")
            if mapper_reasons:
                rejections.append(
                    {
                        "predicate": _render_expression_dsl(predicate),
                        "mapper": _render_expression_dsl(mapper),
                        "reasons": mapper_reasons,
                    }
                )
                continue
            return predicate_signature, predicate, mapper, rejections
    raise ValueError(f"no structurally valid target pair for {task_id}")


def _public_keys_are_safe(value: object, *, path: str = "public") -> None:
    if isinstance(value, dict):
        leaked = _FORBIDDEN_PUBLIC_KEYS.intersection(value)
        if leaked:
            raise ValueError(f"private key(s) {sorted(leaked)} leaked at {path}")
        for key, item in value.items():
            _public_keys_are_safe(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _public_keys_are_safe(item, path=f"{path}[{index}]")


def _joint_behavior(
    predicate_signature: tuple[bool, ...], mapper: AstNode
) -> tuple[tuple[bool, ...], tuple[int, ...]]:
    domain = predicate_probe_domain()
    outputs = tuple(
        _evaluate_int(mapper, item) if keep else 0
        for item, keep in zip(domain, predicate_signature, strict=True)
    )
    return predicate_signature, outputs


def _claim_seed_file(
    seed_file: Path, *, definition: SuiteDefinition, output: Path
) -> tuple[bytes, str]:
    seed_path = seed_file.expanduser().resolve()
    if seed_path.stat().st_mode & 0o077:
        raise ValueError("--seed-file must not be readable or writable by group or others")
    seed, normalized_seed_hex = _generation_seed(seed_hex=None, seed_file=seed_path)
    receipt_path = seed_path.with_name(f"{seed_path.name}.used.json")
    receipt = {
        "schema": "blinded-filter-map-secret-use-v1",
        "suite_schema": definition.schema,
        "seed_commitment_sha256": seed_commitment(seed, suite_version=definition.version),
        "output": str(output),
    }
    _write_private_new(receipt_path, canonical_bytes(receipt) + b"\n")
    return seed, normalized_seed_hex


def _resolve_generation_seed(
    *,
    seed_hex: str | None,
    seed_file: Path | None,
    definition: SuiteDefinition,
    output: Path,
) -> tuple[bytes, str]:
    if seed_hex is not None and seed_file is not None:
        raise ValueError("--seed-hex and --seed-file are mutually exclusive")
    if seed_file is not None:
        try:
            return _claim_seed_file(seed_file, definition=definition, output=output)
        except FileExistsError as error:
            raise FileExistsError(f"secret file was already used: {seed_file}") from error
    return _generation_seed(seed_hex=seed_hex, seed_file=None)


def generate_suite(
    output: Path,
    *,
    seed_hex: str | None = None,
    seed_file: Path | None = None,
    suite_version: str = "v1",
) -> dict[str, object]:
    """Generate a versioned suite and private reveal, refusing any overwrite."""

    definition = _definition_for_version(suite_version)
    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    seed, normalized_seed_hex = _resolve_generation_seed(
        seed_hex=seed_hex,
        seed_file=seed_file,
        definition=definition,
        output=destination,
    )
    public_dir = destination / "public"
    private_dir = destination / "private"
    public_dir.mkdir(parents=True)
    private_dir.mkdir()

    public_tasks: list[dict[str, object]] = []
    private_tasks: list[dict[str, object]] = []
    used_predicate_signatures: set[tuple[bool, ...]] = set()
    used_joint_behaviors: set[tuple[tuple[bool, ...], tuple[int, ...]]] = set()
    for task_id, shape, family in definition.schedule:
        predicate_signature, predicate, mapper, rejections = _select_target(
            task_id=task_id,
            shape=shape,
            family=family,
            seed=seed,
            used_predicate_signatures=used_predicate_signatures,
            used_joint_behaviors=used_joint_behaviors,
            schema=definition.schema,
            require_unique_bounded_predicates=definition.require_unique_bounded_predicates,
        )
        used_predicate_signatures.add(predicate_signature)
        used_joint_behaviors.add(_joint_behavior(predicate_signature, mapper))
        iterations = 4 if definition.version == "v2" else 5
        task = _task_document(
            task_id,
            predicate,
            mapper,
            seed=seed,
            schema=definition.schema,
            iterations=iterations,
        )
        task_bytes = canonical_bytes(task) + b"\n"
        task_path = public_dir / f"{task_id}.json"
        task_path.write_bytes(task_bytes)
        task_canonical_sha256 = sha256(canonical_bytes(task))
        task_file_sha256 = sha256(task_bytes)
        nonce = _digest(
            seed, f"{task_id}\0commitment-nonce", schema=definition.schema
        ).hex()
        preimage = {
            "schema": definition.target_schema,
            "task_id": task_id,
            "task_canonical_sha256": task_canonical_sha256,
            "target": {"predicate": predicate, "mapper": mapper},
            "nonce": nonce,
        }
        commitment = sha256(canonical_bytes(preimage))
        public_tasks.append(
            {
                "task_id": task_id,
                "path": task_path.name,
                "task_file_sha256": task_file_sha256,
                "task_canonical_sha256": task_canonical_sha256,
                "target_commitment_sha256": commitment,
            }
        )
        domain = predicate_probe_domain()
        private_tasks.append(
            {
                "task_id": task_id,
                "mapper_family": family,
                "predicate_shape": shape,
                "predicate_support": [
                    item for item, keep in zip(domain, predicate_signature, strict=True) if keep
                ],
                "target_predicate_dsl": _render_expression_dsl(predicate),
                "target_mapper_dsl": _render_expression_dsl(mapper),
                "structural_rejections_before_acceptance": rejections,
                "commitment_preimage": preimage,
                "target_commitment_sha256": commitment,
            }
        )

    implementation_path = Path(__file__).resolve()
    manifest: dict[str, object] = {
        "schema": definition.schema,
        "task_count": len(definition.schedule),
        "seed_commitment_sha256": seed_commitment(seed, suite_version=definition.version),
        "task_generation": {
            "constants": _encoded_list(CONSTANTS),
            "provider_boundary": "task files only; never copy private/reveal.json",
            "generator_sha256": sha256(implementation_path.read_bytes()),
        },
        "tasks": public_tasks,
    }
    _public_keys_are_safe(manifest)
    for task_record in public_tasks:
        _public_keys_are_safe(_read_object(public_dir / cast(str, task_record["path"])))
    _write_json(public_dir / "manifest.json", manifest)
    reveal = {
        "schema": definition.schema,
        "seed_hex": normalized_seed_hex,
        "seed_commitment_sha256": seed_commitment(seed, suite_version=definition.version),
        "public_manifest_sha256": sha256(canonical_bytes(manifest)),
        "tasks": private_tasks,
    }
    _write_json(private_dir / "reveal.json", reveal)
    return manifest


def _task_by_id(records: object, task_id: str, *, source: str) -> dict[str, Any]:
    if not isinstance(records, list):
        raise ValueError(f"{source}.tasks must be an array")
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get("task_id") == task_id
    ]
    if len(matches) != 1:
        raise ValueError(f"{source} must contain exactly one record for {task_id}")
    return cast(dict[str, Any], matches[0])


def verify_suite(public_dir: Path, reveal_path: Path) -> dict[str, object]:
    """Verify task bytes, target commitments, catalog membership, and oracle outputs."""

    public_root = public_dir.expanduser().resolve()
    reveal_file = reveal_path.expanduser().resolve()
    manifest = _read_object(public_root / "manifest.json")
    reveal = _read_object(reveal_file)
    definition = _definition_for_schema(manifest.get("schema"))
    if reveal.get("schema") != definition.schema:
        raise ValueError("suite schema mismatch")
    seed_hex = reveal.get("seed_hex")
    try:
        revealed_seed = bytes.fromhex(seed_hex) if isinstance(seed_hex, str) else b""
    except ValueError as error:
        raise ValueError("invalid revealed suite seed") from error
    expected_seed_commitment = (
        seed_commitment(revealed_seed, suite_version=definition.version)
        if len(revealed_seed) == 32
        else ""
    )
    if len(revealed_seed) != 32 or manifest.get(
        "seed_commitment_sha256"
    ) != expected_seed_commitment or reveal.get(
        "seed_commitment_sha256"
    ) != expected_seed_commitment:
        raise ValueError("suite seed commitment mismatch")
    _public_keys_are_safe(manifest)
    if reveal.get("public_manifest_sha256") != sha256(canonical_bytes(manifest)):
        raise ValueError("private reveal is bound to a different public manifest")

    predicate_keys = {canonical_key(expression) for expression in filter_predicates(CONSTANTS)}
    mapper_keys = {
        canonical_key(expression) for expression in arithmetic_expressions("Item", CONSTANTS)
    }
    verified = []
    public_records = manifest.get("tasks")
    if (
        manifest.get("task_count") != len(definition.schedule)
        or not isinstance(public_records, list)
        or len(public_records) != len(definition.schedule)
    ):
        raise ValueError(
            f"public manifest must declare exactly {len(definition.schedule)} tasks"
        )
    private_records = reveal.get("tasks")
    if not isinstance(private_records, list) or len(private_records) != len(
        definition.schedule
    ):
        raise ValueError(
            f"private reveal must declare exactly {len(definition.schedule)} tasks"
        )
    used_predicate_signatures: set[tuple[bool, ...]] = set()
    used_joint_behaviors: set[tuple[tuple[bool, ...], tuple[int, ...]]] = set()
    initial_predicate, initial_mapper = _initial_components()
    domain = predicate_probe_domain()
    initial_predicate_signature = tuple(
        _evaluate_bool(initial_predicate, item) for item in domain
    )
    for task_index, raw_public in enumerate(public_records):
        if not isinstance(raw_public, dict) or not isinstance(raw_public.get("task_id"), str):
            raise ValueError("invalid public task record")
        public_record = cast(dict[str, Any], raw_public)
        task_id = cast(str, public_record["task_id"])
        expected_task_id, expected_shape, expected_family = definition.schedule[task_index]
        if task_id != expected_task_id:
            raise ValueError(
                f"task order mismatch: expected {expected_task_id}, observed {task_id}"
            )
        private_record = _task_by_id(reveal.get("tasks"), task_id, source="reveal")
        relative = public_record.get("path")
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise ValueError(f"unsafe public task path for {task_id}")
        task_path = public_root / relative
        task_bytes = task_path.read_bytes()
        task = _read_object(task_path)
        _public_keys_are_safe(task)
        if sha256(task_bytes) != public_record.get("task_file_sha256"):
            raise ValueError(f"task file hash mismatch for {task_id}")
        canonical_task_hash = sha256(canonical_bytes(task))
        if canonical_task_hash != public_record.get("task_canonical_sha256"):
            raise ValueError(f"canonical task hash mismatch for {task_id}")

        preimage = private_record.get("commitment_preimage")
        if not isinstance(preimage, dict):
            raise ValueError(f"missing commitment preimage for {task_id}")
        if set(preimage) != {
            "schema",
            "task_id",
            "task_canonical_sha256",
            "target",
            "nonce",
        } or preimage.get("schema") != definition.target_schema:
            raise ValueError(f"invalid commitment preimage for {task_id}")
        nonce = preimage.get("nonce")
        try:
            nonce_bytes = bytes.fromhex(nonce) if isinstance(nonce, str) else b""
        except ValueError as error:
            raise ValueError(f"invalid commitment nonce for {task_id}") from error
        if len(nonce_bytes) != 32:
            raise ValueError(f"invalid commitment nonce for {task_id}")
        commitment = sha256(canonical_bytes(preimage))
        private_commitment = private_record.get("target_commitment_sha256")
        if (
            commitment != public_record.get("target_commitment_sha256")
            or commitment != private_commitment
        ):
            raise ValueError(f"target commitment mismatch for {task_id}")
        if preimage.get("task_id") != task_id or preimage.get(
            "task_canonical_sha256"
        ) != canonical_task_hash:
            raise ValueError(f"target preimage is bound to a different task for {task_id}")
        target = preimage.get("target")
        if not isinstance(target, dict) or set(target) != {"predicate", "mapper"}:
            raise ValueError(f"missing target for {task_id}")
        predicate = target.get("predicate")
        mapper = target.get("mapper")
        if not isinstance(predicate, dict) or canonical_key(predicate) not in predicate_keys:
            raise ValueError(f"target predicate is outside the catalog for {task_id}")
        if not isinstance(mapper, dict) or canonical_key(mapper) not in mapper_keys:
            raise ValueError(f"target mapper is outside the catalog for {task_id}")

        observed_signature = tuple(
            _evaluate_bool(cast(AstNode, predicate), item)
            for item in predicate_probe_domain()
        )
        if not 3 <= sum(observed_signature) <= 5:
            raise ValueError(f"target predicate support is outside the protocol for {task_id}")
        if not observed_signature[0] and not observed_signature[-1]:
            observed_shape = "bounded"
        elif observed_signature[-1] and not observed_signature[0]:
            observed_shape = "lower"
        elif observed_signature[0] and not observed_signature[-1]:
            observed_shape = "upper"
        else:
            raise ValueError(f"target predicate shape is outside the protocol for {task_id}")
        private_shape = private_record.get("predicate_shape")
        if observed_shape != expected_shape or private_shape != expected_shape:
            raise ValueError(f"target predicate shape mismatch for {task_id}")
        observed_family = _mapper_family(_mapper_signature(cast(AstNode, mapper)))
        private_family = private_record.get("mapper_family")
        if observed_family != expected_family or private_family != expected_family:
            raise ValueError(f"target mapper family mismatch for {task_id}")

        expected_signature, selected_predicate, selected_mapper, expected_rejections = (
            _select_target(
                task_id=task_id,
                shape=expected_shape,
                family=expected_family,
                seed=revealed_seed,
                used_predicate_signatures=used_predicate_signatures,
                used_joint_behaviors=used_joint_behaviors,
                schema=definition.schema,
                require_unique_bounded_predicates=(
                    definition.require_unique_bounded_predicates
                ),
            )
        )
        if (
            observed_signature != expected_signature
            or canonical_key(cast(AstNode, predicate)) != canonical_key(selected_predicate)
            or canonical_key(cast(AstNode, mapper)) != canonical_key(selected_mapper)
            or private_record.get("structural_rejections_before_acceptance")
            != expected_rejections
        ):
            raise ValueError(f"target is not the first deterministic accepted pair for {task_id}")

        predicate_distance = sum(
            left != right
            for left, right in zip(
                observed_signature, initial_predicate_signature, strict=True
            )
        )
        if predicate_distance < 2:
            raise ValueError(f"target predicate is too close to frozen start for {task_id}")
        support = tuple(
            item for item, keep in zip(domain, observed_signature, strict=True) if keep
        )
        target_outputs = tuple(_evaluate_int(cast(AstNode, mapper), item) for item in support)
        initial_outputs = tuple(_evaluate_int(initial_mapper, item) for item in support)
        if len(set(target_outputs)) < 3:
            raise ValueError(f"target mapper has too few retained outputs for {task_id}")
        if sum(
            left != right
            for left, right in zip(target_outputs, initial_outputs, strict=True)
        ) < 2:
            raise ValueError(f"target mapper is too close to frozen start for {task_id}")
        joint_behavior = _joint_behavior(observed_signature, cast(AstNode, mapper))
        if joint_behavior in used_joint_behaviors:
            raise ValueError(f"duplicate joint target behavior for {task_id}")
        if (
            definition.require_unique_bounded_predicates
            and observed_shape == "bounded"
            and observed_signature in used_predicate_signatures
        ):
            raise ValueError(f"duplicate bounded predicate behavior for {task_id}")
        used_predicate_signatures.add(observed_signature)
        used_joint_behaviors.add(joint_behavior)

        config = load_experiment_config(task_path)
        expected_iterations = 4 if definition.version == "v2" else 5
        if (
            tuple(config.spec.integer_constants) != CONSTANTS
            or len(config.spec.examples) != 12
            or config.smc.seed != 17
            or config.smc.iterations != expected_iterations
        ):
            raise ValueError(f"public task structure mismatch for {task_id}")
        example_inputs = [example.input_value for example in config.spec.examples]
        domain = list(predicate_probe_domain())
        if (
            example_inputs[0] != []
            or example_inputs[1:9] != [[item] for item in domain]
            or example_inputs[9] != domain
            or not isinstance(example_inputs[10], list)
            or sorted(cast(list[int], example_inputs[10])) != domain
            or not isinstance(example_inputs[11], list)
            or len(example_inputs[11]) != 2 * len(domain)
            or sorted(cast(list[int], example_inputs[11])[: len(domain)]) != domain
            or sorted(cast(list[int], example_inputs[11])[len(domain) :]) != domain
        ):
            raise ValueError(f"public example schedule mismatch for {task_id}")
        program = _assemble_program(cast(AstNode, predicate), cast(AstNode, mapper))
        initial_program = _assemble_program(initial_predicate, initial_mapper)
        initial_is_exact = True
        for example in config.spec.examples:
            predicted = evaluate_program(cast(Node, program), example.input_value)
            if predicted != example.output_value:
                raise ValueError(f"target fails a public example for {task_id}")
            initial_predicted = evaluate_program(
                cast(Node, initial_program), example.input_value
            )
            if initial_predicted != example.output_value:
                initial_is_exact = False
        if initial_is_exact:
            raise ValueError(f"initial complete program is exact for {task_id}")
        verified.append(task_id)
    if len(set(verified)) != len(definition.schedule):
        raise ValueError("task IDs must be unique")
    return {"schema": definition.schema, "verified": verified, "exact": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare-secret", help="create a custodial raw secret and print only its commitment"
    )
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--suite-version", choices=tuple(_DEFINITIONS_BY_VERSION), default="v2")
    generate = commands.add_parser("generate", help="create a new versioned suite")
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument("--seed-hex")
    generate.add_argument("--seed-file", type=Path)
    generate.add_argument("--suite-version", choices=tuple(_DEFINITIONS_BY_VERSION), default="v1")
    verify = commands.add_parser("verify", help="verify a public suite against its reveal")
    verify.add_argument("--public", required=True, type=Path)
    verify.add_argument("--reveal", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare-secret":
        commitment = prepare_secret(args.output, suite_version=args.suite_version)
        print(json.dumps({"seed_commitment_sha256": commitment}, sort_keys=True))
        return 0
    if args.command == "generate":
        manifest = generate_suite(
            args.output,
            seed_hex=args.seed_hex,
            seed_file=args.seed_file,
            suite_version=args.suite_version,
        )
        print(
            json.dumps(
                {
                    "status": "generated",
                    "public_manifest": str(args.output / "public" / "manifest.json"),
                    "task_count": manifest["task_count"],
                    "private_seed_printed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    result = verify_suite(args.public, args.reveal)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

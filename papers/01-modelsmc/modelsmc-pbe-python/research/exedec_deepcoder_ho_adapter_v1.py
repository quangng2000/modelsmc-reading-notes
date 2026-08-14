"""Debug-only adapter for a strict ExeDec DeepCoder higher-order subset.

This is an independent compatibility adapter, not a copy of ExeDec's runtime.
The semantic authority is the public Google DeepMind ExeDec repository pinned by
``EXEDEC_SOURCE_COMMIT`` below.  The released datasets are public and therefore
must never be described as blind or contamination-free confirmatory evidence.

The supported source shape is intentionally small and exact::

    x0 = INPUT | x1 = Filter P x0 | x2 = Map M x1

where ``P`` is ``(>0)`` or ``(<0)`` and ``M`` is one of the seven integer
lambdas listed in ``SUPPORTED_MAPPERS``.  Such a program is translated to the
existing typed ``FoldRightProgram`` filter-map family without executing source
code or importing TensorFlow, JAX, or ExeDec.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.core.results import RejectedProgram, ScoredProgram
from modelsmc_pbe.core.scorer import ProgramScorer
from modelsmc_pbe.domain.ast import AstNode, ProgramAst, canonical_key

ADAPTER_SCHEMA = "exedec-deepcoder-filter-map-ho-adapter-v1"
BUNDLE_SCHEMA = "exedec-deepcoder-filter-map-ho-debug-bundle-v1"
ORACLE_SCHEMA = "exedec-deepcoder-filter-map-ho-debug-oracle-v1"

EXEDEC_SOURCE_REPOSITORY = "https://github.com/google-deepmind/exedec"
EXEDEC_SOURCE_COMMIT = "ef046ce2cc3fcd024e32f5dfe00e69700dac82ed"
EXEDEC_SOURCE_DATA_PATH = "data/test_data/deepcoder/COMPOSE_DIFFERENT_CONCEPTS.jsonl"
EXEDEC_SOURCE_DATA_SHA256 = "0acdb1d9a502278ea6f3d2ec92db1b903e849464af4b9a078cdbc801f1457cb5"
EXEDEC_SOFTWARE_LICENSE = "Apache-2.0"
EXEDEC_DATA_LICENSE = "CC-BY-4.0"

DEEPCODER_MIN_INT = -50
DEEPCODER_MAX_INT = 50
DEEPCODER_MAX_INPUT_LIST_LENGTH = 5
INTEGER_CONSTANTS = (-1, 0, 1, 2, 3, 4)
SUPPORTED_PREDICATES = ("(<0)", "(>0)")
SUPPORTED_MAPPERS = (
    "(+1)",
    "(-1)",
    "(*2)",
    "(*(-1))",
    "(**2)",
    "(*3)",
    "(*4)",
)
OFFICIAL_LIST_VALUE_RANGES = ((0, 4), (0, 9), (-10, 10), (-50, 50))

_VARIABLE = r"x(?:0|[1-9][0-9]*)"
_INPUT_LINE = re.compile(rf"^(?P<lhs>{_VARIABLE})\s*=\s*INPUT$")
_CALL_LINE = re.compile(
    rf"^(?P<lhs>{_VARIABLE})\s*=\s*(?P<op>Filter|Map)\s+"
    rf"(?P<lambda>\([^\s]+\))\s+(?P<arg>{_VARIABLE})$"
)
_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")


class AdapterError(ValueError):
    """The source or generated adapter record violates a frozen invariant."""


class UnsupportedProgramError(AdapterError):
    """A valid-looking DeepCoder program is outside the declared subset."""


@dataclass(frozen=True, slots=True)
class SupportedProgram:
    """The complete source semantics retained by the adapter."""

    input_variable: str
    predicate: str
    mapper: str


@dataclass(frozen=True, slots=True)
class ReleasedRecord:
    """One validated released PBE record in the supported subset."""

    index: int
    program_text: str
    source_line_sha256: str
    source_program: SupportedProgram
    inputs: tuple[tuple[int, ...], ...]
    outputs: tuple[tuple[int, ...], ...]
    target_ast: ProgramAst
    syntax_fingerprint_sha256: str
    semantic_fingerprint_sha256: str
    public_pbe_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class SourceAudit:
    """Deterministic audit of one released JSONL file."""

    source_sha256: str
    total_records: int
    compatible_occurrences: int
    distinct_semantic_targets: int
    selected_source_indices: tuple[int, ...]
    semantic_collision_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class CollisionRegistry:
    """Released semantic fingerprints and their public provenance."""

    fingerprints: Mapping[str, tuple[str, ...]]

    def collisions(self, target: SupportedProgram) -> tuple[str, ...]:
        """Return released occurrences equivalent on the full scalar domain."""

        return self.fingerprints.get(semantic_fingerprint(target), ())


def canonical_bytes(value: object) -> bytes:
    """Return deterministic JSON bytes for commitments and generated files."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(canonical_bytes(value))


def _parse_int_list(text: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(text, str):
        raise AdapterError(f"{field} must be a DeepCoder result string")
    stripped = text.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        raise AdapterError(f"{field} must be a DeepCoder integer-list result")
    body = stripped[1:-1].strip()
    if not body:
        return ()
    values: list[int] = []
    for token in body.split():
        if _INTEGER.fullmatch(token) is None:
            raise AdapterError(f"{field} contains a noncanonical integer token {token!r}")
        value = int(token)
        if not DEEPCODER_MIN_INT <= value <= DEEPCODER_MAX_INT:
            raise AdapterError(f"{field} contains out-of-range DeepCoder integer {value}")
        values.append(value)
    return tuple(values)


def _parse_single_list_state(text: object, variable: str, *, field: str) -> tuple[int, ...]:
    if not isinstance(text, str):
        raise AdapterError(f"{field} must be a DeepCoder program-state string")
    pieces = [piece.strip() for piece in text.split("|")]
    if len(pieces) != 1:
        raise AdapterError(f"{field} must contain exactly one source input")
    prefix = f"{variable} ="
    if not pieces[0].startswith(prefix):
        raise AdapterError(f"{field} must bind the supported input variable {variable}")
    values = _parse_int_list(pieces[0][len(prefix) :].strip(), field=field)
    if len(values) > DEEPCODER_MAX_INPUT_LIST_LENGTH:
        raise AdapterError(
            f"{field} exceeds the pinned DeepCoder input-list length "
            f"{DEEPCODER_MAX_INPUT_LIST_LENGTH}"
        )
    return values


def parse_supported_program(program_text: object) -> SupportedProgram:
    """Parse exactly the linear two-statement filter-then-map subset."""

    if not isinstance(program_text, str):
        raise AdapterError("program must be a string")
    lines = [line.strip() for line in program_text.split("|")]
    if len(lines) != 3:
        raise UnsupportedProgramError(
            "supported programs require one input and exactly Filter then Map"
        )
    input_match = _INPUT_LINE.fullmatch(lines[0])
    filter_match = _CALL_LINE.fullmatch(lines[1])
    map_match = _CALL_LINE.fullmatch(lines[2])
    if input_match is None or filter_match is None or map_match is None:
        raise UnsupportedProgramError("program is outside the strict linear Filter-Map syntax")
    input_variable = input_match.group("lhs")
    filter_variable = filter_match.group("lhs")
    map_variable = map_match.group("lhs")
    if len({input_variable, filter_variable, map_variable}) != 3:
        raise UnsupportedProgramError("source variables must be distinct")
    if filter_match.group("op") != "Filter" or map_match.group("op") != "Map":
        raise UnsupportedProgramError("operation order must be exactly Filter then Map")
    if filter_match.group("arg") != input_variable or map_match.group("arg") != filter_variable:
        raise UnsupportedProgramError("source data flow must be a single linear chain")
    predicate = filter_match.group("lambda")
    mapper = map_match.group("lambda")
    if predicate not in SUPPORTED_PREDICATES:
        raise UnsupportedProgramError(f"unsupported Filter lambda {predicate}")
    if mapper not in SUPPORTED_MAPPERS:
        raise UnsupportedProgramError(f"unsupported Map lambda {mapper}")
    return SupportedProgram(input_variable=input_variable, predicate=predicate, mapper=mapper)


def _literal(value: int) -> AstNode:
    return {"kind": "IntLiteral", "intValue": str(value)}


def _binary(kind: str, left: AstNode, right: AstNode) -> AstNode:
    return {"kind": kind, "left": left, "right": right}


def translate_target(program: SupportedProgram) -> ProgramAst:
    """Translate supported DeepCoder semantics into the typed filter-map AST."""

    item: AstNode = {"kind": "Item"}
    zero = _literal(0)
    condition = (
        _binary("LessThan", item, zero)
        if program.predicate == "(<0)"
        else _binary("LessThan", zero, item)
    )
    mapper_by_token: dict[str, AstNode] = {
        "(+1)": _binary("Add", item, _literal(1)),
        "(-1)": _binary("Subtract", item, _literal(1)),
        "(*2)": _binary("Multiply", item, _literal(2)),
        "(*(-1))": _binary("Subtract", zero, item),
        "(**2)": _binary("Multiply", item, item),
        "(*3)": _binary("Multiply", item, _literal(3)),
        "(*4)": _binary("Multiply", item, _literal(4)),
    }
    mapper = mapper_by_token[program.mapper]
    accumulator: AstNode = {"kind": "Accumulator"}
    return {
        "kind": "FoldRightProgram",
        "initial": {"kind": "EmptyIntList"},
        "reducer": {
            "kind": "IfThenElse",
            "condition": condition,
            "thenExpr": {
                "kind": "PrependInt",
                "head": mapper,
                "tail": accumulator,
            },
            "elseExpr": accumulator,
        },
    }


def _map_scalar(mapper: str, value: int) -> int:
    if mapper == "(+1)":
        return value + 1
    if mapper == "(-1)":
        return value - 1
    if mapper == "(*2)":
        return value * 2
    if mapper == "(*(-1))":
        return -value
    if mapper == "(**2)":
        return value**2
    if mapper == "(*3)":
        return value * 3
    if mapper == "(*4)":
        return value * 4
    raise AdapterError(f"unknown frozen mapper {mapper}")


def run_reference(program: SupportedProgram, inputs: Sequence[int]) -> tuple[int, ...] | None:
    """Evaluate the pinned source subset, including DeepCoder's integer bounds.

    ``None`` represents an undefined DeepCoder execution.  ExeDec validates the
    result after every operation; Filter preserves already-validated inputs and
    Map is the only value-changing operation in this subset.
    """

    if len(inputs) > DEEPCODER_MAX_INPUT_LIST_LENGTH:
        return None
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not DEEPCODER_MIN_INT <= value <= DEEPCODER_MAX_INT
        for value in inputs
    ):
        return None
    keep = (lambda value: value < 0) if program.predicate == "(<0)" else (lambda value: value > 0)
    mapped = tuple(_map_scalar(program.mapper, value) for value in inputs if keep(value))
    if any(not DEEPCODER_MIN_INT <= value <= DEEPCODER_MAX_INT for value in mapped):
        return None
    return mapped


def semantic_fingerprint(program: SupportedProgram) -> str:
    """Fingerprint complete pointwise behavior over the pinned scalar domain.

    Filter-Map is pointwise and order preserving, so empty-list behavior plus
    every singleton (including undefined executions) determines its behavior on
    every valid list in this restricted family.
    """

    outcomes: list[list[int] | None] = []
    for value in range(DEEPCODER_MIN_INT, DEEPCODER_MAX_INT + 1):
        output = run_reference(program, [value])
        outcomes.append(None if output is None else list(output))
    return _sha256_json(
        {
            "schema": f"{ADAPTER_SCHEMA}-complete-singleton-semantics-v1",
            "scalar_domain": [DEEPCODER_MIN_INT, DEEPCODER_MAX_INT],
            "empty": [],
            "singletons": outcomes,
        }
    )


def _validate_record(raw: object, *, line_number: int, line_bytes: bytes) -> ReleasedRecord:
    if not isinstance(raw, dict):
        raise AdapterError(f"line {line_number} must contain a JSON object")
    if set(raw) != {"index", "inputs", "outputs", "program"}:
        raise AdapterError(f"line {line_number} has an unexpected released-record schema")
    index = raw["index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise AdapterError(f"line {line_number} has an invalid index")
    source_program = parse_supported_program(raw["program"])
    raw_inputs = raw["inputs"]
    raw_outputs = raw["outputs"]
    if not isinstance(raw_inputs, list) or not isinstance(raw_outputs, list):
        raise AdapterError(f"released record {index} inputs and outputs must be lists")
    if not raw_inputs or len(raw_inputs) != len(raw_outputs):
        raise AdapterError(f"released record {index} has inconsistent examples")
    inputs = tuple(
        _parse_single_list_state(value, source_program.input_variable, field=f"inputs[{i}]")
        for i, value in enumerate(raw_inputs)
    )
    outputs = tuple(
        _parse_int_list(value, field=f"outputs[{i}]") for i, value in enumerate(raw_outputs)
    )
    expected = tuple(run_reference(source_program, value) for value in inputs)
    if any(value is None for value in expected):
        raise AdapterError(f"released record {index} is undefined under pinned semantics")
    if cast(tuple[tuple[int, ...], ...], expected) != outputs:
        raise AdapterError(f"released record {index} disagrees with the pinned subset semantics")
    target_ast = translate_target(source_program)
    public_payload = {
        "inputs": [list(value) for value in inputs],
        "outputs": [list(value) for value in outputs],
    }
    return ReleasedRecord(
        index=index,
        program_text=cast(str, raw["program"]),
        source_line_sha256=_sha256_bytes(line_bytes),
        source_program=source_program,
        inputs=inputs,
        outputs=outputs,
        target_ast=target_ast,
        syntax_fingerprint_sha256=_sha256_bytes(canonical_key(target_ast).encode("utf-8")),
        semantic_fingerprint_sha256=semantic_fingerprint(source_program),
        public_pbe_fingerprint_sha256=_sha256_json(public_payload),
    )


def scan_released_source(
    source: Path,
    *,
    enforce_official_binding: bool = True,
) -> tuple[SourceAudit, tuple[ReleasedRecord, ...]]:
    """Scan, validate, and semantically deduplicate one released test file."""

    source_bytes = source.read_bytes()
    source_sha256 = _sha256_bytes(source_bytes)
    if enforce_official_binding and source_sha256 != EXEDEC_SOURCE_DATA_SHA256:
        raise AdapterError(
            "released source digest differs from the pinned ExeDec commit: "
            f"expected {EXEDEC_SOURCE_DATA_SHA256}, received {source_sha256}"
        )
    compatible: list[ReleasedRecord] = []
    seen_indices: set[int] = set()
    lines = source_bytes.splitlines()
    for line_number, line_bytes in enumerate(lines, start=1):
        try:
            raw = json.loads(line_bytes)
        except json.JSONDecodeError as error:
            raise AdapterError(f"line {line_number} is invalid JSON: {error.msg}") from error
        if not isinstance(raw, dict):
            raise AdapterError(f"line {line_number} must contain a JSON object")
        index = raw.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index in seen_indices:
            raise AdapterError(f"line {line_number} has a missing, invalid, or duplicate index")
        seen_indices.add(index)
        try:
            compatible.append(
                _validate_record(raw, line_number=line_number, line_bytes=line_bytes)
            )
        except UnsupportedProgramError:
            continue

    groups: dict[str, list[ReleasedRecord]] = defaultdict(list)
    for record in compatible:
        groups[record.semantic_fingerprint_sha256].append(record)
    selected = tuple(
        sorted(
            (min(records, key=lambda record: record.index) for records in groups.values()),
            key=lambda record: record.index,
        )
    )
    collision_groups = tuple(
        sorted(
            (tuple(sorted(record.index for record in records)) for records in groups.values()),
            key=lambda indices: indices[0],
        )
    )
    audit = SourceAudit(
        source_sha256=source_sha256,
        total_records=len(lines),
        compatible_occurrences=len(compatible),
        distinct_semantic_targets=len(groups),
        selected_source_indices=tuple(record.index for record in selected),
        semantic_collision_groups=collision_groups,
    )
    return audit, selected


def _program_strings_from_release_object(raw: object) -> tuple[str, ...]:
    """Extract target and few-shot program strings from either released schema."""

    if not isinstance(raw, dict):
        return ()
    programs: list[str] = []
    direct = raw.get("program")
    if isinstance(direct, str):
        programs.append(direct)
    test_problem = raw.get("test_problem")
    if isinstance(test_problem, dict) and isinstance(test_problem.get("program"), str):
        programs.append(cast(str, test_problem["program"]))
    few_shot = raw.get("few_shot_examples")
    if isinstance(few_shot, list):
        programs.extend(
            cast(str, record["program"])
            for record in few_shot
            if isinstance(record, dict) and isinstance(record.get("program"), str)
        )
    return tuple(programs)


def build_release_collision_registry(paths: Iterable[Path]) -> CollisionRegistry:
    """Index supported semantics across released test and LLM JSONL files."""

    registry: dict[str, list[str]] = defaultdict(list)
    for path in sorted(paths, key=lambda value: value.as_posix()):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise AdapterError(f"{path}:{line_number} is invalid JSON: {error.msg}") from error
            for occurrence, program_text in enumerate(_program_strings_from_release_object(raw)):
                try:
                    program = parse_supported_program(program_text)
                except UnsupportedProgramError:
                    continue
                fingerprint = semantic_fingerprint(program)
                registry[fingerprint].append(f"{path.as_posix()}:{line_number}:{occurrence}")
    return CollisionRegistry(
        fingerprints={key: tuple(values) for key, values in sorted(registry.items())}
    )


def assert_no_release_collision(
    target: SupportedProgram,
    registry: CollisionRegistry,
) -> None:
    """Fail closed if a proposed future task matches released subset behavior."""

    occurrences = registry.collisions(target)
    if occurrences:
        raise AdapterError(
            "candidate target semantically collides with released ExeDec material: "
            + ", ".join(occurrences)
        )


def public_task(record: ReleasedRecord) -> dict[str, object]:
    """Build the target-free JSON task consumed by the current typed SMC."""

    return {
        "name": f"exedec-public-debug-compose-different-concepts-{record.index:04d}",
        "signature": {"input": "List<Int>", "output": "List<Int>"},
        "examples": [
            {"input": [str(value) for value in inputs], "output": [str(value) for value in output]}
            for inputs, output in zip(record.inputs, record.outputs, strict=True)
        ],
        "integerConstants": [str(value) for value in INTEGER_CONSTANTS],
    }


def hidden_examples(
    target: SupportedProgram,
    *,
    debug_seed: int,
    random_multi_examples: int = 128,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Create deterministic debug-only held-out semantic probes.

    Every target-defined singleton in ``[-50, 50]`` is included.  Seeded
    multi-element inputs sample the four value ranges used by ExeDec's public
    generator, with lengths two through the pinned maximum.  Undefined target
    inputs are rejected because DeepCoder programs are partial.
    """

    if isinstance(debug_seed, bool) or not isinstance(debug_seed, int) or debug_seed < 0:
        raise ValueError("debug_seed must be a nonnegative integer")
    if (
        isinstance(random_multi_examples, bool)
        or not isinstance(random_multi_examples, int)
        or random_multi_examples < 0
    ):
        raise ValueError("random_multi_examples must be a nonnegative integer")
    examples: list[tuple[tuple[int, ...], tuple[int, ...]]] = [((), ())]
    for value in range(DEEPCODER_MIN_INT, DEEPCODER_MAX_INT + 1):
        output = run_reference(target, [value])
        if output is not None:
            examples.append(((value,), output))

    derived_seed = int.from_bytes(
        hashlib.sha256(
            f"{ADAPTER_SCHEMA}\0{debug_seed}\0{semantic_fingerprint(target)}".encode()
        ).digest()[:8],
        "big",
    )
    rng = random.Random(derived_seed)
    seen = {inputs for inputs, _ in examples}
    attempts = 0
    added = 0
    while added < random_multi_examples:
        attempts += 1
        if attempts > max(10_000, random_multi_examples * 1_000):
            raise AdapterError("could not generate enough valid hidden debug probes")
        low, high = OFFICIAL_LIST_VALUE_RANGES[attempts % len(OFFICIAL_LIST_VALUE_RANGES)]
        inputs = tuple(
            rng.randint(low, high)
            for _ in range(rng.randint(2, DEEPCODER_MAX_INPUT_LIST_LENGTH))
        )
        if inputs in seen:
            continue
        output = run_reference(target, inputs)
        if output is None:
            continue
        seen.add(inputs)
        examples.append((inputs, output))
        added += 1
    return tuple(examples)


def debug_oracle(
    record: ReleasedRecord,
    *,
    debug_seed: int,
    random_multi_examples: int = 128,
) -> dict[str, object]:
    """Build a provider-private but non-secret oracle for released debug data."""

    probes = hidden_examples(
        record.source_program,
        debug_seed=debug_seed,
        random_multi_examples=random_multi_examples,
    )
    return {
        "schema": ORACLE_SCHEMA,
        "classification": "debug-only-public-released-target-not-confirmatory",
        "source_index": record.index,
        "source_line_sha256": record.source_line_sha256,
        "target_ast": record.target_ast,
        "syntax_fingerprint_sha256": record.syntax_fingerprint_sha256,
        "semantic_fingerprint_sha256": record.semantic_fingerprint_sha256,
        "evaluation_policy": {
            "target_domain": [DEEPCODER_MIN_INT, DEEPCODER_MAX_INT],
            "maximum_input_list_length": DEEPCODER_MAX_INPUT_LIST_LENGTH,
            "all_target_defined_singletons": True,
            "random_multi_examples": random_multi_examples,
            "undefined_target_inputs": "excluded",
            "candidate_acceptance": "exact output equality on every stored probe",
            "universality_claim": False,
        },
        "examples": [
            {"input": [str(value) for value in inputs], "output": [str(value) for value in output]}
            for inputs, output in probes
        ],
    }


def evaluate_candidate(candidate: object, oracle: Mapping[str, object]) -> dict[str, object]:
    """Evaluate one typed candidate against a stored hidden debug oracle."""

    if oracle.get("schema") != ORACLE_SCHEMA:
        raise AdapterError("hidden oracle schema is not supported")
    raw_examples = oracle.get("examples")
    if not isinstance(raw_examples, list) or not raw_examples:
        raise AdapterError("hidden oracle has no examples")
    config = ExperimentConfig.model_validate(
        {
            "spec": {
                "name": "exedec-hidden-debug-evaluation",
                "signature": {"input": "List<Int>", "output": "List<Int>"},
                "examples": raw_examples,
                "integerConstants": [str(value) for value in INTEGER_CONSTANTS],
            },
            "smc": {
                "maxCost": 64,
                "maxDepth": 32,
                "maxNodes": 256,
                "lossCap": 1_000_000,
            },
        }
    )
    result = ProgramScorer(config).score(candidate)
    if isinstance(result, RejectedProgram):
        return {
            "schema": f"{ORACLE_SCHEMA}-evaluation-v1",
            "exact": False,
            "kind": "Rejected",
            "reason": result.reason,
        }
    assert isinstance(result, ScoredProgram)
    return {
        "schema": f"{ORACLE_SCHEMA}-evaluation-v1",
        "exact": result.exact_program,
        "kind": "Scored",
        "total_loss": result.total_loss,
        "exact_matches": result.exact_matches,
        "example_count": len(result.evaluations),
        "cost": result.cost,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_bytes(value) + b"\n")


def build_debug_bundle(
    source: Path,
    destination: Path,
    *,
    debug_seed: int,
    random_multi_examples: int = 128,
    enforce_official_binding: bool = True,
) -> dict[str, object]:
    """Generate a deterministic target-free/public and oracle/private bundle."""

    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing destination: {destination}")
    audit, records = scan_released_source(
        source,
        enforce_official_binding=enforce_official_binding,
    )
    generated: list[tuple[Path, dict[str, object], Path, dict[str, object]]] = []
    task_entries: list[dict[str, object]] = []
    for record in records:
        task_id = f"exedec-debug-{record.index:04d}"
        public_path = Path("public") / f"{task_id}.json"
        private_path = Path("private") / f"{task_id}.oracle.json"
        task = public_task(record)
        oracle = debug_oracle(
            record,
            debug_seed=debug_seed,
            random_multi_examples=random_multi_examples,
        )
        generated.append((public_path, task, private_path, oracle))
        task_entries.append(
            {
                "task_id": task_id,
                "source_index": record.index,
                "public_path": public_path.as_posix(),
                "public_sha256": _sha256_bytes(canonical_bytes(task) + b"\n"),
                "private_oracle_path": private_path.as_posix(),
                "private_oracle_sha256": _sha256_bytes(canonical_bytes(oracle) + b"\n"),
                "semantic_fingerprint_sha256": record.semantic_fingerprint_sha256,
            }
        )
    manifest: dict[str, object] = {
        "schema": BUNDLE_SCHEMA,
        "classification": "debug-only-public-released-data-no-provider-run-authorized",
        "source": {
            "repository": EXEDEC_SOURCE_REPOSITORY,
            "commit": EXEDEC_SOURCE_COMMIT,
            "relative_path": EXEDEC_SOURCE_DATA_PATH,
            "sha256": audit.source_sha256,
            "software_license": EXEDEC_SOFTWARE_LICENSE,
            "data_license": EXEDEC_DATA_LICENSE,
            "citation": "Shi et al., ExeDec, ICLR 2024",
        },
        "adapter": {
            "schema": ADAPTER_SCHEMA,
            "supported_predicates": list(SUPPORTED_PREDICATES),
            "supported_mappers": list(SUPPORTED_MAPPERS),
            "target_skeleton": "foldr-filter-map",
            "integer_constants": list(INTEGER_CONSTANTS),
        },
        "audit": asdict(audit),
        "debug_seed": debug_seed,
        "random_multi_examples_per_task": random_multi_examples,
        "tasks": task_entries,
        "claim_limits": [
            "released ExeDec tasks are public and may be present in model training data",
            "the adapter subset is not an official ExeDec split or full DeepCoder implementation",
            "hidden debug probes are finite and do not prove arbitrary-program equivalence",
            (
                "no provider run, SMC result, compositional-generalization result, "
                "or speedup is recorded"
            ),
        ],
    }
    destination.mkdir(parents=True)
    (destination / "public").mkdir()
    (destination / "private").mkdir()
    for public_path, task, private_path, oracle in generated:
        _write_json(destination / public_path, task)
        _write_json(destination / private_path, oracle)
    _write_json(destination / "manifest.json", manifest)
    return manifest


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AdapterError(f"invalid JSON in {path}: {error.msg}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="audit the pinned released JSONL")
    audit.add_argument("source", type=Path)
    audit.add_argument("--allow-unbound-source", action="store_true")

    build = subparsers.add_parser("build-debug", help="build debug-only adapted tasks")
    build.add_argument("source", type=Path)
    build.add_argument("destination", type=Path)
    build.add_argument("--debug-seed", type=int, required=True)
    build.add_argument("--random-multi-examples", type=int, default=128)
    build.add_argument("--allow-unbound-source", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="evaluate a typed AST on an oracle")
    evaluate.add_argument("candidate", type=Path)
    evaluate.add_argument("oracle", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        audit, _ = scan_released_source(
            args.source,
            enforce_official_binding=not args.allow_unbound_source,
        )
        print(canonical_bytes(asdict(audit)).decode())
        return 0
    if args.command == "build-debug":
        manifest = build_debug_bundle(
            args.source,
            args.destination,
            debug_seed=args.debug_seed,
            random_multi_examples=args.random_multi_examples,
            enforce_official_binding=not args.allow_unbound_source,
        )
        print(canonical_bytes(manifest).decode())
        return 0
    result = evaluate_candidate(_read_json(args.candidate), _read_json(args.oracle))
    print(canonical_bytes(result).decode())
    return 0 if result["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run full-support evidence-aware SMC from typed GPT repair shortlists.

The provider emits four local repair suggestions, but those suggestions are
not themselves treated as samples with an unknown probability.  Instead, the
application constructs and samples from the exactly evaluable proposal

    q(y | S, x, h) = (1 - epsilon) H_S(y) + epsilon g(y),

where ``H_S`` is the empirical distribution over four deterministic slots.
Malformed slots map to the current program, duplicates retain multiplicity,
and ``g`` is a normalized recursive grammar sampler that gives every complete
program positive probability without enumerating or counting complete programs.
The raw provider response is an auxiliary variable, so its unknown generation
probability cancels from the product-path Feynman--Kac weights.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import httpx
import torch

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.core.evaluate import evaluate_expression
from modelsmc_pbe.core.results import ScoredProgram
from modelsmc_pbe.core.scorer import ProgramScorer
from modelsmc_pbe.core.types import Node
from modelsmc_pbe.domain.ast import AstNode
from modelsmc_pbe.grammar import arithmetic_expressions, filter_predicates
from modelsmc_pbe.smc import effective_sample_size, normalize_log_weights, normalize_weights
from research.automatic_repair_feedback import derive_automatic_feedback
from research.automatic_shortlist_experiment import _provider_inventory, _shortlist_payload
from research.direct_json_repair_choice import _canonical_bytes, _write_json
from research.execution_guided_repair import (
    MODEL_REVISION,
    _assemble_program,
    _dsl_catalog,
)
from research.iterative_beam_experiment import (
    BeamState,
    _evidence_summary,
    derive_singleton_constraints,
    select_repair_hole,
)

SHORTLIST_SIZE = 4
ROUNDS = 4
PROPOSAL_SOURCES = ("llm", "grammar-random")

_MAPPER_TEMPLATES = (
    "item",
    "constant",
    "add(item,item)",
    "sub(item,item)",
    "mul(item,item)",
    "add(item,c)",
    "sub(item,c)",
    "mul(item,c)",
    "add(c,item)",
    "sub(c,item)",
    "mul(c,item)",
)
_ATOM_TEMPLATES = ("lt(item,c)", "lt(c,item)", "eq(item,c)")


@dataclass(frozen=True, slots=True)
class ProgramKey:
    predicate: str
    mapper: str


@dataclass(frozen=True, slots=True)
class Particle:
    key: ProgramKey
    score: ScoredProgram
    predicate_violations: int
    mapper_violations: int
    lineage: str


@dataclass(frozen=True, slots=True)
class ProposalDraw:
    key: ProgramKey
    branch: str
    probability: float


def proposal_probability(
    key: ProgramKey,
    slots: tuple[ProgramKey, ...],
    *,
    grammar_probability: Fraction,
    epsilon: float,
) -> float:
    """Evaluate the normalized shortlist/full-grammar mixture at one program."""

    if not slots:
        raise ValueError("proposal slots must not be empty")
    if not isinstance(grammar_probability, Fraction) or not 0 < grammar_probability <= 1:
        raise ValueError("grammar_probability must be finite and in (0, 1]")
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be finite and in (0, 1]")
    epsilon_fraction = Fraction(str(epsilon))
    local_mass = Fraction(slots.count(key), len(slots))
    return float((1 - epsilon_fraction) * local_mass + epsilon_fraction * grammar_probability)


def recursive_grammar_probabilities(
    predicate_order: tuple[str, ...],
    mapper_order: tuple[str, ...],
) -> tuple[dict[str, Fraction], dict[str, Fraction]]:
    """Define a normalized rule-factorized prior without a full-program count.

    Predicate generation first chooses ``atom`` versus ``and`` uniformly, then
    chooses comparison atoms uniformly inside that form. Mapper generation
    chooses one of five syntactic forms uniformly and then its operator and/or
    constant.  The catalog is used only to bind generated DSL strings to the
    public typed grammar; complete predicate/mapper pairs are never enumerated.
    """

    predicate_set = frozenset(predicate_order)
    mapper_set = frozenset(mapper_order)
    atoms = tuple(dsl for dsl in predicate_order if not dsl.startswith("and("))
    conjunctions = tuple(dsl for dsl in predicate_order if dsl.startswith("and("))
    if not atoms or not conjunctions:
        raise ValueError("predicate catalog must contain atoms and conjunctions")
    predicate_probability = {
        **{dsl: Fraction(1, 2 * len(atoms)) for dsl in atoms},
        **{dsl: Fraction(1, 2 * len(conjunctions)) for dsl in conjunctions},
    }
    if frozenset(predicate_probability) != predicate_set:
        raise ValueError("predicate grammar probability map does not match its catalog")

    forms: dict[str, list[str]] = {
        "item": [],
        "constant": [],
        "binary-item-item": [],
        "binary-item-constant": [],
        "binary-constant-item": [],
    }
    for dsl in mapper_order:
        if dsl == "item":
            form = "item"
        elif "(" not in dsl:
            form = "constant"
        else:
            left, right = dsl[dsl.index("(") + 1 : -1].split(",", maxsplit=1)
            if left == right == "item":
                form = "binary-item-item"
            elif left == "item":
                form = "binary-item-constant"
            elif right == "item":
                form = "binary-constant-item"
            else:
                raise ValueError(f"unrecognized mapper grammar form: {dsl}")
        forms[form].append(dsl)
    if any(not values for values in forms.values()):
        raise ValueError("mapper catalog is missing a recursive grammar form")
    mapper_probability = {
        dsl: Fraction(1, len(forms) * len(values))
        for values in forms.values()
        for dsl in values
    }
    if frozenset(mapper_probability) != mapper_set:
        raise ValueError("mapper grammar probability map does not match its catalog")
    if sum(predicate_probability.values()) != 1:
        raise ValueError("predicate recursive grammar does not normalize")
    if sum(mapper_probability.values()) != 1:
        raise ValueError("mapper recursive grammar does not normalize")
    return predicate_probability, mapper_probability


def program_grammar_probability(
    key: ProgramKey,
    predicate_probability: dict[str, Fraction],
    mapper_probability: dict[str, Fraction],
) -> Fraction:
    """Return the product of local recursive grammar-rule probabilities."""

    try:
        return predicate_probability[key.predicate] * mapper_probability[key.mapper]
    except KeyError as error:
        raise ValueError(f"program is outside the recursive grammar: {key}") from error


def _fraction_weighted_choice(
    values: tuple[str, ...],
    probabilities: dict[str, Fraction],
    *,
    rng: random.Random,
) -> str:
    """Draw exactly from unnormalized rational weights."""

    if not values:
        raise ValueError("weighted choice requires at least one value")
    denominator = 1
    for value in values:
        try:
            weight = probabilities[value]
        except KeyError as error:
            raise ValueError(f"weighted choice value has no probability: {value}") from error
        if weight <= 0:
            raise ValueError("weighted choice probabilities must be positive")
        denominator = math.lcm(denominator, weight.denominator)
    integer_weights = tuple(
        probabilities[value].numerator * (denominator // probabilities[value].denominator)
        for value in values
    )
    position = rng.randrange(sum(integer_weights))
    cumulative = 0
    for value, weight in zip(values, integer_weights, strict=True):
        cumulative += weight
        if position < cumulative:
            return value
    raise AssertionError("exact weighted choice exhausted its support")


def random_local_slots(
    *,
    parent: ProgramKey,
    hole: str,
    predicate_order: tuple[str, ...],
    mapper_order: tuple[str, ...],
    predicate_probability: dict[str, Fraction],
    mapper_probability: dict[str, Fraction],
    seed: int,
) -> tuple[ProgramKey, ...]:
    """Return four unique non-no-op local repairs from the recursive grammar.

    Sampling is weighted without replacement.  The unknown probability of this
    auxiliary shortlist is not used in downstream particle weights, exactly as
    for an LLM-produced shortlist.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("random shortlist seed must be an integer")
    if hole == "predicate":
        current = parent.predicate
        available = tuple(value for value in predicate_order if value != current)
        probabilities = predicate_probability
    elif hole == "mapper":
        current = parent.mapper
        available = tuple(value for value in mapper_order if value != current)
        probabilities = mapper_probability
    else:
        raise ValueError("random shortlist hole must be predicate or mapper")
    if len(available) < SHORTLIST_SIZE:
        raise ValueError("local grammar has fewer than four non-no-op repairs")
    rng = random.Random(seed)
    selected: list[str] = []
    for _ in range(SHORTLIST_SIZE):
        choice = _fraction_weighted_choice(available, probabilities, rng=rng)
        selected.append(choice)
        available = tuple(value for value in available if value != choice)
    if hole == "predicate":
        return tuple(ProgramKey(value, parent.mapper) for value in selected)
    return tuple(ProgramKey(parent.predicate, value) for value in selected)


def sample_proposal(
    *,
    slots: tuple[ProgramKey, ...],
    predicate_order: tuple[str, ...],
    mapper_order: tuple[str, ...],
    predicate_probability: dict[str, Fraction],
    mapper_probability: dict[str, Fraction],
    epsilon: float,
    rng: random.Random,
) -> ProposalDraw:
    """Sample one program from the application-controlled mixture."""

    if not predicate_order or not mapper_order:
        raise ValueError("grammar orders must not be empty")
    epsilon_fraction = Fraction(str(epsilon))
    restart = (
        rng.randrange(epsilon_fraction.denominator) < epsilon_fraction.numerator
    )
    if restart:
        atom_order = tuple(dsl for dsl in predicate_order if not dsl.startswith("and("))
        conjunction_order = tuple(dsl for dsl in predicate_order if dsl.startswith("and("))
        predicate = rng.choice(atom_order if rng.randrange(2) == 0 else conjunction_order)
        mapper_forms = (
            tuple(dsl for dsl in mapper_order if dsl == "item"),
            tuple(dsl for dsl in mapper_order if "(" not in dsl and dsl != "item"),
            tuple(dsl for dsl in mapper_order if "(item,item)" in dsl),
            tuple(
                dsl
                for dsl in mapper_order
                if dsl != "item" and dsl.endswith(",item)") and "(item,item)" not in dsl
            ),
            tuple(
                dsl
                for dsl in mapper_order
                if dsl.startswith(("add(item,", "sub(item,", "mul(item,"))
                and "(item,item)" not in dsl
            ),
        )
        mapper = rng.choice(mapper_forms[rng.randrange(len(mapper_forms))])
        key = ProgramKey(predicate, mapper)
        branch = "full-grammar-restart"
    else:
        key = rng.choice(slots)
        branch = "shortlist-slot"
    probability = proposal_probability(
        key,
        slots,
        grammar_probability=program_grammar_probability(
            key,
            predicate_probability,
            mapper_probability,
        ),
        epsilon=epsilon,
    )
    return ProposalDraw(key=key, branch=branch, probability=probability)


def stage_log_gamma(
    *,
    round_number: int,
    score: ScoredProgram,
    predicate_violations: int,
    mapper_violations: int,
    evidence_scale: float,
    log_prior: float,
) -> float:
    """Return the frozen four-stage unnormalized target log density."""

    if round_number not in {1, 2, 3, 4}:
        raise ValueError("round_number must be in 1..4")
    if evidence_scale < 0.0 or not math.isfinite(evidence_scale):
        raise ValueError("evidence_scale must be finite and nonnegative")
    if predicate_violations < 0 or mapper_violations < 0:
        raise ValueError("violation counts must be nonnegative")
    if round_number == 1:
        evidence = evidence_scale * predicate_violations
        execution = 0.0
    elif round_number == 2:
        evidence = evidence_scale * (predicate_violations + mapper_violations)
        execution = 0.0
    elif round_number == 3:
        evidence = 0.5 * evidence_scale * (predicate_violations + mapper_violations)
        execution = 0.5 * score.log_target
    else:
        evidence = 0.0
        execution = score.log_target
    return log_prior - evidence + execution


def normalized_child_weights(log_weights: list[float]) -> tuple[torch.Tensor, float]:
    """Normalize child weights and return ESS for artifact/test reuse."""

    normalized = normalize_log_weights(torch.tensor(log_weights, dtype=torch.float64))
    return normalized.weights, effective_sample_size(normalized.weights)


def population_schedule(
    *,
    rounds: int,
    parent_count: int,
    offspring_per_parent: int,
    first_round_offspring: int | None = None,
) -> tuple[tuple[int, ...], int]:
    """Return logical execution checkpoints and the provider-call cap."""

    if min(rounds, parent_count, offspring_per_parent) < 1:
        raise ValueError("rounds, parent_count, and offspring_per_parent must be positive")
    first = offspring_per_parent if first_round_offspring is None else first_round_offspring
    if isinstance(first, bool) or not isinstance(first, int) or first < 1:
        raise ValueError("first_round_offspring must be a positive integer")
    checkpoints = [1, 1 + first]
    checkpoints.extend(
        1 + first + offset * parent_count * offspring_per_parent
        for offset in range(1, rounds)
    )
    return tuple(checkpoints), 1 + (rounds - 1) * parent_count


def systematic_resample_count(
    weights: torch.Tensor,
    *,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Systematically draw an explicit number of ancestors."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    probabilities = normalize_weights(weights)
    offset = torch.rand((), dtype=torch.float64, generator=generator) / count
    positions = offset + torch.arange(count, dtype=torch.float64) / count
    cumulative = torch.cumsum(probabilities, dim=0)
    cumulative[-1] = 1.0
    return torch.searchsorted(cumulative, positions, right=True).to(dtype=torch.int64)


def decode_compact_candidate(hole: str, candidate: object) -> str:
    """Decode one integer-valued typed grammar object to canonical DSL."""

    if not isinstance(candidate, dict):
        raise ValueError("structured candidate must be an object")

    def constant(value: object) -> str:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("candidate constant must be an integer")
        return str(value)

    if hole == "mapper":
        if set(candidate) != {"template", "constant"}:
            raise ValueError("mapper candidate has the wrong fields")
        template = candidate["template"]
        if template not in _MAPPER_TEMPLATES:
            raise ValueError("mapper candidate has an invalid template")
        literal = constant(candidate["constant"])
        if template == "item":
            return "item"
        if template == "constant":
            return literal
        return cast(str, template).replace("c", literal)

    if hole != "predicate" or set(candidate) != {"combine", "left", "right"}:
        raise ValueError("predicate candidate has the wrong fields")

    def atom(value: object) -> str:
        if not isinstance(value, dict) or set(value) != {"template", "constant"}:
            raise ValueError("predicate atom has the wrong fields")
        template = value["template"]
        if template not in _ATOM_TEMPLATES:
            raise ValueError("predicate atom has an invalid template")
        return cast(str, template).replace("c", constant(value["constant"]))

    combine = candidate["combine"]
    left = atom(candidate["left"])
    right = atom(candidate["right"])
    if combine == "single":
        return left
    if combine == "and":
        return f"and({left},{right})"
    raise ValueError("predicate combine must be single or and")


def _constant_schema(constants: tuple[int, ...]) -> dict[str, object]:
    stable = tuple(sorted(set(constants)))
    if stable == tuple(range(stable[0], stable[-1] + 1)) and len(stable) > 32:
        return {"type": "integer", "minimum": stable[0], "maximum": stable[-1]}
    return {"type": "integer", "enum": list(stable)}


def _compact_structured_payload(
    payload: dict[str, object],
    *,
    hole: str,
    constants: tuple[int, ...],
) -> dict[str, object]:
    """Use integer constants and a compact interval schema for large catalogs."""

    constant_schema = _constant_schema(constants)
    messages = cast(list[dict[str, str]], payload["messages"])
    document = json.loads(messages[1]["content"])
    stable = tuple(sorted(set(constants)))
    constant_rule: object = (
        {"inclusive_integer_interval": [stable[0], stable[-1]]}
        if "minimum" in constant_schema
        else {"allowed_integers": list(stable)}
    )
    if hole == "mapper":
        candidate_schema: dict[str, object] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["template", "constant"],
            "properties": {
                "template": {"type": "string", "enum": list(_MAPPER_TEMPLATES)},
                "constant": constant_schema,
            },
        }
        document["grammar"] = {
            "mapper_templates": list(_MAPPER_TEMPLATES),
            "constant_rule": constant_rule,
            "note": "c is one allowed integer; ignored when the template contains no c",
        }
    else:
        atom_schema: dict[str, object] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["template", "constant"],
            "properties": {
                "template": {"type": "string", "enum": list(_ATOM_TEMPLATES)},
                "constant": constant_schema,
            },
        }
        candidate_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["combine", "left", "right"],
            "properties": {
                "combine": {"type": "string", "enum": ["single", "and"]},
                "left": atom_schema,
                "right": atom_schema,
            },
        }
        document["grammar"] = {
            "predicate_combine": ["single", "and"],
            "atom_templates": list(_ATOM_TEMPLATES),
            "constant_rule": constant_rule,
            "note": "right is ignored when combine is single",
        }
    messages[1]["content"] = _canonical_bytes(document).decode()
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": SHORTLIST_SIZE,
                "maxItems": SHORTLIST_SIZE,
                "items": candidate_schema,
            }
        },
    }
    cast(dict[str, Any], payload["response_format"])["json_schema"]["schema"] = schema
    return payload


def _augment_smc_prompt(
    payload: dict[str, object],
    *,
    round_number: int,
    state: BeamState,
    feedback: dict[str, object],
    hole: str,
    decision: dict[str, object],
) -> dict[str, object]:
    messages = cast(list[dict[str, str]], payload["messages"])
    document = json.loads(messages[1]["content"])
    fallback = cast(bool, decision["fallback_used"])
    document["smc_context"] = {
        "round": round_number,
        "maximum_rounds": ROUNDS,
        "current_loss": state.score.total_loss,
        "note": (
            "The application will construct a normalized proposal from this shortlist; "
            "do not provide probabilities."
        ),
    }
    document["hole_selection"] = {
        "selected_hole": hole,
        "evidence_backed": cast(bool, decision["evidence_backed"]),
        "fallback_used": fallback,
        "reason": decision["reason"],
    }
    document["mechanical_evidence_interpretation"] = _evidence_summary(feedback, hole)
    document["forbidden_no_op_expression"] = (
        state.predicate if hole == "predicate" else state.mapper
    )
    document["requirements"] = [
        "Return exactly four typed grammar expressions.",
        "Do not return the current expression.",
        "Ground every suggestion in public examples and interpreter feedback.",
        "Order alternatives from most to least promising.",
    ]
    messages[1]["content"] = _canonical_bytes(document).decode()
    return payload


def _sentinel_slots(parent: ProgramKey) -> tuple[ProgramKey, ...]:
    return (parent,) * SHORTLIST_SIZE


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def python_tree_binding(project_root: Path, binding: dict[str, object]) -> dict[str, object]:
    """Recompute a canonical all-Python source-tree binding."""

    if binding.get("schema") != "sha256-python-tree-v1":
        raise ValueError("Python tree binding has an unsupported schema")
    if binding.get("include") != "**/*.py":
        raise ValueError("Python tree binding must include exactly **/*.py")
    relative_root = binding.get("root")
    if not isinstance(relative_root, str):
        raise ValueError("Python tree binding root must be a string")
    tree_root = project_root / relative_root
    if not tree_root.is_dir():
        raise ValueError("Python tree binding root is not a directory")
    for path in tree_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Python tree binding rejects symlink: {path}")
    paths = sorted(
        (path for path in tree_root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(tree_root).as_posix().encode("utf-8"),
    )
    entries = [
        {
            "path": path.relative_to(tree_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]
    manifest = {
        "schema": "sha256-python-tree-v1",
        "root": relative_root,
        "include": "**/*.py",
        "entries": entries,
    }
    actual = {
        "schema": "sha256-python-tree-v1",
        "root": relative_root,
        "include": "**/*.py",
        "file_count": len(entries),
        "manifest_sha256": hashlib.sha256(_canonical_bytes(manifest)).hexdigest(),
    }
    if actual != binding:
        raise ValueError("frozen Python source-tree binding differs")
    return actual


def validate_frozen_invocation(args: argparse.Namespace) -> dict[str, object] | None:
    """Fail closed on every frozen source, task, and semantic run argument."""

    if args.study_protocol is None and args.run_id is None:
        if not args.allow_unfrozen_developmental:
            raise ValueError(
                "provider runs require a frozen protocol, or explicit "
                "--allow-unfrozen-developmental"
            )
        return {"mode": "explicit-unfrozen-developmental"}
    if args.allow_unfrozen_developmental:
        raise ValueError("do not combine frozen and unfrozen invocation modes")
    if args.study_protocol is None or args.run_id is None:
        raise ValueError("--study-protocol and --run-id must be provided together")
    if args.expected_study_protocol_sha256 is None:
        raise ValueError("frozen runs require --expected-study-protocol-sha256")
    study_path = args.study_protocol.resolve()
    study_sha256 = _sha256_file(study_path)
    if study_sha256 != args.expected_study_protocol_sha256:
        raise ValueError("study protocol differs from the external expected SHA-256")
    study = json.loads(study_path.read_text(encoding="utf-8"))
    if study.get("status") not in {
        "frozen-before-large-space-provider-calls",
        "frozen-before-provider-calls",
    }:
        raise ValueError("study protocol is not frozen")

    project_root = Path(__file__).resolve().parent.parent
    bindings = study.get("source_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("study protocol has no source bindings")
    records: list[dict[str, str]] = []
    for label, binding in sorted(bindings.items()):
        if label == "large_public_task":
            continue
        if label == "bounded_public_task":
            continue
        if isinstance(binding, dict) and "path" in binding and "sha256" in binding:
            path = project_root / cast(str, binding["path"])
            actual = _sha256_file(path)
            if actual != binding["sha256"]:
                raise ValueError(f"frozen source hash mismatch: {label}")
            records.append({"label": label, "path": path.as_posix(), "sha256": actual})
    dependency_bundle = bindings.get("dependency_bundle")
    if not isinstance(dependency_bundle, list) or not dependency_bundle:
        raise ValueError("study protocol has no dependency bundle")
    for binding in dependency_bundle:
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError("dependency binding has the wrong shape")
        path = project_root / cast(str, binding["path"])
        actual = _sha256_file(path)
        if actual != binding["sha256"]:
            raise ValueError(f"frozen dependency hash mismatch: {binding['path']}")
        records.append(
            {"label": "dependency", "path": path.as_posix(), "sha256": actual}
        )
    source_tree = bindings.get("modelsmc_python_tree")
    if not isinstance(source_tree, dict):
        raise ValueError("study protocol has no complete Python source-tree binding")
    validated_source_tree = python_tree_binding(project_root, source_tree)

    runs = study.get("runs")
    if not isinstance(runs, list):
        raise ValueError("study protocol has no runs")
    matches = [run for run in runs if isinstance(run, dict) and run.get("id") == args.run_id]
    if len(matches) != 1:
        raise ValueError("run ID is not uniquely frozen")
    run = cast(dict[str, object], matches[0])
    task_sha256 = _sha256_file(args.task.resolve())
    if task_sha256 != run.get("task_sha256"):
        raise ValueError("task bytes differ from the frozen run")
    expected_checkpoints, llm_call_cap = population_schedule(
        rounds=ROUNDS,
        parent_count=cast(int, run["parent_count"]),
        offspring_per_parent=cast(int, run["offspring_per_parent"]),
        first_round_offspring=cast(int, run["first_round_offspring"]),
    )
    proposal_source = cast(str, run.get("proposal_source", "llm"))
    if proposal_source not in PROPOSAL_SOURCES:
        raise ValueError("frozen run has an unsupported proposal source")
    random_shortlist_seed = run.get("random_shortlist_seed")
    if (proposal_source == "grammar-random") != (random_shortlist_seed is not None):
        raise ValueError(
            "frozen run requires random_shortlist_seed iff proposal_source is grammar-random"
        )
    expected_call_cap = llm_call_cap if proposal_source == "llm" else 0
    derived = {
        "logical_execution_cap": expected_checkpoints[-1],
        "provider_call_cap": expected_call_cap,
        "terminal_weighted_particles": (
            cast(int, run["parent_count"]) * cast(int, run["offspring_per_parent"])
        ),
    }
    for name, expected_value in derived.items():
        if run.get(name) != expected_value:
            raise ValueError(f"frozen run has an inconsistent derived field: {name}")

    provider = cast(dict[str, object], study["provider"])
    if MODEL_REVISION != provider.get("model_revision"):
        raise ValueError("served-model revision binding differs from the harness constant")
    expected = {
        "base_url": provider["base_url"],
        "model": provider["model"],
        "reasoning_effort": provider["reasoning_effort"],
        "temperature": provider["temperature"],
        "max_tokens": provider["max_output_tokens"],
        "timeout_seconds": provider["timeout_seconds"],
        "epsilon": run["epsilon"],
        "evidence_scale": run["evidence_scale"],
        "start_seed": run["start_seed"],
        "parent_count": run["parent_count"],
        "offspring_per_parent": run["offspring_per_parent"],
        "first_round_offspring": run["first_round_offspring"],
        "provider_seed": run["provider_seed"],
        "sample_seed": run["sample_seed"],
        "resample_seed": run["resample_seed"],
        "max_concurrency": run["max_concurrency"],
        "exact_reference_limit": run["exact_reference_limit"],
        "proposal_source": proposal_source,
        "random_shortlist_seed": random_shortlist_seed,
    }
    actual_args = {
        name: getattr(
            args,
            name,
            "llm" if name == "proposal_source" else None,
        )
        for name in expected
    }
    if actual_args != expected:
        differences = {
            name: {"expected": expected[name], "actual": actual_args[name]}
            for name in expected
            if actual_args[name] != expected[name]
        }
        raise ValueError(f"frozen run arguments differ: {differences}")
    return {
        "run_id": args.run_id,
        "study_protocol_path": study_path.as_posix(),
        "study_protocol_sha256": study_sha256,
        "task_sha256": task_sha256,
        "validated_sources": records,
        "validated_source_trees": [validated_source_tree],
        "validated_arguments": actual_args,
        "validated_derived_fields": derived,
    }


def provider_metadata(base_url: str, model: str, expected_revision: str) -> dict[str, object]:
    """Query and validate the served checkpoint before any completion request."""

    response = httpx.get(base_url.rstrip("/") + "/models", timeout=20.0)
    response.raise_for_status()
    body = response.json()
    records = body.get("data") if isinstance(body, dict) else None
    matches = [
        record
        for record in records or ()
        if isinstance(record, dict) and record.get("id") == model
    ]
    if len(matches) != 1:
        raise ValueError("served model ID is not uniquely available")
    record = cast(dict[str, object], matches[0])
    root = record.get("root")
    if not isinstance(root, str) or Path(root).name != expected_revision:
        raise ValueError("served model revision differs from the frozen checkpoint")
    return {
        "id": model,
        "root": root,
        "revision": expected_revision,
        "max_model_len": record.get("max_model_len"),
        "response_sha256": hashlib.sha256(response.content).hexdigest(),
    }


async def _call_proposal_slots(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, object],
    stage_dir: Path,
    parent: ProgramKey,
    hole: str,
    catalog: dict[str, AstNode],
) -> tuple[ProgramKey, ...]:
    """Call GPT once and totalize every outcome into four complete slots."""

    stage_dir.mkdir(parents=True)
    request = _canonical_bytes(payload)
    (stage_dir / "request.json").write_bytes(request)
    started = time.perf_counter()
    slots = _sentinel_slots(parent)
    record: dict[str, object]
    try:
        response = await client.post(
            base_url.rstrip("/") + "/chat/completions",
            content=request,
            headers={"Content-Type": "application/json"},
        )
        (stage_dir / "response.json").write_bytes(response.content)
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("provider returned the wrong number of choices")
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
            raise ValueError(f"finish_reason={choice.get('finish_reason')}")
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError("provider returned no final JSON content")
        decoded = json.loads(content)
        candidates = decoded.get("candidates") if isinstance(decoded, dict) else None
        if not isinstance(candidates, list):
            raise ValueError("final JSON has no candidates array")
        mapped: list[ProgramKey] = []
        slot_records: list[dict[str, object]] = []
        for index in range(SHORTLIST_SIZE):
            candidate = candidates[index] if index < len(candidates) else None
            try:
                dsl = decode_compact_candidate(hole, candidate)
                if dsl not in catalog:
                    raise ValueError("decoded expression is outside the catalog")
                key = (
                    ProgramKey(dsl, parent.mapper)
                    if hole == "predicate"
                    else ProgramKey(parent.predicate, dsl)
                )
                status = "valid" if key != parent else "explicit-no-op"
            except Exception as error:
                dsl = None
                key = parent
                status = f"sentinel:{type(error).__name__}"
            mapped.append(key)
            slot_records.append(
                {
                    "slot": index,
                    "raw_candidate": candidate,
                    "decoded_dsl": dsl,
                    "mapped_program": asdict(key),
                    "status": status,
                }
            )
        slots = tuple(mapped)
        record = {
            "status": "parsed",
            "finish_reason": choice.get("finish_reason"),
            "elapsed_seconds": time.perf_counter() - started,
            "usage": body.get("usage"),
            "slots": slot_records,
            "extra_candidates_ignored": max(0, len(candidates) - SHORTLIST_SIZE),
        }
    except Exception as error:
        record = {
            "status": "whole-response-sentinel",
            "error_type": type(error).__name__,
            "detail": str(error),
            "elapsed_seconds": time.perf_counter() - started,
            "slots": [
                {
                    "slot": index,
                    "mapped_program": asdict(parent),
                    "status": "sentinel",
                }
                for index in range(SHORTLIST_SIZE)
            ],
        }
    _write_json(stage_dir / "result.json", record)
    return slots


async def run(args: argparse.Namespace) -> None:
    frozen_invocation = validate_frozen_invocation(args)
    served_model = (
        None
        if args.proposal_source != "llm"
        or frozen_invocation.get("mode") == "explicit-unfrozen-developmental"
        else provider_metadata(args.base_url, args.model, MODEL_REVISION)
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    config = load_experiment_config(args.task)
    scorer = ProgramScorer(config)
    constants = tuple(config.spec.integer_constants)
    predicate_catalog = _dsl_catalog(filter_predicates(constants))
    mapper_catalog = _dsl_catalog(arithmetic_expressions("Item", constants))
    predicate_order = tuple(sorted(predicate_catalog))
    mapper_order = tuple(sorted(mapper_catalog))
    if len(predicate_order) != len(predicate_catalog) or len(mapper_order) != len(mapper_catalog):
        raise ValueError("canonical DSL catalogs contain duplicate syntax")
    predicate_probability, mapper_probability = recursive_grammar_probabilities(
        predicate_order,
        mapper_order,
    )
    examples = [example.model_dump(mode="json", by_alias=True) for example in config.spec.examples]
    observed_items = tuple(
        sorted(
            {
                item
                for example in config.spec.examples
                for item in cast(list[int], example.input_value)
            }
        )
    )
    if not observed_items:
        raise ValueError("the task has no observed scalar items")
    observed_index = {item: index for index, item in enumerate(observed_items)}
    singleton = derive_singleton_constraints(config.spec)

    predicate_signatures: dict[str, tuple[bool, ...]] = {}
    for dsl, expression in predicate_catalog.items():
        values = tuple(
            evaluate_expression(cast(Node, expression), list(observed_items), item=item)
            for item in observed_items
        )
        if any(not isinstance(value, bool) for value in values):
            raise TypeError("predicate catalog member returned a non-Boolean value")
        predicate_signatures[dsl] = cast(tuple[bool, ...], values)
    mapper_signatures: dict[str, tuple[int, ...]] = {}
    for dsl, expression in mapper_catalog.items():
        values = tuple(
            evaluate_expression(cast(Node, expression), list(observed_items), item=item)
            for item in observed_items
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise TypeError("mapper catalog member returned a non-integer value")
        mapper_signatures[dsl] = cast(tuple[int, ...], values)

    score_cache: dict[ProgramKey, ScoredProgram] = {}
    cache_hits = 0

    def score_key(key: ProgramKey) -> ScoredProgram:
        nonlocal cache_hits
        cached = score_cache.get(key)
        if cached is not None:
            cache_hits += 1
            return cached
        score = scorer.score(
            _assemble_program(predicate_catalog[key.predicate], mapper_catalog[key.mapper])
        )
        if not isinstance(score, ScoredProgram):
            raise ValueError(f"grammar-valid complete program was rejected: {score.reason}")
        score_cache[key] = score
        return score

    def violation_counts(key: ProgramKey) -> tuple[int, int]:
        predicate_signature = predicate_signatures[key.predicate]
        mapper_signature = mapper_signatures[key.mapper]
        predicate_count = sum(
            predicate_signature[observed_index[cast(int, fact["item"])]]
            != cast(bool, fact["keep"])
            for fact in singleton["predicate"]
        )
        mapper_count = sum(
            mapper_signature[observed_index[cast(int, fact["item"])]]
            != cast(int, fact["expected_value"])
            for fact in singleton["mapper"]
        )
        return predicate_count, mapper_count

    def make_particle(key: ProgramKey, lineage: str) -> Particle:
        predicate_count, mapper_count = violation_counts(key)
        return Particle(
            key=key,
            score=score_key(key),
            predicate_violations=predicate_count,
            mapper_violations=mapper_count,
            lineage=lineage,
        )

    def feedback_for(particle: Particle) -> dict[str, object]:
        feedback = derive_automatic_feedback(
            config.spec,
            cast(Node, predicate_catalog[particle.key.predicate]),
            cast(Node, mapper_catalog[particle.key.mapper]),
        ).to_dict()
        predicate_violations = tuple(
            fact
            for fact in singleton["predicate"]
            if predicate_signatures[particle.key.predicate][
                observed_index[cast(int, fact["item"])]
            ]
            != cast(bool, fact["keep"])
        )
        mapper_violations = tuple(
            fact
            for fact in singleton["mapper"]
            if mapper_signatures[particle.key.mapper][
                observed_index[cast(int, fact["item"])]
            ]
            != cast(int, fact["expected_value"])
        )
        return feedback | {
            "singleton_predicate_constraints": singleton["predicate"],
            "singleton_mapper_constraints": singleton["mapper"],
            "singleton_predicate_violations": predicate_violations,
            "singleton_mapper_violations": mapper_violations,
        }

    start_rng = random.Random(args.start_seed)
    initial_mapper = mapper_order[start_rng.randrange(len(mapper_order))]
    initial_predicate = predicate_order[start_rng.randrange(len(predicate_order))]
    initial_key = ProgramKey(initial_predicate, initial_mapper)
    initial = make_particle(initial_key, "root")
    task_sha256 = hashlib.sha256(args.task.resolve().read_bytes()).hexdigest()
    checkpoints, llm_provider_call_cap = population_schedule(
        rounds=ROUNDS,
        parent_count=args.parent_count,
        offspring_per_parent=args.offspring_per_parent,
        first_round_offspring=args.first_round_offspring,
    )
    maximum_provider_calls = (
        llm_provider_call_cap if args.proposal_source == "llm" else 0
    )
    protocol = {
        "schema": "evidence-shortlist-product-path-smc-v1",
        "task_sha256": task_sha256,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": args.model,
        "model_revision": MODEL_REVISION,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout_seconds,
        "max_concurrency": args.max_concurrency,
        "exact_reference_limit": args.exact_reference_limit,
        "epsilon": args.epsilon,
        "evidence_scale": args.evidence_scale,
        "rounds": ROUNDS,
        "llm_shortlist_size": SHORTLIST_SIZE,
        "offspring_per_parent": args.offspring_per_parent,
        "first_round_offspring": args.first_round_offspring,
        "resampled_parent_count": args.parent_count,
        "logical_complete_program_execution_cap": checkpoints[-1],
        "proposal_slot_checkpoints": checkpoints,
        "maximum_provider_calls": maximum_provider_calls,
        "proposal_source": args.proposal_source,
        "random_shortlist_seed": args.random_shortlist_seed,
        "exact_parent_policy": (
            "sticky with full-support restart: skip provider, use four current-program local "
            "slots, then independently sample every child from (1-epsilon) delta_current "
            "+ epsilon recursive grammar"
        ),
        "provider_seed": args.provider_seed,
        "sample_seed": args.sample_seed,
        "resample_seed": args.resample_seed,
        "start_seed": args.start_seed,
        "initial_program": asdict(initial_key),
        "catalog": {
            "predicate_syntaxes": len(predicate_order),
            "mapper_syntaxes": len(mapper_order),
            "provider_visible_cardinalities": False,
            "provider_visible_complete_catalog": False,
        },
        "proposal": (
            "q(y|S,x,h)=(1-epsilon) empirical_uniform(four totalized shortlist slots) "
            "+ epsilon g(y), where g is a normalized recursive grammar sampler whose "
            "probability is the product of recorded local rule probabilities"
        ),
        "search_requires_complete_program_count": False,
        "recursive_grammar_prior": {
            "predicate": (
                "choose atom versus ordered conjunction uniformly, then choose uniformly "
                "within the selected public typed form"
            ),
            "mapper": (
                "choose one of five public syntactic forms uniformly, then choose uniformly "
                "within that form"
            ),
        },
        "slot_totalization": (
            "malformed, missing, out-of-catalog, explicit no-op, and whole-response failures "
            "map to the current program; duplicates retain multiplicity; no retry/backfill"
        ),
        "stage_targets": {
            "1": "RecursiveGrammarPrior * exp(-lambda * predicate violations)",
            "2": "RecursiveGrammarPrior * exp(-lambda * (predicate+mapper violations))",
            "3": (
                "RecursiveGrammarPrior * exp(-lambda/2 * (predicate+mapper violations) "
                "+ scorer.log_target/2)"
            ),
            "4": "RecursiveGrammarPrior * exp(scorer.log_target)",
        },
        "inference_semantics": (
            "product-path Feynman-Kac target; raw provider response and deterministic hole "
            "selection are auxiliary kernels and cancel; final marginal is stage-4 target"
        ),
        "early_stop": False,
        "claim_scope": (
            "importance-accounting plumbing and bounded search; exact discovery remains a "
            "separate algorithmic metric"
        ),
        "frozen_invocation": frozen_invocation,
        "served_model": served_model,
    }
    _write_json(output / "protocol.json", protocol)

    proposal_rng = random.Random(args.sample_seed)
    parent_particles = [initial]
    parent_weights = torch.tensor([1.0], dtype=torch.float64)
    logical_executions = 1
    provider_calls = 0
    exact_parent_expansions_skipped = 0
    first_exact_slot = 1 if initial.score.exact_program else None
    best = initial
    execution_records: list[dict[str, object]] = [
        {
            "slot": 1,
            "round": 0,
            "lineage": initial.lineage,
            "program": asdict(initial.key),
            "score": asdict(initial.score),
            "proposal": "fixed-shared-initial-state",
        }
    ]
    round_records: list[dict[str, object]] = []
    log_product_normalizer_estimate = 0.0
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async with httpx.AsyncClient(timeout=args.timeout_seconds) as client:
        for round_number in range(1, ROUNDS + 1):
            async def prepare_parent(
                parent_index: int,
                particle: Particle,
                active_round: int,
            ) -> tuple[
                int,
                Particle,
                str,
                dict[str, object],
                tuple[ProgramKey, ...],
                bool,
            ]:
                if particle.score.exact_program:
                    return (
                        parent_index,
                        particle,
                        "exact-absorbing",
                        {
                            "policy": "skip provider for an exact parent",
                            "selected_first_hole": None,
                            "reason": "the current complete program has zero training loss",
                            "evidence_backed": True,
                            "fallback_used": False,
                        },
                        _sentinel_slots(particle.key),
                        False,
                    )
                feedback = feedback_for(particle)
                state = BeamState(
                    predicate=particle.key.predicate,
                    mapper=particle.key.mapper,
                    score=particle.score,
                    predicate_signature=predicate_signatures[particle.key.predicate],
                    mapper_signature=mapper_signatures[particle.key.mapper],
                    singleton_predicate_violations=particle.predicate_violations,
                    singleton_mapper_violations=particle.mapper_violations,
                )
                hole, decision = select_repair_hole(
                    feedback,
                    state,
                    round_number=active_round,
                    stall_policy="alternate-hole",
                )
                if args.proposal_source == "grammar-random":
                    slots = random_local_slots(
                        parent=particle.key,
                        hole=hole,
                        predicate_order=predicate_order,
                        mapper_order=mapper_order,
                        predicate_probability=predicate_probability,
                        mapper_probability=mapper_probability,
                        seed=args.random_shortlist_seed
                        + 100 * active_round
                        + parent_index,
                    )
                    return parent_index, particle, hole, decision, slots, False
                payload = _compact_structured_payload(
                    _shortlist_payload(
                        model=args.model,
                        reasoning_effort=args.reasoning_effort,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        seed=args.provider_seed + 100 * active_round + parent_index,
                        shortlist_size=SHORTLIST_SIZE,
                        examples=examples,
                        predicate=predicate_catalog[particle.key.predicate],
                        mapper=mapper_catalog[particle.key.mapper],
                        score=particle.score,
                        feedback=feedback,
                        hole=hole,
                        constants=constants,
                    ),
                    hole=hole,
                    constants=constants,
                )
                payload = _augment_smc_prompt(
                    payload,
                    round_number=active_round,
                    state=state,
                    feedback=feedback,
                    hole=hole,
                    decision=decision,
                )
                async with semaphore:
                    slots = await _call_proposal_slots(
                        client=client,
                        base_url=args.base_url,
                        payload=payload,
                        stage_dir=(
                            output
                            / "provider"
                            / f"round-{active_round:02d}"
                            / f"parent-{parent_index:03d}-{hole}"
                        ),
                        parent=particle.key,
                        hole=hole,
                        catalog=(predicate_catalog if hole == "predicate" else mapper_catalog),
                    )
                return parent_index, particle, hole, decision, slots, True

            prepared = await asyncio.gather(
                *(
                    prepare_parent(parent_index, particle, round_number)
                    for parent_index, particle in enumerate(parent_particles)
                )
            )
            provider_calls += sum(provider_called for *_, provider_called in prepared)
            exact_parent_expansions_skipped += sum(
                hole == "exact-absorbing" for _, _, hole, _, _, _ in prepared
            )
            children: list[Particle] = []
            child_log_weights: list[float] = []
            proposals: list[dict[str, object]] = []
            for parent_index, parent, hole, decision, slots, provider_called in prepared:
                offspring_count = (
                    args.first_round_offspring
                    if round_number == 1
                    else args.offspring_per_parent
                )
                for branch_index in range(offspring_count):
                    draw = sample_proposal(
                        slots=slots,
                        predicate_order=predicate_order,
                        mapper_order=mapper_order,
                        predicate_probability=predicate_probability,
                        mapper_probability=mapper_probability,
                        epsilon=args.epsilon,
                        rng=proposal_rng,
                    )
                    logical_executions += 1
                    lineage = f"{parent.lineage}.{round_number}:{parent_index}:{branch_index}"
                    child = make_particle(draw.key, lineage)
                    children.append(child)
                    log_gamma = stage_log_gamma(
                        round_number=round_number,
                        score=child.score,
                        predicate_violations=child.predicate_violations,
                        mapper_violations=child.mapper_violations,
                        evidence_scale=args.evidence_scale,
                        log_prior=math.log(
                            program_grammar_probability(
                                child.key,
                                predicate_probability,
                                mapper_probability,
                            )
                        ),
                    )
                    log_weight = (
                        math.log(float(parent_weights[parent_index].item()))
                        - math.log(offspring_count)
                        + log_gamma
                        - math.log(draw.probability)
                    )
                    child_log_weights.append(log_weight)
                    slot = logical_executions
                    if child.score.exact_program and first_exact_slot is None:
                        first_exact_slot = slot
                    if (child.score.total_loss, child.score.cost, child.lineage) < (
                        best.score.total_loss,
                        best.score.cost,
                        best.lineage,
                    ):
                        best = child
                    proposal_record = {
                        "slot": slot,
                        "round": round_number,
                        "parent_index": parent_index,
                        "parent_lineage": parent.lineage,
                        "child_lineage": child.lineage,
                        "hole": hole,
                        "hole_decision": decision,
                        "provider_called": provider_called,
                        "proposal_source": args.proposal_source,
                        "shortlist_slots": [asdict(key) for key in slots],
                        "sampled_branch": draw.branch,
                        "sampled_program": asdict(draw.key),
                        "q": draw.probability,
                        "log_gamma": log_gamma,
                        "log_weight": log_weight,
                        "predicate_violations": child.predicate_violations,
                        "mapper_violations": child.mapper_violations,
                        "score": asdict(child.score),
                    }
                    proposals.append(proposal_record)
                    execution_records.append(proposal_record)

            weights, ess = normalized_child_weights(child_log_weights)
            log_stage_normalizer = normalize_log_weights(
                torch.tensor(child_log_weights, dtype=torch.float64)
            ).log_normalizer
            log_product_normalizer_estimate += log_stage_normalizer
            round_record: dict[str, object] = {
                "round": round_number,
                "parent_count": len(parent_particles),
                "provider_calls": sum(provider_called for *_, provider_called in prepared),
                "proposal_count": len(children),
                "proposals": proposals,
                "normalized_weights": weights.tolist(),
                "ess": ess,
                "log_stage_normalizer_estimate": log_stage_normalizer,
            }
            if round_number < ROUNDS:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(args.resample_seed + round_number)
                ancestors = systematic_resample_count(
                    weights,
                    count=args.parent_count,
                    generator=generator,
                ).tolist()
                parent_particles = [children[index] for index in ancestors]
                parent_weights = torch.full(
                    (args.parent_count,), 1.0 / args.parent_count, dtype=torch.float64
                )
                round_record["systematic_resample_ancestors"] = ancestors
                round_record["resampled_lineages"] = [
                    particle.lineage for particle in parent_particles
                ]
            else:
                final_particles = children
                final_weights = weights
            round_records.append(round_record)
            _write_json(output / f"round-{round_number:02d}.json", round_record)

    expected_executions = cast(
        int,
        protocol["logical_complete_program_execution_cap"],
    )
    maximum_provider_calls = cast(int, protocol["maximum_provider_calls"])
    if (
        logical_executions != expected_executions
        or not 0 <= provider_calls <= maximum_provider_calls
    ):
        raise RuntimeError(
            f"budget invariant failed: executions={logical_executions}, calls={provider_calls}"
        )
    provider_completed = time.perf_counter()
    search_cache_hits = cache_hits
    search_physical_scorer_calls = len(score_cache)
    inventory = _provider_inventory(output)
    _write_json(output / "provider-seal.json", inventory)
    provider_statuses: Counter[str] = Counter()
    provider_slot_statuses: Counter[str] = Counter()
    provider_usage_totals: Counter[str] = Counter()
    for provider_result in sorted((output / "provider").glob("**/result.json")):
        provider_record = json.loads(provider_result.read_text(encoding="utf-8"))
        provider_statuses[str(provider_record.get("status", "missing-status"))] += 1
        for slot in provider_record.get("slots", ()):
            if isinstance(slot, dict):
                provider_slot_statuses[str(slot.get("status", "missing-status"))] += 1
        usage = provider_record.get("usage")
        if isinstance(usage, dict):
            for name, value in usage.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    provider_usage_totals[name] += value

    reference: dict[str, object] | None = None
    if args.exact_reference_limit > 0:
        reference_program_count = len(predicate_order) * len(mapper_order)
    else:
        reference_program_count = None
    if (
        reference_program_count is not None
        and reference_program_count <= args.exact_reference_limit
    ):
        reference_log_gammas: list[float] = []
        reference_losses: list[float] = []
        reference_exact: list[bool] = []
        for predicate in predicate_order:
            for mapper in mapper_order:
                particle = make_particle(ProgramKey(predicate, mapper), "posthoc-reference")
                reference_log_gammas.append(
                    stage_log_gamma(
                        round_number=4,
                        score=particle.score,
                        predicate_violations=particle.predicate_violations,
                        mapper_violations=particle.mapper_violations,
                        evidence_scale=args.evidence_scale,
                        log_prior=math.log(
                            program_grammar_probability(
                                particle.key,
                                predicate_probability,
                                mapper_probability,
                            )
                        ),
                    )
                )
                reference_losses.append(particle.score.total_loss)
                reference_exact.append(particle.score.exact_program)
        reference_weights = normalize_log_weights(
            torch.tensor(reference_log_gammas, dtype=torch.float64)
        ).weights
        reference = {
            "post_provider_exhaustive_programs": reference_program_count,
            "log_normalizer": normalize_log_weights(
                torch.tensor(reference_log_gammas, dtype=torch.float64)
            ).log_normalizer,
            "exact_target_mass": sum(
                weight
                for weight, exact in zip(reference_weights.tolist(), reference_exact, strict=True)
                if exact
            ),
            "target_mean_loss": sum(
                weight * loss
                for weight, loss in zip(reference_weights.tolist(), reference_losses, strict=True)
            ),
        }

    final_exact_weight = sum(
        float(weight)
        for weight, particle in zip(final_weights.tolist(), final_particles, strict=True)
        if particle.score.exact_program
    )
    final_mean_loss = sum(
        float(weight) * particle.score.total_loss
        for weight, particle in zip(final_weights.tolist(), final_particles, strict=True)
    )
    reference_comparison = (
        None
        if reference is None
        else {
            "absolute_exact_mass_error": abs(
                final_exact_weight - cast(float, reference["exact_target_mass"])
            ),
            "absolute_mean_loss_error": abs(
                final_mean_loss - cast(float, reference["target_mean_loss"])
            ),
        }
    )
    result = {
        "schema": "evidence-shortlist-product-path-smc-result-v1",
        "protocol": protocol,
        "provider_inventory_sha256": inventory["inventory_sha256"],
        "search": {
            "logical_complete_program_executions": logical_executions,
            "physical_scorer_calls_before_reference": search_physical_scorer_calls,
            "score_cache_hits_before_reference": search_cache_hits,
            "provider_calls": provider_calls,
            "provider_status_counts": dict(sorted(provider_statuses.items())),
            "provider_slot_status_counts": dict(sorted(provider_slot_statuses.items())),
            "provider_usage_totals": dict(sorted(provider_usage_totals.items())),
            "grammar_restart_draws": sum(
                record.get("sampled_branch") == "full-grammar-restart"
                for record in execution_records
            ),
            "shortlist_draws": sum(
                record.get("sampled_branch") == "shortlist-slot"
                for record in execution_records
            ),
            "exact_parent_expansions_skipped": exact_parent_expansions_skipped,
            "first_exact_slot": first_exact_slot,
            "found_exact": first_exact_slot is not None,
            "best_program": asdict(best.key),
            "best_score": asdict(best.score),
        },
        "inference": {
            "final_particles": [
                {
                    "program": asdict(particle.key),
                    "lineage": particle.lineage,
                    "score": asdict(particle.score),
                    "normalized_weight": float(weight),
                }
                for particle, weight in zip(final_particles, final_weights.tolist(), strict=True)
            ],
            "final_ess": effective_sample_size(final_weights),
            "final_relative_ess": effective_sample_size(final_weights) / len(final_particles),
            "maximum_normalized_weight": max(final_weights.tolist()),
            "unique_terminal_programs": len(
                {(particle.key.predicate, particle.key.mapper) for particle in final_particles}
            ),
            "self_normalized_exact_mass": final_exact_weight,
            "self_normalized_target_mean_loss": final_mean_loss,
            "log_product_stage_normalizer_estimate": log_product_normalizer_estimate,
            "normalizer_note": (
                "estimates the product of four stage normalizers under the declared "
                "product-path target, not the terminal normalizer alone"
            ),
        },
        "exact_reference": reference,
        "reference_comparison": reference_comparison,
        "rounds": round_records,
        "timing": {
            "provider_and_search_seconds": provider_completed - started,
            "posthoc_reference_seconds": time.perf_counter() - provider_completed,
            "total_seconds": time.perf_counter() - started,
        },
    }
    _write_json(output / "executions.json", execution_records)
    _write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "found_exact": first_exact_slot is not None,
                "first_exact_slot": first_exact_slot,
                "best_program": asdict(best.key),
                "best_loss": best.score.total_loss,
                "final_ess": result["inference"]["final_ess"],
                "self_normalized_exact_mass": final_exact_weight,
                "exact_reference": reference,
                "logical_executions": logical_executions,
                "provider_calls": provider_calls,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout-seconds", type=float, default=420.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--evidence-scale", type=float, default=2.0)
    parser.add_argument("--start-seed", type=int, default=17)
    parser.add_argument("--provider-seed", type=int, default=140000)
    parser.add_argument("--sample-seed", type=int, default=140100)
    parser.add_argument("--resample-seed", type=int, default=140200)
    parser.add_argument("--proposal-source", choices=PROPOSAL_SOURCES, default="llm")
    parser.add_argument("--random-shortlist-seed", type=int)
    parser.add_argument("--parent-count", type=int, default=2)
    parser.add_argument("--offspring-per-parent", type=int, default=4)
    parser.add_argument("--first-round-offspring", type=int)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--exact-reference-limit", type=int, default=100000)
    parser.add_argument("--study-protocol", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--expected-study-protocol-sha256")
    parser.add_argument("--allow-unfrozen-developmental", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.epsilon <= 1.0:
        parser.error("--epsilon must be in (0, 1]")
    if args.evidence_scale < 0.0:
        parser.error("--evidence-scale must be nonnegative")
    if args.exact_reference_limit < 0:
        parser.error("--exact-reference-limit must be nonnegative")
    if args.parent_count < 1:
        parser.error("--parent-count must be positive")
    if args.offspring_per_parent < 1:
        parser.error("--offspring-per-parent must be positive")
    if args.first_round_offspring is None:
        args.first_round_offspring = args.offspring_per_parent
    if args.first_round_offspring < 1:
        parser.error("--first-round-offspring must be positive")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    if (args.proposal_source == "grammar-random") != (
        args.random_shortlist_seed is not None
    ):
        parser.error(
            "--random-shortlist-seed is required iff --proposal-source=grammar-random"
        )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

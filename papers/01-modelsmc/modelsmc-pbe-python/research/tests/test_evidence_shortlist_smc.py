from __future__ import annotations

import json
import math
import random
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from modelsmc_pbe.core.results import ScoredProgram
from research.evidence_shortlist_smc import (
    ProgramKey,
    _constant_schema,
    decode_compact_candidate,
    normalized_child_weights,
    population_schedule,
    program_grammar_probability,
    proposal_probability,
    python_tree_binding,
    random_local_slots,
    recursive_grammar_probabilities,
    sample_proposal,
    stage_log_gamma,
    systematic_resample_count,
    validate_frozen_invocation,
)


def _score(log_target: float) -> ScoredProgram:
    return ScoredProgram(
        kind="Scored",
        inferred_type="List<Int>",
        total_loss=3.0,
        exact_matches=0,
        cost=7,
        log_target=log_target,
        exact_program=False,
        evaluations=(),
    )


def test_duplicate_shortlist_mixture_normalizes_over_complete_grammar() -> None:
    programs = tuple(
        ProgramKey(predicate, mapper)
        for predicate in ("p0", "p1")
        for mapper in ("m0", "m1", "m2")
    )
    slots = (programs[0], programs[0], programs[1], programs[2])
    probabilities = [
        proposal_probability(
            program,
            slots,
            grammar_probability=Fraction(1, len(programs)),
            epsilon=0.05,
        )
        for program in programs
    ]
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-15)
    assert probabilities[0] == pytest.approx(0.95 * 0.5 + 0.05 / 6)
    assert min(probabilities) == pytest.approx(0.05 / 6)


def test_all_sentinel_slots_still_have_full_support() -> None:
    current = ProgramKey("p0", "m0")
    other = ProgramKey("p1", "m1")
    slots = (current,) * 4
    assert proposal_probability(
        current,
        slots,
        grammar_probability=Fraction(1, 10),
        epsilon=0.1,
    ) == pytest.approx(0.91)
    assert proposal_probability(
        other,
        slots,
        grammar_probability=Fraction(1, 10),
        epsilon=0.1,
    ) == pytest.approx(0.01)


def test_sampling_branch_matches_evaluated_mixture() -> None:
    predicate_order = (
        "eq(item,0)",
        "and(eq(item,0),eq(item,0))",
    )
    mapper_order = (
        "item",
        "0",
        "add(item,item)",
        "add(item,0)",
        "add(0,item)",
    )
    predicate_probability, mapper_probability = recursive_grammar_probabilities(
        predicate_order,
        mapper_order,
    )
    local = ProgramKey("eq(item,0)", "item")
    slots = (local,) * 4
    rng = random.Random(7321)
    draws = [
        sample_proposal(
            slots=slots,
            predicate_order=predicate_order,
            mapper_order=mapper_order,
            predicate_probability=predicate_probability,
            mapper_probability=mapper_probability,
            epsilon=0.2,
            rng=rng,
        )
        for _ in range(50_000)
    ]
    observed = sum(draw.key == local for draw in draws) / len(draws)
    expected = proposal_probability(
        local,
        slots,
        grammar_probability=program_grammar_probability(
            local,
            predicate_probability,
            mapper_probability,
        ),
        epsilon=0.2,
    )
    assert observed == pytest.approx(expected, abs=0.006)
    assert all(draw.probability > 0.0 for draw in draws)


def test_four_stage_target_ends_at_execution_gibbs_target() -> None:
    score = _score(-7.5)
    values = [
        stage_log_gamma(
            round_number=round_number,
            score=score,
            predicate_violations=2,
            mapper_violations=3,
            evidence_scale=2.0,
            log_prior=-10.0,
        )
        for round_number in range(1, 5)
    ]
    assert values == pytest.approx([-14.0, -20.0, -18.75, -17.5])


def test_recursive_grammar_normalizes_without_complete_program_count() -> None:
    predicates = (
        "eq(item,0)",
        "lt(item,0)",
        "and(eq(item,0),eq(item,0))",
        "and(eq(item,0),lt(item,0))",
        "and(lt(item,0),eq(item,0))",
        "and(lt(item,0),lt(item,0))",
    )
    mappers = (
        "item",
        "0",
        "add(item,item)",
        "add(item,0)",
        "add(0,item)",
    )
    predicate_probability, mapper_probability = recursive_grammar_probabilities(
        predicates,
        mappers,
    )
    assert sum(predicate_probability.values()) == pytest.approx(1.0)
    assert sum(mapper_probability.values()) == pytest.approx(1.0)
    joint_sum = sum(
        program_grammar_probability(
            ProgramKey(predicate, mapper),
            predicate_probability,
            mapper_probability,
        )
        for predicate in predicates
        for mapper in mappers
    )
    assert joint_sum == pytest.approx(1.0)


def test_random_local_slots_are_unique_non_noop_local_and_deterministic() -> None:
    predicates = (
        "eq(item,0)",
        "lt(item,0)",
        "lt(0,item)",
        "and(eq(item,0),eq(item,0))",
        "and(eq(item,0),lt(item,0))",
        "and(lt(item,0),eq(item,0))",
    )
    mappers = (
        "item",
        "0",
        "1",
        "add(item,item)",
        "add(item,0)",
        "add(item,1)",
        "add(0,item)",
        "add(1,item)",
    )
    predicate_probability, mapper_probability = recursive_grammar_probabilities(
        predicates,
        mappers,
    )
    parent = ProgramKey("eq(item,0)", "item")
    keyword = {
        "parent": parent,
        "predicate_order": predicates,
        "mapper_order": mappers,
        "predicate_probability": predicate_probability,
        "mapper_probability": mapper_probability,
        "seed": 991,
    }
    predicate_slots = random_local_slots(hole="predicate", **keyword)
    assert predicate_slots == random_local_slots(hole="predicate", **keyword)
    assert len(set(predicate_slots)) == 4
    assert all(slot != parent and slot.mapper == parent.mapper for slot in predicate_slots)
    mapper_slots = random_local_slots(hole="mapper", **keyword)
    assert len(set(mapper_slots)) == 4
    assert all(slot != parent and slot.predicate == parent.predicate for slot in mapper_slots)


def test_random_local_first_draw_tracks_recursive_grammar_mass() -> None:
    predicates = (
        "eq(item,0)",
        "lt(item,0)",
        "lt(0,item)",
        "and(eq(item,0),eq(item,0))",
        "and(eq(item,0),lt(item,0))",
        "and(lt(item,0),eq(item,0))",
    )
    mappers = (
        "item",
        "0",
        "1",
        "add(item,item)",
        "add(item,0)",
        "add(item,1)",
        "add(0,item)",
        "add(1,item)",
    )
    predicate_probability, mapper_probability = recursive_grammar_probabilities(
        predicates,
        mappers,
    )
    parent = ProgramKey("eq(item,0)", "item")
    first_draws = [
        random_local_slots(
            parent=parent,
            hole="predicate",
            predicate_order=predicates,
            mapper_order=mappers,
            predicate_probability=predicate_probability,
            mapper_probability=mapper_probability,
            seed=seed,
        )[0].predicate
        for seed in range(4_000)
    ]
    observed_conjunction = sum(value.startswith("and(") for value in first_draws) / len(
        first_draws
    )
    remaining_mass = 1 - predicate_probability[parent.predicate]
    expected_conjunction = sum(
        probability
        for value, probability in predicate_probability.items()
        if value.startswith("and(")
    ) / remaining_mass
    assert observed_conjunction == pytest.approx(float(expected_conjunction), abs=0.025)


def test_compact_typed_candidates_use_integer_constants() -> None:
    mapper = decode_compact_candidate(
        "mapper",
        {"template": "mul(item,c)", "constant": -64},
    )
    predicate = decode_compact_candidate(
        "predicate",
        {
            "combine": "and",
            "left": {"template": "lt(c,item)", "constant": -5},
            "right": {"template": "lt(item,c)", "constant": 7},
        },
    )
    assert mapper == "mul(item,-64)"
    assert predicate == "and(lt(-5,item),lt(item,7))"


def test_large_consecutive_constant_catalog_uses_interval_schema() -> None:
    schema = _constant_schema(tuple(range(-64, 65)))
    assert schema == {"type": "integer", "minimum": -64, "maximum": 64}
    assert "enum" not in schema


def test_large_population_schedule_is_exact() -> None:
    checkpoints, provider_calls = population_schedule(
        rounds=4,
        parent_count=32,
        offspring_per_parent=8,
        first_round_offspring=256,
    )
    assert checkpoints == (1, 257, 513, 769, 1025)
    assert provider_calls == 97


def test_python_tree_binding_rejects_any_tree_drift(tmp_path: Path) -> None:
    source = tmp_path / "src/modelsmc_pbe"
    source.mkdir(parents=True)
    first = source / "a.py"
    first.write_text("VALUE = 1\n", encoding="utf-8")
    entries = [
        {
            "path": "a.py",
            "bytes": first.stat().st_size,
            "sha256": sha256(first.read_bytes()).hexdigest(),
        }
    ]
    manifest = {
        "schema": "sha256-python-tree-v1",
        "root": "src/modelsmc_pbe",
        "include": "**/*.py",
        "entries": entries,
    }
    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    binding = {
        "schema": "sha256-python-tree-v1",
        "root": "src/modelsmc_pbe",
        "include": "**/*.py",
        "file_count": 1,
        "manifest_sha256": sha256(encoded).hexdigest(),
    }
    assert python_tree_binding(tmp_path, binding) == binding
    first.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="binding differs"):
        python_tree_binding(tmp_path, binding)


def test_hierarchical_restart_probability_matches_large_nominal_grammar() -> None:
    constants = tuple(range(-64, 65))
    atoms = tuple(
        template.replace("c", str(constant))
        for constant in constants
        for template in ("lt(item,c)", "lt(c,item)", "eq(item,c)")
    )
    predicates = (*atoms, *(f"and({left},{right})" for left in atoms for right in atoms))
    mapper_templates = (
        "add",
        "sub",
        "mul",
    )
    mappers = (
        "item",
        *(str(value) for value in constants),
        *(f"{op}(item,item)" for op in mapper_templates),
        *(f"{op}(item,{value})" for op in mapper_templates for value in constants),
        *(f"{op}({value},item)" for op in mapper_templates for value in constants),
    )
    predicate_probability, mapper_probability = recursive_grammar_probabilities(
        predicates,
        mappers,
    )
    sentinel = ProgramKey(predicates[0], mappers[0])
    rng = random.Random(99)
    for _ in range(100):
        draw = sample_proposal(
            slots=(sentinel,) * 4,
            predicate_order=predicates,
            mapper_order=mappers,
            predicate_probability=predicate_probability,
            mapper_probability=mapper_probability,
            epsilon=1.0,
            rng=rng,
        )
        assert draw.branch == "full-grammar-restart"
        assert draw.probability == float(
            program_grammar_probability(
                draw.key,
                predicate_probability,
                mapper_probability,
            )
        )


def test_historical_frozen_large_invocation_fails_closed_on_source_drift() -> None:
    project = Path(__file__).resolve().parents[2]
    study = project / "research/protocol-developmental-cardinality-free-smc-v1.json"
    args = SimpleNamespace(
        study_protocol=study,
        run_id="large-136m-discovery-p2-b4-v1",
        expected_study_protocol_sha256=sha256(study.read_bytes()).hexdigest(),
        task=project / "examples/foldr-large-opaque-v1.json",
        base_url="http://127.0.0.1:18000/v1",
        model="gpt-oss-120b",
        reasoning_effort="low",
        temperature=0.0,
        max_tokens=1200,
        timeout_seconds=420.0,
        epsilon=0.05,
        evidence_scale=2.0,
        start_seed=17,
        parent_count=2,
        offspring_per_parent=4,
        first_round_offspring=4,
        provider_seed=152000,
        sample_seed=152100,
        resample_seed=152200,
        max_concurrency=2,
        exact_reference_limit=0,
        allow_unfrozen_developmental=False,
    )
    with pytest.raises(ValueError, match="frozen source hash mismatch"):
        validate_frozen_invocation(args)


def test_provider_run_requires_explicit_frozen_or_unfrozen_mode() -> None:
    args = SimpleNamespace(
        study_protocol=None,
        run_id=None,
        allow_unfrozen_developmental=False,
    )
    with pytest.raises(ValueError, match="require a frozen protocol"):
        validate_frozen_invocation(args)
    args.allow_unfrozen_developmental = True
    assert validate_frozen_invocation(args) == {"mode": "explicit-unfrozen-developmental"}


def test_arm_specific_frozen_epsilon_validates_without_global_fallback() -> None:
    project = Path(__file__).resolve().parents[2]
    study = project / "research/protocol-developmental-smc-benchmark-v1.json"
    relative_suite = Path(
        "artifacts/execution-guided-repair/"
        "blind-evidence-frontier-v2/suite/public"
    )
    candidates = tuple(root / relative_suite for root in (project, *project.parents))
    suite = next(candidate for candidate in candidates if candidate.is_dir())
    task = suite / "blind-v2-01.json"
    args = SimpleNamespace(
        study_protocol=study,
        run_id="blind-v2-01--grammar-only",
        expected_study_protocol_sha256=sha256(study.read_bytes()).hexdigest(),
        task=task,
        base_url="http://127.0.0.1:18000/v1",
        model="gpt-oss-120b",
        reasoning_effort="low",
        temperature=0.0,
        max_tokens=1200,
        timeout_seconds=420.0,
        epsilon=1.0,
        evidence_scale=2.0,
        start_seed=17,
        parent_count=2,
        offspring_per_parent=4,
        first_round_offspring=4,
        provider_seed=401000,
        sample_seed=401100,
        resample_seed=401200,
        max_concurrency=2,
        exact_reference_limit=0,
        proposal_source="grammar-random",
        random_shortlist_seed=401400,
        allow_unfrozen_developmental=False,
    )
    validated = validate_frozen_invocation(args)
    assert validated is not None
    assert validated["run_id"] == "blind-v2-01--grammar-only"


def test_product_path_terminal_marginal_is_final_stage_target() -> None:
    states = (ProgramKey("p0", "m0"), ProgramKey("p1", "m0"))
    gamma1 = {states[0]: 2.0, states[1]: 5.0}
    gamma2 = {states[0]: 11.0, states[1]: 3.0}
    q1 = {states[0]: 0.9, states[1]: 0.1}
    q2 = {
        states[0]: {states[0]: 0.2, states[1]: 0.8},
        states[1]: {states[0]: 0.7, states[1]: 0.3},
    }
    terminal_mass = {state: 0.0 for state in states}
    for first in states:
        for second in states:
            forward = q1[first] * q2[first][second]
            path_weight = gamma1[first] * gamma2[second] / forward
            terminal_mass[second] += forward * path_weight
    total = sum(terminal_mass.values())
    normalized = {state: mass / total for state, mass in terminal_mass.items()}
    expected_total = sum(gamma2.values())
    assert normalized[states[0]] == pytest.approx(gamma2[states[0]] / expected_total)
    assert normalized[states[1]] == pytest.approx(gamma2[states[1]] / expected_total)


def test_explicit_count_systematic_resampling() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(11)
    ancestors = systematic_resample_count(
        torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64),
        count=2,
        generator=generator,
    )
    assert ancestors.shape == (2,)
    assert ancestors.tolist() == sorted(ancestors.tolist())
    assert all(0 <= index < 4 for index in ancestors.tolist())


def test_weight_normalization_is_stable() -> None:
    weights, ess = normalized_child_weights([-1000.0, -1001.0, -1002.0])
    assert math.isclose(float(weights.sum().item()), 1.0, abs_tol=1e-15)
    assert 1.0 <= ess <= 3.0

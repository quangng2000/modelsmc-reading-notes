from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.domain import canonical_key
from modelsmc_pbe.grammar import bounded_square_target
from modelsmc_pbe.induction import assemble_program
from modelsmc_pbe.search.importance.factorized import (
    family_guide_probabilities,
    trace_fillings,
    trace_log_prior,
)
from modelsmc_pbe.search.importance.lazy_prompts import lazy_family_prompt_prefix
from modelsmc_pbe.search.importance.lazy_records import (
    ConstructionTrace,
    LazyImportanceState,
)
from modelsmc_pbe.search.importance.prompts import (
    family_candidate,
    family_prompt_prefix,
    hole_prompt_prefix,
)
from research.audit_math import (
    IDENTIFIED,
    NONIDENTIFYING_SPLICE_PROXY,
    NOT_IDENTIFIED,
    REFUSED_NOT_IDENTIFIED,
    _audit_catalog_and_deduction_guides,
    _audit_final_particle_math,
    _audit_finite_reference,
    _audit_ledger_particle_paths,
    _audit_protocol_configuration,
    _audit_sampled_best,
    _audit_score_prompt_provenance,
    _materialize_support_math,
    _options,
    _particle_path_signature,
    _result_search_summary,
    _runtime_support,
    audit_matrix,
    audit_qd_target_ledgers,
    draws_for_probability,
    identified_probability,
    probability_at_least_one,
    reconcile_telemetry,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
MATRIX_DIR = PROJECT_DIR / "research" / "outputs" / "deduction-stress-paired-v1"
V2_MATRIX_DIR = (
    PROJECT_DIR / "research" / "outputs" / "deduction-stress-v2-paired-seed101"
)
V2_PROTOCOL_PATH = PROJECT_DIR / "research" / "protocol-deduction-stress-v2.json"
V2_SPEC_PATH = PROJECT_DIR / "examples" / "foldr-sparse-bounded-square-v2.json"
PREDICATE_FORWARD = (
    '{"kind":"And","left":{"kind":"LessThan","left":{"intValue":"-2",'
    '"kind":"IntLiteral"},"right":{"kind":"Item"}},"right":{"kind":"LessThan",'
    '"left":{"kind":"Item"},"right":{"intValue":"3","kind":"IntLiteral"}}}'
)
PREDICATE_SWAPPED = (
    '{"kind":"And","left":{"kind":"LessThan","left":{"kind":"Item"},'
    '"right":{"intValue":"3","kind":"IntLiteral"}},"right":{"kind":"LessThan",'
    '"left":{"intValue":"-2","kind":"IntLiteral"},"right":{"kind":"Item"}}}'
)
MAPPER = '{"kind":"Multiply","left":{"kind":"Item"},"right":{"kind":"Item"}}'


@pytest.fixture(scope="module")
def v2_support_audit() -> tuple[object, ...]:
    protocol = json.loads(V2_PROTOCOL_PATH.read_text(encoding="utf-8"))
    options = _options(protocol)
    config = load_experiment_config(V2_SPEC_PATH)
    support_math, support, _, states = _materialize_support_math(
        config,
        options,
        bounded_square_target(),
    )
    return config, support_math, support, states, _runtime_support(config, options)


def _candidate(name: str, probability: float) -> dict[str, object]:
    return {
        "canonical_candidate": name,
        "token_ids": [],
        "token_logprobs": [],
        "scored_token_count": 0,
        "total_sequence_logprob": 0.0,
        "normalized_energy": 0.0,
        "qwen_probability": probability,
        "deduction_probability": probability,
        "proposal_probability": probability,
    }


def _ledger(
    *,
    wave: str,
    candidates: list[str],
    selected_index: int,
    prompt_prefix: str,
) -> dict[str, object]:
    probability = 1.0 / len(candidates)
    return {
        "stage": 1,
        "beta": 1.0,
        "wave": wave,
        "request_index": 0,
        "prompt_prefix": prompt_prefix,
        "prompt_prefix_sha256": hashlib.sha256(prompt_prefix.encode()).hexdigest(),
        "candidate_kind": "expression",
        "source": "test",
        "model": "test",
        "model_revision": None,
        "tokenizer_revision": None,
        "score_semantics": "teacher-forced-full-prompt",
        "score_origin": "synthetic",
        "cache_key_sha256": None,
        "cache_hit": None,
        "energy_normalization": "mean-full-prompt-conditional-logprob",
        "temperature": 0.7,
        "proposal_epsilon": 0.05,
        "deduction_mix": 0.75,
        "candidates": [_candidate(candidate, probability) for candidate in candidates],
        "selections": [
            {
                "slot": 0,
                "ancestor_state_index": None,
                "selected_index": selected_index,
                "selected_probability": probability,
                "selected_qwen_probability": probability,
                "selected_deduction_probability": probability,
                "forced": False,
                "cloned": False,
            }
        ],
    }


def test_derived_probability_helpers_refuse_splice_proxies() -> None:
    proxy = {"status": NONIDENTIFYING_SPLICE_PROXY, "mean": 0.01}
    with pytest.raises(ValueError, match="identified probability"):
        probability_at_least_one(proxy, 20)
    with pytest.raises(ValueError, match="identified probability"):
        draws_for_probability(proxy)

    identified = identified_probability(0.01, provenance="unit test")
    assert identified["status"] == IDENTIFIED
    assert probability_at_least_one(identified, 2) == pytest.approx(0.0199)
    assert draws_for_probability(identified) == 69


def test_telemetry_reconciliation_separates_cache_and_provider_tokens() -> None:
    result = {
        "scored_candidates": 2,
        "score_ledger": [
            {
                "score_origin": "cache",
                "cache_hit": True,
                "cache_key_sha256": "a" * 64,
                "candidates": [{"scored_token_count": 5}],
            },
            {
                "score_origin": "provider",
                "cache_hit": False,
                "cache_key_sha256": "b" * 64,
                "candidates": [{"scored_token_count": 7}],
            },
        ],
    }
    manifest = {
        "metrics": {
            "candidate_score_cache": {
                "mode": "read-write",
                "lookup_requests": 2,
                "hit_requests": 1,
                "miss_requests": 1,
                "provider_score_requests": 1,
                "lookup_candidates": 2,
                "hit_candidates": 1,
                "miss_candidates": 1,
                "provider_candidates": 1,
                "provider_scored_tokens": 7,
                "cache_served_scored_tokens": 5,
                "provider_await_wall_seconds": 0.5,
            },
            "candidate_score_provider": {
                "http_requests": 1,
                "scored_token_positions": 7,
                "http_request_seconds_sum": 0.75,
            },
        }
    }
    telemetry, violations = reconcile_telemetry(result, manifest, cell_id="test-cell")
    assert violations == []
    assert telemetry["provider_scored_token_positions"] == 7
    assert telemetry["cache_served_token_positions"] == 5
    assert telemetry["ledger_token_positions"] == 12
    assert telemetry["provider_score_requests"] == 1
    assert telemetry["http_requests"] == 1

    mutated = copy.deepcopy(result)
    mutated["score_ledger"][1]["score_origin"] = "cache"
    _, violations = reconcile_telemetry(mutated, manifest, cell_id="test-cell")
    assert {
        "LEDGER_CACHE_PROVENANCE_INVALID",
        "CACHE_LEDGER_REQUEST_RECONCILIATION_MISMATCH",
        "CACHE_LEDGER_CANDIDATE_RECONCILIATION_MISMATCH",
        "CACHE_LEDGER_TOKEN_RECONCILIATION_MISMATCH",
    }.issubset({violation["code"] for violation in violations})


def test_cache_off_provider_telemetry_is_not_erased() -> None:
    result = {
        "scored_candidates": 2,
        "score_ledger": [
            {
                "score_origin": "provider",
                "cache_hit": None,
                "cache_key_sha256": None,
                "stage": 1,
                "wave": "family",
                "candidates": [{"scored_token_count": 7}],
            },
            {
                "score_origin": "provider",
                "cache_hit": None,
                "cache_key_sha256": None,
                "stage": 1,
                "wave": "family",
                "candidates": [{"scored_token_count": 5}],
            },
        ],
    }
    manifest = {
        "metrics": {
            "candidate_score_cache": {"mode": "off"},
            "candidate_score_provider": {
                "http_requests": 2,
                "http_failures": 0,
                "scored_token_positions": 12,
                "http_request_seconds_sum": 0.25,
            },
        }
    }
    telemetry, violations = reconcile_telemetry(result, manifest, cell_id="test-cell")
    assert violations == []
    assert telemetry["provider_score_requests"] == 2
    assert telemetry["provider_invocations"] == 1
    assert telemetry["provider_candidates"] == 2
    assert telemetry["provider_scored_token_positions"] == 12
    assert telemetry["ledger_token_positions"] == 12
    assert telemetry["http_requests"] == 2
    assert telemetry["provider_await_wall_seconds"] is None


def test_finite_reference_mutation_is_detected() -> None:
    result = {
        "reference": {
            "enumeration_exact_mass": 0.9,
            "log_path_z_enumeration": -1.0,
        }
    }
    violations: list[dict[str, str]] = []
    _audit_finite_reference(
        result,
        {"exact_target_mass": 0.8, "log_target_normalizer": -1.0},
        cell_id="test-cell",
        violations=violations,
    )
    assert [violation["code"] for violation in violations] == [
        "FINITE_REFERENCE_REPLAY_MISMATCH"
    ]


def test_runtime_state_index_and_sampled_best_mutations_are_detected(
    v2_support_audit: tuple[object, ...],
) -> None:
    _, support_math, _, states, runtime_support = v2_support_audit
    runtime_state = runtime_support.states[0]
    factorized_state = next(
        state for state in states if state["key"] == runtime_state.key
    )
    particle = {
        "particle_index": 0,
        "state_index": 0,
        "ancestor_state_index": 0,
        "family": runtime_state.family,
        "program": runtime_state.program,
        "total_loss": runtime_state.score.total_loss,
        "cost": runtime_state.score.cost,
        "exact_program": runtime_state.score.exact_program,
        "weight": 1.0,
        "log_q_mixture": 0.0,
        "log_incremental_weight": factorized_state["log_target"],
    }
    violations: list[dict[str, str]] = []
    _audit_final_particle_math(
        [particle],
        states,
        runtime_support,
        cell_id="test-cell",
        violations=violations,
    )
    assert violations == []

    wrong_index = next(
        state.state_index for state in runtime_support.states if state.key != runtime_state.key
    )
    mutated_particle = {**particle, "state_index": wrong_index}
    violations = []
    _audit_final_particle_math(
        [mutated_particle],
        states,
        runtime_support,
        cell_id="test-cell",
        violations=violations,
    )
    assert "PARTICLE_STATE_INDEX_PROGRAM_MISMATCH" in {
        violation["code"] for violation in violations
    }

    enumeration_probability = math.exp(
        factorized_state["log_target"] - support_math["log_target_normalizer"]
    )
    result = {
        "sampled_best": {
            "state_index": 0,
            "family": runtime_state.family,
            "program": runtime_state.program,
            "total_loss": runtime_state.score.total_loss,
            "cost": runtime_state.score.cost,
            "exact_program": not runtime_state.score.exact_program,
            "empirical_mass": 1.0,
            "enumeration_probability": enumeration_probability,
        }
    }
    violations = []
    _audit_sampled_best(
        result,
        [particle],
        states,
        runtime_support,
        support_math,
        cell_id="test-cell",
        violations=violations,
    )
    assert [violation["code"] for violation in violations] == [
        "SAMPLED_BEST_REPLAY_MISMATCH"
    ]


def test_ledger_to_particle_log_q_mutation_is_detected(
    v2_support_audit: tuple[object, ...],
) -> None:
    config, _, support, _, _ = v2_support_audit
    family = support.families[0]
    trace = ConstructionTrace(
        hypothesis_index=family.hypothesis_index,
        filling_indices=tuple(0 for _ in family.catalogs),
    )
    fillings = trace_fillings(family, trace)
    program = assemble_program(
        family.hypothesis,
        {filling.hole_name: filling.expression for filling in fillings},
        allowed_integer_constants=config.spec.integer_constants,
    )
    ledgers: list[dict[str, object]] = [
        {
            "wave": "family",
            "candidates": [
                {"canonical_candidate": json.dumps(family.hypothesis.kind.value)}
            ],
            "selections": [
                {
                    "slot": 0,
                    "ancestor_state_index": 3,
                    "selected_index": 0,
                    "selected_probability": 0.5,
                    "forced": False,
                    "cloned": False,
                }
            ],
        }
    ]
    log_q = math.log(0.5)
    for hole_index, filling in enumerate(fillings):
        probability = 0.25
        log_q += math.log(probability)
        ledgers.append(
            {
                "wave": f"hole-{hole_index}",
                "candidates": [{"canonical_candidate": filling.key}],
                "selections": [
                    {
                        "slot": 0,
                        "ancestor_state_index": 3,
                        "selected_index": 0,
                        "selected_probability": probability,
                        "forced": False,
                        "cloned": False,
                    }
                ],
            }
        )
    particle = {
        "particle_index": 0,
        "ancestor_state_index": 3,
        "family": family.hypothesis.kind.value,
        "program": program,
        "log_q_mixture": log_q + 0.1,
        "cloned": False,
    }
    violations: list[dict[str, str]] = []
    _audit_ledger_particle_paths(
        [particle],
        ledgers,
        support,
        allowed_integer_constants=config.spec.integer_constants,
        cell_id="test-cell",
        violations=violations,
    )
    assert [violation["code"] for violation in violations] == [
        "PARTICLE_LOG_Q_LEDGER_MISMATCH"
    ]

    correct_particle = {**particle, "log_q_mixture": log_q}
    violations = []
    _audit_ledger_particle_paths(
        [{**correct_particle, "ancestor_state_index": 4, "cloned": True}],
        ledgers,
        support,
        allowed_integer_constants=config.spec.integer_constants,
        cell_id="test-cell",
        violations=violations,
    )
    assert {
        "PARTICLE_CLONE_FLAG_INVALID",
        "PARTICLE_LEDGER_ANCESTOR_MISMATCH",
    }.issubset({violation["code"] for violation in violations})

    renumbered_ledgers = [
        {
            **ledger,
            "selections": [
                {**selection, "slot": 1}
                for selection in ledger["selections"]
            ],
        }
        for ledger in ledgers
    ]
    violations = []
    _audit_ledger_particle_paths(
        [{**correct_particle, "particle_index": 1}],
        renumbered_ledgers,
        support,
        allowed_integer_constants=config.spec.integer_constants,
        cell_id="test-cell",
        violations=violations,
    )
    assert {
        "PARTICLE_SLOT_DOMAIN_MISMATCH",
        "LEDGER_PARTICLE_SLOT_SET_MISMATCH",
    }.issubset({violation["code"] for violation in violations})


def test_unselected_catalog_and_deduction_guide_mutations_are_detected(
    v2_support_audit: tuple[object, ...],
) -> None:
    config, _, support, _, _ = v2_support_audit
    protocol = json.loads(V2_PROTOCOL_PATH.read_text(encoding="utf-8"))
    options = _options(protocol)
    guide = family_guide_probabilities(
        support,
        cost_scale=float(config.smc.cost_scale),
        violation_scale=options.deduction_strength,
    )
    guide_by_hypothesis = {
        family.hypothesis_index: float(guide[index])
        for index, family in enumerate(support.families)
    }
    families = sorted(support.families, key=lambda family: family.hypothesis.kind.value)
    ledger = {
        "wave": "family",
        "candidates": [
            {
                "canonical_candidate": json.dumps(family.hypothesis.kind.value),
                "deduction_probability": guide_by_hypothesis[family.hypothesis_index],
            }
            for family in families
        ],
        "selections": [{"slot": 0, "selected_index": 0}],
    }
    violations: list[dict[str, str]] = []
    _audit_catalog_and_deduction_guides(
        [ledger],
        support,
        config,
        options,
        cell_id="test-cell",
        violations=violations,
    )
    assert violations == []

    mutated = copy.deepcopy(ledger)
    mutated["candidates"][1]["canonical_candidate"] = '"not-a-real-family"'
    mutated["candidates"][1]["deduction_probability"] += 0.1
    violations = []
    _audit_catalog_and_deduction_guides(
        [mutated],
        support,
        config,
        options,
        cell_id="test-cell",
        violations=violations,
    )
    assert {
        "LEDGER_CATALOG_REPLAY_MISMATCH",
        "DEDUCTION_GUIDE_REPLAY_MISMATCH",
    } == {violation["code"] for violation in violations}


def test_prompt_prefix_is_bound_to_materialized_and_lazy_ancestors(
    v2_support_audit: tuple[object, ...],
) -> None:
    config, _, support, states, runtime_support = v2_support_audit
    runtime_families = tuple(sorted(runtime_support.families, key=family_candidate))
    materialized_ancestor = runtime_support.states[0]
    materialized_prefix = family_prompt_prefix(
        config=config,
        families=runtime_families,
        ancestor=materialized_ancestor,
        stage=1,
        beta=1.0,
    )
    family_candidates = [
        {"canonical_candidate": family_candidate(family)}
        for family in runtime_families
    ]
    materialized_ledger = {
        "stage": 1,
        "beta": 1.0,
        "wave": "family",
        "prompt_prefix": materialized_prefix,
        "prompt_prefix_sha256": hashlib.sha256(
            materialized_prefix.encode()
        ).hexdigest(),
        "candidates": family_candidates,
        "selections": [{"slot": 0, "selected_index": 0}],
    }
    materialized_particle = {
        "particle_index": 0,
        "state_index": 0,
        "ancestor_state_index": 0,
    }
    violations: list[dict[str, str]] = []
    _audit_score_prompt_provenance(
        [materialized_particle],
        [materialized_ledger],
        support,
        runtime_support,
        config,
        cell_id="test-cell",
        violations=violations,
    )
    assert violations == []

    swapped_prefix = family_prompt_prefix(
        config=config,
        families=runtime_families,
        ancestor=runtime_support.states[1],
        stage=1,
        beta=1.0,
    )
    swapped_ledger = {
        **materialized_ledger,
        "prompt_prefix": swapped_prefix,
        "prompt_prefix_sha256": hashlib.sha256(swapped_prefix.encode()).hexdigest(),
    }
    violations = []
    _audit_score_prompt_provenance(
        [materialized_particle],
        [swapped_ledger],
        support,
        runtime_support,
        config,
        cell_id="test-cell",
        violations=violations,
    )
    assert [violation["code"] for violation in violations] == [
        "PROMPT_PREFIX_PROVENANCE_MISMATCH"
    ]

    specialized_runtime = next(
        family
        for family in runtime_families
        if family.hypothesis.kind.value == "foldr-filter-map"
    )
    specialized_factorized = support.family(specialized_runtime.hypothesis_index)
    specialized_index = runtime_families.index(specialized_runtime)
    predicate = specialized_factorized.catalogs[0].fillings[0]
    mapper = specialized_factorized.catalogs[1].fillings[0]
    specialized_family_ledger = {
        **materialized_ledger,
        "selections": [{"slot": 0, "selected_index": specialized_index}],
    }
    hole_zero_prefix = hole_prompt_prefix(
        config=config,
        report=specialized_runtime.deduction,
        hole=specialized_runtime.hypothesis.holes[0],
        ancestor=materialized_ancestor,
        previous_fillings=(),
        stage=1,
        beta=1.0,
        candidate_count=1,
    )
    hole_zero_ledger = {
        "stage": 1,
        "beta": 1.0,
        "wave": "hole-0",
        "prompt_prefix": hole_zero_prefix,
        "prompt_prefix_sha256": hashlib.sha256(hole_zero_prefix.encode()).hexdigest(),
        "candidates": [{"canonical_candidate": predicate.key}],
        "selections": [{"slot": 0, "selected_index": 0}],
    }
    hole_one_prefix = hole_prompt_prefix(
        config=config,
        report=specialized_runtime.deduction,
        hole=specialized_runtime.hypothesis.holes[1],
        ancestor=materialized_ancestor,
        previous_fillings=(predicate,),
        stage=1,
        beta=1.0,
        candidate_count=1,
    )
    hole_one_ledger = {
        "stage": 1,
        "beta": 1.0,
        "wave": "hole-1",
        "prompt_prefix": hole_one_prefix,
        "prompt_prefix_sha256": hashlib.sha256(hole_one_prefix.encode()).hexdigest(),
        "candidates": [{"canonical_candidate": mapper.key}],
        "selections": [{"slot": 0, "selected_index": 0}],
    }
    previous_json = json.dumps(
        {predicate.hole_name: json.loads(predicate.key)},
        sort_keys=True,
        separators=(",", ":"),
    )
    empty_previous = hole_one_prefix.replace(
        f"PreviousFillings={previous_json}", "PreviousFillings={}"
    )
    assert empty_previous != hole_one_prefix
    tampered_hole_one = {
        **hole_one_ledger,
        "prompt_prefix": empty_previous,
        "prompt_prefix_sha256": hashlib.sha256(empty_previous.encode()).hexdigest(),
    }
    violations = []
    _audit_score_prompt_provenance(
        [materialized_particle],
        [specialized_family_ledger, hole_zero_ledger, tampered_hole_one],
        support,
        runtime_support,
        config,
        cell_id="test-cell",
        violations=violations,
    )
    assert [violation["code"] for violation in violations] == [
        "PROMPT_PREFIX_PROVENANCE_MISMATCH"
    ]

    factorized_state = states[0]
    trace = factorized_state["trace_record"]
    factorized_family = support.family(trace.hypothesis_index)
    fillings = trace_fillings(factorized_family, trace)
    program = assemble_program(
        factorized_family.hypothesis,
        {filling.hole_name: filling.expression for filling in fillings},
        allowed_integer_constants=config.spec.integer_constants,
    )
    key = canonical_key(program)
    rescored = next(state for state in runtime_support.states if state.key == key)
    lazy_ancestor = LazyImportanceState(
        trace=trace,
        family=factorized_family.hypothesis.kind.value,
        fillings=fillings,
        program=program,
        key=key,
        score=rescored.score,
        log_prior=trace_log_prior(
            support,
            trace,
            cost_scale=float(config.smc.cost_scale),
        ),
    )
    factorized_families = tuple(
        sorted(support.families, key=lambda family: family.hypothesis.kind.value)
    )
    lazy_prefix = lazy_family_prompt_prefix(
        config=config,
        families=factorized_families,
        ancestor=lazy_ancestor,
        stage=1,
        beta=1.0,
    )
    trace_json = {
        "hypothesis_index": trace.hypothesis_index,
        "filling_indices": list(trace.filling_indices),
    }
    lazy_particle = {
        "particle_index": 0,
        "trace": trace_json,
        "ancestor_trace": trace_json,
    }
    lazy_ledger = {
        **materialized_ledger,
        "prompt_prefix": lazy_prefix,
        "prompt_prefix_sha256": hashlib.sha256(lazy_prefix.encode()).hexdigest(),
        "candidates": [
            {"canonical_candidate": json.dumps(family.hypothesis.kind.value)}
            for family in factorized_families
        ],
    }
    violations = []
    _audit_score_prompt_provenance(
        [lazy_particle],
        [lazy_ledger],
        support,
        runtime_support,
        config,
        cell_id="test-cell",
        violations=violations,
    )
    assert violations == []


def test_protocol_configuration_mutation_is_detected(
    v2_support_audit: tuple[object, ...],
) -> None:
    config, _, _, _, _ = v2_support_audit
    protocol = json.loads(V2_PROTOCOL_PATH.read_text(encoding="utf-8"))
    caps = protocol["caps"]
    shared = protocol["shared_arguments"]
    cell_plan = {
        "arm": "D",
        "seed": 101,
        "model_id": None,
        "model_hf_repository": None,
        "model_revision": None,
        "tokenizer_revision": None,
    }
    runtime_configuration = {
        "mode": "importance-smc",
        "beta_max": shared["beta_max"],
        "candidate_batch_size": shared["candidate_batch_size"],
        "deduction_mix": 0.75,
        "family_deduction_mix": 0.75,
        "hole_deduction_mix": 0.0,
        "deduction_strength": shared["deduction_strength"],
        "proposal_epsilon": shared["proposal_epsilon"],
        "temperature": shared["temperature"],
        "llm_energy_normalization": shared["llm_energy_normalization"],
        "max_scored_candidates": caps["max_scored_candidates"],
        "hole_max_cost": caps["hole_max_cost"],
        "hole_state_limit": caps["hole_state_limit"],
        "support_limit": caps["support_limit"],
        "proposal": "catalog",
        "timeout_seconds": caps["provider_timeout_seconds"],
        "materialize_reference": True,
        "requested_skeleton": "auto",
        "skeleton": "multi-family",
        "score_cache_mode": "off",
        "score_cache_dir": None,
        "vllm_server_config": None,
        "model": None,
        "model_repository": None,
        "model_revision": None,
        "tokenizer_revision": None,
        "experiment": {
            "spec": config.spec.model_dump(mode="json", by_alias=True),
            "smc": {
                **config.smc.model_dump(mode="json", by_alias=True),
                "cloneProbability": caps["alpha"],
                "essThreshold": caps["ess_threshold"],
                "iterations": caps["iterations"],
                "particles": caps["particles"],
                "seed": 101,
            },
        },
    }
    result = {
        "mode": "importance-smc",
        "proposal_source": "uniform-finite-candidates",
        "deduction_strength": shared["deduction_strength"],
        "llm_energy_normalization": shared["llm_energy_normalization"],
        "max_scored_candidates": caps["max_scored_candidates"],
        "stages": [{}],
        "reference": {},
    }
    envelope = {"particle_count": caps["particles"]}
    command = [
        "/tmp/modelsmc-pbe",
        "synthesize",
        "/alternate/checkout/examples/foldr-sparse-bounded-square-v2.json",
        "--mode",
        "importance-smc",
        "--proposal",
        "catalog",
        "--model",
        "size-invariant-catalog-control",
        "--skeleton",
        "auto",
        "--particles",
        "4",
        "--iterations",
        "1",
        "--alpha",
        "0.0",
        "--ess-threshold",
        "1.0",
        "--seed",
        "101",
        "--max-scored-candidates",
        "2652",
        "--support-limit",
        "250000",
        "--hole-state-limit",
        "250000",
        "--hole-max-cost",
        "3",
        "--timeout-seconds",
        "300.0",
        "--deduction-mix",
        "0.75",
        "--family-deduction-mix",
        "0.75",
        "--hole-deduction-mix",
        "0.0",
        "--beta-max",
        "1.0",
        "--candidate-batch-size",
        "32",
        "--deduction-strength",
        "4.0",
        "--llm-energy-normalization",
        "mean-full-prompt-conditional-logprob",
        "--proposal-epsilon",
        "0.05",
        "--temperature",
        "0.7",
        "--materialize-reference",
    ]
    core_manifest = {
        "seed": 101,
        "configuration": runtime_configuration,
        "command": command,
        "metrics": {
            "candidate_score_cache": {"mode": "off", "cache_dir": None}
        },
    }
    cell_document = {"command": command}
    matrix_manifest = {"effective_caps": caps}
    ledger = {
        "stage": 1,
        "beta": shared["beta_max"],
        "temperature": shared["temperature"],
        "proposal_epsilon": shared["proposal_epsilon"],
        "energy_normalization": shared["llm_energy_normalization"],
        "model": "none",
        "model_revision": None,
        "tokenizer_revision": None,
    }
    violations: list[dict[str, str]] = []
    _audit_protocol_configuration(
        result,
        envelope,
        cell_document,
        core_manifest,
        matrix_manifest,
        protocol,
        cell_plan,
        [ledger],
        config,
        cell_id="test-cell",
        violations=violations,
    )
    assert violations == []

    violations = []
    _audit_protocol_configuration(
        result,
        envelope,
        cell_document,
        core_manifest,
        matrix_manifest,
        protocol,
        cell_plan,
        [{**ledger, "temperature": 99.0}],
        config,
        cell_id="test-cell",
        violations=violations,
    )
    assert [violation["code"] for violation in violations] == [
        "PROTOCOL_CONFIGURATION_MISMATCH"
    ]

    model = protocol["models"][0]
    provider = protocol["provider"]
    provider_cell_plan = {
        **cell_plan,
        "arm": "QD",
        "model_id": model["id"],
        "model_alias": model["alias"],
        "model_hf_repository": model["hf_repository"],
        "model_revision": model["model_revision"],
        "tokenizer_revision": model["tokenizer_revision"],
    }
    provider_configuration = copy.deepcopy(runtime_configuration)
    provider_configuration.update(
        {
            "proposal": "vllm",
            "model": model["id"],
            "model_repository": model["hf_repository"],
            "model_revision": model["model_revision"],
            "tokenizer_revision": model["tokenizer_revision"],
            "score_cache_mode": provider["score_cache_mode"],
            "score_cache_dir": (
                "/alternate/checkout/research/" + provider["score_cache_dir"]
            ),
            "vllm_server_config": provider["vllm_server_config"],
        }
    )
    provider_result = {
        **result,
        "proposal_source": "vllm-prompt-logprobs",
    }
    provider_ledger = {
        **ledger,
        "model": model["id"],
        "model_revision": model["model_revision"],
        "tokenizer_revision": model["tokenizer_revision"],
    }
    provider_command = list(command)

    def replace_flag(command_values: list[str], flag: str, value: str) -> None:
        command_values[command_values.index(flag) + 1] = value

    replace_flag(provider_command, "--proposal", "vllm")
    replace_flag(provider_command, "--model", model["alias"])
    provider_command.extend(
        [
            "--model-repository",
            model["hf_repository"],
            "--model-revision",
            model["model_revision"],
            "--tokenizer-revision",
            model["tokenizer_revision"],
            "--base-url",
            "https://provider.invalid/v1",
            "--score-cache-dir",
            "/alternate/checkout/research/" + provider["score_cache_dir"],
            "--score-cache-mode",
            provider["score_cache_mode"],
            "--vllm-server-config",
            provider["vllm_server_config"],
        ]
    )
    provider_manifest = {
        "seed": 101,
        "configuration": provider_configuration,
        "command": provider_command,
        "metrics": {
            "candidate_score_cache": {
                "mode": provider["score_cache_mode"],
                "cache_dir": (
                    "/alternate/checkout/research/" + provider["score_cache_dir"]
                ),
            }
        },
    }
    provider_cell_document = {"command": provider_command}
    violations = []
    _audit_protocol_configuration(
        provider_result,
        envelope,
        provider_cell_document,
        provider_manifest,
        matrix_manifest,
        protocol,
        provider_cell_plan,
        [provider_ledger],
        config,
        cell_id="test-cell",
        violations=violations,
    )
    assert violations == []

    mutated_provider_manifest = copy.deepcopy(provider_manifest)
    mutated_provider_manifest["configuration"]["vllm_server_config"] = "wrong"
    mutated_provider_manifest["configuration"]["timeout_seconds"] = 999.0
    mutated_provider_manifest["configuration"]["score_cache_mode"] = "off"
    mutated_provider_manifest["configuration"]["score_cache_dir"] = None
    mutated_provider_manifest["configuration"]["materialize_reference"] = False
    mutated_provider_manifest["metrics"]["candidate_score_cache"] = {
        "mode": "off",
        "cache_dir": None,
    }
    violations = []
    _audit_protocol_configuration(
        provider_result,
        envelope,
        provider_cell_document,
        mutated_provider_manifest,
        matrix_manifest,
        protocol,
        provider_cell_plan,
        [provider_ledger],
        config,
        cell_id="test-cell",
        violations=violations,
    )
    assert [violation["code"] for violation in violations] == [
        "PROTOCOL_CONFIGURATION_MISMATCH"
    ]

    mutated_provider_command = list(provider_command)
    replace_flag(mutated_provider_command, "--model", "wrong")
    replace_flag(mutated_provider_command, "--skeleton", "wrong")
    replace_flag(mutated_provider_command, "--timeout-seconds", "999.0")
    replace_flag(mutated_provider_command, "--model-revision", "wrong")
    replace_flag(mutated_provider_command, "--score-cache-mode", "replay-only")
    replace_flag(mutated_provider_command, "--vllm-server-config", "wrong")
    mutated_command_manifest = {
        **provider_manifest,
        "command": mutated_provider_command,
    }
    violations = []
    _audit_protocol_configuration(
        provider_result,
        envelope,
        {"command": mutated_provider_command},
        mutated_command_manifest,
        matrix_manifest,
        protocol,
        provider_cell_plan,
        [provider_ledger],
        config,
        cell_id="test-cell",
        violations=violations,
    )
    assert [violation["code"] for violation in violations] == [
        "PROTOCOL_CONFIGURATION_MISMATCH"
    ]

    violations = []
    _audit_protocol_configuration(
        provider_result,
        envelope,
        {"command": [*provider_command, "--unexpected"]},
        provider_manifest,
        matrix_manifest,
        protocol,
        provider_cell_plan,
        [provider_ledger],
        config,
        cell_id="test-cell",
        violations=violations,
    )
    assert [violation["code"] for violation in violations] == [
        "PROTOCOL_CONFIGURATION_MISMATCH"
    ]

    mutated_manifest = copy.deepcopy(core_manifest)
    mutated_configuration = mutated_manifest["configuration"]
    mutated_configuration["mode"] = "paper-search"
    mutated_configuration["experiment"]["smc"]["maxCost"] = 999
    mutated_configuration["experiment"]["spec"]["examples"][0]["output"] = [1]
    violations = []
    _audit_protocol_configuration(
        result,
        envelope,
        cell_document,
        mutated_manifest,
        matrix_manifest,
        protocol,
        cell_plan,
        [ledger],
        config,
        cell_id="test-cell",
        violations=violations,
    )
    assert [violation["code"] for violation in violations] == [
        "PROTOCOL_CONFIGURATION_MISMATCH"
    ]


def test_qd_target_audit_does_not_splice_a_wrong_predicate_prefix() -> None:
    wrong_predicate = {
        "kind": "LessThan",
        "left": {"kind": "Item"},
        "right": {"intValue": "0", "kind": "IntLiteral"},
    }
    ledgers = [
        _ledger(
            wave="family",
            candidates=['"expression"', '"foldr-filter-map"'],
            selected_index=1,
            prompt_prefix="family",
        ),
        _ledger(
            wave="hole-0",
            candidates=[PREDICATE_FORWARD, PREDICATE_SWAPPED, "other"],
            selected_index=2,
            prompt_prefix="predicate",
        ),
        _ledger(
            wave="hole-1",
            candidates=[MAPPER, "other"],
            selected_index=1,
            prompt_prefix=f'PreviousFillings={{"predicate":{wrong_predicate!r}}}',
        ),
    ]
    # Use valid JSON in the prompt line while keeping it visibly distinct from both targets.
    ledgers[-1]["prompt_prefix"] = (
        "PreviousFillings={\"predicate\":{\"kind\":\"LessThan\","
        "\"left\":{\"kind\":\"Item\"},\"right\":{\"intValue\":\"0\","
        "\"kind\":\"IntLiteral\"}}}"
    )
    ledgers[-1]["prompt_prefix_sha256"] = hashlib.sha256(
        str(ledgers[-1]["prompt_prefix"]).encode()
    ).hexdigest()

    report, violations = audit_qd_target_ledgers(
        ledgers,
        exact_predicate_keys=frozenset((PREDICATE_FORWARD, PREDICATE_SWAPPED)),
        mapper_key=MAPPER,
        cell_id="test-cell",
    )
    assert violations == []
    assert report["exact_prefix_mapper_requests"] == 0
    assert report["exact_proposal_probability"]["status"] == NOT_IDENTIFIED
    assert report["splice_proxy"]["status"] == NONIDENTIFYING_SPLICE_PROXY
    assert report["splice_proxy"]["observations"] == 1
    assert report["discovery_probability_20_draws"]["status"] == REFUSED_NOT_IDENTIFIED
    assert report["draws_for_50_percent"]["status"] == REFUSED_NOT_IDENTIFIED


def test_audit_adapts_trace_and_state_index_result_shapes_strictly() -> None:
    trace = {"hypothesis_index": 2, "filling_indices": [1, 2]}
    ancestor = {"hypothesis_index": 2, "filling_indices": [3, 4]}
    assert _particle_path_signature(
        {"trace": trace, "ancestor_trace": ancestor}
    ) == ["trace", trace, ancestor]
    assert _particle_path_signature(
        {"state_index": 8_890, "ancestor_state_index": 27_222}
    ) == ["state_index", 8_890, 27_222]

    with pytest.raises(ValueError, match="incomplete"):
        _particle_path_signature({"trace": trace})
    with pytest.raises(ValueError, match="mixes"):
        _particle_path_signature(
            {
                "trace": trace,
                "ancestor_trace": ancestor,
                "state_index": 8_890,
                "ancestor_state_index": 27_222,
            }
        )
    with pytest.raises(ValueError, match="lacks"):
        _particle_path_signature({})

    with pytest.raises(ValueError, match="must be an object"):
        _result_search_summary(
            {"search": "invalid", "sampled_best": {"exact_program": False}}
        )
    with pytest.raises(ValueError, match="requires"):
        _result_search_summary({"best_visited": {"exact_program": False}})

    best, evaluated, shape = _result_search_summary(
        {
            "search": {"evaluated_programs": 8},
            "best_visited": {"exact_program": False},
        }
    )
    assert best["exact_program"] is False
    assert evaluated == 8
    assert shape == "best_visited"

    best, evaluated, shape = _result_search_summary(
        {"sampled_best": {"exact_program": False}}
    )
    assert best["exact_program"] is False
    assert evaluated is None
    assert shape == "sampled_best_final_population"


def test_v2_audit_resolves_split_deduction_mixes_from_the_d_arm() -> None:
    protocol = json.loads(V2_PROTOCOL_PATH.read_text(encoding="utf-8"))
    options = _options(protocol)
    assert options.deduction_mix == 0.75
    assert options.resolved_family_deduction_mix == 0.75
    assert options.resolved_hole_deduction_mix == 0.0


@pytest.mark.skipif(not MATRIX_DIR.is_dir(), reason="sealed deduction-stress artifacts absent")
def test_current_deduction_stress_artifacts_pass_math_audit() -> None:
    report = audit_matrix(MATRIX_DIR)
    assert report["invariants_passed"] is True
    assert report["violations"] == []

    target = report["target_math"]
    assert target["support_states"] == 36_198
    assert target["exact_states"] == 2
    assert target["exact_support_fraction"] == pytest.approx(2 / 36_198)
    assert target["exact_prior_mass"] == pytest.approx(1.8344729850634e-5)
    assert target["exact_is_global_map"] is False
    assert target["minimum_loss_scale_for_exact_map_tie"] == pytest.approx(
        1.2676345133088531
    )

    d_probability = report["d_guided_exact_proposal"]
    assert d_probability["status"] == IDENTIFIED
    assert d_probability["value"] == pytest.approx(4.004265371543205e-5)
    assert d_probability["draws_for_50_percent"] == 17_310

    qd_probability = report["qd_guided_exact_proposal"]
    assert qd_probability["exact_proposal_probability"]["status"] == NOT_IDENTIFIED
    assert qd_probability["exact_prefix_mapper_requests"] == 0
    assert qd_probability["splice_proxy"]["mean"] == pytest.approx(4.006930138642335e-5)
    assert qd_probability["mapper_qwen_rank_range"] == [49, 56]
    assert qd_probability["mapper_proposal_rank_range"] == [58, 60]

    telemetry = report["telemetry"]
    assert telemetry["provider_scored_token_positions"] == 4_715_582
    assert telemetry["cache_served_token_positions"] == 1_870_206
    assert telemetry["ledger_token_positions"] == 6_585_788
    assert telemetry["provider_score_requests"] == 43
    assert telemetry["http_requests"] == 272
    assert all(
        pair["same_ancestor_and_final_traces"] and pair["same_selected_indices"]
        for pair in report["matrix"]["paired_draws"]
    )

    evaluated = {
        (cell["arm"], cell["seed"]): cell["evaluated_programs"]
        for cell in report["cells"]
    }
    assert evaluated[("D", 503)] == evaluated[("QD", 503)] == 7
    assert all(
        evaluated[(arm, seed)] == 8
        for arm in ("D", "QD")
        for seed in (101, 211, 307, 401)
    )

    assert math.isclose(
        telemetry["provider_scored_token_positions"]
        + telemetry["cache_served_token_positions"],
        telemetry["ledger_token_positions"],
    )


@pytest.mark.skipif(not V2_MATRIX_DIR.is_dir(), reason="v2 pilot artifacts absent")
def test_current_v2_pilot_artifacts_pass_strict_math_audit() -> None:
    report = audit_matrix(V2_MATRIX_DIR)
    assert report["invariants_passed"] is True
    assert report["violations"] == []
    audit_source = PROJECT_DIR / "research" / "audit_math.py"
    assert report["audit_source_sha256"] == hashlib.sha256(
        audit_source.read_bytes()
    ).hexdigest()

    target = report["target_math"]
    assert target["support_states"] == 36_198
    assert target["exact_states"] == 2
    assert target["exact_is_global_map"] is True
    assert target["exact_target_mass"] == pytest.approx(0.9207000842137091)
    assert target["best_exact"]["log_target"] - target["best_nonexact_at_configured_scale"][
        "log_target"
    ] == pytest.approx(4.0)

    d_probability = report["d_guided_exact_proposal"]
    assert d_probability["status"] == IDENTIFIED
    assert d_probability["family_deduction_mix"] == 0.75
    assert d_probability["hole_deduction_mix"] == 0.0
    assert d_probability["value"] == pytest.approx(4.0312369971377614e-5)

    qd_probability = report["qd_guided_exact_proposal"]
    assert qd_probability["exact_proposal_probability"]["status"] == NOT_IDENTIFIED
    assert qd_probability["exact_prefix_mapper_requests"] == 0
    assert qd_probability["splice_proxy"]["status"] == NONIDENTIFYING_SPLICE_PROXY
    assert qd_probability["predicate_qwen_rank_range"] == [17, 53]
    assert qd_probability["predicate_proposal_rank_range"] == [17, 53]

    assert report["matrix"]["paired_draws"] == [
        {
            "seed": 101,
            "paired_arms_present": True,
            "same_ancestor_and_final_traces": True,
            "same_selected_indices": True,
        }
    ]
    assert all(cell["final_exact_particles"] == 0 for cell in report["cells"])
    assert all(
        cell["result_shape"] == "sampled_best_final_population"
        for cell in report["cells"]
    )

    telemetry = report["telemetry"]
    assert telemetry["provider_scored_token_positions"] == 447_816
    assert telemetry["cache_served_token_positions"] == 0
    assert telemetry["ledger_token_positions"] == 447_816
    assert telemetry["provider_score_requests"] == 10
    assert telemetry["provider_invocations"] == 3
    assert telemetry["provider_invocation_failures"] == 0
    assert telemetry["http_requests"] == 29
    assert telemetry["provider_http_failures"] == 0

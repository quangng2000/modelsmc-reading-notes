from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from research.audit_math import (
    IDENTIFIED,
    NONIDENTIFYING_SPLICE_PROXY,
    NOT_IDENTIFIED,
    REFUSED_NOT_IDENTIFIED,
    audit_matrix,
    audit_qd_target_ledgers,
    draws_for_probability,
    identified_probability,
    probability_at_least_one,
    reconcile_telemetry,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
MATRIX_DIR = PROJECT_DIR / "research" / "outputs" / "deduction-stress-paired-v1"
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
                "candidates": [
                    {"scored_token_count": 5},
                    {"scored_token_count": 7},
                ]
            }
        ],
    }
    manifest = {
        "metrics": {
            "candidate_score_cache": {
                "mode": "read-write",
                "lookup_requests": 1,
                "hit_requests": 0,
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

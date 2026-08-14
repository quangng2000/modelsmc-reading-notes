from __future__ import annotations

import json
import math
from pathlib import Path

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.domain import canonical_key
from modelsmc_pbe.grammar import arithmetic_expressions, filter_predicates
from research.conditional_ranking import (
    NULL_DIGEST,
    _mapper_context,
    _target_keys,
    competition_rank,
    independent_n50,
    ranked_keys,
    smoothed_top_k_probability,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROTOCOL = PROJECT_DIR / "research" / "protocol-conditional-ranking-gptoss120b-v1.json"
TASK = PROJECT_DIR / "examples" / "foldr-sparse-bounded-square-v2.json"


def test_protocol_freezes_the_counterfactual_prefixes_and_gold_mapper() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen-before-provider-execution"
    assert protocol["oracle_reference"] == {
        "predicate_indices_zero_based": [238, 537],
        "mapper_index_zero_based": 42,
        "usage": (
            "Predicate indices define the two disclosed counterfactual mapper prefixes. "
            "The mapper index and all target-key joins are accessed only by the evaluation "
            "phase after the complete scores bundle is sealed."
        ),
    }


def test_gold_keys_match_the_frozen_factorized_catalog_indices() -> None:
    config = load_experiment_config(TASK)
    predicates = tuple(
        canonical_key(item) for item in filter_predicates(config.spec.integer_constants)
    )
    mappers = tuple(
        canonical_key(item)
        for item in arithmetic_expressions("Item", config.spec.integer_constants)
    )
    exact_predicates, exact_mapper = _target_keys()
    assert tuple(predicates.index(item) for item in exact_predicates) == (238, 537)
    assert mappers.index(exact_mapper) == 42


def test_competition_rank_retains_the_full_tie_interval() -> None:
    scores = {"a": 3.0, "b": 2.0, "c": 2.0, "d": -1.0}
    assert competition_rank(scores, "a") == (1, (1, 1))
    assert competition_rank(scores, "b") == (2, (2, 3))
    assert competition_rank(scores, "c") == (2, (2, 3))
    assert competition_rank(scores, "d") == (4, (4, 4))


def test_operational_tie_break_is_deterministic_but_seed_specific() -> None:
    scores = {str(index): 0.0 for index in range(30)}
    first = ranked_keys(
        scores,
        seed=1729,
        phase="predicate",
        prefix_sha256=NULL_DIGEST,
    )
    repeated = ranked_keys(
        scores,
        seed=1729,
        phase="predicate",
        prefix_sha256=NULL_DIGEST,
    )
    another = ranked_keys(
        scores,
        seed=2718,
        phase="predicate",
        prefix_sha256=NULL_DIGEST,
    )
    assert first == repeated
    assert first != another
    assert set(first) == set(scores)


def test_two_shortlisted_exact_paths_have_n50_nineteen() -> None:
    predicate_ranking = ("p1", "p2", *(f"p{index}" for index in range(3, 601)))
    mapper_ranking = ("m", *(f"m{index}" for index in range(1, 60)))
    q_predicate = smoothed_top_k_probability(
        "p1",
        predicate_ranking,
        catalog_size=600,
        top_k=10,
        epsilon=0.05,
    )
    q_mapper = smoothed_top_k_probability(
        "m",
        mapper_ranking,
        catalog_size=60,
        top_k=5,
        epsilon=0.05,
    )
    exact_mass = 2.0 * q_predicate * q_mapper
    assert math.isclose(exact_mass, 0.03629013888888889, rel_tol=0.0, abs_tol=1e-15)
    assert independent_n50(exact_mass) == 19


def test_mapper_prompt_never_calls_the_counterfactual_prefix_correct() -> None:
    prompt = _mapper_context("raw task", '{"kind":"Item"}')
    assert "correct predicate" not in prompt.lower()
    assert "predicate under evaluation" in prompt.lower()

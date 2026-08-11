"""Adversarial replay tests for compact joint-semantic evidence."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import get_type_hints

import pytest

from modelsmc_pbe.search.importance.joint_semantic.ledger import (
    SemanticBoundaryLedger,
    SemanticProgramScoreLedger,
    SemanticProposalLedger,
    SemanticSelectionLedger,
    SemanticTraceIdentity,
    SemanticTraceProbabilityLedger,
    slate_digest,
    validate_final_semantic_selections,
)
from modelsmc_pbe.search.importance.lazy_records import LazyImportanceSMCResult


def _boundary(mapping: str, marker: str) -> SemanticBoundaryLedger:
    shared_ids = marker * 64
    label_a_prefix_logprobs = chr(ord(marker) + 1) * 64
    label_b_prefix_logprobs = chr(ord(marker) + 2) * 64
    return SemanticBoundaryLedger(
        mapping=mapping,
        label_a_candidate_sha256="a" * 64,
        label_b_candidate_sha256="b" * 64,
        label_a_token_ids_sha256="c" * 64,
        label_b_token_ids_sha256="d" * 64,
        label_a_token_logprobs_sha256="e" * 64,
        label_b_token_logprobs_sha256="f" * 64,
        label_a_prefix_token_ids_sha256=shared_ids,
        label_b_prefix_token_ids_sha256=shared_ids,
        label_a_prefix_token_logprobs_sha256=label_a_prefix_logprobs,
        label_b_prefix_token_logprobs_sha256=label_b_prefix_logprobs,
        shared_token_count=8,
        shared_token_ids_sha256=shared_ids,
        max_abs_prefix_logprob_delta=0.25,
        label_a_token_id=10,
        label_b_token_id=11,
        label_a_logprob=-0.1 if mapping == "a-is-compatible" else -1.1,
        label_b_logprob=-1.1 if mapping == "a-is-compatible" else -0.1,
    )


def _ledger() -> SemanticProposalLedger:
    program = SemanticProgramScoreLedger(
        program_sha256="9" * 64,
        compatibility_log_score=1.0,
        template_version="joint-semantic-v1",
        prompt_sha256="0" * 64,
        source="test",
        model="test-model",
        score_semantics="teacher-forced-full-prompt",
        model_revision="revision",
        tokenizer_revision="tokenizer-revision",
        score_origin="synthetic",
        cache_key_sha256=None,
        cache_hit=None,
        boundaries=(
            _boundary("a-is-compatible", "1"),
            _boundary("b-is-compatible", "3"),
        ),
    )
    selected = SemanticTraceIdentity(0, (0,))
    prior = -math.log(2.0)
    inside_q = math.log(0.9)
    trace = SemanticTraceProbabilityLedger(
        trace=selected,
        program_sha256=program.program_sha256,
        log_prior=prior,
        compatibility_log_score=1.0,
        log_q_semantic=0.0,
        log_q_proposal=inside_q,
    )
    outside_q = math.log(0.1)
    selections = (
        SemanticSelectionLedger(
            stage=1,
            beta=1.0,
            slot=0,
            ancestor=selected,
            selected=selected,
            family_log_probability=0.0,
            hole_log_probabilities=(inside_q,),
            selected_log_prior=prior,
            selected_log_q_semantic=0.0,
            log_q_proposal=inside_q,
        ),
        SemanticSelectionLedger(
            stage=1,
            beta=1.0,
            slot=1,
            ancestor=selected,
            selected=SemanticTraceIdentity(1, (0,)),
            family_log_probability=0.0,
            hole_log_probabilities=(outside_q,),
            selected_log_prior=prior,
            selected_log_q_semantic=None,
            log_q_proposal=outside_q,
        ),
    )
    traces = (trace,)
    return SemanticProposalLedger(
        schema_version="joint-semantic-proposal-ledger-v2",
        formula="q=epsilon*pi+(1-epsilon)*normalize_slate(pi*exp(eta*a_llm))",
        slate_selection="seeded-sha256-rank",
        slate_seed=7,
        support_states=2,
        requested_slate_size=1,
        slate_traces=1,
        unique_programs=1,
        raw_candidate_count=4,
        proposal_epsilon=0.2,
        semantic_scale=2.0,
        template_version="joint-semantic-v1",
        prompt_sha256="0" * 64,
        slate_sha256=slate_digest(traces),
        program_scores=(program,),
        trace_probabilities=traces,
        selections=selections,
    )


def test_semantic_ledger_replays_inside_and_outside_slate_draws() -> None:
    ledger = _ledger()

    assert ledger.selections[0].selected_log_q_semantic == 0.0
    assert ledger.selections[1].selected_log_q_semantic is None

    validate_final_semantic_selections(
        ledger,
        iterations=1,
        final_traces=tuple(selection.selected for selection in ledger.selections),
        final_ancestors=tuple(selection.ancestor for selection in ledger.selections),
        final_log_q=tuple(selection.log_q_proposal for selection in ledger.selections),
    )

    with pytest.raises(ValueError, match="incomplete"):
        validate_final_semantic_selections(
            replace(ledger, selections=()),
            iterations=1,
            final_traces=tuple(selection.selected for selection in ledger.selections),
            final_ancestors=tuple(selection.ancestor for selection in ledger.selections),
            final_log_q=tuple(selection.log_q_proposal for selection in ledger.selections),
        )


def test_public_lazy_result_resolves_semantic_ledger_type() -> None:
    annotation = get_type_hints(LazyImportanceSMCResult)["semantic_score_ledger"]

    assert "SemanticProposalLedger" in str(annotation)


@pytest.mark.parametrize(
    "mutation, message",
    (
        (
            lambda ledger: replace(ledger, schema_version="joint-semantic-proposal-ledger-v1"),
            "unknown joint-semantic proposal ledger schema",
        ),
        (
            lambda ledger: replace(
                ledger,
                program_scores=(replace(ledger.program_scores[0], compatibility_log_score=2.0),),
            ),
            "program score disagrees",
        ),
        (
            lambda ledger: replace(
                ledger,
                trace_probabilities=(replace(ledger.trace_probabilities[0], log_q_semantic=-0.2),),
            ),
            "component probability failed replay",
        ),
        (
            lambda ledger: replace(
                ledger,
                selections=(
                    ledger.selections[0],
                    replace(ledger.selections[1], selected_log_prior=-2.0),
                ),
            ),
            "selected defensive semantic probability failed replay",
        ),
        (
            lambda ledger: replace(
                ledger,
                selections=(
                    replace(ledger.selections[0], log_q_proposal=-0.5),
                    ledger.selections[1],
                ),
            ),
            "conditionals failed to telescope",
        ),
        (
            lambda ledger: replace(
                ledger,
                unique_programs=2,
                raw_candidate_count=8,
                program_scores=(
                    *ledger.program_scores,
                    replace(ledger.program_scores[0], program_sha256="8" * 64),
                ),
            ),
            "inventories disagree",
        ),
        (
            lambda ledger: replace(
                ledger,
                program_scores=(
                    replace(
                        ledger.program_scores[0],
                        score_origin="cache",
                        cache_key_sha256=None,
                        cache_hit=False,
                    ),
                ),
            ),
            "cache-origin semantic scores require",
        ),
        (
            lambda ledger: replace(
                ledger,
                support_states=1,
                requested_slate_size=None,
                slate_selection="full-bounded-support",
            ),
            "must normalize",
        ),
        (
            lambda ledger: replace(
                ledger,
                selections=(
                    replace(ledger.selections[0], hole_log_probabilities=()),
                    ledger.selections[1],
                ),
            ),
            "one factor per selected hole",
        ),
        (
            lambda ledger: replace(
                ledger,
                program_scores=(
                    replace(
                        ledger.program_scores[0],
                        boundaries=(
                            replace(
                                ledger.program_scores[0].boundaries[0],
                                max_abs_prefix_logprob_delta=-0.1,
                            ),
                            ledger.program_scores[0].boundaries[1],
                        ),
                    ),
                ),
            ),
            "prefix logprob delta must be finite and nonnegative",
        ),
    ),
)
def test_semantic_ledger_rejects_tampering(
    mutation: Callable[[SemanticProposalLedger], SemanticProposalLedger],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        mutation(_ledger())

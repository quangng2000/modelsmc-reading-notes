"""Strict extraction of the single-token answer-label boundary."""

from __future__ import annotations

import hashlib
import json

from modelsmc_pbe.proposals.base import ProposalError
from modelsmc_pbe.proposals.candidate_scoring import CandidateSequenceScore
from modelsmc_pbe.proposals.labels.contracts import (
    CompatibilityLabel,
    CompatibilityMapping,
    CompatibilityPathSpec,
    LabelBoundaryProof,
    LabelPathEvidence,
)


class LabelBoundaryError(ProposalError):
    """A provider path did not isolate one final A/B token."""


def extract_label_pair(
    mapping: CompatibilityMapping,
    label_a_spec: CompatibilityPathSpec,
    label_b_spec: CompatibilityPathSpec,
    label_a_score: CandidateSequenceScore,
    label_b_score: CandidateSequenceScore,
) -> tuple[LabelPathEvidence, LabelPathEvidence, LabelBoundaryProof]:
    """Validate and retain one A/B path pair differing only at its final token."""

    _validate_spec(mapping, CompatibilityLabel.A, label_a_spec, label_a_score)
    _validate_spec(mapping, CompatibilityLabel.B, label_b_spec, label_b_score)
    if len(label_a_score.token_ids) != len(label_b_score.token_ids):
        raise LabelBoundaryError("mapped A/B paths must have equal scored-token lengths")
    if not label_a_score.token_ids:
        raise LabelBoundaryError("mapped A/B paths must include a scored final label token")
    if label_a_score.token_ids[:-1] != label_b_score.token_ids[:-1]:
        raise LabelBoundaryError("mapped A/B paths may differ only at the final token ID")
    if label_a_score.token_ids[-1] == label_b_score.token_ids[-1]:
        raise LabelBoundaryError("mapped A/B labels must have distinct final token IDs")
    if label_a_score.token_logprobs[:-1] != label_b_score.token_logprobs[:-1]:
        raise LabelBoundaryError("mapped A/B paths must have aligned shared-token logprobs")
    evidence_a = _evidence(label_a_spec, label_a_score)
    evidence_b = _evidence(label_b_spec, label_b_score)
    proof = LabelBoundaryProof(
        mapping=mapping,
        shared_token_count=len(label_a_score.token_ids) - 1,
        shared_token_ids_sha256=_sequence_sha256(label_a_score.token_ids[:-1]),
        shared_token_logprobs_sha256=_sequence_sha256(
            label_a_score.token_logprobs[:-1]
        ),
        label_a_token_id=label_a_score.token_ids[-1],
        label_b_token_id=label_b_score.token_ids[-1],
        label_a_logprob=label_a_score.token_logprobs[-1],
        label_b_logprob=label_b_score.token_logprobs[-1],
    )
    return evidence_a, evidence_b, proof


def _validate_spec(
    mapping: CompatibilityMapping,
    label: CompatibilityLabel,
    spec: CompatibilityPathSpec,
    score: CandidateSequenceScore,
) -> None:
    if spec.mapping is not mapping or spec.label is not label:
        raise LabelBoundaryError("label path specifications are not in canonical order")
    if score.candidate != spec.candidate:
        raise LabelBoundaryError("provider label path does not match its requested candidate")


def _evidence(
    spec: CompatibilityPathSpec,
    score: CandidateSequenceScore,
) -> LabelPathEvidence:
    return LabelPathEvidence(
        mapping=spec.mapping,
        label=spec.label,
        candidate_sha256=hashlib.sha256(score.candidate.encode()).hexdigest(),
        scored_token_count=len(score.token_ids),
        prefix_token_count=len(score.token_ids) - 1,
        token_ids_sha256=_sequence_sha256(score.token_ids),
        token_logprobs_sha256=_sequence_sha256(score.token_logprobs),
        prefix_token_ids_sha256=_sequence_sha256(score.token_ids[:-1]),
        prefix_token_logprobs_sha256=_sequence_sha256(score.token_logprobs[:-1]),
        final_token_id=score.token_ids[-1],
        final_logprob=score.token_logprobs[-1],
        sequence_logprob=score.sequence_logprob,
    )


def _sequence_sha256(values: tuple[int, ...] | tuple[float, ...]) -> str:
    encoded = json.dumps(
        values,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()

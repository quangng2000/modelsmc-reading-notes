"""Batch identity and de-duplication for finite scoring requests."""

from __future__ import annotations

import hashlib

from modelsmc_pbe.proposals import CandidateScoreRequest


def request_batch_sha256(requests: list[CandidateScoreRequest]) -> str:
    digest = hashlib.sha256()
    for request in requests:
        digest.update(request.prompt_prefix.encode("utf-8"))
        digest.update(b"\0")
        for candidate in request.candidates:
            digest.update(candidate.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def deduplicate_requests(
    requests: list[CandidateScoreRequest],
) -> tuple[list[CandidateScoreRequest], tuple[int, ...]]:
    """Collapse identical scorer work while retaining every path result."""

    unique: list[CandidateScoreRequest] = []
    positions: dict[tuple[object, ...], int] = {}
    inverse: list[int] = []
    for request in requests:
        key = (
            request.prompt_prefix,
            request.candidates,
            request.hole,
            request.integer_constants,
            request.max_depth,
            request.max_nodes,
            request.candidate_kind,
        )
        position = positions.get(key)
        if position is None:
            position = len(unique)
            positions[key] = position
            unique.append(request)
        inverse.append(position)
    return unique, tuple(inverse)

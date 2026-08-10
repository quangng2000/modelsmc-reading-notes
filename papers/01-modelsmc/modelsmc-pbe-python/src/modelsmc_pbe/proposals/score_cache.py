"""Immutable, content-addressed replay cache for finite candidate scores."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import cast

from modelsmc_pbe.proposals.base import ProposalError
from modelsmc_pbe.proposals.candidate_scoring import (
    CandidateKind,
    CandidateLogprobSemantics,
    CandidateScoreBatch,
    CandidateScoreOrigin,
    CandidateScoreProvenance,
    CandidateScorer,
    CandidateScoreRequest,
    CandidateSequenceScore,
    LLMEnergyNormalization,
    ProviderMetricSource,
    ProviderScoreMetrics,
)
from modelsmc_pbe.proposals.json_extract import parse_canonical_expression_content

_SCHEMA_VERSION = 1


class ScoreCacheMode(StrEnum):
    """Allowed persistent-cache policies."""

    OFF = "off"
    READ_WRITE = "read-write"
    REPLAY_ONLY = "replay-only"


class ScoreCacheError(ProposalError):
    """Base class for cache failures that must stop the experiment."""


class ScoreCacheMissError(ScoreCacheError):
    """A replay-only run requested evidence absent from the cache."""


class ScoreCacheCorruptionError(ScoreCacheError):
    """A cache entry exists but cannot be trusted or reconstructed."""


@dataclass(frozen=True, slots=True)
class ScoreCacheIdentity:
    """Provider facts that can change teacher-forced token scores."""

    scorer_name: str
    model_alias: str
    model_repository: str
    model_revision: str
    tokenizer_revision: str
    server_config: str
    semantics: CandidateLogprobSemantics
    energy_normalization: LLMEnergyNormalization
    add_special_tokens: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("scorer_name", self.scorer_name),
            ("model_alias", self.model_alias),
            ("model_repository", self.model_repository),
            ("model_revision", self.model_revision),
            ("tokenizer_revision", self.tokenizer_revision),
            ("server_config", self.server_config),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.semantics, CandidateLogprobSemantics):
            raise TypeError("semantics must be CandidateLogprobSemantics")
        if not isinstance(self.energy_normalization, LLMEnergyNormalization):
            raise TypeError("energy_normalization must be LLMEnergyNormalization")
        if not isinstance(self.add_special_tokens, bool):
            raise TypeError("add_special_tokens must be a boolean")


@dataclass(frozen=True, slots=True)
class ScoreCacheMetrics:
    """Cumulative cache/provider accounting for one scorer instance."""

    lookup_requests: int
    lookup_candidates: int
    hit_requests: int
    hit_candidates: int
    miss_requests: int
    miss_candidates: int
    provider_invocations: int
    provider_score_requests: int
    provider_candidates: int
    provider_failures: int
    cache_served_scored_tokens: int
    provider_scored_tokens: int
    provider_await_wall_seconds: float


@dataclass(frozen=True, slots=True)
class ScoreCacheProvenance:
    """Cache disposition for one exact finite-score request."""

    key_sha256: str
    hit: bool


@dataclass(slots=True)
class _MutableMetrics:
    lookup_requests: int = 0
    lookup_candidates: int = 0
    hit_requests: int = 0
    hit_candidates: int = 0
    miss_requests: int = 0
    miss_candidates: int = 0
    provider_invocations: int = 0
    provider_score_requests: int = 0
    provider_candidates: int = 0
    provider_failures: int = 0
    cache_served_scored_tokens: int = 0
    provider_scored_tokens: int = 0
    provider_await_wall_seconds: float = 0.0

    def snapshot(self) -> ScoreCacheMetrics:
        return ScoreCacheMetrics(
            lookup_requests=self.lookup_requests,
            lookup_candidates=self.lookup_candidates,
            hit_requests=self.hit_requests,
            hit_candidates=self.hit_candidates,
            miss_requests=self.miss_requests,
            miss_candidates=self.miss_candidates,
            provider_invocations=self.provider_invocations,
            provider_score_requests=self.provider_score_requests,
            provider_candidates=self.provider_candidates,
            provider_failures=self.provider_failures,
            cache_served_scored_tokens=self.cache_served_scored_tokens,
            provider_scored_tokens=self.provider_scored_tokens,
            provider_await_wall_seconds=self.provider_await_wall_seconds,
        )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _identity_record(identity: ScoreCacheIdentity) -> dict[str, object]:
    return {
        "scorer_name": identity.scorer_name,
        "model_alias": identity.model_alias,
        "model_repository": identity.model_repository,
        "model_revision": identity.model_revision,
        "tokenizer_revision": identity.tokenizer_revision,
        "server_config": identity.server_config,
        "semantics": identity.semantics.value,
        "energy_normalization": identity.energy_normalization.value,
        "add_special_tokens": identity.add_special_tokens,
    }


def _hole_record(request: CandidateScoreRequest) -> dict[str, object] | None:
    hole = request.hole
    if hole is None:
        return None
    return {
        "expected_type": hole.expected_type.value,
        "scope": {
            "input_type": hole.scope.input_type.value,
            "input_available": hole.scope.input_available,
            "item_type": (None if hole.scope.item_type is None else hole.scope.item_type.value),
            "accumulator_type": (
                None if hole.scope.accumulator_type is None else hole.scope.accumulator_type.value
            ),
        },
    }


def _request_key(identity: ScoreCacheIdentity, request: CandidateScoreRequest) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "identity": _identity_record(identity),
        "prompt_prefix": request.prompt_prefix,
        "canonical_candidates": list(request.candidates),
        "candidate_kind": request.candidate_kind.value,
        "validation_context": {
            "integer_constants": list(request.integer_constants),
            "max_depth": request.max_depth,
            "max_nodes": request.max_nodes,
            "hole": _hole_record(request),
        },
    }


class CachedCandidateScorer:
    """Replay complete finite score batches or atomically add immutable evidence."""

    def __init__(
        self,
        scorer: CandidateScorer,
        *,
        cache_dir: Path,
        mode: ScoreCacheMode,
        identity: ScoreCacheIdentity,
    ) -> None:
        if mode is ScoreCacheMode.OFF:
            raise ValueError("CachedCandidateScorer requires a persistent cache mode")
        self._scorer = scorer
        self.name = scorer.name
        self.cache_dir = cache_dir.expanduser().resolve()
        self.mode = mode
        self.identity = identity
        self._metrics = _MutableMetrics()
        self._provenance: dict[str, ScoreCacheProvenance] = {}
        self._lock = asyncio.Lock()

    @property
    def source_name(self) -> str:
        """Expose the provider source retained in replayed score batches."""

        return self.identity.scorer_name

    def metrics(self) -> ScoreCacheMetrics:
        """Return an immutable snapshot of all cache/provider work so far."""

        return self._metrics.snapshot()

    def provider_metrics(self) -> ProviderScoreMetrics:
        """Return underlying HTTP telemetry, or explicit zeros for local scorers."""

        if isinstance(self._scorer, ProviderMetricSource):
            return self._scorer.provider_metrics()
        return ProviderScoreMetrics(0, 0, 0, 0.0)

    def provenance(self, request: CandidateScoreRequest) -> ScoreCacheProvenance | None:
        """Return the latest disposition for an exact request, if it was scored."""

        digest = self.key_sha256(request)
        return self._provenance.get(digest)

    def key_sha256(self, request: CandidateScoreRequest) -> str:
        """Return the stable content address for an exact scoring contract."""

        return _sha256(_request_key(self.identity, request))

    def entry_path(self, request: CandidateScoreRequest) -> Path:
        """Return the immutable entry path used for one request."""

        digest = self.key_sha256(request)
        return self.cache_dir / f"v{_SCHEMA_VERSION}" / digest[:2] / f"{digest}.json"

    async def score_candidates(self, request: CandidateScoreRequest) -> CandidateScoreBatch:
        return (await self.score_many([request]))[0]

    async def score_many(self, requests: list[CandidateScoreRequest]) -> list[CandidateScoreBatch]:
        if not requests:
            return []
        async with self._lock:
            return await self._score_many_locked(requests)

    async def _score_many_locked(
        self, requests: list[CandidateScoreRequest]
    ) -> list[CandidateScoreBatch]:
        results: list[CandidateScoreBatch | None] = [None] * len(requests)
        misses: dict[str, tuple[CandidateScoreRequest, list[int]]] = {}
        for index, request in enumerate(requests):
            self._metrics.lookup_requests += 1
            self._metrics.lookup_candidates += len(request.candidates)
            digest = self.key_sha256(request)
            path = self.entry_path(request)
            if path.exists():
                batch = replace(
                    self._load(path, digest, request),
                    provenance=CandidateScoreProvenance(
                        CandidateScoreOrigin.CACHE,
                        cache_key_sha256=digest,
                        cache_hit=True,
                    ),
                )
                results[index] = batch
                self._metrics.hit_requests += 1
                self._metrics.hit_candidates += len(request.candidates)
                self._metrics.cache_served_scored_tokens += _scored_tokens(batch)
                self._provenance[digest] = ScoreCacheProvenance(digest, True)
                continue
            self._metrics.miss_requests += 1
            self._metrics.miss_candidates += len(request.candidates)
            self._provenance[digest] = ScoreCacheProvenance(digest, False)
            if digest in misses:
                misses[digest][1].append(index)
            else:
                misses[digest] = (request, [index])

        if misses and self.mode is ScoreCacheMode.REPLAY_ONLY:
            missing = ", ".join(sorted(misses)[:3])
            raise ScoreCacheMissError(
                f"replay-only candidate-score cache miss ({len(misses)} entries): {missing}"
            )
        if misses:
            await self._fetch_and_store(misses, results)
        if any(batch is None for batch in results):  # pragma: no cover - internal invariant
            raise RuntimeError("candidate-score cache did not resolve every request")
        return [cast(CandidateScoreBatch, batch) for batch in results]

    async def _fetch_and_store(
        self,
        misses: Mapping[str, tuple[CandidateScoreRequest, list[int]]],
        results: list[CandidateScoreBatch | None],
    ) -> None:
        unique = [request for request, _ in misses.values()]
        self._metrics.provider_invocations += 1
        self._metrics.provider_score_requests += len(unique)
        self._metrics.provider_candidates += sum(len(request.candidates) for request in unique)
        started = time.perf_counter()
        try:
            batches = await self._scorer.score_many(unique)
        except Exception:
            self._metrics.provider_failures += 1
            raise
        finally:
            self._metrics.provider_await_wall_seconds += time.perf_counter() - started
        if len(batches) != len(unique):
            raise ProposalError("candidate scorer did not return one batch per cache miss")
        for (digest, (request, positions)), batch in zip(misses.items(), batches, strict=True):
            self._validate_batch(request, batch)
            self._metrics.provider_scored_tokens += _scored_tokens(batch)
            self._store(self.entry_path(request), digest, request, batch)
            batch = replace(
                batch,
                provenance=CandidateScoreProvenance(
                    CandidateScoreOrigin.PROVIDER,
                    cache_key_sha256=digest,
                    cache_hit=False,
                ),
            )
            for position in positions:
                results[position] = batch

    def _validate_batch(self, request: CandidateScoreRequest, batch: CandidateScoreBatch) -> None:
        if tuple(score.candidate for score in batch.scores) != request.candidates:
            raise ProposalError("candidate scorer reordered or changed the cache request")
        if batch.source != self.identity.scorer_name:
            raise ProposalError("candidate scorer source does not match cache identity")
        if batch.model != self.identity.model_alias:
            raise ProposalError("candidate scorer model does not match cache identity")
        if batch.model_revision != self.identity.model_revision:
            raise ProposalError("candidate scorer model revision does not match cache identity")
        if batch.tokenizer_revision != self.identity.tokenizer_revision:
            raise ProposalError("candidate scorer tokenizer revision does not match cache identity")
        if batch.semantics is not self.identity.semantics:
            raise ProposalError("candidate scorer semantics do not match cache identity")

    def _store(
        self,
        path: Path,
        digest: str,
        request: CandidateScoreRequest,
        batch: CandidateScoreBatch,
    ) -> None:
        body: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "key_sha256": digest,
            "key": _request_key(self.identity, request),
            "batch": {
                "source": batch.source,
                "model": batch.model,
                "semantics": batch.semantics.value,
                "model_revision": batch.model_revision,
                "tokenizer_revision": batch.tokenizer_revision,
                "scores": [
                    {
                        "canonical_candidate": score.candidate,
                        "token_ids": list(score.token_ids),
                        "token_logprobs": list(score.token_logprobs),
                        "sequence_logprob": score.sequence_logprob,
                    }
                    for score in batch.scores
                ],
            },
        }
        body["content_sha256"] = _sha256(body)
        encoded = _canonical_bytes(body) + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{digest}.", suffix=".tmp", dir=path.parent
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_name, path)
                _fsync_directory(path.parent)
            except FileExistsError:
                # A concurrent writer won. Never replace it; trust it only after
                # the same strict validation used for ordinary reads.
                self._load(path, digest, request)
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

    def _load(self, path: Path, digest: str, request: CandidateScoreRequest) -> CandidateScoreBatch:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ScoreCacheCorruptionError(f"cannot read score cache entry {path}") from error
        if not isinstance(raw, dict):
            raise ScoreCacheCorruptionError(f"score cache entry is not an object: {path}")
        expected_fields = {
            "schema_version",
            "key_sha256",
            "key",
            "batch",
            "content_sha256",
        }
        if set(raw) != expected_fields:
            raise ScoreCacheCorruptionError(f"score cache entry fields are invalid: {path}")
        checksum = raw.get("content_sha256")
        unsigned = {key: value for key, value in raw.items() if key != "content_sha256"}
        if not isinstance(checksum, str) or checksum != _sha256(unsigned):
            raise ScoreCacheCorruptionError(f"score cache entry checksum mismatch: {path}")
        expected_key = _request_key(self.identity, request)
        if (
            raw.get("schema_version") != _SCHEMA_VERSION
            or raw.get("key_sha256") != digest
            or raw.get("key") != expected_key
            or _sha256(expected_key) != digest
        ):
            raise ScoreCacheCorruptionError(f"score cache entry key mismatch: {path}")
        batch = self._decode_batch(raw.get("batch"), request, path)
        try:
            self._validate_batch(request, batch)
        except ProposalError as error:
            raise ScoreCacheCorruptionError(
                f"score cache entry provider metadata mismatch: {path}"
            ) from error
        return batch

    def _decode_batch(
        self, value: object, request: CandidateScoreRequest, path: Path
    ) -> CandidateScoreBatch:
        try:
            record = _mapping(value)
            if set(record) != {
                "source",
                "model",
                "semantics",
                "model_revision",
                "tokenizer_revision",
                "scores",
            }:
                raise ValueError("invalid batch fields")
            raw_scores = _sequence(record["scores"])
            if len(raw_scores) != len(request.candidates):
                raise ValueError("score count differs from candidate count")
            scores = tuple(
                self._decode_score(raw_score, request, index)
                for index, raw_score in enumerate(raw_scores)
            )
            return CandidateScoreBatch(
                scores=scores,
                source=_string(record["source"]),
                model=_string(record["model"]),
                semantics=CandidateLogprobSemantics(_string(record["semantics"])),
                model_revision=_optional_string(record["model_revision"]),
                tokenizer_revision=_optional_string(record["tokenizer_revision"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ScoreCacheCorruptionError(
                f"score cache batch cannot be reconstructed: {path}"
            ) from error

    def _decode_score(
        self, value: object, request: CandidateScoreRequest, index: int
    ) -> CandidateSequenceScore:
        record = _mapping(value)
        if set(record) != {
            "canonical_candidate",
            "token_ids",
            "token_logprobs",
            "sequence_logprob",
        }:
            raise ValueError("invalid score fields")
        candidate = _string(record["canonical_candidate"])
        if candidate != request.candidates[index]:
            raise ValueError("cached candidate order differs from request")
        token_ids = tuple(_integer(item) for item in _sequence(record["token_ids"]))
        token_logprobs = tuple(_number(item) for item in _sequence(record["token_logprobs"]))
        expression = (
            parse_canonical_expression_content(candidate, request)
            if request.candidate_kind is CandidateKind.EXPRESSION
            else None
        )
        return CandidateSequenceScore(
            candidate=candidate,
            expression=expression,
            token_ids=token_ids,
            token_logprobs=token_logprobs,
            sequence_logprob=_number(record["sequence_logprob"]),
        )


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("expected an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("expected an array")
    return cast(Sequence[object], value)


def _string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("expected a nonempty string")
    return value


def _optional_string(value: object) -> str | None:
    return None if value is None else _string(value)


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("expected a nonnegative integer")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed > 0:
        raise ValueError("expected a finite nonpositive number")
    return parsed


def _scored_tokens(batch: CandidateScoreBatch) -> int:
    return sum(len(score.token_logprobs) for score in batch.scores)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

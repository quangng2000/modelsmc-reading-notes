"""Serializable finite-score evidence and independent Qwen categorical replay."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from modelsmc_pbe.proposals import LLMEnergyNormalization


@dataclass(frozen=True, slots=True)
class LLMScoreCandidateLedger:
    """One canonical choice and all values needed to replay its local mass."""

    canonical_candidate: str
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    scored_token_count: int
    total_sequence_logprob: float
    normalized_energy: float
    qwen_probability: float
    deduction_probability: float
    proposal_probability: float


@dataclass(frozen=True, slots=True)
class LLMScoreSelectionLedger:
    """One sampled or forced use of a deduplicated finite score request."""

    slot: int
    ancestor_state_index: int | None
    selected_index: int
    selected_probability: float
    selected_qwen_probability: float
    selected_deduction_probability: float
    forced: bool
    cloned: bool


@dataclass(frozen=True, slots=True)
class LLMScoreWaveLedger:
    """Compact, reconstructible evidence for one unique scored request in a wave."""

    stage: int
    beta: float
    wave: str
    request_index: int
    prompt_prefix: str
    prompt_prefix_sha256: str
    candidate_kind: str
    source: str
    model: str
    model_revision: str | None
    tokenizer_revision: str | None
    score_semantics: str
    energy_normalization: LLMEnergyNormalization
    temperature: float
    proposal_epsilon: float
    deduction_mix: float
    candidates: tuple[LLMScoreCandidateLedger, ...]
    selections: tuple[LLMScoreSelectionLedger, ...]


def prompt_prefix_sha256(prompt_prefix: str) -> str:
    """Return the stable digest stored beside a reconstructible prefix."""

    return hashlib.sha256(prompt_prefix.encode("utf-8")).hexdigest()


def _field(record: Mapping[str, Any], name: str) -> Any:
    try:
        return record[name]
    except KeyError as error:
        raise ValueError(f"score ledger is missing {name!r}") from error


def _float_sequence(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"score ledger {name} must be an array")
    parsed: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"score ledger {name} must contain only numbers")
        parsed.append(float(item))
    return tuple(parsed)


def _candidate_energy(
    token_logprobs: tuple[float, ...],
    normalization: LLMEnergyNormalization,
) -> float:
    total = math.fsum(token_logprobs)
    if normalization is LLMEnergyNormalization.TOTAL_FULL_PROMPT_LOGPROB:
        return total
    if normalization is LLMEnergyNormalization.MEAN_FULL_PROMPT_CONDITIONAL_LOGPROB:
        return 0.0 if not token_logprobs else total / len(token_logprobs)
    raise ValueError(f"unsupported score-ledger energy normalization {normalization!r}")


def replay_qwen_categorical(
    ledger: LLMScoreWaveLedger | Mapping[str, Any],
) -> tuple[float, ...]:
    """Validate a score ledger and replay Qwen's pre-mixture categorical.

    A mapping may come directly from a JSON round trip.  Replay recomputes each
    total and normalized energy from the persisted full-prompt token logprobs,
    then applies the persisted positive temperature.  It does not trust the
    stored categorical probabilities.
    """

    if isinstance(ledger, LLMScoreWaveLedger):
        prompt_prefix = ledger.prompt_prefix
        prompt_digest = ledger.prompt_prefix_sha256
        normalization = ledger.energy_normalization
        temperature = ledger.temperature
        candidates: Sequence[LLMScoreCandidateLedger | Mapping[str, Any]] = ledger.candidates
    else:
        raw_prefix = _field(ledger, "prompt_prefix")
        raw_digest = _field(ledger, "prompt_prefix_sha256")
        raw_normalization = _field(ledger, "energy_normalization")
        raw_temperature = _field(ledger, "temperature")
        raw_candidates = _field(ledger, "candidates")
        if not isinstance(raw_prefix, str) or not isinstance(raw_digest, str):
            raise ValueError("score ledger prompt prefix and digest must be strings")
        if not isinstance(raw_candidates, Sequence) or isinstance(
            raw_candidates, (str, bytes, bytearray)
        ):
            raise ValueError("score ledger candidates must be an array")
        if isinstance(raw_temperature, bool) or not isinstance(
            raw_temperature, (int, float)
        ):
            raise ValueError("score ledger temperature must be a number")
        prompt_prefix = raw_prefix
        prompt_digest = raw_digest
        normalization = LLMEnergyNormalization(raw_normalization)
        temperature = float(raw_temperature)
        candidates = cast(
            Sequence[LLMScoreCandidateLedger | Mapping[str, Any]], raw_candidates
        )

    if prompt_prefix_sha256(prompt_prefix) != prompt_digest:
        raise ValueError("score ledger prompt-prefix digest does not match its prefix")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("score ledger temperature must be finite and positive")
    if not candidates:
        raise ValueError("score ledger must contain at least one candidate")

    energies: list[float] = []
    stored_probabilities: list[float] = []
    for candidate in candidates:
        if isinstance(candidate, LLMScoreCandidateLedger):
            token_logprobs = candidate.token_logprobs
            scored_token_count = candidate.scored_token_count
            total_sequence_logprob = candidate.total_sequence_logprob
            normalized_energy = candidate.normalized_energy
            qwen_probability = candidate.qwen_probability
        elif isinstance(candidate, Mapping):
            token_logprobs = _float_sequence(
                _field(candidate, "token_logprobs"), "candidate token_logprobs"
            )
            scored_token_count = int(_field(candidate, "scored_token_count"))
            total_sequence_logprob = float(_field(candidate, "total_sequence_logprob"))
            normalized_energy = float(_field(candidate, "normalized_energy"))
            qwen_probability = float(_field(candidate, "qwen_probability"))
        else:
            raise ValueError("score ledger candidate must be an object")
        if scored_token_count != len(token_logprobs):
            raise ValueError("score ledger scored-token count is inconsistent")
        total = math.fsum(token_logprobs)
        if not math.isclose(total, total_sequence_logprob, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("score ledger total logprob is inconsistent")
        energy = _candidate_energy(token_logprobs, normalization)
        if not math.isclose(energy, normalized_energy, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("score ledger normalized energy is inconsistent")
        energies.append(energy)
        stored_probabilities.append(qwen_probability)

    logits = [energy / temperature for energy in energies]
    maximum = max(logits)
    unnormalized = [math.exp(logit - maximum) for logit in logits]
    total_mass = math.fsum(unnormalized)
    probabilities = tuple(value / total_mass for value in unnormalized)
    if any(
        not math.isclose(actual, stored, rel_tol=1e-12, abs_tol=1e-12)
        for actual, stored in zip(probabilities, stored_probabilities, strict=True)
    ):
        raise ValueError("score ledger Qwen probabilities do not replay")
    return probabilities


__all__ = [
    "LLMScoreCandidateLedger",
    "LLMScoreSelectionLedger",
    "LLMScoreWaveLedger",
    "prompt_prefix_sha256",
    "replay_qwen_categorical",
]

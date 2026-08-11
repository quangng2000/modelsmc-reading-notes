"""Provider-neutral contracts for contrastive semantic compatibility scores."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import StrEnum

from modelsmc_pbe.proposals.candidate_scoring import (
    CandidateLogprobSemantics,
    CandidateScoreProvenance,
)


class CompatibilityLabel(StrEnum):
    """The two deliberately content-free answer labels."""

    A = "A"
    B = "B"


class CompatibilityMapping(StrEnum):
    """Which answer label denotes semantic compatibility on one path pair."""

    A_IS_COMPATIBLE = "a-is-compatible"
    B_IS_COMPATIBLE = "b-is-compatible"


@dataclass(frozen=True, slots=True)
class CompatibilityProgram:
    """One unique complete program to assess without executing it."""

    program_key: str
    program_text: str

    def __post_init__(self) -> None:
        if not self.program_key.strip() or not self.program_text.strip():
            raise ValueError("program_key and program_text must not be empty")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.program_text.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CompatibilityPathSpec:
    """Expected mapping, final label, and complete candidate continuation."""

    mapping: CompatibilityMapping
    label: CompatibilityLabel
    candidate: str


@dataclass(frozen=True, slots=True)
class LabelPathEvidence:
    """Compact digest of one raw teacher-forced label path."""

    mapping: CompatibilityMapping
    label: CompatibilityLabel
    candidate_sha256: str
    scored_token_count: int
    prefix_token_count: int
    token_ids_sha256: str
    token_logprobs_sha256: str
    prefix_token_ids_sha256: str
    prefix_token_logprobs_sha256: str
    final_token_id: int
    final_logprob: float
    sequence_logprob: float

    def __post_init__(self) -> None:
        if self.scored_token_count < 1:
            raise ValueError("label evidence requires at least one scored token")
        if self.prefix_token_count != self.scored_token_count - 1:
            raise ValueError("label prefix must contain every scored token except the label")
        for name, digest in (
            ("candidate_sha256", self.candidate_sha256),
            ("token_ids_sha256", self.token_ids_sha256),
            ("token_logprobs_sha256", self.token_logprobs_sha256),
            ("prefix_token_ids_sha256", self.prefix_token_ids_sha256),
            ("prefix_token_logprobs_sha256", self.prefix_token_logprobs_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.final_token_id < 0:
            raise ValueError("final token ID must be nonnegative")
        if any(
            not math.isfinite(value) or value > 0
            for value in (self.final_logprob, self.sequence_logprob)
        ):
            raise ValueError("label path logprobs must be finite and nonpositive")


@dataclass(frozen=True, slots=True)
class LabelBoundaryProof:
    """Shared token-ID boundary with separate raw prefix-score commitments."""

    mapping: CompatibilityMapping
    shared_token_count: int
    shared_token_ids_sha256: str
    label_a_prefix_token_logprobs_sha256: str
    label_b_prefix_token_logprobs_sha256: str
    max_abs_prefix_logprob_delta: float
    label_a_token_id: int
    label_b_token_id: int
    label_a_logprob: float
    label_b_logprob: float

    def __post_init__(self) -> None:
        if self.label_a_token_id == self.label_b_token_id:
            raise ValueError("final A and B labels must have distinct token IDs")
        if self.shared_token_count < 0:
            raise ValueError("shared token count must be nonnegative")
        for digest in (
            self.shared_token_ids_sha256,
            self.label_a_prefix_token_logprobs_sha256,
            self.label_b_prefix_token_logprobs_sha256,
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("shared token evidence must use lowercase SHA-256")
        if (
            not math.isfinite(self.max_abs_prefix_logprob_delta)
            or self.max_abs_prefix_logprob_delta < 0
        ):
            raise ValueError("prefix logprob delta must be finite and nonnegative")
        if (
            self.label_a_prefix_token_logprobs_sha256 == self.label_b_prefix_token_logprobs_sha256
            and self.max_abs_prefix_logprob_delta != 0.0
        ):
            raise ValueError("identical prefix logprob commitments require zero delta")
        if any(
            not math.isfinite(value) or value > 0
            for value in (self.label_a_logprob, self.label_b_logprob)
        ):
            raise ValueError("final-label logprobs must be finite and nonpositive")


@dataclass(frozen=True, slots=True)
class RawCompatibilityScoreIdentity:
    """Provider identity attached to the four raw paths behind one score."""

    source: str
    model: str
    semantics: CandidateLogprobSemantics
    model_revision: str | None
    tokenizer_revision: str | None
    provenance: CandidateScoreProvenance | None


@dataclass(frozen=True, slots=True)
class CompatibilityScore:
    """Symmetrized label-score contrast plus commitments to raw path evidence."""

    program_key: str
    program_sha256: str
    compatibility_log_odds: float
    paths: tuple[LabelPathEvidence, ...]
    boundary_proofs: tuple[LabelBoundaryProof, ...]
    template_version: str
    prompt_sha256: str
    raw_identity: RawCompatibilityScoreIdentity

    def __post_init__(self) -> None:
        if not self.program_key.strip() or not self.template_version.strip():
            raise ValueError("program_key and template_version must not be empty")
        for name, digest in (
            ("program_sha256", self.program_sha256),
            ("prompt_sha256", self.prompt_sha256),
        ):
            invalid_character = any(character not in "0123456789abcdef" for character in digest)
            if len(digest) != 64 or invalid_character:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not math.isfinite(self.compatibility_log_odds):
            raise ValueError("compatibility_log_odds must be finite")
        expected_paths = (
            (CompatibilityMapping.A_IS_COMPATIBLE, CompatibilityLabel.A),
            (CompatibilityMapping.A_IS_COMPATIBLE, CompatibilityLabel.B),
            (CompatibilityMapping.B_IS_COMPATIBLE, CompatibilityLabel.A),
            (CompatibilityMapping.B_IS_COMPATIBLE, CompatibilityLabel.B),
        )
        if tuple((path.mapping, path.label) for path in self.paths) != expected_paths:
            raise ValueError("compatibility score requires the four canonical raw paths")
        if tuple(proof.mapping for proof in self.boundary_proofs) != (
            CompatibilityMapping.A_IS_COMPATIBLE,
            CompatibilityMapping.B_IS_COMPATIBLE,
        ):
            raise ValueError("compatibility score requires one proof per swapped mapping")
        for index, proof in enumerate(self.boundary_proofs):
            label_a = self.paths[2 * index]
            label_b = self.paths[2 * index + 1]
            if (
                label_a.prefix_token_count != proof.shared_token_count
                or label_b.prefix_token_count != proof.shared_token_count
                or label_a.prefix_token_ids_sha256 != proof.shared_token_ids_sha256
                or label_b.prefix_token_ids_sha256 != proof.shared_token_ids_sha256
                or label_a.prefix_token_logprobs_sha256
                != proof.label_a_prefix_token_logprobs_sha256
                or label_b.prefix_token_logprobs_sha256
                != proof.label_b_prefix_token_logprobs_sha256
                or label_a.final_token_id != proof.label_a_token_id
                or label_b.final_token_id != proof.label_b_token_id
                or label_a.final_logprob != proof.label_a_logprob
                or label_b.final_logprob != proof.label_b_logprob
            ):
                raise ValueError("label paths and boundary proof are inconsistent")
        reconstructed = 0.5 * (
            self.boundary_proofs[0].label_a_logprob
            - self.boundary_proofs[0].label_b_logprob
            + self.boundary_proofs[1].label_b_logprob
            - self.boundary_proofs[1].label_a_logprob
        )
        if not math.isclose(
            self.compatibility_log_odds,
            reconstructed,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("compatibility score disagrees with its label evidence")


@dataclass(frozen=True, slots=True)
class CompatibilityScoreBatch:
    """Semantic scores corresponding one-for-one with requested programs."""

    scores: tuple[CompatibilityScore, ...]
    template_version: str
    prompt_sha256: str

    def __post_init__(self) -> None:
        if not self.scores:
            raise ValueError("compatibility score batch must not be empty")
        if len({score.program_key for score in self.scores}) != len(self.scores):
            raise ValueError("compatibility score program keys must be unique")
        if any(score.template_version != self.template_version for score in self.scores):
            raise ValueError("compatibility score template versions must agree")
        if any(score.prompt_sha256 != self.prompt_sha256 for score in self.scores):
            raise ValueError("compatibility score prompt digests must agree")

    @property
    def raw_candidate_count(self) -> int:
        return 4 * len(self.scores)


@dataclass(frozen=True, slots=True)
class CompatibilityScoringRequest:
    """A dataset context and unique program slate for semantic assessment."""

    dataset_context: str
    programs: tuple[CompatibilityProgram, ...]
    integer_constants: tuple[int, ...] = ()
    request_index: int = 0
    max_depth: int = 32
    max_nodes: int = 256
    raw_candidate_batch_size: int = 128
    raw_request_batch_size: int = 32

    def __post_init__(self) -> None:
        if not self.dataset_context.strip() or not self.programs:
            raise ValueError("dataset_context and programs must not be empty")
        if len({program.program_key for program in self.programs}) != len(self.programs):
            raise ValueError("program keys must be unique")
        if len({program.program_text for program in self.programs}) != len(self.programs):
            raise ValueError("program texts must be unique")
        if self.request_index < 0:
            raise ValueError("request_index must be nonnegative")
        if self.max_depth < 1 or self.max_nodes < 1:
            raise ValueError("depth and node limits must be positive")
        if (
            isinstance(self.raw_candidate_batch_size, bool)
            or not isinstance(self.raw_candidate_batch_size, int)
            or self.raw_candidate_batch_size < 4
        ):
            raise ValueError("raw_candidate_batch_size must be an integer of at least four")
        if (
            isinstance(self.raw_request_batch_size, bool)
            or not isinstance(self.raw_request_batch_size, int)
            or self.raw_request_batch_size < 1
        ):
            raise ValueError("raw_request_batch_size must be a positive integer")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.integer_constants
        ):
            raise TypeError("integer_constants must contain only integers")

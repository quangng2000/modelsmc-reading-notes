"""Lazy LLM-semantic joint proposal with an exact defensive density."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.proposals import CandidateScorer
from modelsmc_pbe.proposals.labels import (
    CompatibilityScoringRequest,
    SymmetrizedLabelCompatibilityScorer,
)

from ..factorized import trace_log_prior
from ..lazy_records import (
    FactorizedImportanceSupport,
    LazyImportanceState,
    LazyProposedTrace,
)
from ..options import ImportanceProposalBudgetExceeded, ImportanceSMCOptions
from ..score_ledger import LLMScoreWaveLedger
from .law import SemanticStageLaw
from .ledger import (
    SEMANTIC_PROPOSAL_LEDGER_SCHEMA_VERSION,
    SemanticProgramScoreLedger,
    SemanticProposalLedger,
    SemanticSelectionLedger,
    SemanticTraceProbabilityLedger,
    slate_digest,
    summarize_score,
    trace_identity,
)
from .programs import dataset_context, prepare_semantic_programs
from .slate import (
    attach_semantic_scores,
    enumerate_construction_traces,
    select_deterministic_slate,
)

JOINT_SEMANTIC_PROPOSAL_SOURCE = "joint-semantic-symmetrized-label-logscore-contrast"
type EventEmitter = Callable[..., None]


class LazyJointSemanticProposalKernel:
    """Score a complete-program slate, then sample exact joint marginals."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        options: ImportanceSMCOptions,
        support: FactorizedImportanceSupport,
        scorer: CandidateScorer,
        generator: torch.Generator,
        emit: EventEmitter | None = None,
    ) -> None:
        if generator.device.type != "cpu":
            raise ValueError("joint-semantic proposal requires a CPU generator")
        if config.smc.alpha != 0.0:
            raise ValueError("joint-semantic proposal requires alpha=0")
        if options.proposal_strategy != "joint-semantic":
            raise ValueError("joint-semantic kernel requires its proposal strategy")
        self._config = config
        self._options = options
        self._support = support
        self._scorer = SymmetrizedLabelCompatibilityScorer(
            scorer,
            prompt_protocol=options.semantic_prompt_protocol,
        )
        self._generator = generator
        self._emit = emit
        self._law: SemanticStageLaw | None = None
        self._ledger: SemanticProposalLedger | None = None
        self._selections: list[SemanticSelectionLedger] = []
        self._scored_candidates = 0

    @property
    def source(self) -> str:
        return JOINT_SEMANTIC_PROPOSAL_SOURCE

    @property
    def scored_candidates(self) -> int:
        return self._scored_candidates

    @property
    def score_ledger(self) -> tuple[LLMScoreWaveLedger, ...]:
        """The guided full-prompt categorical ledger is inactive in this mode."""

        return ()

    @property
    def semantic_score_ledger(self) -> SemanticProposalLedger:
        """Return the immutable score law plus every sequential categorical use."""

        if self._ledger is None:
            raise RuntimeError("joint-semantic slate has not been scored")
        return replace(self._ledger, selections=tuple(self._selections))

    def final_family_proposal(self) -> torch.Tensor:
        """Return the fixed defensive proposal mass of every family."""

        law = self._require_law()
        return torch.tensor(law.family_distribution().probabilities, dtype=torch.float64)

    async def sample_many(
        self,
        ancestors: tuple[LazyImportanceState, ...],
        *,
        stage: int,
        beta: float,
    ) -> tuple[LazyProposedTrace, ...]:
        await self._initialize()
        law = self._require_law()
        proposals: list[LazyProposedTrace] = []
        for slot, ancestor in enumerate(ancestors):
            path = law.sample(generator=self._generator)
            trace = path.trace
            family = self._support.family(trace.hypothesis_index)
            log_q_semantic = law.q_a_log_probability(trace)
            self._selections.append(
                SemanticSelectionLedger(
                    stage=stage,
                    beta=beta,
                    slot=slot,
                    ancestor=trace_identity(
                        ancestor.trace.hypothesis_index,
                        ancestor.trace.filling_indices,
                    ),
                    selected=trace_identity(trace.hypothesis_index, trace.filling_indices),
                    family_log_probability=path.family_log_probability,
                    hole_log_probabilities=path.hole_log_probabilities,
                    selected_log_prior=trace_log_prior(
                        self._support,
                        trace,
                        cost_scale=float(self._config.smc.cost_scale),
                    ),
                    selected_log_q_semantic=(
                        None if log_q_semantic == -math.inf else log_q_semantic
                    ),
                    log_q_proposal=path.log_probability,
                )
            )
            proposals.append(
                LazyProposedTrace(
                    trace=trace,
                    ancestor_trace=ancestor.trace,
                    log_q_construct=path.log_probability,
                    log_q_mixture=path.log_probability,
                    cloned=False,
                    family=family.hypothesis.kind.value,
                )
            )
        return tuple(proposals)

    async def _initialize(self) -> None:
        if self._law is not None:
            return
        all_traces = enumerate_construction_traces(self._support)
        traces = select_deterministic_slate(
            all_traces,
            size=self._options.semantic_slate_size,
            seed=self._config.smc.seed,
        )
        prepared = prepare_semantic_programs(
            config=self._config,
            support=self._support,
            traces=traces,
        )
        attempted = 4 * len(prepared.programs)
        if attempted > self._options.max_scored_candidates:
            raise ImportanceProposalBudgetExceeded(
                "joint-semantic label scoring would use "
                f"{attempted} raw candidates, exceeding --max-scored-candidates="
                f"{self._options.max_scored_candidates}"
            )
        self._scored_candidates = attempted
        self._event(
            "importance.lazy.semantic.scoring.started",
            message="joint semantic compatibility slate scoring started",
            level="info",
            slate_traces=len(traces),
            unique_programs=len(prepared.programs),
            raw_candidates=attempted,
        )
        batch = await self._scorer.score(
            CompatibilityScoringRequest(
                dataset_context=dataset_context(self._config),
                programs=prepared.programs,
                integer_constants=tuple(self._config.spec.integer_constants),
                max_depth=self._config.smc.max_depth,
                max_nodes=self._config.smc.max_nodes,
                raw_candidate_batch_size=self._options.semantic_candidate_batch_size,
            )
        )
        score_by_key = {score.program_key: score for score in batch.scores}
        semantic_scores = tuple(
            score_by_key[key].compatibility_log_score for key in prepared.trace_program_keys
        )
        slate = attach_semantic_scores(
            self._support,
            traces,
            semantic_scores,
            cost_scale=float(self._config.smc.cost_scale),
        )
        self._law = SemanticStageLaw(
            support=self._support,
            slate=slate,
            cost_scale=float(self._config.smc.cost_scale),
            semantic_scale=self._options.semantic_scale,
            stage_fraction=1.0,
            epsilon=self._options.proposal_epsilon,
        )
        self._ledger = self._build_ledger(
            prepared.trace_program_keys,
            score_by_key,
            batch.template_version,
            batch.prompt_sha256,
        )
        self._event(
            "importance.lazy.semantic.scoring.completed",
            message="joint semantic proposal law normalized without program execution",
            level="info",
            slate_sha256=self._ledger.slate_sha256,
            proposal_epsilon=self._options.proposal_epsilon,
            semantic_scale=self._options.semantic_scale,
        )

    def _build_ledger(
        self,
        trace_program_keys: tuple[str, ...],
        score_by_key: dict[str, Any],
        template_version: str,
        prompt_sha256: str,
    ) -> SemanticProposalLedger:
        law = self._require_law()
        program_scores: tuple[SemanticProgramScoreLedger, ...] = tuple(
            summarize_score(score_by_key[key]) for key in sorted(score_by_key)
        )
        probabilities = law.slate_probabilities
        traces = tuple(
            SemanticTraceProbabilityLedger(
                trace=trace_identity(
                    item.entry.trace.hypothesis_index,
                    item.entry.trace.filling_indices,
                ),
                program_sha256=score_by_key[key].program_sha256,
                log_prior=item.entry.log_prior,
                compatibility_log_score=item.entry.semantic_score,
                log_q_semantic=(None if item.log_q_a == -math.inf else item.log_q_a),
                log_q_proposal=item.log_q,
            )
            for item, key in zip(probabilities, trace_program_keys, strict=True)
        )
        return SemanticProposalLedger(
            schema_version=SEMANTIC_PROPOSAL_LEDGER_SCHEMA_VERSION,
            formula="q=epsilon*pi+(1-epsilon)*normalize_slate(pi*exp(eta*a_llm))",
            slate_selection=(
                "full-bounded-support"
                if len(traces) == self._support.support_states
                else "seeded-sha256-rank"
            ),
            slate_seed=self._config.smc.seed,
            support_states=self._support.support_states,
            requested_slate_size=self._options.semantic_slate_size,
            slate_traces=len(traces),
            unique_programs=len(program_scores),
            raw_candidate_count=self._scored_candidates,
            proposal_epsilon=self._options.proposal_epsilon,
            semantic_scale=self._options.semantic_scale,
            template_version=template_version,
            prompt_sha256=prompt_sha256,
            slate_sha256=slate_digest(traces),
            program_scores=program_scores,
            trace_probabilities=traces,
            selections=(),
        )

    def _require_law(self) -> SemanticStageLaw:
        if self._law is None:
            raise RuntimeError("joint-semantic slate has not been scored")
        return self._law

    def _event(self, name: str, *, message: str, level: str, **data: Any) -> None:
        if self._emit is not None:
            self._emit(name, message=message, level=level, **data)

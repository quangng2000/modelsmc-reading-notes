"""Human-readable shell views for lazy importance-SMC strategies."""

from __future__ import annotations

from modelsmc_pbe.search.importance import LazyImportanceSMCResult

from .output import ProgramResultView
from .run_records import CompletedRun


def lazy_completed_run(result: LazyImportanceSMCResult) -> CompletedRun:
    """Render strategy-specific diagnostics without weakening persisted records."""

    if result.proposal_strategy == "joint-semantic":
        details = _semantic_details(result)
        family_details = tuple(
            "family "
            f"{family.family}: traces={family.support_states} "
            f"prior={family.prior_mass:.5g} "
            f"proposal={_required(family.proposal_mass):.5g} "
            f"particles={family.particle_mass:.5g}"
            for family in result.families
        )
    else:
        details = _guided_details(result)
        family_details = tuple(
            "family "
            f"{family.family}: traces={family.support_states} "
            f"prior={family.prior_mass:.5g} "
            f"guide={_required(family.deduction_guide_mass):.5g} "
            f"particles={family.particle_mass:.5g}"
            for family in result.families
        )
    best = result.best_visited
    return CompletedRun(
        persisted_result=result,
        final_particles=result.final_particles,
        view=ProgramResultView(
            mode=result.mode,
            exact=result.exact,
            program=best.program,
            total_loss=best.total_loss,
            cost=best.cost,
            details=(
                "execution=lazy-factorized (exact reference disabled)",
                f"proposal={result.proposal_source}",
                f"support traces={result.support_states} materialized=false",
                "visited programs: "
                f"{result.search.evaluated_programs}/{result.support_states} "
                f"({result.search.evaluated_fraction_of_support:.3%})",
                "search success: "
                f"exact-found={result.search.exact_found} "
                f"final-exact-mass={result.search.final_exact_particle_mass:.7g}",
                "reference metrics=unavailable; joint-target is the external finite control",
                *details,
                *family_details,
            ),
        ),
    )


def _semantic_details(result: LazyImportanceSMCResult) -> tuple[str, ...]:
    ledger = result.semantic_score_ledger
    if ledger is None or result.semantic_scale is None or result.proposal_epsilon is None:
        raise RuntimeError("joint-semantic result is missing its proposal evidence")
    return (
        "semantic score=symmetrized final-label compatibility log-score contrast",
        "joint law=q=epsilon*prior+(1-epsilon)*softmax(log-prior+eta*score)",
        f"semantic slate={ledger.slate_traces}/{ledger.support_states} "
        f"unique-programs={ledger.unique_programs}",
        f"semantic eta={result.semantic_scale:.5g} "
        f"prior-floor={result.proposal_epsilon:.5g}",
        "raw label scores: "
        f"used={result.scored_candidates} limit={result.max_scored_candidates}",
        "only sampled complete programs were executed; slate ASTs were prompt material only",
    )


def _guided_details(result: LazyImportanceSMCResult) -> tuple[str, ...]:
    return (
        "candidate scores: "
        f"used={result.scored_candidates} limit={result.max_scored_candidates}",
        "proposal mixture: "
        f"family-deduction={_required(result.family_deduction_mix):.5g} "
        f"hole-deduction={_required(result.hole_deduction_mix):.5g} "
        f"strength={_required(result.deduction_strength):.5g}",
    )


def _required(value: float | None) -> float:
    if value is None:
        raise RuntimeError("strategy-specific result metric is missing")
    return value

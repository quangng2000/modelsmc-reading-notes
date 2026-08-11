from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path

import pytest

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.core import ProgramScorer
from modelsmc_pbe.proposals import LLMEnergyNormalization, UniformCandidateScorer
from modelsmc_pbe.runtime import make_cpu_generator, resolve_device
from modelsmc_pbe.search.importance import FactorizedSupportBuilder, ImportanceSMCOptions
from modelsmc_pbe.search.importance.factorized import trace_log_prior
from modelsmc_pbe.search.importance.lazy_proposal import LazyGuidedProposalKernel
from modelsmc_pbe.search.importance.lazy_records import (
    ConstructionTrace,
    LazyImportanceState,
)
from modelsmc_pbe.search.importance.proposal import FiniteGuidedProposalKernel
from modelsmc_pbe.search.importance.support import ImportanceSupportBuilder
from modelsmc_pbe.search.importance.target import FiniteImportanceTarget
from research.heldout import generate_inputs
from research.protocol import effective_caps, load_protocol
from research.run_matrix import build_plan, command_for, find_stage, main

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_DIR / "research" / "protocol-deduction-stress-v2.json"
V1_PROTOCOL_PATH = PROJECT_DIR / "research" / "protocol-deduction-stress-v1.json"
SPEC_PATH = PROJECT_DIR / "examples" / "foldr-sparse-bounded-square-v2.json"
V1_SPEC_PATH = PROJECT_DIR / "examples" / "foldr-sparse-bounded-square.json"


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_v2_changes_only_the_declared_loss_scale_in_the_task_spec() -> None:
    original_bytes = V1_SPEC_PATH.read_bytes()
    assert hashlib.sha256(original_bytes).hexdigest() == (
        "651e1a853d0212896c32c0ccfdd7b3459506b82ab4921f6694d7808180419f0b"
    )
    original = json.loads(original_bytes)
    corrected = json.loads(SPEC_PATH.read_bytes())

    assert original["lossScale"] == 0.75
    assert corrected["lossScale"] == 2.0
    corrected["lossScale"] = original["lossScale"]
    assert corrected == original


def test_v2_protocol_declares_split_deduction_and_small_gated_plan() -> None:
    original_bytes = V1_PROTOCOL_PATH.read_bytes()
    assert hashlib.sha256(original_bytes).hexdigest() == (
        "e261f910da00eb61a825728d9f1ef1b13a74a3c872b6f7af16a1401259c76156"
    )
    document = json.loads(PROTOCOL_PATH.read_bytes())
    protocol = load_protocol(PROTOCOL_PATH)

    assert document["schema_version"] == 3
    assert protocol.protocol_id == "modelsmc-pbe-deduction-stress-v2"
    assert "exploratory" in protocol.status
    assert "corrected" in protocol.status
    assert protocol.materialize_reference is True
    assert document["target_contract"]["loss_scale"] == 2.0
    assert protocol.target_contract is not None
    assert protocol.target_contract.reference_audit_required is True
    assert protocol.target_contract.reference_audit_stage == (
        "provider-free-reference-audit"
    )
    assert protocol.tasks[0].task_id == "foldr-sparse-bounded-square"

    inputs = generate_inputs(protocol.tasks[0])
    corpus = json.dumps(inputs, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(corpus).hexdigest() == (
        "4b1ac93177418236e597c91bef5ebf7f192a7de04f344baf137c7460539271ab"
    )

    arms = {arm["id"]: arm for arm in document["arms"]}
    assert {
        name: (arm["family_deduction_mix"], arm["hole_deduction_mix"])
        for name, arm in arms.items()
    } == {
        "U": (0.0, 0.0),
        "D": (0.75, 0.0),
        "Q": (0.0, 0.0),
        "QD": (0.75, 0.0),
    }

    audit = find_stage(protocol, "provider-free-reference-audit")
    assert audit is not None
    assert audit.max_provider_scored_candidates == 0
    audit_plans = build_plan(protocol, stage=audit)
    assert [(plan.arm, plan.seed, plan.model_id) for plan in audit_plans] == [
        ("D", 101, None)
    ]
    audit_command = command_for(
        protocol,
        protocol.tasks[0],
        next(arm for arm in protocol.arms if arm.name == "D"),
        audit_plans[0],
        caps=effective_caps(protocol, audit),
        executable="modelsmc-pbe",
        artifacts_dir=Path("/tmp/deduction-stress-v2-audit"),
        base_url=None,
    )
    assert "--materialize-reference" in audit_command
    assert _value_after(audit_command, "--deduction-mix") == "0.75"
    assert _value_after(audit_command, "--family-deduction-mix") == "0.75"
    assert _value_after(audit_command, "--hole-deduction-mix") == "0.0"

    paired = find_stage(protocol, "paired-d-qd-32b-pilot")
    assert paired is not None
    assert paired.requires_audit_stage == "provider-free-reference-audit"
    assert paired.max_provider_scored_candidates == 2_652
    paired_plans = build_plan(protocol, stage=paired)
    assert [(plan.arm, plan.seed) for plan in paired_plans] == [
        ("D", 101),
        ("QD", 101),
    ]
    assert sum(plan.model_id is not None for plan in paired_plans) == 1
    qd_plan = next(plan for plan in paired_plans if plan.arm == "QD")
    qd_command = command_for(
        protocol,
        protocol.tasks[0],
        next(arm for arm in protocol.arms if arm.name == "QD"),
        qd_plan,
        caps=effective_caps(protocol, paired),
        executable="modelsmc-pbe",
        artifacts_dir=Path("/tmp/deduction-stress-v2-paired"),
        base_url="https://provider.invalid/v1",
    )
    assert _value_after(qd_command, "--family-deduction-mix") == "0.75"
    assert _value_after(qd_command, "--hole-deduction-mix") == "0.0"


def test_v2_paid_stage_refuses_to_start_without_audit_certificate(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires --audit-certificate"):
        main(
            [
                "--protocol",
                str(PROTOCOL_PATH),
                "--output",
                str(tmp_path / "paid-stage"),
                "--stage",
                "paired-d-qd-32b-pilot",
                "--base-url",
                "https://provider.invalid/v1",
            ]
        )
    assert not (tmp_path / "paid-stage").exists()


def test_v2_provider_cells_cannot_bypass_the_named_audit_stage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="forbidden outside a named stage"):
        main(
            [
                "--protocol",
                str(PROTOCOL_PATH),
                "--output",
                str(tmp_path / "ungated-provider"),
                "--arms",
                "QD",
                "--base-url",
                "https://provider.invalid/v1",
            ]
        )
    assert not (tmp_path / "ungated-provider").exists()


def test_v3_provider_stage_cannot_add_an_unaudited_task(tmp_path: Path) -> None:
    document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    for task in document["tasks"]:
        task["spec"] = str((PROTOCOL_PATH.parent / task["spec"]).resolve())
    uncovered = dict(document["tasks"][0])
    uncovered["id"] = "uncovered-task"
    document["tasks"].append(uncovered)
    paid = next(stage for stage in document["stages"] if stage["id"] == "paired-d-qd-32b-pilot")
    paid["tasks"].append("uncovered-task")
    path = tmp_path / "uncovered-protocol.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="tasks not covered"):
        load_protocol(path)


def test_v2_terminal_target_exhaustively_ranks_every_exact_trace_first() -> None:
    config = load_experiment_config(SPEC_PATH)
    assert float(config.smc.loss_scale) == 2.0
    options = ImportanceSMCOptions(
        multi_family=True,
        support_limit=250_000,
        hole_state_limit=250_000,
        hole_max_cost=3,
        proposal_temperature=0.7,
        proposal_epsilon=0.05,
        deduction_mix=0.75,
        family_deduction_mix=0.75,
        hole_deduction_mix=0.0,
        deduction_strength=4.0,
        beta_max=1.0,
        llm_energy_normalization=(
            LLMEnergyNormalization.MEAN_FULL_PROMPT_CONDITIONAL_LOGPROB
        ),
    )
    with ProgramScorer(config) as scorer:
        support = ImportanceSupportBuilder(
            spec=config.spec,
            smc=config.smc,
            options=options,
            scorer=scorer,
        ).build()
    target = FiniteImportanceTarget.build(
        support,
        smc=config.smc,
        device=resolve_device("cpu"),
    )

    assert len(support.states) == 36_198
    assert support.exact_programs == 2
    assert int(target.exact_mask.sum().item()) == 2
    assert bool(target.exact_mask.any().item())
    assert bool((~target.exact_mask).any().item())

    terminal = target.log_unnormalized(beta=1.0)
    minimum_exact = float(terminal[target.exact_mask].min().item())
    maximum_inexact = float(terminal[~target.exact_mask].max().item())
    assert minimum_exact > maximum_inexact
    assert minimum_exact - maximum_inexact == pytest.approx(4.0)

    positive_losses = target.losses_cpu[~target.exact_mask]
    critical = float(
        (
            (target.log_prior_cpu[~target.exact_mask] - minimum_exact)
            / positive_losses
        ).max().item()
    )
    assert critical == pytest.approx(1.267634513308855)
    assert float(config.smc.loss_scale) > critical

    exact_states = tuple(state for state in support.states if state.score.exact_program)
    exact_indices = tuple(state.state_index for state in exact_states)
    finite_kernel = FiniteGuidedProposalKernel(
        config=config,
        options=options,
        support=support,
        scorer=UniformCandidateScorer(),
        generator=make_cpu_generator(101),
    )
    finite_proposals = asyncio.run(
        finite_kernel.evaluate_many(
            exact_indices[0],
            exact_indices,
            stage=1,
            beta=1.0,
        )
    )

    # The lazy kernel's clone route forces the same canonical traces while still
    # evaluating their complete construction density. Alpha affects only the
    # clone/construct mixture, not log_q_construct, which is the quantity audited.
    factorized = FactorizedSupportBuilder(
        spec=config.spec,
        smc=config.smc,
        options=options,
    ).build()
    lazy_ancestors: list[LazyImportanceState] = []
    for state in exact_states:
        family = factorized.family(state.hypothesis_index)
        filling_indices = tuple(
            next(
                index
                for index, candidate in enumerate(catalog.fillings)
                if candidate.key == filling.key
            )
            for catalog, filling in zip(family.catalogs, state.fillings, strict=True)
        )
        trace = ConstructionTrace(
            hypothesis_index=state.hypothesis_index,
            filling_indices=filling_indices,
        )
        lazy_ancestors.append(
            LazyImportanceState(
                trace=trace,
                family=state.family,
                fillings=state.fillings,
                program=state.program,
                key=state.key,
                score=state.score,
                log_prior=trace_log_prior(
                    factorized,
                    trace,
                    cost_scale=float(config.smc.cost_scale),
                ),
            )
        )
    cloning_config = config.model_copy(
        update={"smc": config.smc.model_copy(update={"clone_probability": 0.5})}
    )
    lazy_kernel = LazyGuidedProposalKernel(
        config=cloning_config,
        options=options,
        support=factorized,
        scorer=UniformCandidateScorer(),
        # The first two float64 draws for this seed are below 0.5, forcing both
        # ancestors through the clone route without patching kernel internals.
        generator=make_cpu_generator(1),
    )
    lazy_proposals = asyncio.run(
        lazy_kernel.sample_many(tuple(lazy_ancestors), stage=1, beta=1.0)
    )

    expected_family_probability = 0.7256226594847973
    expected_per_exact_trace = 2.0156184985688807e-5
    expected_total_exact_mass = 4.0312369971377614e-5
    assert expected_per_exact_trace == pytest.approx(
        expected_family_probability / (600 * 60),
        rel=1e-15,
    )

    finite_masses = tuple(math.exp(proposal.log_q_construct) for proposal in finite_proposals)
    lazy_masses = tuple(math.exp(proposal.log_q_construct) for proposal in lazy_proposals)
    assert finite_masses == pytest.approx(
        (expected_per_exact_trace, expected_per_exact_trace),
        rel=1e-13,
    )
    assert lazy_masses == pytest.approx(finite_masses, rel=1e-13)
    assert math.fsum(finite_masses) == pytest.approx(expected_total_exact_mass, rel=1e-13)
    assert math.fsum(lazy_masses) == pytest.approx(expected_total_exact_mass, rel=1e-13)
    assert all(proposal.cloned for proposal in lazy_proposals)
    assert tuple(proposal.trace for proposal in lazy_proposals) == tuple(
        ancestor.trace for ancestor in lazy_ancestors
    )

    for kernel in (finite_kernel, lazy_kernel):
        family_ledgers = tuple(
            ledger for ledger in kernel.score_ledger if ledger.wave == "family"
        )
        assert family_ledgers
        assert all(ledger.deduction_mix == 0.75 for ledger in family_ledgers)
        assert all(
            selection.selected_probability
            == pytest.approx(expected_family_probability, rel=1e-13)
            for ledger in family_ledgers
            for selection in ledger.selections
        )
        assert all(
            ledger.deduction_mix == 0.0
            for ledger in kernel.score_ledger
            if ledger.wave.startswith("hole-")
        )

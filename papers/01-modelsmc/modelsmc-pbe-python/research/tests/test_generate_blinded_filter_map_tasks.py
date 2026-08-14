from __future__ import annotations

import copy
import json
import stat
from pathlib import Path
from typing import cast

import pytest

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.core.evaluate import evaluate_expression
from modelsmc_pbe.core.types import Node
from modelsmc_pbe.domain.ast import AstNode, canonical_key
from modelsmc_pbe.grammar import arithmetic_expressions, filter_predicates
from research.generate_blinded_filter_map_tasks import (
    CONSTANTS,
    MAPPER_FAMILIES,
    V2_DEFINITION,
    V2_PROTOCOL_TASK_SCHEDULE,
    V2_SCHEMA,
    _public_keys_are_safe,
    canonical_bytes,
    generate_suite,
    predicate_probe_domain,
    prepare_secret,
    seed_commitment,
    verify_suite,
)

SEED = "0123456789abcdef" * 4


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _reveal(root: Path) -> dict[str, object]:
    value = json.loads((root / "private" / "reveal.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def test_seeded_generation_is_byte_deterministic_and_provider_blind(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_suite(first, seed_hex=SEED)
    generate_suite(second, seed_hex=SEED)

    assert _files(first) == _files(second)
    public_files = _files(first / "public")
    public_blob = b"".join(public_files.values())
    assert SEED.encode() not in public_blob
    assert b"target_predicate" not in public_blob
    assert b"target_mapper" not in public_blob
    for content in public_files.values():
        _public_keys_are_safe(json.loads(content))

    reveal = _reveal(first)
    private_tasks = cast(list[dict[str, object]], reveal["tasks"])
    assert tuple(cast(str, task["mapper_family"]) for task in private_tasks) == MAPPER_FAMILIES
    assert tuple(cast(str, task["predicate_shape"]) for task in private_tasks) == (
        "bounded",
        "lower",
        "upper",
        "bounded",
    )
    assert len({cast(str, task["target_commitment_sha256"]) for task in private_tasks}) == 4


def test_commitments_catalog_membership_and_examples_verify(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    generate_suite(suite, seed_hex=SEED)

    result = verify_suite(suite / "public", suite / "private" / "reveal.json")

    assert result["exact"] is True
    assert result["verified"] == [f"blind-{index:02d}" for index in range(1, 5)]
    predicate_keys = {canonical_key(value) for value in filter_predicates(CONSTANTS)}
    mapper_keys = {
        canonical_key(value) for value in arithmetic_expressions("Item", CONSTANTS)
    }
    reveal = _reveal(suite)
    for record in cast(list[dict[str, object]], reveal["tasks"]):
        preimage = cast(dict[str, object], record["commitment_preimage"])
        target = cast(dict[str, AstNode], preimage["target"])
        assert canonical_key(target["predicate"]) in predicate_keys
        assert canonical_key(target["mapper"]) in mapper_keys
        support = cast(list[int], record["predicate_support"])
        assert 3 <= len(support) <= 5
        task_id = cast(str, record["task_id"])
        config = load_experiment_config(suite / "public" / f"{task_id}.json")
        assert len(config.spec.examples) == 12
        domain = list(predicate_probe_domain())
        assert config.spec.examples[9].input_value == domain
        first = cast(list[int], config.spec.examples[10].input_value)
        doubled = cast(list[int], config.spec.examples[11].input_value)
        assert sorted(first) == domain
        assert sorted(doubled[:8]) == domain
        assert sorted(doubled[8:]) == domain


def test_commitment_rejects_private_target_tampering(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    generate_suite(suite, seed_hex=SEED)
    reveal_path = suite / "private" / "reveal.json"
    reveal = _reveal(suite)
    tampered = copy.deepcopy(reveal)
    tasks = cast(list[dict[str, object]], tampered["tasks"])
    preimage = cast(dict[str, object], tasks[0]["commitment_preimage"])
    target = cast(dict[str, object], preimage["target"])
    target["mapper"] = {"kind": "Item"}
    reveal_path.write_bytes(canonical_bytes(tampered) + b"\n")

    with pytest.raises(ValueError, match="target commitment mismatch"):
        verify_suite(suite / "public", reveal_path)


def test_singletons_identify_behavior_on_the_frozen_observed_domain(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    generate_suite(suite, seed_hex=SEED)
    reveal = _reveal(suite)
    domain = predicate_probe_domain()
    predicates = filter_predicates(CONSTANTS)

    for record in cast(list[dict[str, object]], reveal["tasks"]):
        preimage = cast(dict[str, object], record["commitment_preimage"])
        target = cast(dict[str, AstNode], preimage["target"])
        target_predicate = target["predicate"]
        target_signature = tuple(
            cast(bool, evaluate_expression(cast(Node, target_predicate), [], item=item))
            for item in domain
        )
        for alternative in predicates:
            signature = tuple(
                cast(bool, evaluate_expression(cast(Node, alternative), [], item=item))
                for item in domain
            )
            if signature != target_signature:
                continue
            assert signature == target_signature


def test_invalid_seed_and_overwrite_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="64 hexadecimal"):
        generate_suite(tmp_path / "bad", seed_hex="abcd")
    destination = tmp_path / "suite"
    generate_suite(destination, seed_hex=SEED)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_suite(destination, seed_hex=SEED)


def test_v2_schedule_exactly_matches_frozen_protocol_template() -> None:
    protocol_path = Path(__file__).parents[1] / "protocol-blind-evidence-frontier-v2-template.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    declared = protocol["fresh_task_generator"]["fixed_balanced_schedule"]

    assert tuple(tuple(record) for record in declared) == V2_PROTOCOL_TASK_SCHEDULE
    assert tuple(task_id for task_id, _, _ in V2_DEFINITION.schedule) == tuple(
        f"blind-v2-{index:02d}" for index in range(1, 13)
    )


def test_v2_custodial_secret_is_private_committed_and_single_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_seed = bytes.fromhex(SEED)
    monkeypatch.setattr("secrets.token_bytes", lambda length: raw_seed)
    secret_path = tmp_path / "custodian" / "blind-v2.secret"

    commitment = prepare_secret(secret_path, suite_version="v2")

    assert commitment == seed_commitment(raw_seed, suite_version="v2")
    assert stat.S_IMODE(secret_path.stat().st_mode) == 0o600
    manifest = generate_suite(
        tmp_path / "suite",
        seed_file=secret_path,
        suite_version="v2",
    )
    assert manifest["schema"] == V2_SCHEMA
    assert manifest["task_count"] == 12
    with pytest.raises(FileExistsError, match="secret file was already used"):
        generate_suite(
            tmp_path / "second-suite",
            seed_file=secret_path,
            suite_version="v2",
        )


def test_v2_generation_and_verification_preserve_protocol_order(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    manifest = generate_suite(suite, seed_hex=SEED, suite_version="v2")

    task_records = cast(list[dict[str, object]], manifest["tasks"])
    assert [record["task_id"] for record in task_records] == [
        task_id for task_id, _, _ in V2_PROTOCOL_TASK_SCHEDULE
    ]
    assert [record["path"] for record in task_records] == [
        f"blind-v2-{index:02d}.json" for index in range(1, 13)
    ]
    result = verify_suite(suite / "public", suite / "private" / "reveal.json")
    assert result == {
        "schema": V2_SCHEMA,
        "verified": [f"blind-v2-{index:02d}" for index in range(1, 13)],
        "exact": True,
    }

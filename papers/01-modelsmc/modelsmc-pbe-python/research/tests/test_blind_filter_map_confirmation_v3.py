from __future__ import annotations

import json
from pathlib import Path

import pytest

import research.generate_blind_filter_map_confirmation_v3 as generator
from research.blind_filter_map_confirmation_v3 import (
    CUSTODY_SEAL_SCHEMA,
    DOMAIN,
    METHOD_SCHEMA,
    METHOD_SEAL_SCHEMA,
    METHOD_STATUS,
    TASK_IDS,
    _evaluate_bool,
    _evaluate_int,
    canonical_bytes,
    frozen_run_seeds,
    paired_sign_test_p_value,
    sample_target,
    seed_commitment,
    sha256_file,
)
from research.freeze_blind_filter_map_confirmation_v3 import (
    _validate_research_dependency_closure,
)


def _method_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    source = repo / "research/generate_blind_filter_map_confirmation_v3.py"
    source.parent.mkdir(parents=True)
    source.write_text("# frozen fixture generator\n", encoding="utf-8")
    common = repo / "research/blind_filter_map_confirmation_v3.py"
    common.write_text("# frozen fixture common definitions\n", encoding="utf-8")
    monkeypatch.setattr(generator, "__file__", str(source))
    protocol = repo / "research/protocol-v3.json"
    protocol.write_bytes(
        canonical_bytes(
            {
                "schema": METHOD_SCHEMA,
                "protocol_status": METHOD_STATUS,
                "task_distribution": {
                    "distribution_id": "target-independent-recursive-grammar-rejection-v1"
                },
                "freeze_requirements": {
                    "common": {
                        "path": common.relative_to(repo).as_posix(),
                        "sha256": sha256_file(common),
                    },
                    "generator": {
                        "path": source.relative_to(repo).as_posix(),
                        "sha256": sha256_file(source),
                    }
                },
            }
        )
        + b"\n"
    )
    seal = repo / "research/method-seal-v3.json"
    seal.write_bytes(
        canonical_bytes(
            {
                "schema": METHOD_SEAL_SCHEMA,
                "sealed_before_secret_preparation": True,
                "protocol": {
                    "path": protocol.relative_to(repo).as_posix(),
                    "sha256": sha256_file(protocol),
                },
            }
        )
        + b"\n"
    )
    return repo, protocol, seal


def test_target_sampler_is_deterministic_and_uses_only_frozen_checks() -> None:
    seed = bytes(range(32))
    first = sample_target(seed, "blind-v3-01")
    second = sample_target(seed, "blind-v3-01")
    assert first == second
    assert first.predicate_dsl == "and(lt(item,4),lt(item,1))"
    assert first.mapper_dsl == "add(item,item)"
    assert first.accepted_attempt == 11
    support = [item for item in DOMAIN if _evaluate_bool(first.predicate, item)]
    outputs = [_evaluate_int(first.mapper, item) for item in support]
    assert 3 <= len(support) <= 5
    assert len(set(outputs)) >= 3
    assert all(
        set(record["reasons"])
        <= {
            "retained-domain-cardinality-outside-3-to-5",
            "fewer-than-three-distinct-retained-outputs",
        }
        for record in first.rejected
    )


def test_frozen_seed_schedule_is_explicit_unique_and_matched() -> None:
    records = frozen_run_seeds()
    assert tuple(record.task_id for record in records) == TASK_IDS
    values = [
        value
        for record in records
        for key, value in record.to_dict().items()
        if key != "task_id"
    ]
    assert len(values) == len(set(values))
    assert records[0].to_dict() == {
        "task_id": "blind-v3-01",
        "start_seed": 401010,
        "provider_seed": 401020,
        "sample_seed": 401030,
        "resample_seed": 401040,
        "random_shortlist_seed": 401050,
    }


def test_exact_paired_sign_test_has_frozen_tail() -> None:
    assert paired_sign_test_p_value(5, 0) == pytest.approx(1 / 32)
    assert paired_sign_test_p_value(4, 1) == pytest.approx(6 / 32)
    assert paired_sign_test_p_value(0, 0) == 1.0


def test_runtime_research_import_closure_is_fully_bound() -> None:
    project = Path(__file__).resolve().parents[2]
    _validate_research_dependency_closure(project)


def test_secret_preparation_fails_if_method_seal_bytes_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, protocol, seal = _method_fixture(tmp_path, monkeypatch)
    expected = sha256_file(seal)
    payload = json.loads(seal.read_text(encoding="utf-8"))
    payload["sealed_before_secret_preparation"] = False
    seal.write_bytes(canonical_bytes(payload) + b"\n")
    with pytest.raises(ValueError, match="external method-seal SHA-256"):
        generator.prepare_secret(
            tmp_path / "secret.bin",
            repo_root=repo,
            protocol_path=protocol,
            method_seal_path=seal,
            expected_method_seal_sha256=expected,
        )


def test_dummy_suite_round_trips_only_after_both_seals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, protocol, method_seal = _method_fixture(tmp_path, monkeypatch)
    secret = tmp_path / "dummy-secret.bin"
    secret.write_bytes(bytes(range(32)))
    secret.chmod(0o600)
    custody = repo / "evidence/custody.json"
    custody.parent.mkdir(parents=True)
    generator.seal_custody(
        custody,
        secret_file=secret,
        repo_root=repo,
        protocol_path=protocol,
        method_seal_path=method_seal,
        expected_method_seal_sha256=sha256_file(method_seal),
    )
    custody_payload = json.loads(custody.read_text(encoding="utf-8"))
    assert custody_payload["schema"] == CUSTODY_SEAL_SCHEMA
    assert custody_payload["task_secret_commitment_sha256"] == seed_commitment(
        bytes(range(32))
    )
    suite = repo / "artifacts/dummy-suite"
    manifest = generator.generate_suite(
        suite,
        secret_file=secret,
        repo_root=repo,
        protocol_path=protocol,
        method_seal_path=method_seal,
        expected_method_seal_sha256=sha256_file(method_seal),
        custody_seal_path=custody,
        expected_custody_seal_sha256=sha256_file(custody),
    )
    assert manifest["task_count"] == 12
    public_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (suite / "public").glob("*.json")
    )
    assert '"target"' not in public_text
    assert '"seed_hex"' not in public_text
    verified = generator.verify_suite(
        public_dir=suite / "public",
        reveal_path=suite / "private/reveal.json",
        custody_seal_path=custody,
    )
    assert verified == {
        "schema": "blinded-filter-map-suite-v3",
        "verified": list(TASK_IDS),
        "exact": True,
    }
    task = suite / "public/blind-v3-01.json"
    task.write_text(task.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="file SHA-256"):
        generator.verify_suite(
            public_dir=suite / "public",
            reveal_path=suite / "private/reveal.json",
            custody_seal_path=custody,
        )

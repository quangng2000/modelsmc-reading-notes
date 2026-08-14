from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.core.results import ScoredProgram
from modelsmc_pbe.core.scorer import ProgramScorer
from modelsmc_pbe.domain.ast import ProgramAst, canonical_key
from modelsmc_pbe.grammar import enumerate_skeleton
from research.exedec_deepcoder_ho_adapter_v1 import (
    ADAPTER_SCHEMA,
    BUNDLE_SCHEMA,
    EXEDEC_SOURCE_COMMIT,
    EXEDEC_SOURCE_DATA_SHA256,
    INTEGER_CONSTANTS,
    SUPPORTED_MAPPERS,
    SUPPORTED_PREDICATES,
    AdapterError,
    SupportedProgram,
    UnsupportedProgramError,
    assert_no_release_collision,
    build_debug_bundle,
    build_release_collision_registry,
    canonical_bytes,
    debug_oracle,
    evaluate_candidate,
    hidden_examples,
    parse_supported_program,
    scan_released_source,
    semantic_fingerprint,
    translate_target,
)


def _record(index: int, predicate: str, mapper: str) -> dict[str, object]:
    inputs = (
        "x0 = [ -3 0 2 ]",
        "x0 = [ -1 1 4 ]",
        "x0 = [ 0 ]",
    )
    target = SupportedProgram("x0", predicate, mapper)
    from research.exedec_deepcoder_ho_adapter_v1 import run_reference

    outputs = [run_reference(target, [-3, 0, 2]), run_reference(target, [-1, 1, 4]), ()]
    assert all(output is not None for output in outputs)

    def render(output: tuple[int, ...] | None) -> str:
        assert output is not None
        return "[ " + " ".join(str(value) for value in output) + " ]"

    return {
        "index": index,
        "inputs": list(inputs),
        "outputs": [render(output) for output in outputs],
        "program": (
            f"x0 = INPUT | x1 = Filter {predicate} x0 | "
            f"x2 = Map {mapper} x1"
        ),
    }


def _write_source(path: Path, records: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(canonical_bytes(record) + b"\n" for record in records))


def _fixture_source(path: Path) -> None:
    records = [
        _record(8, "(<0)", "(*(-1))"),
        _record(9, "(<0)", "(*(-1))"),
        _record(10, "(>0)", "(+1)"),
        {
            "index": 11,
            "inputs": ["x0 = [ 2 ]"],
            "outputs": ["[ 1 ]"],
            "program": "x0 = INPUT | x1 = Filter (>0) x0 | x2 = Map (/2) x1",
        },
    ]
    _write_source(path, records)


def test_strict_subset_parser_and_translation_are_exact(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source, [_record(4, "(<0)", "(*(-1))")])
    _, records = scan_released_source(source, enforce_official_binding=False)
    assert len(records) == 1
    record = records[0]
    assert record.source_program == SupportedProgram("x0", "(<0)", "(*(-1))")

    task_path = tmp_path / "task.json"
    from research.exedec_deepcoder_ho_adapter_v1 import public_task

    task_path.write_bytes(canonical_bytes(public_task(record)) + b"\n")
    config = load_experiment_config(task_path)
    score = ProgramScorer(config).score(record.target_ast)
    assert isinstance(score, ScoredProgram)
    assert score.exact_program is True
    assert score.total_loss == 0


def test_every_declared_target_is_in_current_typed_smc_grammar() -> None:
    support = {
        canonical_key(program)
        for program in enumerate_skeleton(
            "foldr-filter-map",
            INTEGER_CONSTANTS,
            limit=20_000,
        )
    }
    for predicate in SUPPORTED_PREDICATES:
        for mapper in SUPPORTED_MAPPERS:
            target = translate_target(SupportedProgram("x0", predicate, mapper))
            assert canonical_key(target) in support


@pytest.mark.parametrize(
    "program",
    [
        "x0 = INPUT | x1 = Map (+1) x0 | x2 = Filter (>0) x1",
        "x0 = INPUT | x1 = Filter (%2==0) x0 | x2 = Map (+1) x1",
        "x0 = INPUT | x1 = Filter (>0) x0 | x2 = Map (/2) x1",
        "x0 = INPUT | x1 = Filter (>0) x0 | x2 = Map (+1) x0",
        "x0 = INPUT | x1 = INPUT | x2 = Filter (>0) x0 | x3 = Map (+1) x2",
        "x0 = INPUT | x1 = Filter (>0) x0 | x2 = Map (+1) x1 | x3 = Sum x2",
    ],
)
def test_unsupported_deepcoder_constructs_fail_closed(program: str) -> None:
    with pytest.raises(UnsupportedProgramError):
        parse_supported_program(program)


def test_source_audit_deduplicates_semantic_targets(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _fixture_source(source)

    audit, selected = scan_released_source(source, enforce_official_binding=False)

    assert audit.total_records == 4
    assert audit.compatible_occurrences == 3
    assert audit.distinct_semantic_targets == 2
    assert audit.selected_source_indices == (8, 10)
    assert audit.semantic_collision_groups == ((8, 9), (10,))
    assert tuple(record.index for record in selected) == (8, 10)


def test_source_binding_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _fixture_source(source)
    with pytest.raises(AdapterError, match="digest differs"):
        scan_released_source(source)


def test_hidden_debug_probes_are_deterministic_and_score_typed_ast(tmp_path: Path) -> None:
    target = SupportedProgram("x0", "(>0)", "(+1)")
    first = hidden_examples(target, debug_seed=71, random_multi_examples=12)
    second = hidden_examples(target, debug_seed=71, random_multi_examples=12)
    different = hidden_examples(target, debug_seed=72, random_multi_examples=12)
    assert first == second
    assert first != different
    assert first[0] == ((), ())
    assert ((50,), (51,)) not in first  # The source program is undefined at +50.

    source_record = _record(10, "(>0)", "(+1)")
    source_path = tmp_path / "source.jsonl"
    _write_source(source_path, [source_record])
    _, records = scan_released_source(source_path, enforce_official_binding=False)
    record = records[0]
    oracle = debug_oracle(record, debug_seed=71, random_multi_examples=12)
    exact = evaluate_candidate(record.target_ast, oracle)
    wrong = evaluate_candidate(
        translate_target(SupportedProgram("x0", "(<0)", "(+1)")),
        oracle,
    )
    assert exact["exact"] is True
    assert exact["total_loss"] == 0
    assert wrong["exact"] is False


def test_debug_bundle_is_byte_deterministic_and_target_free_publicly(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _fixture_source(source)
    first = tmp_path / "first"
    second = tmp_path / "second"

    manifest = build_debug_bundle(
        source,
        first,
        debug_seed=808,
        random_multi_examples=8,
        enforce_official_binding=False,
    )
    build_debug_bundle(
        source,
        second,
        debug_seed=808,
        random_multi_examples=8,
        enforce_official_binding=False,
    )

    first_files = {
        path.relative_to(first): path.read_bytes()
        for path in sorted(first.rglob("*"))
        if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in sorted(second.rglob("*"))
        if path.is_file()
    }
    assert first_files == second_files
    assert manifest["schema"] == BUNDLE_SCHEMA
    task_entries = cast(list[dict[str, object]], manifest["tasks"])
    assert [entry["source_index"] for entry in task_entries] == [8, 10]
    public_blob = b"".join(
        path.read_bytes() for path in sorted((first / "public").glob("*.json"))
    )
    assert b"target_ast" not in public_blob
    assert b"semantic_fingerprint" not in public_blob
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_debug_bundle(
            source,
            first,
            debug_seed=808,
            enforce_official_binding=False,
        )


def test_release_collision_registry_includes_test_and_llm_schemas(tmp_path: Path) -> None:
    test_path = tmp_path / "test.jsonl"
    llm_path = tmp_path / "llm.jsonl"
    target = _record(1, "(<0)", "(*(-1))")
    _write_source(test_path, [target])
    llm_path.write_bytes(
        canonical_bytes(
            {
                "index": 0,
                "test_problem": {"program": _record(2, "(>0)", "(+1)")["program"]},
                "few_shot_examples": [{"program": target["program"]}],
            }
        )
        + b"\n"
    )

    registry = build_release_collision_registry([test_path, llm_path])
    negative_abs = SupportedProgram("x7", "(<0)", "(*(-1))")
    assert len(registry.collisions(negative_abs)) == 2
    with pytest.raises(AdapterError, match="semantically collides"):
        assert_no_release_collision(negative_abs, registry)
    assert_no_release_collision(SupportedProgram("x0", "(<0)", "(**2)"), registry)


def test_protocol_metadata_matches_adapter_constants() -> None:
    protocol_path = Path(__file__).parents[1] / "protocol-exedec-deepcoder-ho-debug-v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["schema"] == ADAPTER_SCHEMA
    assert protocol["source"]["commit"] == EXEDEC_SOURCE_COMMIT
    assert protocol["source"]["released_data_sha256"] == EXEDEC_SOURCE_DATA_SHA256
    assert protocol["supported_subset"]["target_skeleton"] == "foldr-filter-map"
    assert protocol["authorization"]["provider_calls"] is False


def test_semantic_fingerprint_ignores_source_variable_spelling() -> None:
    first = SupportedProgram("x0", "(<0)", "(+1)")
    second = SupportedProgram("x8", "(<0)", "(+1)")
    other = SupportedProgram("x0", "(>0)", "(+1)")
    assert semantic_fingerprint(first) == semantic_fingerprint(second)
    assert semantic_fingerprint(first) != semantic_fingerprint(other)


def test_translated_target_is_canonical_json() -> None:
    target = translate_target(SupportedProgram("x0", "(<0)", "(**2)"))
    decoded = json.loads(canonical_bytes(target))
    assert cast(ProgramAst, decoded) == target

"""Run the 36-way predicate-behavior compatibility smoke."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from modelsmc_pbe.config import load_experiment_config
from modelsmc_pbe.proposals import VLLMPromptLogprobConfig, VLLMPromptLogprobScorer
from modelsmc_pbe.proposals.labels import (
    CompatibilityProgram,
    CompatibilityScoringRequest,
    SemanticPromptProtocol,
    SymmetrizedLabelCompatibilityScorer,
)

MODEL_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


async def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    classes_path = args.classes.resolve()
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    config = load_experiment_config(args.task)
    task = config.spec.model_dump(mode="json", by_alias=True)
    domain = classes["domain"]
    context = (
        "Fixed skeleton: foldr((item,acc) => if predicate(item) then "
        "mapper(item)::acc else acc, [], xs). The predicate candidate below is "
        "specified only by its keep mask over DomainOrder="
        + _canonical(domain)
        + ". One unknown deterministic small arithmetic mapper using item and declared "
        "constants must be shared by every retained element. Judge the predicate behavior "
        "compatible exactly when some such shared mapper can reproduce every raw output "
        "while preserving retained input order. RawExamples=" + _canonical(task["examples"])
    )
    if args.reasoning_prefix:
        context = f"DerivedReasoning={args.reasoning_prefix}\n{context}"
    selected_masks = set(args.masks.split(",")) if args.masks else None
    programs = tuple(
        CompatibilityProgram(
            program_key=item["id"],
            program_text=(
                f"Predicate behavior candidate: DomainOrder={_canonical(domain)}; "
                f"KeepMask={item['mask']}"
            ),
        )
        for item in classes["classes"]
        if selected_masks is None or item["mask"] in selected_masks
    )
    if selected_masks is not None and len(programs) != len(selected_masks):
        raise ValueError("each requested mask must identify exactly one behavior class")
    (output / "context.txt").write_text(context, encoding="utf-8")
    (output / "candidates.json").write_text(
        _canonical(
            {
                "source_classes_sha256": hashlib.sha256(classes_path.read_bytes()).hexdigest(),
                "programs": [
                    {
                        "id": program.program_key,
                        "text": program.program_text,
                        "text_sha256": program.sha256,
                    }
                    for program in programs
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    raw = VLLMPromptLogprobScorer(
        VLLMPromptLogprobConfig(
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            max_concurrency=8,
            max_batch_size=128,
            add_special_tokens=False,
            model_revision=MODEL_REVISION,
            tokenizer_revision=MODEL_REVISION,
        )
    )
    scorer = SymmetrizedLabelCompatibilityScorer(
        raw,
        prompt_protocol=SemanticPromptProtocol.HARMONY_GPT_OSS_V1,
    )
    started = time.perf_counter()
    batch = await scorer.score(
        CompatibilityScoringRequest(
            dataset_context=context,
            programs=programs,
            request_index=0,
            raw_candidate_batch_size=128,
            raw_request_batch_size=8,
        )
    )
    elapsed = time.perf_counter() - started
    records: list[dict[str, Any]] = []
    for score in batch.scores:
        record = asdict(score)
        record["raw_identity"]["provenance"] = None
        records.append(record)
    result = {
        "schema": "behavior-class-pointwise-v1",
        "analysis_label": "raw-example-36-behavior-class-symmetrized-label-score",
        "gold_visible": False,
        "template_version": batch.template_version,
        "prompt_sha256": batch.prompt_sha256,
        "candidate_count": len(batch.scores),
        "raw_path_count": batch.raw_candidate_count,
        "elapsed_seconds": elapsed,
        "provider_metrics": asdict(raw.provider_metrics()),
        "scores": records,
    }
    scores_path = output / "scores.json"
    scores_path.write_text(_canonical(result) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_count": len(batch.scores),
                "elapsed_seconds": elapsed,
                "provider_metrics": asdict(raw.provider_metrics()),
                "raw_path_count": batch.raw_candidate_count,
                "scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--masks", default="")
    parser.add_argument("--reasoning-prefix", default="")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

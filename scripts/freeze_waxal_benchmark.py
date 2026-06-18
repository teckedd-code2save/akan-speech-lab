from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from akan_speech.data.manifest import read_jsonl


def stable_order(row: dict, seed: int) -> str:
    key = f"{seed}:{row.get('sample_id')}:{row.get('dataset_row')}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def select_balanced_test_rows(
    records: list[dict], *, per_speaker: int, seed: int, exclude_train_text_overlap: bool
) -> tuple[list[dict], dict]:
    train_texts = {
        row.get("normalized_text")
        for row in records
        if row.get("split") in {"train", "validation"} and row.get("normalized_text")
    }
    grouped = defaultdict(list)
    excluded_overlap = 0
    for row in records:
        if row.get("split") != "test":
            continue
        if exclude_train_text_overlap and row.get("normalized_text") in train_texts:
            excluded_overlap += 1
            continue
        grouped[str(row.get("speaker_id") or "unknown")].append(row)

    selected = []
    shortages = {}
    for speaker, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: stable_order(row, seed))
        chosen = ordered[:per_speaker]
        selected.extend(chosen)
        if len(chosen) < per_speaker:
            shortages[speaker] = len(chosen)
    random.Random(seed).shuffle(selected)
    for index, row in enumerate(selected):
        row["benchmark_index"] = index
        row["benchmark_seed"] = seed
    return selected, {
        "rows": len(selected),
        "speakers": len(grouped),
        "per_speaker": per_speaker,
        "seed": seed,
        "excluded_train_text_overlap": excluded_overlap,
        "speaker_shortages": shortages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a speaker-balanced Waxal test benchmark.")
    parser.add_argument("--manifest", default="data/manifests/waxal_aka_labeled_metadata.jsonl")
    parser.add_argument("--output", default="evals/samples/waxal_aka_benchmark_remote.jsonl")
    parser.add_argument("--report", default="evals/reports/waxal_aka_benchmark.json")
    parser.add_argument("--index", default="evals/waxal_aka_benchmark_v1.json")
    parser.add_argument("--per-speaker", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--allow-train-text-overlap", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    selected, report = select_balanced_test_rows(
        records,
        per_speaker=args.per_speaker,
        seed=args.seed,
        exclude_train_text_overlap=not args.allow_train_text_overlap,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report.update({"source_manifest": args.manifest, "output_manifest": args.output})
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    index_payload = {
        "dataset": "google/WaxalNLP",
        "config": "aka_asr",
        "split": "test",
        "seed": args.seed,
        "per_speaker": args.per_speaker,
        "selection": [
            {
                "benchmark_index": row["benchmark_index"],
                "dataset_row": row.get("dataset_row"),
                "sample_id": row.get("sample_id"),
                "speaker_id": row.get("speaker_id"),
            }
            for row in sorted(selected, key=lambda item: item["benchmark_index"])
        ],
    }
    index_payload["selection_sha256"] = hashlib.sha256(
        json.dumps(index_payload["selection"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    serialized_index = json.dumps(index_payload, indent=2, ensure_ascii=False) + "\n"
    Path(args.index).write_text(serialized_index, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

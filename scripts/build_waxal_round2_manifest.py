from __future__ import annotations

import argparse
import json
from pathlib import Path

from akan_speech.data.hf_viewer import fetch_split_rows
from akan_speech.data.round2 import build_speaker_component_split


DATASET = "google/WaxalNLP"
CONFIG = "aka_asr"
MAIN_REVISION = "e0a62aaebc61bd5bb8cac17a08d1b42c65551dd2"
PARQUET_REVISION = "fe897206a41cad1b26f39f4c4088a45538ccfced"


def viewer_records(split: str, workers: int) -> list[dict]:
    items = fetch_split_rows(DATASET, CONFIG, split, workers=workers)
    records = []
    for item in items:
        row = item.get("row") or {}
        index = int(item["row_idx"])
        records.append(
            {
                "sample_id": str(row.get("id") or ""),
                "speaker_id": str(row.get("speaker_id") or ""),
                "text": str(row.get("transcription") or ""),
                "source_split": split,
                "dataset_row": index,
            }
        )
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze contamination-safe Waxal Round 2 metadata")
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests/waxal_round2"))
    parser.add_argument("--audit", type=Path, default=Path("evals/reports/waxal_round2_split_audit.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.85)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    published_train = viewer_records("train", args.workers)
    published_validation = viewer_records("validation", args.workers)
    published_test = viewer_records("test", args.workers)
    pool = published_train + published_validation
    split_rows, audit = build_speaker_component_split(
        pool,
        published_test,
        seed=args.seed,
        train_fraction=args.train_fraction,
    )
    train_rows = [row for row in split_rows if row["split"] == "train"]
    dev_rows = [row for row in split_rows if row["split"] == "dev"]
    test_rows = [{**row, "split": "test"} for row in published_test]
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "dev.jsonl", dev_rows)
    write_jsonl(args.output_dir / "test.jsonl", test_rows)
    report = {
        "dataset": DATASET,
        "config": CONFIG,
        "main_revision": MAIN_REVISION,
        "parquet_revision": PARQUET_REVISION,
        **audit,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

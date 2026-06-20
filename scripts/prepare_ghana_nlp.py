from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from akan_speech.data.audit import corpus_audit
from akan_speech.data.hf_viewer import fetch_split_rows
from akan_speech.data.manifest import make_record, write_jsonl
from akan_speech.data.normalize import normalize_akan_text
from akan_speech.data.split import stable_group_split


DATASET_ID = "ghananlpcommunity/twi-speech-text-multispeaker-16k"


def prepare_records(rows: list[dict], *, dataset_id: str, config: str, seed: int):
    records = []
    for item in rows:
        idx = int(item.get("row_idx", len(records)))
        row = item.get("row") or {}
        text = str(row.get("text") or row.get("transcription") or row.get("sentence") or "")
        normalized = normalize_akan_text(text)
        if not normalized:
            continue
        split = stable_group_split(normalized, seed=seed)
        records.append(
            make_record(
                audio_path=f"hf://{dataset_id}/{config}/train/{idx}",
                text=text,
                language="aka",
                speaker_id=None,
                duration_seconds=float(row["duration"]) if row.get("duration") is not None else None,
                split=split,
                source=f"{dataset_id}/{config}/train",
                dataset_row=idx,
                sample_id=f"ghana-nlp-{idx}",
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GhanaNLP Twi ASR manifest.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--config", default="default")
    parser.add_argument("--output", default="data/manifests/ghana_nlp_twi.jsonl")
    parser.add_argument("--audit-output", default="evals/reports/ghana_nlp_corpus_audit.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    rows = fetch_split_rows(
        args.dataset_id,
        args.config,
        "train",
        workers=args.workers,
        cache_dir="data/processed/viewer_cache",
    )
    if args.limit:
        rows = rows[: args.limit]
    records = prepare_records(rows, dataset_id=args.dataset_id, config=args.config, seed=args.seed)

    count = write_jsonl(records, Path(args.output))
    print(f"Wrote {count} records to {args.output}")
    durations = [record.duration_seconds for record in records if record.duration_seconds is not None]
    short_clips = sum(duration < 0.4 for duration in durations)
    long_clips = sum(duration > 30 for duration in durations)
    audit = corpus_audit([record.to_json() for record in records])
    audit.update(
        {
            "dataset": args.dataset_id,
            "config": args.config,
            "source_split": "train",
            "split_method": "SHA-256 group split over normalized transcript (90/5/5)",
            "speaker_split_status": "unavailable: published schema has no speaker identifier",
            "dataset_card_count": 21138,
            "viewer_count": 15560 if not args.limit else len(rows),
            "source_rows_scanned": len(rows),
            "usable_transcript_rows": len(records),
            "dropped_empty_transcript_rows": len(rows) - len(records),
            "count_mismatch": not args.limit and len(rows) != 21138,
            "durations": {
                "known": len(durations),
                "total_hours": round(sum(durations) / 3600, 3),
                "minimum_seconds": min(durations, default=None),
                "maximum_seconds": max(durations, default=None),
                "below_training_minimum": short_clips,
                "above_training_maximum": long_clips,
                "training_eligible": len(durations) - short_clips - long_clips,
            },
            "split_counts": dict(Counter(record.split for record in records)),
        }
    )
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

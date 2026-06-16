from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from datasets import Audio, concatenate_datasets, load_dataset

from akan_speech.data.manifest import make_record, write_jsonl


DEFAULT_DATASET_ID = "google/WaxalNLP"
DEFAULT_CONFIG = "aka_asr"


def stable_bucket(value: str, modulo: int = 1000) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def split_from_speaker(speaker_id: str, validation_pct: int, test_pct: int) -> str:
    bucket = stable_bucket(speaker_id)
    if bucket < test_pct * 10:
        return "test"
    if bucket < (test_pct + validation_pct) * 10:
        return "validation"
    return "train"


def load_waxal_splits(
    dataset_id: str,
    config: str,
    splits: list[str],
    sample_rate: int,
    *,
    streaming: bool = False,
):
    datasets = []
    for split in splits:
        ds = load_dataset(dataset_id, config, split=split, streaming=streaming)
        if "audio" in ds.column_names:
            ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
        datasets.append(ds)
    if streaming:
        return datasets
    return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]


def duration_from_audio(audio: dict | None) -> float | None:
    if not audio:
        return None
    array = audio.get("array")
    sample_rate = audio.get("sampling_rate") or audio.get("sample_rate")
    if array is None or not sample_rate:
        return None
    return round(float(len(array)) / float(sample_rate), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WaxalNLP Akan ASR data.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Waxal config, e.g. aka_asr.")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--output", default="data/manifests/waxal_aka_asr.jsonl")
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--speaker-safe-split",
        action="store_true",
        help="Ignore Waxal split labels and re-split deterministically by speaker_id.",
    )
    parser.add_argument("--validation-pct", type=int, default=10)
    parser.add_argument("--test-pct", type=int, default=10)
    parser.add_argument("--min-duration", type=float, default=0.4)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Materialize the dataset locally. Default streams rows, which is safer for smoke tests.",
    )
    args = parser.parse_args()

    datasets = load_waxal_splits(
        args.dataset_id,
        args.config,
        args.splits,
        args.target_sample_rate,
        streaming=not args.no_streaming,
    )
    if not isinstance(datasets, list):
        datasets = [datasets]
    records = []
    skipped = defaultdict(int)
    seen = 0

    for source_split, ds in zip(args.splits, datasets, strict=False):
        for idx, row in enumerate(ds):
            if args.limit and seen >= args.limit:
                break
            seen += 1
            text = row.get("transcription") or row.get("text") or ""
            audio = row.get("audio") or {}
            duration = duration_from_audio(audio)
            speaker_id = row.get("speaker_id") or f"unknown-{source_split}-{idx}"
            language = row.get("language") or row.get("locale") or "aka"
            if not text.strip():
                skipped["empty_text"] += 1
                continue
            if duration is not None and duration < args.min_duration:
                skipped["too_short"] += 1
                continue
            if duration is not None and duration > args.max_duration:
                skipped["too_long"] += 1
                continue
            split = (
                split_from_speaker(str(speaker_id), args.validation_pct, args.test_pct)
                if args.speaker_safe_split
                else source_split
            )
            audio_path = audio.get("path") or f"{args.dataset_id}:{args.config}:{source_split}:{idx}"
            records.append(
                make_record(
                    audio_path=audio_path,
                    text=text,
                    language=language,
                    speaker_id=speaker_id,
                    duration_seconds=duration,
                    split=split,
                    source=f"{args.dataset_id}/{args.config}/{source_split}",
                )
            )
        if args.limit and seen >= args.limit:
            break

    count = write_jsonl(records, Path(args.output))
    split_counts = defaultdict(int)
    speakers_by_split = defaultdict(set)
    for record in records:
        split_counts[record.split] += 1
        if record.speaker_id:
            speakers_by_split[record.split].add(record.speaker_id)

    print(f"Wrote {count} records to {args.output}")
    print(f"Skipped: {dict(skipped)}")
    print(f"Split counts: {dict(split_counts)}")
    print(f"Speaker counts: { {key: len(value) for key, value in speakers_by_split.items()} }")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from akan_speech.data.manifest import make_record, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GhanaNLP Twi ASR manifest.")
    parser.add_argument("--dataset-id", default="ghananlpcommunity/twi-speech-text-multispeaker-16k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/manifests/ghana_nlp_twi.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_id, split=args.split)
    records = []
    for idx, row in enumerate(dataset):
        if args.limit and idx >= args.limit:
            break
        audio = row.get("audio") or {}
        audio_path = audio.get("path") or f"{args.dataset_id}:{args.split}:{idx}"
        text = row.get("transcription") or row.get("text") or row.get("sentence") or ""
        speaker_id = row.get("speaker_id") or row.get("speaker") or None
        records.append(
            make_record(
                audio_path=audio_path,
                text=text,
                language="aka",
                speaker_id=speaker_id,
                split=args.split,
                source=args.dataset_id,
            )
        )

    count = write_jsonl(records, Path(args.output))
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()


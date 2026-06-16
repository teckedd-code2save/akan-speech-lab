from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

from akan_speech.data.manifest import make_record, write_jsonl


BASE_URL = "https://datasets-server.huggingface.co"


def fetch_json(path: str, params: dict[str, str]) -> dict:
    query = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{BASE_URL}{path}?{query}", timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def first_audio_url(audio_cell) -> str:
    if isinstance(audio_cell, list) and audio_cell:
        first = audio_cell[0] or {}
        return str(first.get("src") or "")
    if isinstance(audio_cell, dict):
        return str(audio_cell.get("src") or audio_cell.get("path") or "")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a small Waxal ASR manifest from HF Dataset Viewer preview rows."
    )
    parser.add_argument("--dataset", default="google/WaxalNLP")
    parser.add_argument("--config", default="aka_asr")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", default="evals/samples/waxal_aka_asr_preview.jsonl")
    args = parser.parse_args()

    payload = fetch_json(
        "/first-rows",
        {"dataset": args.dataset, "config": args.config, "split": args.split},
    )
    records = []
    for item in payload.get("rows", [])[: args.limit]:
        row = item.get("row", {})
        audio_url = first_audio_url(row.get("audio"))
        records.append(
            make_record(
                audio_path=audio_url,
                text=row.get("transcription") or "",
                language=row.get("language") or "aka",
                speaker_id=row.get("speaker_id"),
                split=args.split,
                source=f"{args.dataset}/{args.config}/{args.split}/viewer/{item.get('row_idx')}",
            )
        )
    count = write_jsonl(records, Path(args.output))
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()


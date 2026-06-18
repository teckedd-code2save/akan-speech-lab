from __future__ import annotations

import argparse
import json
import random
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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


def select_row_indices(
    *, total: int, limit: int, mode: str, start_row: int = 0, seed: int = 42
) -> list[int]:
    if total <= 0 or limit <= 0:
        return []
    limit = min(limit, total)
    if mode == "random":
        return sorted(random.Random(seed).sample(range(total), limit))
    start = max(0, min(start_row, max(0, total - limit)))
    return list(range(start, start + limit))


def fetch_rows(dataset: str, config: str, split: str, indices: list[int]) -> list[dict]:
    def fetch_one(index: int) -> dict | None:
        payload = fetch_json(
            "/rows",
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": str(index),
                "length": "1",
            },
        )
        rows = payload.get("rows", [])
        return rows[0] if rows else None

    with ThreadPoolExecutor(max_workers=min(8, len(indices) or 1)) as executor:
        items = list(executor.map(fetch_one, indices))
    return [item for item in items if item is not None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a small Waxal ASR manifest from HF Dataset Viewer preview rows."
    )
    parser.add_argument("--dataset", default="google/WaxalNLP")
    parser.add_argument("--config", default="aka_asr")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--mode", choices=["sequential", "random"], default="sequential")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="evals/samples/waxal_aka_asr_preview.jsonl")
    args = parser.parse_args()

    first_page = fetch_json(
        "/rows",
        {
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "offset": "0",
            "length": "1",
        },
    )
    total = int(first_page.get("num_rows_total") or len(first_page.get("rows", [])))
    indices = select_row_indices(
        total=total,
        limit=args.limit,
        mode=args.mode,
        start_row=args.start_row,
        seed=args.seed,
    )
    items = fetch_rows(args.dataset, args.config, args.split, indices)
    records = []
    for item in items:
        row = item.get("row", {})
        row_idx = int(item.get("row_idx", 0))
        audio_url = first_audio_url(row.get("audio"))
        records.append(
            make_record(
                audio_path=audio_url,
                text=row.get("transcription") or "",
                language=row.get("language") or "aka",
                speaker_id=row.get("speaker_id"),
                split=args.split,
                source=f"{args.dataset}/{args.config}/{args.split}/viewer/{row_idx}",
                dataset_row=row_idx,
            )
        )
    count = write_jsonl(records, Path(args.output))
    print(
        f"Wrote {count} records to {args.output} from {args.split} rows "
        f"{indices} (mode={args.mode}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()

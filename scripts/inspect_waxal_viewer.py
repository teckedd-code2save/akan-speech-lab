from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from akan_speech.data.normalize import normalize_akan_text


BASE_URL = "https://datasets-server.huggingface.co"


def fetch_json(path: str, params: dict[str, str]) -> dict:
    query = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{BASE_URL}{path}?{query}", timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast Waxal metadata inspection via HF Dataset Viewer.")
    parser.add_argument("--dataset", default="google/WaxalNLP")
    parser.add_argument("--config", default="aka_asr")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="evals/reports/waxal_aka_asr_viewer_preview.json")
    args = parser.parse_args()

    splits_payload = fetch_json("/splits", {"dataset": args.dataset})
    relevant_splits = [
        row["split"]
        for row in splits_payload.get("splits", [])
        if row.get("config") == args.config
    ]
    rows_payload = fetch_json(
        "/first-rows",
        {"dataset": args.dataset, "config": args.config, "split": args.split},
    )
    rows = [item.get("row", {}) for item in rows_payload.get("rows", [])]
    speakers = Counter(str(row.get("speaker_id") or "unknown") for row in rows)
    languages = Counter(str(row.get("language") or "unknown") for row in rows)
    examples = []
    for row in rows[:10]:
        text = row.get("transcription") or ""
        examples.append(
            {
                "id": row.get("id"),
                "speaker_id": row.get("speaker_id"),
                "language": row.get("language"),
                "text": text,
                "normalized_text": normalize_akan_text(text),
            }
        )

    report = {
        "dataset": args.dataset,
        "config": args.config,
        "available_splits": sorted(set(relevant_splits)),
        "features": rows_payload.get("features", []),
        "preview_rows": len(rows),
        "speaker_counts": dict(speakers),
        "language_counts": dict(languages),
        "examples": examples,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


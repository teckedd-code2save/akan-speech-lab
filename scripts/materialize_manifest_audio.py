from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from akan_speech.data.manifest import read_jsonl


def extension_from_url(url: str) -> str:
    path = urlparse(url).path
    suffix = Path(path).suffix
    return suffix if suffix else ".mp3"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download remote audio in a manifest to local files.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--audio-dir", default="data/processed/audio_samples")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    materialized = []
    for idx, row in enumerate(rows):
        source = row["audio_path"]
        if source.startswith("http://") or source.startswith("https://"):
            output_path = audio_dir / f"sample_{idx:04d}{extension_from_url(source)}"
            if not output_path.exists():
                print(f"Downloading {idx + 1}/{len(rows)} -> {output_path}")
                urllib.request.urlretrieve(source, output_path)
            row = dict(row)
            row["remote_audio_path"] = source
            row["audio_path"] = str(output_path)
        materialized.append(row)

    output = Path(args.output_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in materialized:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(materialized)} rows to {output}")


if __name__ == "__main__":
    main()


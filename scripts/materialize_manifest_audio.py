from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from akan_speech.data.manifest import read_jsonl


def extension_from_url(url: str) -> str:
    path = urlparse(url).path
    suffix = Path(path).suffix
    return suffix if suffix else ".mp3"


def audio_filename(index: int, url: str) -> str:
    """Keep cached audio aligned with its manifest row across split changes."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"sample_{index:04d}_{digest}{extension_from_url(url)}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_duration(path: Path) -> float:
    import soundfile as sf

    return round(float(sf.info(path).duration), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download remote audio in a manifest to local files.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--audio-dir", default="data/processed/audio_samples")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    def materialize(item: tuple[int, dict]) -> dict:
        idx, row = item
        source = row["audio_path"]
        if source.startswith("http://") or source.startswith("https://"):
            output_path = audio_dir / audio_filename(idx, source)
            if not output_path.exists():
                print(f"Downloading {idx + 1}/{len(rows)} -> {output_path}")
                urllib.request.urlretrieve(source, output_path)
            row = dict(row)
            row["remote_audio_path"] = source
            row["audio_path"] = str(output_path)
            row["duration_seconds"] = audio_duration(output_path)
            row["audio_sha256"] = file_sha256(output_path)
        return row

    with ThreadPoolExecutor(max_workers=min(args.workers, len(rows) or 1)) as executor:
        materialized = list(executor.map(materialize, enumerate(rows)))

    output = Path(args.output_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in materialized:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(materialized)} rows to {output}")


if __name__ == "__main__":
    main()

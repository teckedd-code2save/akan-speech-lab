from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akan_speech.asr.v06_manifest import V06_CODE_NAME, V06ManifestPaths, build_v06_manifest  # noqa: E402


def default_paths() -> V06ManifestPaths:
    return V06ManifestPaths(
        source_manifest=ROOT
        / "data/manifests/serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-replay-fullft-v0.1.jsonl",
        clean_gnlp_candidates=ROOT / "outputs/audits/gnlp_manifest_v0.5/clean_candidates.jsonl",
        output_manifest=ROOT / "data/manifests" / f"{V06_CODE_NAME}.jsonl",
        output_report=ROOT / "evals/reports" / f"{V06_CODE_NAME}-manifest.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the v0.6 cleaned GhanaNLP + Waxal replay ASR manifest."
    )
    parser.parse_args()
    report = build_v06_manifest(default_paths())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

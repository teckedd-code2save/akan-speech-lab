from __future__ import annotations

import argparse
import json
from pathlib import Path

from akan_speech.asr.artifacts import FIRST_ASR_REVIEW
from akan_speech.asr.sanitize import SanitizePaths, sanitize_v01


ROOT = Path(__file__).resolve().parents[1]


def default_paths() -> SanitizePaths:
    code = FIRST_ASR_REVIEW.code_name
    return SanitizePaths(
        waxal_metadata=ROOT / "data/manifests/waxal_aka_labeled_metadata.jsonl",
        waxal_round2_dir=ROOT / "data/manifests/waxal_round2",
        ghana_nlp_manifest=ROOT / "data/manifests/ghana_nlp_twi.jsonl",
        output_manifest=ROOT / "data/manifests" / f"{code}.jsonl",
        output_report=ROOT / "evals/reports" / f"{code}-sanitize.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitize the first ASR v0.1 corpus mix.")
    parser.parse_args()
    report = sanitize_v01(default_paths())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


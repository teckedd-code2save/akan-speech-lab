from __future__ import annotations

import argparse
import json
from pathlib import Path

from akan_speech.asr.artifacts import FIRST_ASR_REVIEW
from akan_speech.asr.pipeline import evaluate_pipeline, pipeline_next_action, render_pipeline_markdown


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the ASR artifact pipeline.")
    parser.add_argument("action", choices=["status", "next", "json"])
    args = parser.parse_args()

    if args.action == "status":
        print(render_pipeline_markdown(ROOT))
    elif args.action == "next":
        print(pipeline_next_action(ROOT))
    else:
        payload = {
            "code_name": FIRST_ASR_REVIEW.code_name,
            "hf_repo": FIRST_ASR_REVIEW.hf_repo,
            "stages": [
                {
                    "key": item.stage.key,
                    "title": item.stage.title,
                    "status": item.status,
                    "missing": [evidence.path for evidence in item.missing],
                    "present": [evidence.path for evidence in item.present],
                    "next_action": item.stage.next_action,
                }
                for item in evaluate_pipeline(ROOT)
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


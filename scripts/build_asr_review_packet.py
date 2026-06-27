from __future__ import annotations

import argparse
import json
from pathlib import Path

from akan_speech.asr.artifacts import FIRST_ASR_REVIEW, render_review_model_card


ROOT = Path(__file__).resolve().parents[1]


def build_review_packet(output_dir: Path, model_card_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_card_dir.mkdir(parents=True, exist_ok=True)

    spec = FIRST_ASR_REVIEW
    packet = {
        **spec.to_json(),
        "required_before_training": [
            "corpus audit for Waxal",
            "corpus audit for GhanaNLP",
            "manifest hashes recorded",
            "punctuated and WER-normalized transcript fields separated",
            "split leakage report complete",
            "replay-mixing ratios chosen",
        ],
        "required_before_hf_publish": [
            "held-out metrics recorded",
            "paired comparison complete",
            "failure taxonomy recorded",
            "Ghanaian qualitative review notes recorded",
            "model card updated from planned to trained artifact",
        ],
        "decision_after_review": "learn | relearn | unlearn | stop",
    }

    packet_path = output_dir / "review_packet.json"
    card_path = model_card_dir / "README.md"
    packet_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    card_path.write_text(render_review_model_card(spec), encoding="utf-8")
    return {
        "packet_path": str(packet_path.relative_to(ROOT)),
        "model_card_path": str(card_path.relative_to(ROOT)),
        "hf_repo": spec.hf_repo,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the first ASR review packet.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/review_packets" / FIRST_ASR_REVIEW.code_name,
    )
    parser.add_argument(
        "--model-card-dir",
        type=Path,
        default=ROOT / "model_cards" / FIRST_ASR_REVIEW.code_name,
    )
    args = parser.parse_args()

    result = build_review_packet(args.output_dir, args.model_card_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


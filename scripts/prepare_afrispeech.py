from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect/prepare AfriSpeech YouVersion data.")
    parser.add_argument("--dataset-id", default="AfriSpeech/youversion-african-speech")
    parser.parse_args()
    raise SystemExit(
        "AfriSpeech prep is intentionally gated. First inspect subsets, licensing, "
        "language labels, and transcript quality before mixing it into GhanaNLP."
    )


if __name__ == "__main__":
    main()


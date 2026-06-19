from __future__ import annotations

import argparse
import json
from pathlib import Path

from akan_speech.eval.bootstrap import paired_bootstrap_wer_difference


def load_run(path: Path, strategy: str) -> dict:
    report = json.loads(path.read_text())
    try:
        return report["runs"][strategy]
    except KeyError as error:
        raise ValueError(f"Strategy {strategy!r} is missing from {path}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare aligned ASR benchmark predictions")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--strategy", default="no_forced_language")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_run(args.baseline, args.strategy)
    candidate = load_run(args.candidate, args.strategy)
    comparison = {
        "strategy": args.strategy,
        "baseline_wer": baseline["metrics"]["wer"],
        "candidate_wer": candidate["metrics"]["wer"],
        "absolute_wer_change": candidate["metrics"]["wer"] - baseline["metrics"]["wer"],
        "paired_bootstrap_candidate_minus_baseline": paired_bootstrap_wer_difference(
            baseline["predictions"],
            candidate["predictions"],
            samples=args.samples,
            seed=args.seed,
        ),
        "bootstrap_samples": args.samples,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from akan_speech.eval.bootstrap import paired_bootstrap_wer_difference


def load_run(path: Path, run_name: str) -> dict:
    report = json.loads(path.read_text())
    try:
        return report["runs"][run_name]
    except KeyError as error:
        raise ValueError(f"Run {run_name!r} is missing from {path}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare aligned ASR benchmark predictions")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--strategy", default="no_forced_language")
    parser.add_argument("--baseline-run", default="")
    parser.add_argument("--candidate-run", default="")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_name = args.baseline_run or args.strategy
    candidate_name = args.candidate_run or args.strategy
    baseline = load_run(args.baseline, baseline_name)
    candidate = load_run(args.candidate, candidate_name)
    row_differences = [
        {
            "dataset_row": candidate_row.get("dataset_row"),
            "baseline_wer": baseline_row["wer"],
            "candidate_wer": candidate_row["wer"],
            "wer_change": candidate_row["wer"] - baseline_row["wer"],
            "reference": candidate_row["reference"],
            "baseline_prediction": baseline_row["prediction"],
            "candidate_prediction": candidate_row["prediction"],
        }
        for baseline_row, candidate_row in zip(
            baseline["predictions"], candidate["predictions"], strict=True
        )
    ]
    comparison = {
        "baseline_run": baseline_name,
        "candidate_run": candidate_name,
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
        "row_outcomes": {
            "candidate_better": sum(row["wer_change"] < 0 for row in row_differences),
            "tied": sum(row["wer_change"] == 0 for row in row_differences),
            "candidate_worse": sum(row["wer_change"] > 0 for row in row_differences),
            "candidate_catastrophic_regressions": sum(
                row["baseline_wer"] < 1 <= row["candidate_wer"] for row in row_differences
            ),
        },
        "largest_regressions": sorted(
            row_differences, key=lambda row: row["wer_change"], reverse=True
        )[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

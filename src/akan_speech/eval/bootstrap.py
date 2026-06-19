from __future__ import annotations

import random
from typing import Any

import numpy as np

from akan_speech.eval.wer import speech_error_rates


def bootstrap_wer_interval(
    rows: list[dict[str, Any]], *, samples: int = 1000, seed: int = 42
) -> dict[str, float]:
    if not rows:
        return {"low": 0.0, "median": 0.0, "high": 0.0}
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [rows[rng.randrange(len(rows))] for _ in rows]
        estimates.append(
            float(
                speech_error_rates(
                    [str(row.get("reference") or "") for row in selected],
                    [str(row.get("prediction") or "") for row in selected],
                )["wer"]
            )
        )
    low, median, high = np.percentile(estimates, [2.5, 50, 97.5])
    return {"low": float(low), "median": float(median), "high": float(high)}


def paired_bootstrap_wer_difference(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    samples: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError("Baseline and candidate must contain the same number of rows")
    if not baseline_rows:
        return {
            "low": 0.0,
            "median": 0.0,
            "high": 0.0,
            "probability_candidate_improves": 0.0,
        }
    for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True):
        if baseline.get("dataset_row") != candidate.get("dataset_row"):
            raise ValueError("Baseline and candidate rows are not aligned")
        if baseline.get("reference") != candidate.get("reference"):
            raise ValueError("Baseline and candidate references differ")

    baseline_counts = []
    candidate_counts = []
    for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True):
        reference = str(baseline.get("reference") or "")
        baseline_metrics = speech_error_rates(
            [reference], [str(baseline.get("prediction") or "")]
        )
        candidate_metrics = speech_error_rates(
            [reference], [str(candidate.get("prediction") or "")]
        )
        reference_words = int(baseline_metrics["reference_words"])
        baseline_errors = sum(
            int(baseline_metrics[key]) for key in ("substitutions", "deletions", "insertions")
        )
        candidate_errors = sum(
            int(candidate_metrics[key])
            for key in ("substitutions", "deletions", "insertions")
        )
        baseline_counts.append((baseline_errors, reference_words))
        candidate_counts.append((candidate_errors, reference_words))

    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        indices = [rng.randrange(len(baseline_rows)) for _ in baseline_rows]
        reference_words = sum(baseline_counts[index][1] for index in indices)
        baseline_wer = sum(baseline_counts[index][0] for index in indices) / reference_words
        candidate_wer = sum(candidate_counts[index][0] for index in indices) / reference_words
        differences.append(candidate_wer - baseline_wer)

    low, median, high = np.percentile(differences, [2.5, 50, 97.5])
    return {
        "low": float(low),
        "median": float(median),
        "high": float(high),
        "probability_candidate_improves": float(
            sum(difference < 0 for difference in differences) / samples
        ),
    }

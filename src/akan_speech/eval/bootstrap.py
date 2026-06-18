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

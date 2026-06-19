import pytest

from akan_speech.eval.bootstrap import (
    bootstrap_wer_interval,
    paired_bootstrap_wer_difference,
)


def test_bootstrap_interval_is_zero_for_perfect_predictions():
    rows = [{"reference": "me ho yɛ", "prediction": "me ho yɛ"}]

    interval = bootstrap_wer_interval(rows, samples=20, seed=1)

    assert interval == {"low": 0.0, "median": 0.0, "high": 0.0}


def test_paired_bootstrap_detects_better_candidate():
    baseline = [
        {"dataset_row": 1, "reference": "me ho yɛ", "prediction": "me yɛ"},
        {"dataset_row": 2, "reference": "maakye oo", "prediction": "maakye"},
    ]
    candidate = [
        {"dataset_row": 1, "reference": "me ho yɛ", "prediction": "me ho yɛ"},
        {"dataset_row": 2, "reference": "maakye oo", "prediction": "maakye oo"},
    ]

    result = paired_bootstrap_wer_difference(baseline, candidate, samples=100, seed=1)

    assert result["high"] < 0
    assert result["probability_candidate_improves"] == 1.0


def test_paired_bootstrap_rejects_misaligned_rows():
    with pytest.raises(ValueError, match="not aligned"):
        paired_bootstrap_wer_difference(
            [{"dataset_row": 1, "reference": "a", "prediction": "a"}],
            [{"dataset_row": 2, "reference": "a", "prediction": "a"}],
        )

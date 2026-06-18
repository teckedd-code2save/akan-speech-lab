from akan_speech.eval.bootstrap import bootstrap_wer_interval


def test_bootstrap_interval_is_zero_for_perfect_predictions():
    rows = [{"reference": "me ho yɛ", "prediction": "me ho yɛ"}]

    interval = bootstrap_wer_interval(rows, samples=20, seed=1)

    assert interval == {"low": 0.0, "median": 0.0, "high": 0.0}

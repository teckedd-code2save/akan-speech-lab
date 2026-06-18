from akan_speech.eval.breakdown import grouped_error_rates


def test_grouped_error_rates_reports_speakers_and_durations():
    rows = [
        {
            "speaker_id": "a",
            "duration_seconds": 2.0,
            "reference": "me ho yɛ",
            "prediction": "me ho yɛ",
        },
        {
            "speaker_id": "b",
            "duration_seconds": 12.0,
            "reference": "me ho yɛ",
            "prediction": "me ho nye",
        },
    ]

    speakers = grouped_error_rates(rows, "speaker_id")
    durations = grouped_error_rates(rows, "duration_bucket")

    assert speakers["a"]["wer"] == 0.0
    assert speakers["b"]["wer"] > 0.0
    assert durations["1s_to_3s"]["rows"] == 1
    assert durations["10s_to_20s"]["rows"] == 1

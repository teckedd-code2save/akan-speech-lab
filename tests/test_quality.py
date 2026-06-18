from akan_speech.data.quality import manifest_quality_report


def test_quality_report_summarizes_duration_and_audio_duplicates():
    rows = [
        {
            "split": "test",
            "language": "aka",
            "speaker_id": "a",
            "source": "waxal",
            "normalized_text": "one",
            "audio_path": "one.wav",
            "audio_sha256": "same",
            "duration_seconds": 10.0,
        },
        {
            "split": "test",
            "language": "aka",
            "speaker_id": "b",
            "source": "waxal",
            "normalized_text": "two",
            "audio_path": "two.wav",
            "audio_sha256": "same",
            "duration_seconds": 20.0,
        },
    ]

    report = manifest_quality_report(rows)

    assert report["measured_duration_rows"] == 2
    assert report["minimum_duration_seconds"] == 10.0
    assert report["maximum_duration_seconds"] == 20.0
    assert report["duplicate_audio_hash_groups"] == 1

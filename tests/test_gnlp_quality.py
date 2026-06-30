from akan_speech.data.gnlp_quality import audit_gnlp_rows


def test_gnlp_audit_flags_duplicate_and_bad_timing():
    rows = [
        {
            "record_id": "a",
            "split": "train",
            "audio_path": "hf://row-a",
            "text_normalized": "me ho yɛ",
            "text_hash": "same",
            "audio_hash": "audio-a",
            "duration_seconds": 0.4,
        },
        {
            "record_id": "b",
            "split": "validation",
            "audio_path": "hf://row-b",
            "text_normalized": "me ho yɛ",
            "text_hash": "same",
            "audio_hash": "audio-b",
            "duration_seconds": 3.0,
        },
    ]

    report, audited_rows = audit_gnlp_rows(rows)

    assert report["rows"] == 2
    assert report["flag_counts"]["duplicate_text"] == 2
    assert report["flag_counts"]["too_short_audio"] == 1
    assert not audited_rows[0]["clean_candidate"]


def test_gnlp_audit_keeps_clean_missing_speaker_as_candidate():
    report, audited_rows = audit_gnlp_rows(
        [
            {
                "record_id": "clean",
                "split": "train",
                "audio_path": "hf://row-clean",
                "text_normalized": "me ho yɛ papa paa",
                "text_hash": "clean-text",
                "audio_hash": "clean-audio",
                "duration_seconds": 3.5,
            }
        ]
    )

    assert report["clean_candidates"] == 1
    assert audited_rows[0]["flags"] == ["missing_speaker_id"]
    assert audited_rows[0]["clean_candidate"]

from akan_speech.data.audit import corpus_audit


def test_corpus_audit_detects_cross_split_leakage():
    rows = [
        {
            "split": "train",
            "speaker_id": "speaker-a",
            "sample_id": "train-1",
            "normalized_text": "me ho yɛ papa",
            "dataset_row": 1,
            "language": "aka",
        },
        {
            "split": "test",
            "speaker_id": "speaker-a",
            "sample_id": "test-1",
            "normalized_text": "me ho yɛ papa",
            "dataset_row": 2,
            "language": "aka",
        },
    ]

    report = corpus_audit(rows)

    assert report["cross_split_text_groups"] == 1
    assert report["speaker_overlap"]["test_x_train"]["count"] == 1


def test_corpus_audit_does_not_count_missing_speaker_as_one_person():
    report = corpus_audit(
        [
            {
                "split": "train",
                "speaker_id": None,
                "sample_id": "row-1",
                "normalized_text": "maakye",
                "dataset_row": 1,
                "language": "aka",
            }
        ]
    )

    assert report["unique_speakers"] == 0
    assert report["speakers_by_split"] == {}

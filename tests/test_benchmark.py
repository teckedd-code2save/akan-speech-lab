from scripts.freeze_waxal_benchmark import select_balanced_test_rows


def test_benchmark_is_balanced_and_excludes_train_text_overlap():
    rows = [
        {"split": "train", "speaker_id": "train", "normalized_text": "duplicate"},
        {"split": "test", "speaker_id": "a", "sample_id": "a0", "dataset_row": 0, "normalized_text": "duplicate"},
        {"split": "test", "speaker_id": "a", "sample_id": "a1", "dataset_row": 1, "normalized_text": "one"},
        {"split": "test", "speaker_id": "a", "sample_id": "a2", "dataset_row": 2, "normalized_text": "two"},
        {"split": "test", "speaker_id": "b", "sample_id": "b1", "dataset_row": 3, "normalized_text": "three"},
        {"split": "test", "speaker_id": "b", "sample_id": "b2", "dataset_row": 4, "normalized_text": "four"},
    ]

    selected, report = select_balanced_test_rows(
        rows, per_speaker=2, seed=42, exclude_train_text_overlap=True
    )

    assert len(selected) == 4
    assert report["excluded_train_text_overlap"] == 1
    assert {row["speaker_id"] for row in selected} == {"a", "b"}

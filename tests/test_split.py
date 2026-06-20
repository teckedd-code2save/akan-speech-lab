import pytest

from akan_speech.data.split import stable_group_split


def test_stable_group_split_keeps_duplicate_text_together():
    assert stable_group_split("me ho yɛ", seed=42) == stable_group_split("me ho yɛ", seed=42)


def test_stable_group_split_produces_all_partitions():
    splits = {stable_group_split(f"utterance {index}") for index in range(1000)}
    assert splits == {"train", "validation", "test"}


def test_stable_group_split_rejects_invalid_fractions():
    with pytest.raises(ValueError):
        stable_group_split("text", train_fraction=0.98, validation_fraction=0.05)

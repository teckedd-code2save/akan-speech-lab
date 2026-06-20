import pytest

from akan_speech.data.round2 import build_speaker_component_split


def row(sample_id: str, speaker: str, text: str) -> dict:
    return {"sample_id": sample_id, "speaker_id": speaker, "text": text}


def test_round2_split_is_speaker_and_transcript_disjoint():
    pool = [
        row("a1", "a", "maakye"),
        row("a2", "a", "me ho yɛ"),
        row("b1", "b", "maakye"),
        row("c1", "c", "medaase"),
        row("d1", "d", "yɛbɛhyia"),
        row("e1", "e", "akwaaba"),
    ]
    test = [row("t1", "test-speaker", "nante yie")]

    assigned, audit = build_speaker_component_split(pool, test, seed=42, train_fraction=0.6)

    split_by_speaker = {item["speaker_id"]: item["split"] for item in assigned}
    assert split_by_speaker["a"] == split_by_speaker["b"]
    assert audit["passed"] is True
    assert not any(audit["assertions"].values())


def test_round2_split_rejects_test_transcript_overlap():
    with pytest.raises(ValueError, match="transcripts overlap"):
        build_speaker_component_split(
            [row("a1", "a", "maakye")],
            [row("t1", "test-speaker", "Maakye!")],
        )


def test_round2_split_rejects_missing_speaker():
    with pytest.raises(ValueError, match="requires speaker_id"):
        build_speaker_component_split(
            [row("a1", "", "maakye")],
            [row("t1", "test-speaker", "medaase")],
        )

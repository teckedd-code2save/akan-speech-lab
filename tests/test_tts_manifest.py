from akan_speech.tts.manifest import finalize_manifest


def test_manifest_is_text_disjoint_and_rejects_duplicates_and_numbers():
    rows = [
        {"text": "Me ho yɛ", "audio_sha256": "a", "flags": []},
        {"text": "Me ho yɛ", "audio_sha256": "b", "flags": []},
        {"text": "Merekɔ Kumasi", "audio_sha256": "c", "flags": []},
        {"text": "Fa 20 bra", "audio_sha256": "d", "flags": []},
    ]
    manifest, report = finalize_manifest(rows)
    assert manifest[0]["accepted"] is False
    assert manifest[1]["accepted"] is False
    assert "numeric_review" in manifest[3]["flags"]
    assert report["normalized_text_overlap"] == {
        "train_x_validation": 0,
        "train_x_test": 0,
        "validation_x_test": 0,
    }

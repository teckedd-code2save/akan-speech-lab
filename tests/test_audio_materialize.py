from scripts.materialize_manifest_audio import audio_filename


def test_audio_cache_key_changes_with_source_url():
    first = audio_filename(0, "https://example.com/train/sample.mp3")
    second = audio_filename(0, "https://example.com/test/sample.mp3")

    assert first != second
    assert first.endswith(".mp3")

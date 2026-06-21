from akan_speech.tts.text import normalize_tts_text, unresolved_numeric_tokens


def test_normalization_preserves_akan_orthography():
    assert normalize_tts_text("  Ɔbaa no…  ɛyɛ  ") == "Ɔbaa no... ɛyɛ"


def test_numeric_tokens_are_quarantined_not_guessed():
    assert unresolved_numeric_tokens("Fa GH₵20 kɔ 2026 mu") == ["GH₵20", "2026"]

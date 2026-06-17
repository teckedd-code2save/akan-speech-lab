from akan_speech.data.normalize import normalize_akan_text, normalize_language_code
from akan_speech.eval.wer import speech_error_rates


def test_normalize_akan_text_keeps_diacritics():
    assert normalize_akan_text("Ɛnnɛ, me ho yɛ!") == "ɛnnɛ me ho yɛ"


def test_normalize_language_aliases():
    assert normalize_language_code("twi") == "aka"
    assert normalize_language_code("fante") == "fat"


def test_speech_error_rates_expose_word_error_counts():
    metrics = speech_error_rates(
        ["me ho yɛ papa"],
        ["me ho yɛ yie paa"],
    )

    assert metrics["reference_words"] == 4
    assert metrics["hits"] == 3
    assert metrics["substitutions"] == 1
    assert metrics["deletions"] == 0
    assert metrics["insertions"] == 1
    assert metrics["wer"] == 0.5

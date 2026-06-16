from akan_speech.data.normalize import normalize_akan_text, normalize_language_code


def test_normalize_akan_text_keeps_diacritics():
    assert normalize_akan_text("Ɛnnɛ, me ho yɛ!") == "ɛnnɛ me ho yɛ"


def test_normalize_language_aliases():
    assert normalize_language_code("twi") == "aka"
    assert normalize_language_code("fante") == "fat"


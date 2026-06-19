from akan_speech.inference.guard import (
    has_repetition_collapse,
    longest_consecutive_token_repeat,
)


def test_detects_repeated_token_collapse():
    assert longest_consecutive_token_repeat("me ho ye ye ye ye ye") == 5
    assert has_repetition_collapse("me ho ye ye ye ye ye")


def test_allows_normal_repetition():
    assert longest_consecutive_token_repeat("se se, na afei se bio") == 2
    assert not has_repetition_collapse("se se, na afei se bio")


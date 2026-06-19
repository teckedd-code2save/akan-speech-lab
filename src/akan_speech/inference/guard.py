from __future__ import annotations

from akan_speech.data.normalize import normalize_akan_text


def longest_consecutive_token_repeat(text: str) -> int:
    tokens = normalize_akan_text(text).split()
    if not tokens:
        return 0
    longest = current = 1
    for previous, token in zip(tokens, tokens[1:], strict=False):
        current = current + 1 if token == previous else 1
        longest = max(longest, current)
    return longest


def has_repetition_collapse(text: str, *, threshold: int = 5) -> bool:
    return longest_consecutive_token_repeat(text) >= threshold


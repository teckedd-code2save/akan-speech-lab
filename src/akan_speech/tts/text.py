from __future__ import annotations

import re
import unicodedata


WHITESPACE_RE = re.compile(r"\s+")
NUMERIC_RE = re.compile(r"(?<!\w)(?:(?:GH)?₵|[$£€])?\d[\d,.]*(?!\w)", re.IGNORECASE)


def normalize_tts_text(text: str) -> str:
    """Normalize typography while preserving Akan letters and spoken punctuation cues."""

    value = unicodedata.normalize("NFC", text or "").strip()
    value = value.translate(
        str.maketrans(
            {
                "“": '"',
                "”": '"',
                "‘": "'",
                "’": "'",
                "–": "-",
                "—": "-",
                "…": "...",
            }
        )
    )
    return WHITESPACE_RE.sub(" ", value).strip()


def unresolved_numeric_tokens(text: str) -> list[str]:
    """Return numbers/currencies that need an explicitly reviewed spoken-Twi expansion."""

    return NUMERIC_RE.findall(normalize_tts_text(text))


def character_inventory(texts: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        for character in normalize_tts_text(text):
            counts[character] = counts.get(character, 0) + 1
    return dict(sorted(counts.items()))

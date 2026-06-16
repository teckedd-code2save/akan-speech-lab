from __future__ import annotations

import re
import unicodedata

PUNCTUATION_RE = re.compile(r"[“”\"'‘’.,!?;:()\[\]{}<>/\\|*_+=~`]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_akan_text(text: str, *, keep_apostrophe: bool = False) -> str:
    """Normalize Akan transcripts for ASR training and WER evaluation.

    The function keeps Akan diacritics, lowercases text, removes punctuation,
    and collapses whitespace. It avoids aggressive English-centric spelling
    changes because those can damage dialectal forms.
    """

    value = unicodedata.normalize("NFC", text or "").strip().lower()
    if keep_apostrophe:
        value = re.sub(r"[“”\"‘’.,!?;:()\[\]{}<>/\\|*_+=~`]", " ", value)
    else:
        value = PUNCTUATION_RE.sub(" ", value)
    value = WHITESPACE_RE.sub(" ", value)
    return value.strip()


def normalize_language_code(language: str | None) -> str:
    value = (language or "aka").strip().lower()
    aliases = {
        "twi": "aka",
        "akan": "aka",
        "ak": "aka",
        "fante": "fat",
        "fanti": "fat",
    }
    return aliases.get(value, value)


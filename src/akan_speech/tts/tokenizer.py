from __future__ import annotations

from typing import Any

from akan_speech.tts.text import character_inventory, normalize_tts_text


def tokenizer_coverage(tokenizer: Any, texts: list[str]) -> dict:
    unknown_id = tokenizer.unk_token_id
    unknown_rows = []
    for index, text in enumerate(texts):
        normalized = normalize_tts_text(text)
        ids = tokenizer(normalized, add_special_tokens=False).input_ids
        if unknown_id in ids:
            unknown_rows.append(index)
    return {
        "rows": len(texts),
        "unknown_rows": unknown_rows,
        "unknown_row_count": len(unknown_rows),
        "characters": character_inventory(texts),
        "passed": bool(texts) and not unknown_rows,
    }


def extend_character_tokenizer(tokenizer: Any, texts: list[str]) -> dict:
    inventory = character_inventory(texts)
    missing = []
    for character in inventory:
        if character.isspace():
            continue
        ids = tokenizer(character, add_special_tokens=False).input_ids
        if tokenizer.unk_token_id in ids:
            missing.append(character)
    added = tokenizer.add_tokens(missing)
    coverage = tokenizer_coverage(tokenizer, texts)
    return {"missing_characters": missing, "added_tokens": added, **coverage}

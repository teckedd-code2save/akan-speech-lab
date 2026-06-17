from __future__ import annotations

from jiwer import process_characters, process_words

from akan_speech.data.normalize import normalize_akan_text


def speech_error_rates(references: list[str], predictions: list[str]) -> dict[str, float | int]:
    norm_refs = [normalize_akan_text(item) for item in references]
    norm_preds = [normalize_akan_text(item) for item in predictions]
    word_result = process_words(norm_refs, norm_preds)
    char_result = process_characters(norm_refs, norm_preds)
    return {
        "wer": float(word_result.wer),
        "cer": float(char_result.cer),
        "reference_words": int(word_result.hits + word_result.substitutions + word_result.deletions),
        "hits": int(word_result.hits),
        "substitutions": int(word_result.substitutions),
        "deletions": int(word_result.deletions),
        "insertions": int(word_result.insertions),
    }

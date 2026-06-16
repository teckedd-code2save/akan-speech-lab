from __future__ import annotations

from jiwer import cer, wer

from akan_speech.data.normalize import normalize_akan_text


def speech_error_rates(references: list[str], predictions: list[str]) -> dict[str, float]:
    norm_refs = [normalize_akan_text(item) for item in references]
    norm_preds = [normalize_akan_text(item) for item in predictions]
    return {
        "wer": float(wer(norm_refs, norm_preds)),
        "cer": float(cer(norm_refs, norm_preds)),
    }


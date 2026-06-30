from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Any


ENGLISH_MARKERS = {
    "and",
    "because",
    "book",
    "church",
    "for",
    "god",
    "lord",
    "school",
    "the",
    "to",
    "you",
}
RELIGIOUS_MARKERS = {
    "adom",
    "adwura",
    "adwurade",
    "asɔre",
    "honhom",
    "kronkron",
    "mpae",
    "nyame",
}
HEALTH_MARKERS = {
    "ayaresa",
    "ayaresabea",
    "aduro",
    "yare",
    "yaredɔm",
    "apɔmuden",
    "yaw",
}


@dataclass(frozen=True)
class RowQuality:
    flags: list[str]
    tags: list[str]
    word_count: int
    char_count: int
    words_per_second: float | None
    clean_candidate: bool


def words(text: str) -> list[str]:
    return [item for item in re.split(r"\s+", (text or "").strip()) if item]


def summarize_numbers(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": round(ordered[0], 4),
        "median": round(float(median(ordered)), 4),
        "max": round(ordered[-1], 4),
    }


def quality_for_row(
    row: dict[str, Any], duplicate_texts: set[str], duplicate_audios: set[str]
) -> RowQuality:
    text = str(row.get("text_normalized") or row.get("normalized_text") or "").strip()
    row_words = words(text)
    duration = row.get("duration_seconds")
    duration_float = float(duration) if duration is not None else None
    word_count = len(row_words)
    char_count = len(text)
    words_per_second = (
        round(word_count / duration_float, 4)
        if duration_float is not None and duration_float > 0
        else None
    )
    text_hash = str(row.get("text_hash") or "")
    audio_hash = str(row.get("audio_hash") or "")

    flags: list[str] = []
    tags: list[str] = []
    token_set = set(row_words)

    if not text:
        flags.append("empty_text")
    if not row.get("audio_path"):
        flags.append("missing_audio")
    if row.get("speaker_id") in {None, "", "unknown"}:
        flags.append("missing_speaker_id")
    if duration_float is None:
        flags.append("missing_duration")
    elif duration_float < 1.0:
        flags.append("too_short_audio")
    elif duration_float > 18.0:
        flags.append("too_long_audio")
    if word_count < 2:
        flags.append("too_few_words")
    elif word_count > 45:
        flags.append("too_many_words")
    if words_per_second is not None and (words_per_second < 0.75 or words_per_second > 4.75):
        flags.append("suspicious_words_per_second")
    if text_hash and text_hash in duplicate_texts:
        flags.append("duplicate_text")
    if audio_hash and audio_hash in duplicate_audios:
        flags.append("duplicate_audio")
    if any(char.isdigit() for char in text):
        flags.append("contains_digits")

    if token_set & ENGLISH_MARKERS:
        tags.append("english_or_codeswitch_marker")
    if token_set & RELIGIOUS_MARKERS:
        tags.append("religious_domain")
    if token_set & HEALTH_MARKERS:
        tags.append("health_domain")

    severe_flags = {
        "empty_text",
        "missing_audio",
        "missing_duration",
        "too_short_audio",
        "too_long_audio",
        "too_few_words",
        "too_many_words",
        "suspicious_words_per_second",
        "duplicate_text",
        "duplicate_audio",
        "contains_digits",
    }
    clean_candidate = not (set(flags) & severe_flags)
    return RowQuality(
        flags=flags,
        tags=tags,
        word_count=word_count,
        char_count=char_count,
        words_per_second=words_per_second,
        clean_candidate=clean_candidate,
    )


def audit_gnlp_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    text_counts = Counter(str(row.get("text_hash") or "") for row in rows if row.get("text_hash"))
    audio_counts = Counter(
        str(row.get("audio_hash") or "") for row in rows if row.get("audio_hash")
    )
    duplicate_texts = {key for key, count in text_counts.items() if count > 1}
    duplicate_audios = {key for key, count in audio_counts.items() if count > 1}
    audited_rows = []

    for row in rows:
        quality = quality_for_row(row, duplicate_texts, duplicate_audios)
        audited_rows.append(
            {
                "record_id": row.get("record_id"),
                "split": row.get("split"),
                "dataset_row": row.get("dataset_row"),
                "sample_id": row.get("sample_id"),
                "duration_seconds": row.get("duration_seconds"),
                "word_count": quality.word_count,
                "char_count": quality.char_count,
                "words_per_second": quality.words_per_second,
                "clean_candidate": quality.clean_candidate,
                "flags": quality.flags,
                "tags": quality.tags,
                "text_normalized": row.get("text_normalized") or row.get("normalized_text"),
                "audio_path": row.get("audio_path"),
                "text_hash": row.get("text_hash"),
                "audio_hash": row.get("audio_hash"),
            }
        )

    flag_counts = Counter(flag for row in audited_rows for flag in row["flags"])
    tag_counts = Counter(tag for row in audited_rows for tag in row["tags"])
    split_counts = Counter(str(row.get("split") or "unknown") for row in audited_rows)
    clean_by_split = Counter(
        str(row.get("split") or "unknown") for row in audited_rows if row["clean_candidate"]
    )
    report = {
        "corpus": "ghananlpcommunity/twi-speech-text-multispeaker-16k",
        "rows": len(rows),
        "splits": dict(split_counts),
        "clean_candidates": sum(row["clean_candidate"] for row in audited_rows),
        "clean_candidates_by_split": dict(clean_by_split),
        "flag_counts": dict(flag_counts),
        "tag_counts": dict(tag_counts),
        "duration_seconds": summarize_numbers(
            [
                float(row["duration_seconds"])
                for row in audited_rows
                if row["duration_seconds"] is not None
            ]
        ),
        "word_count": summarize_numbers([float(row["word_count"]) for row in audited_rows]),
        "words_per_second": summarize_numbers(
            [
                float(row["words_per_second"])
                for row in audited_rows
                if row["words_per_second"] is not None
            ]
        ),
        "missing_speaker_note": (
            "GhanaNLP rows in the frozen manifest do not expose stable speaker IDs, so this corpus "
            "cannot support a speaker-disjoint split without extra source metadata."
        ),
        "next_training_use": (
            "Use clean candidates as adaptation data only, replay Waxal in every run, and keep Waxal "
            "dev/test as the regression gate."
        ),
    }
    return report, audited_rows

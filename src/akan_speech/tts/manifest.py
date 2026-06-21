from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from akan_speech.data.split import stable_group_split
from akan_speech.tts.text import normalize_tts_text, unresolved_numeric_tokens


REJECTING_FLAGS = {
    "invalid_audio",
    "duration_outside_1_15s",
    "clipping",
    "excessive_silence",
    "low_loudness",
    "empty_text",
    "numeric_review",
    "duplicate_audio",
    "duplicate_text",
}


def text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_tts_text(text).casefold().encode("utf-8")).hexdigest()


def finalize_manifest(rows: list[dict[str, Any]], *, seed: int = 42) -> tuple[list[dict], dict]:
    audio_counts = Counter(str(row.get("audio_sha256") or "") for row in rows)
    text_counts = Counter(
        text_sha256(str(row.get("normalized_text") or row.get("text") or "")) for row in rows
    )
    finalized = []
    for source in rows:
        row = dict(source)
        normalized = normalize_tts_text(str(row.get("normalized_text") or row.get("text") or ""))
        flags = set(row.get("flags") or [])
        if not normalized:
            flags.add("empty_text")
        if unresolved_numeric_tokens(normalized):
            flags.add("numeric_review")
        audio_hash = str(row.get("audio_sha256") or "")
        normalized_hash = text_sha256(normalized)
        if audio_hash and audio_counts[audio_hash] > 1:
            flags.add("duplicate_audio")
        if normalized and text_counts[normalized_hash] > 1:
            flags.add("duplicate_text")
        accepted = not bool(flags & REJECTING_FLAGS)
        row.update(
            {
                "normalized_text": normalized,
                "text_sha256": normalized_hash,
                "flags": sorted(flags),
                "accepted": accepted,
                "split": stable_group_split(
                    normalized_hash,
                    seed=seed,
                    train_fraction=0.8,
                    validation_fraction=0.1,
                ),
            }
        )
        finalized.append(row)

    accepted_rows = [row for row in finalized if row["accepted"]]
    overlap = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_hashes = {row["text_sha256"] for row in accepted_rows if row["split"] == left}
        right_hashes = {row["text_sha256"] for row in accepted_rows if row["split"] == right}
        overlap[f"{left}_x_{right}"] = len(left_hashes & right_hashes)
    report = {
        "rows": len(finalized),
        "accepted": len(accepted_rows),
        "rejected": len(finalized) - len(accepted_rows),
        "splits": dict(Counter(row["split"] for row in accepted_rows)),
        "flags": dict(Counter(flag for row in finalized for flag in row["flags"])),
        "normalized_text_overlap": overlap,
        "passed": bool(accepted_rows) and not any(overlap.values()),
    }
    return finalized, report

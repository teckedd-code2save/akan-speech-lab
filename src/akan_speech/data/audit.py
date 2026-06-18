from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any


ORTHOGRAPHIC_MARKERS = {
    "fante_leaning": {"dze", "dɛ", "hɔn", "nyimpa", "mbofra", "mbaa", "kyerɛ dɛ"},
    "twi_leaning": {"de", "sɛ", "wɔn", "nipa", "mmofra", "maa", "kyerɛ sɛ"},
}


def text_fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def corpus_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    splits = Counter(str(row.get("split") or "unknown") for row in records)
    languages = Counter(str(row.get("language") or "unknown") for row in records)
    speakers_by_split: dict[str, set[str]] = defaultdict(set)
    ids: dict[str, list[tuple[str, int | None]]] = defaultdict(list)
    texts: dict[str, list[tuple[str, int | None]]] = defaultdict(list)
    marker_counts = Counter()
    empty_text = 0

    for row in records:
        split = str(row.get("split") or "unknown")
        speaker = str(row.get("speaker_id") or "unknown")
        speakers_by_split[split].add(speaker)
        sample_id = str(row.get("sample_id") or "")
        normalized = str(row.get("normalized_text") or "").strip()
        location = (split, row.get("dataset_row"))
        if sample_id:
            ids[sample_id].append(location)
        if normalized:
            texts[text_fingerprint(normalized)].append(location)
        else:
            empty_text += 1
        padded = f" {normalized} "
        for family, markers in ORTHOGRAPHIC_MARKERS.items():
            for marker in markers:
                marker_counts[f"{family}:{marker}"] += padded.count(f" {marker} ")

    speaker_overlap = {}
    for left, right in combinations(sorted(speakers_by_split), 2):
        shared = speakers_by_split[left] & speakers_by_split[right]
        speaker_overlap[f"{left}_x_{right}"] = {
            "count": len(shared),
            "speakers": sorted(shared),
        }

    duplicate_ids = {key: value for key, value in ids.items() if len(value) > 1}
    duplicate_texts = {key: value for key, value in texts.items() if len(value) > 1}
    cross_split_texts = {
        key: value
        for key, value in duplicate_texts.items()
        if len({split for split, _ in value}) > 1
    }
    return {
        "records": len(records),
        "splits": dict(splits),
        "languages": dict(languages),
        "unique_speakers": len(set().union(*speakers_by_split.values())) if speakers_by_split else 0,
        "speakers_by_split": {key: len(value) for key, value in speakers_by_split.items()},
        "speaker_overlap": speaker_overlap,
        "empty_text": empty_text,
        "duplicate_id_groups": len(duplicate_ids),
        "duplicate_text_groups": len(duplicate_texts),
        "cross_split_text_groups": len(cross_split_texts),
        "cross_split_text_examples": list(cross_split_texts.values())[:20],
        "orthographic_marker_counts": dict(marker_counts),
        "dialect_note": (
            "Waxal labels these rows as aka and does not provide dialect labels. Marker counts are "
            "orthographic evidence only, not automatic Twi/Fante ground truth."
        ),
    }

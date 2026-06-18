from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any


@dataclass(frozen=True)
class DurationBucket:
    name: str
    minimum: float
    maximum: float | None


DURATION_BUCKETS = [
    DurationBucket("under_1s", 0.0, 1.0),
    DurationBucket("1s_to_3s", 1.0, 3.0),
    DurationBucket("3s_to_10s", 3.0, 10.0),
    DurationBucket("10s_to_20s", 10.0, 20.0),
    DurationBucket("over_20s", 20.0, None),
]


def duration_bucket(duration_seconds: float | None) -> str:
    if duration_seconds is None:
        return "unknown"
    for bucket in DURATION_BUCKETS:
        if duration_seconds >= bucket.minimum and (
            bucket.maximum is None or duration_seconds < bucket.maximum
        ):
            return bucket.name
    return "unknown"


def manifest_quality_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = Counter(row.get("split", "train") for row in records)
    language_counts = Counter(row.get("language", "unknown") for row in records)
    source_counts = Counter(row.get("source", "unknown") for row in records)
    duration_counts = Counter(duration_bucket(row.get("duration_seconds")) for row in records)
    speaker_counts = Counter(row.get("speaker_id") or "unknown" for row in records)
    speakers_by_split: dict[str, set[str]] = defaultdict(set)
    empty_text = 0
    missing_audio = 0
    audio_hashes = Counter()
    durations = []

    for row in records:
        split = row.get("split", "train")
        speaker_id = row.get("speaker_id") or "unknown"
        speakers_by_split[split].add(speaker_id)
        if not str(row.get("normalized_text") or "").strip():
            empty_text += 1
        if not str(row.get("audio_path") or "").strip():
            missing_audio += 1
        if row.get("audio_sha256"):
            audio_hashes[str(row["audio_sha256"])] += 1
        if row.get("duration_seconds") is not None:
            durations.append(float(row["duration_seconds"]))

    overlap = {}
    for left, right in combinations(sorted(speakers_by_split), 2):
        shared = speakers_by_split[left] & speakers_by_split[right]
        overlap[f"{left}_x_{right}"] = len(shared)

    return {
        "records": len(records),
        "splits": dict(split_counts),
        "languages": dict(language_counts),
        "sources": dict(source_counts),
        "duration_buckets": dict(duration_counts),
        "unique_speakers": len(speaker_counts),
        "top_speakers": speaker_counts.most_common(10),
        "speakers_by_split": {key: len(value) for key, value in speakers_by_split.items()},
        "speaker_overlap": overlap,
        "empty_text": empty_text,
        "missing_audio": missing_audio,
        "measured_duration_rows": len(durations),
        "total_audio_hours": round(sum(durations) / 3600, 4),
        "minimum_duration_seconds": round(min(durations), 4) if durations else None,
        "maximum_duration_seconds": round(max(durations), 4) if durations else None,
        "duplicate_audio_hash_groups": sum(1 for count in audio_hashes.values() if count > 1),
        "valid": bool(records) and empty_text == 0 and missing_audio == 0,
    }

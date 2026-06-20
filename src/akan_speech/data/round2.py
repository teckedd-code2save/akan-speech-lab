from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from akan_speech.data.normalize import normalize_akan_text


class UnionFind:
    def __init__(self, values: set[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def build_speaker_component_split(
    pool_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    seed: int = 42,
    train_fraction: float = 0.85,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    rows = []
    for row in pool_rows:
        item = dict(row)
        item["speaker_id"] = str(item.get("speaker_id") or "").strip()
        item["sample_id"] = str(item.get("sample_id") or "").strip()
        item["normalized_text"] = normalize_akan_text(str(item.get("text") or ""))
        if not item["speaker_id"] or not item["sample_id"] or not item["normalized_text"]:
            raise ValueError("Every pool row requires speaker_id, sample_id, and non-empty text")
        rows.append(item)

    normalized_test = [
        {
            **row,
            "speaker_id": str(row.get("speaker_id") or "").strip(),
            "sample_id": str(row.get("sample_id") or "").strip(),
            "normalized_text": normalize_akan_text(str(row.get("text") or "")),
        }
        for row in test_rows
    ]
    test_speakers = {row["speaker_id"] for row in normalized_test}
    test_ids = {row["sample_id"] for row in normalized_test}
    test_texts = {row["normalized_text"] for row in normalized_test}
    if {row["speaker_id"] for row in rows} & test_speakers:
        raise ValueError("Published test speakers overlap the training pool")
    if {row["sample_id"] for row in rows} & test_ids:
        raise ValueError("Published test sample IDs overlap the training pool")
    if {row["normalized_text"] for row in rows} & test_texts:
        raise ValueError("Published test transcripts overlap the training pool")

    speakers = {row["speaker_id"] for row in rows}
    components = UnionFind(speakers)
    speakers_by_text: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        speakers_by_text[row["normalized_text"]].append(row["speaker_id"])
    for text_speakers in speakers_by_text.values():
        first = text_speakers[0]
        for speaker in text_speakers[1:]:
            components.union(first, speaker)

    component_members: dict[str, set[str]] = defaultdict(set)
    for speaker in speakers:
        component_members[components.find(speaker)].add(speaker)
    component_split = {
        root: "train" if stable_fraction("|".join(sorted(members)), seed) < train_fraction else "dev"
        for root, members in component_members.items()
    }
    for row in rows:
        row["split"] = component_split[components.find(row["speaker_id"])]

    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    train_speakers = {row["speaker_id"] for row in train_rows}
    dev_speakers = {row["speaker_id"] for row in dev_rows}
    train_texts = {row["normalized_text"] for row in train_rows}
    dev_texts = {row["normalized_text"] for row in dev_rows}
    assertions = {
        "train_dev_speaker_overlap": len(train_speakers & dev_speakers),
        "train_dev_transcript_overlap": len(train_texts & dev_texts),
        "train_test_speaker_overlap": len(train_speakers & test_speakers),
        "dev_test_speaker_overlap": len(dev_speakers & test_speakers),
        "train_test_transcript_overlap": len(train_texts & test_texts),
        "dev_test_transcript_overlap": len(dev_texts & test_texts),
    }
    if any(assertions.values()):
        raise ValueError(f"Contamination assertions failed: {assertions}")
    audit = {
        "seed": seed,
        "train_fraction": train_fraction,
        "rows": {"train": len(train_rows), "dev": len(dev_rows), "test": len(test_rows)},
        "speakers": {
            "train": len(train_speakers),
            "dev": len(dev_speakers),
            "test": len(test_speakers),
        },
        "speaker_components": len(component_members),
        "assertions": assertions,
        "passed": not any(assertions.values()) and bool(train_rows) and bool(dev_rows),
    }
    return rows, audit

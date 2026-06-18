from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from akan_speech.data.normalize import normalize_akan_text, normalize_language_code


@dataclass(frozen=True)
class SpeechRecord:
    audio_path: str
    text: str
    normalized_text: str
    language: str = "aka"
    speaker_id: str | None = None
    duration_seconds: float | None = None
    split: str = "train"
    source: str = ""
    dataset_row: int | None = None

    def to_json(self) -> dict:
        return {
            "audio_path": self.audio_path,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "language": self.language,
            "speaker_id": self.speaker_id,
            "duration_seconds": self.duration_seconds,
            "split": self.split,
            "source": self.source,
            "dataset_row": self.dataset_row,
        }


def make_record(
    *,
    audio_path: str,
    text: str,
    language: str = "aka",
    speaker_id: str | None = None,
    duration_seconds: float | None = None,
    split: str = "train",
    source: str = "",
    dataset_row: int | None = None,
) -> SpeechRecord:
    return SpeechRecord(
        audio_path=audio_path,
        text=(text or "").strip(),
        normalized_text=normalize_akan_text(text),
        language=normalize_language_code(language),
        speaker_id=speaker_id,
        duration_seconds=duration_seconds,
        split=split,
        source=source,
        dataset_row=dataset_row,
    )


def write_jsonl(records: Iterable[SpeechRecord], output_path: str | Path) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if not record.audio_path or not record.normalized_text:
                continue
            handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def validate_manifest(path: str | Path) -> dict:
    records = read_jsonl(path)
    missing_audio = [idx for idx, row in enumerate(records) if not row.get("audio_path")]
    missing_text = [idx for idx, row in enumerate(records) if not row.get("normalized_text")]
    splits = sorted({row.get("split", "train") for row in records})
    languages = sorted({row.get("language", "aka") for row in records})
    return {
        "records": len(records),
        "splits": splits,
        "languages": languages,
        "missing_audio": len(missing_audio),
        "missing_text": len(missing_text),
        "valid": bool(records) and not missing_audio and not missing_text,
    }

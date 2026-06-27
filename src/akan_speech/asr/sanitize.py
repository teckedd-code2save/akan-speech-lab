from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from akan_speech.asr.artifacts import FIRST_ASR_REVIEW, AsrArtifactSpec
from akan_speech.data.normalize import normalize_akan_text


@dataclass(frozen=True)
class SanitizePaths:
    waxal_metadata: Path
    waxal_round2_dir: Path
    ghana_nlp_manifest: Path
    output_manifest: Path
    output_report: Path


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def base_record(
    *,
    spec: AsrArtifactSpec,
    corpus: str,
    source_dataset: str,
    source_config: str,
    row: dict[str, Any],
    split: str,
    audio_path: str,
    text: str,
    source_split: str,
    duration_seconds: float | None,
    speaker_id: str | None,
    sample_id: str,
    dataset_row: int | None,
) -> dict[str, Any]:
    normalized = normalize_akan_text(text)
    return {
        "artifact_code": spec.code_name,
        "record_id": stable_hash(f"{corpus}:{sample_id}:{split}:{dataset_row}")[:24],
        "corpus": corpus,
        "source_dataset": source_dataset,
        "source_config": source_config,
        "source_split": source_split,
        "split": split,
        "audio_path": audio_path,
        "audio_hash": stable_hash(audio_path),
        "text_hash": stable_hash(text or ""),
        "text_raw": (text or "").strip(),
        "text_punctuated": (text or "").strip(),
        "text_normalized": normalized,
        "expressive_tags": [],
        "language": "aka",
        "speaker_id": speaker_id,
        "duration_seconds": duration_seconds,
        "sample_id": sample_id,
        "dataset_row": dataset_row,
        "source_row": row,
    }


def quarantine_reason(record: dict[str, Any]) -> str | None:
    if not record["audio_path"]:
        return "missing_audio"
    if not record["text_normalized"]:
        return "missing_text"
    duration = record.get("duration_seconds")
    if duration is not None and duration < 0.4:
        return "duration_below_0.4s"
    if duration is not None and duration > 30.0:
        return "duration_above_30s"
    return None


def load_waxal_records(paths: SanitizePaths, spec: AsrArtifactSpec) -> list[dict[str, Any]]:
    metadata = {row.get("sample_id"): row for row in read_jsonl(paths.waxal_metadata)}
    records: list[dict[str, Any]] = []
    for split_file, split in [("train.jsonl", "train"), ("dev.jsonl", "dev"), ("test.jsonl", "test")]:
        for row in read_jsonl(paths.waxal_round2_dir / split_file):
            sample_id = str(row.get("sample_id") or "")
            meta = metadata.get(sample_id, {})
            text = str(row.get("text") or meta.get("text") or "")
            records.append(
                base_record(
                    spec=spec,
                    corpus="waxal",
                    source_dataset="google/WaxalNLP",
                    source_config="aka_asr",
                    row=row,
                    split=split,
                    audio_path=str(meta.get("audio_path") or ""),
                    text=text,
                    source_split=str(row.get("source_split") or meta.get("split") or split),
                    duration_seconds=meta.get("duration_seconds"),
                    speaker_id=str(row.get("speaker_id") or meta.get("speaker_id") or "") or None,
                    sample_id=sample_id,
                    dataset_row=row.get("dataset_row"),
                )
            )
    return records


def load_ghana_nlp_records(paths: SanitizePaths, spec: AsrArtifactSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in read_jsonl(paths.ghana_nlp_manifest):
        split = str(row.get("split") or "train")
        records.append(
            base_record(
                spec=spec,
                corpus="gnlp",
                source_dataset="ghananlpcommunity/twi-speech-text-multispeaker-16k",
                source_config="default",
                row=row,
                split=split,
                audio_path=str(row.get("audio_path") or ""),
                text=str(row.get("text") or ""),
                source_split=str(row.get("source") or "train"),
                duration_seconds=row.get("duration_seconds"),
                speaker_id=row.get("speaker_id"),
                sample_id=str(row.get("sample_id") or ""),
                dataset_row=row.get("dataset_row"),
            )
        )
    return records


def sanitize_v01(paths: SanitizePaths, spec: AsrArtifactSpec = FIRST_ASR_REVIEW) -> dict[str, Any]:
    candidates = load_waxal_records(paths, spec) + load_ghana_nlp_records(paths, spec)
    accepted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    seen_audio: dict[str, str] = {}

    for record in candidates:
        reason = quarantine_reason(record)
        if reason is None and record["audio_hash"] in seen_audio:
            reason = "duplicate_audio_path"
        if reason:
            quarantined.append({**record, "quarantine_reason": reason})
            continue
        seen_audio[record["audio_hash"]] = record["record_id"]
        accepted.append(record)

    output_rows = [
        {key: value for key, value in row.items() if key != "source_row"}
        for row in accepted
    ]
    write_jsonl(paths.output_manifest, output_rows)

    accepted_by_corpus = Counter(row["corpus"] for row in accepted)
    accepted_by_split = Counter(f"{row['corpus']}:{row['split']}" for row in accepted)
    quarantine_reasons = Counter(row["quarantine_reason"] for row in quarantined)
    report = {
        "artifact_code": spec.code_name,
        "hf_repo": spec.hf_repo,
        "status": "complete",
        "inputs": {
            "waxal_metadata": str(paths.waxal_metadata),
            "waxal_round2_dir": str(paths.waxal_round2_dir),
            "ghana_nlp_manifest": str(paths.ghana_nlp_manifest),
        },
        "outputs": {
            "manifest": str(paths.output_manifest),
            "report": str(paths.output_report),
        },
        "candidate_rows": len(candidates),
        "accepted_rows": len(accepted),
        "quarantined_rows": len(quarantined),
        "accepted_by_corpus": dict(accepted_by_corpus),
        "accepted_by_split": dict(accepted_by_split),
        "quarantine_reasons": dict(quarantine_reasons),
        "policy": {
            "min_duration_seconds": 0.4,
            "max_duration_seconds": 30.0,
            "preserve_raw_text": True,
            "preserve_punctuated_text": True,
            "wer_normalized_text": True,
            "expressive_tags_field": True,
            "test_rows_in_manifest": True,
            "training_code_must_exclude_test_split": True,
        },
    }
    paths.output_report.parent.mkdir(parents=True, exist_ok=True)
    paths.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


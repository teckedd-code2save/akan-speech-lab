from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


V06_CODE_NAME = "serendepify-gsl-asr-ak-waxal-gnlpclean-whisper-small-replay-fullft-v0.6"
V06_PARENT_CODE_NAME = "serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-replay-fullft-v0.1"


@dataclass(frozen=True)
class V06ManifestPaths:
    source_manifest: Path
    clean_gnlp_candidates: Path
    output_manifest: Path
    output_report: Path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def row_role(row: dict[str, Any]) -> str:
    corpus = str(row.get("corpus") or "")
    split = str(row.get("split") or "")
    if split == "train":
        return "train"
    if split in {"dev", "validation"}:
        return "validation"
    if corpus == "waxal" and split == "test":
        return "regression_test"
    return "excluded"


def training_bucket(row: dict[str, Any]) -> str:
    corpus = str(row.get("corpus") or "")
    role = row_role(row)
    if corpus == "waxal" and role == "train":
        return "waxal_replay_train"
    if corpus == "waxal" and role == "validation":
        return "waxal_replay_validation"
    if corpus == "waxal" and role == "regression_test":
        return "waxal_regression_test"
    if corpus == "gnlp" and role == "train":
        return "gnlp_clean_adaptation_train"
    if corpus == "gnlp" and role == "validation":
        return "gnlp_clean_adaptation_validation"
    return "excluded"


def retarget_row(
    row: dict[str, Any], *, gnlp_audit: dict[str, Any] | None = None
) -> dict[str, Any]:
    cleaned = {key: value for key, value in row.items() if key != "source_row"}
    cleaned["artifact_code"] = V06_CODE_NAME
    cleaned["parent_artifact_code"] = V06_PARENT_CODE_NAME
    cleaned["row_role"] = row_role(row)
    cleaned["training_bucket"] = training_bucket(row)
    if gnlp_audit is not None:
        cleaned["quality_flags"] = gnlp_audit.get("flags") or []
        cleaned["quality_tags"] = gnlp_audit.get("tags") or []
        cleaned["quality_clean_candidate"] = bool(gnlp_audit.get("clean_candidate"))
    else:
        cleaned["quality_flags"] = []
        cleaned["quality_tags"] = []
        cleaned["quality_clean_candidate"] = True
    return cleaned


def build_v06_manifest(paths: V06ManifestPaths) -> dict[str, Any]:
    source_rows = read_jsonl(paths.source_manifest)
    clean_gnlp_rows = read_jsonl(paths.clean_gnlp_candidates)
    clean_gnlp_by_record = {str(row["record_id"]): row for row in clean_gnlp_rows}
    output_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    for row in source_rows:
        corpus = str(row.get("corpus") or "")
        if corpus == "waxal":
            candidate = retarget_row(row)
        elif corpus == "gnlp":
            audit_row = clean_gnlp_by_record.get(str(row.get("record_id")))
            if audit_row is None:
                excluded_rows.append({**row, "exclude_reason": "gnlp_not_clean_candidate"})
                continue
            if row.get("split") == "test":
                excluded_rows.append({**row, "exclude_reason": "gnlp_test_reserved"})
                continue
            candidate = retarget_row(row, gnlp_audit=audit_row)
        else:
            excluded_rows.append({**row, "exclude_reason": "unsupported_corpus"})
            continue

        if candidate["row_role"] == "excluded":
            excluded_rows.append({**candidate, "exclude_reason": "unsupported_split"})
            continue
        output_rows.append(candidate)

    write_jsonl(paths.output_manifest, output_rows)
    by_bucket = Counter(row["training_bucket"] for row in output_rows)
    by_corpus_split = Counter(f"{row['corpus']}:{row['split']}" for row in output_rows)
    excluded_reasons = Counter(row["exclude_reason"] for row in excluded_rows)
    report = {
        "artifact_code": V06_CODE_NAME,
        "parent_artifact_code": V06_PARENT_CODE_NAME,
        "status": "complete",
        "inputs": {
            "source_manifest": str(paths.source_manifest),
            "clean_gnlp_candidates": str(paths.clean_gnlp_candidates),
        },
        "outputs": {
            "manifest": str(paths.output_manifest),
            "report": str(paths.output_report),
        },
        "rows": len(output_rows),
        "rows_by_bucket": dict(by_bucket),
        "rows_by_corpus_split": dict(by_corpus_split),
        "excluded_rows": len(excluded_rows),
        "excluded_reasons": dict(excluded_reasons),
        "policy": {
            "gnlp_source": "first-pass clean candidates from v0.5 manifest audit",
            "gnlp_test_rows": "reserved and excluded from v0.6 manifest",
            "waxal_role": "replay anchor plus dev/test regression evidence",
            "trainable_rows": (
                by_bucket["waxal_replay_train"] + by_bucket["gnlp_clean_adaptation_train"]
            ),
            "validation_rows": (
                by_bucket["waxal_replay_validation"] + by_bucket["gnlp_clean_adaptation_validation"]
            ),
            "regression_test_rows": by_bucket["waxal_regression_test"],
            "training_code_must_exclude": ["row_role=validation", "row_role=regression_test"],
        },
    }
    paths.output_report.parent.mkdir(parents=True, exist_ok=True)
    paths.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report

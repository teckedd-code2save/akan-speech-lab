from __future__ import annotations

from collections import defaultdict
from typing import Any

from akan_speech.data.quality import duration_bucket
from akan_speech.eval.wer import speech_error_rates


def grouped_error_rates(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if field == "duration_bucket":
            key = duration_bucket(row.get("duration_seconds"))
        else:
            key = str(row.get(field) or "unknown")
        groups[key].append(row)
    return {
        key: {
            "rows": len(group_rows),
            **speech_error_rates(
                [str(row.get("reference") or "") for row in group_rows],
                [str(row.get("prediction") or "") for row in group_rows],
            ),
        }
        for key, group_rows in sorted(groups.items())
    }

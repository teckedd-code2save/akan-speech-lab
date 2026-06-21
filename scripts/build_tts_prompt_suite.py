from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from akan_speech.tts.text import normalize_tts_text, unresolved_numeric_tokens


ROOT = Path(__file__).resolve().parents[1]
BLUEPRINT = ROOT / "configs/tts/eval_suite_blueprint.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_prompt_suite(rows: list[dict], blueprint: dict) -> dict:
    expected = blueprint["categories"]
    counts = Counter(str(row.get("category") or "") for row in rows)
    unapproved = [row.get("prompt_id") for row in rows if row.get("review_status") != "approved"]
    empty = [row.get("prompt_id") for row in rows if not normalize_tts_text(row.get("text") or "")]
    numeric_review = [
        row.get("prompt_id")
        for row in rows
        if unresolved_numeric_tokens(row.get("text") or "")
        and not row.get("spoken_number_reviewed")
    ]
    duplicates = len(rows) - len({normalize_tts_text(row.get("text") or "").casefold() for row in rows})
    category_errors = {
        category: {"expected": required, "actual": counts.get(category, 0)}
        for category, required in expected.items()
        if counts.get(category, 0) != required
    }
    return {
        "version": blueprint["version"],
        "rows": len(rows),
        "expected_rows": blueprint["total_prompts"],
        "categories": dict(counts),
        "category_errors": category_errors,
        "unapproved": unapproved,
        "empty": empty,
        "numeric_review": numeric_review,
        "duplicate_normalized_texts": duplicates,
        "passed": (
            len(rows) == blueprint["total_prompts"]
            and not category_errors
            and not unapproved
            and not empty
            and not numeric_review
            and duplicates == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and freeze the 120-prompt TTS suite")
    parser.add_argument("--input", required=True, help="Reviewed JSONL prompt draft")
    parser.add_argument("--output", default="evals/tts/asante_twi_120_v1.jsonl")
    parser.add_argument("--report", default="evals/tts/asante_twi_120_v1.audit.json")
    args = parser.parse_args()
    blueprint = json.loads(BLUEPRINT.read_text())
    rows = read_jsonl(ROOT / args.input)
    report = validate_prompt_suite(rows, blueprint)
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if not report["passed"]:
        raise SystemExit(f"Prompt suite audit failed; see {report_path.relative_to(ROOT)}")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

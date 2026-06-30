from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akan_speech.data.gnlp_quality import audit_gnlp_rows  # noqa: E402
from akan_speech.data.manifest import read_jsonl  # noqa: E402


DEFAULT_MANIFEST = (
    ROOT / "data/manifests/serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-replay-fullft-v0.1.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs/audits/gnlp_manifest_v0.5"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit GhanaNLP rows in the frozen GSL ASR manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    all_rows = read_jsonl(args.manifest)
    gnlp_rows = [row for row in all_rows if row.get("corpus") == "gnlp"]
    if not gnlp_rows:
        raise SystemExit(f"No GhanaNLP rows found in {args.manifest}")

    report, audited_rows = audit_gnlp_rows(gnlp_rows)
    clean_rows = [row for row in audited_rows if row["clean_candidate"]]
    flagged_rows = [row for row in audited_rows if row["flags"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.output_dir / "audited_rows.jsonl", audited_rows)
    write_jsonl(args.output_dir / "clean_candidates.jsonl", clean_rows)
    write_jsonl(
        args.output_dir / "clean_train_candidates.jsonl",
        [row for row in clean_rows if row["split"] == "train"],
    )
    write_jsonl(
        args.output_dir / "clean_validation_candidates.jsonl",
        [row for row in clean_rows if row["split"] == "validation"],
    )

    csv_path = args.output_dir / "flagged_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "record_id",
            "split",
            "dataset_row",
            "sample_id",
            "duration_seconds",
            "word_count",
            "words_per_second",
            "clean_candidate",
            "flags",
            "tags",
            "text_normalized",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in flagged_rows:
            writer.writerow(
                {
                    key: "|".join(row[key]) if key in {"flags", "tags"} else row.get(key)
                    for key in fieldnames
                }
            )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

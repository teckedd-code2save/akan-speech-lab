from __future__ import annotations

import argparse
import json
from pathlib import Path

from akan_speech.data.audit import corpus_audit
from akan_speech.data.hf_viewer import fetch_split_rows, first_audio_url
from akan_speech.data.manifest import make_record, write_jsonl


def render_audit(report: dict) -> str:
    lines = [
        "# Waxal Akan corpus audit",
        "",
        f"- Records: **{report['records']:,}**",
        f"- Splits: `{report['splits']}`",
        f"- Unique speakers: **{report['unique_speakers']}**",
        f"- Speakers by split: `{report['speakers_by_split']}`",
        f"- Duplicate ID groups: **{report['duplicate_id_groups']}**",
        f"- Duplicate normalized-text groups: **{report['duplicate_text_groups']}**",
        f"- Cross-split normalized-text groups: **{report['cross_split_text_groups']}**",
        "",
        "## Speaker overlap",
        "",
    ]
    for pair, details in report["speaker_overlap"].items():
        lines.append(f"- `{pair}`: **{details['count']}** shared speakers")
    lines.extend(
        [
            "",
            "## Language and orthography",
            "",
            f"- Dataset language labels: `{report['languages']}`",
            f"- {report['dialect_note']}",
            f"- Marker counts: `{report['orthographic_marker_counts']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build complete labeled Waxal Akan metadata.")
    parser.add_argument("--dataset", default="google/WaxalNLP")
    parser.add_argument("--config", default="aka_asr")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--output", default="data/manifests/waxal_aka_labeled_metadata.jsonl")
    parser.add_argument("--audit-json", default="evals/reports/waxal_aka_corpus_audit.json")
    parser.add_argument("--audit-md", default="evals/reports/waxal_aka_corpus_audit.md")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    records = []
    for split in args.splits:
        print(f"Fetching {split} metadata...", flush=True)
        for item in fetch_split_rows(args.dataset, args.config, split, workers=args.workers):
            row = item.get("row", {})
            row_idx = int(item.get("row_idx", 0))
            records.append(
                make_record(
                    audio_path=first_audio_url(row.get("audio")),
                    text=row.get("transcription") or "",
                    language=row.get("language") or "aka",
                    speaker_id=row.get("speaker_id"),
                    split=split,
                    source=f"{args.dataset}/{args.config}/{split}",
                    dataset_row=row_idx,
                    sample_id=row.get("id"),
                    gender=row.get("gender"),
                )
            )

    count = write_jsonl(records, args.output)
    serialized = [record.to_json() for record in records]
    audit = corpus_audit(serialized)
    audit["manifest"] = args.output
    audit["dataset"] = args.dataset
    audit["config"] = args.config
    Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_json).write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    Path(args.audit_md).write_text(render_audit(audit), encoding="utf-8")
    print(f"Wrote {count} records to {args.output}")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from akan_speech.eval.wer import speech_error_rates


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an ASR eval JSON report as Markdown.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    report = payload["report"]
    predictions = payload.get("predictions", [])
    decoder_strategy = (
        report.get("decoder_strategy")
        or report.get("generate_kwargs", {}).get("language")
        or "no_forced_language"
    )
    if predictions:
        for row in predictions:
            if "wer" not in row:
                row.update(speech_error_rates([row.get("reference", "")], [row.get("prediction", "")]))
        if "reference_words" not in report:
            report.update(
                speech_error_rates(
                    [row.get("reference", "") for row in predictions],
                    [row.get("prediction", "") for row in predictions],
                )
            )
    lines = [
        f"# ASR Eval: `{report['model_id']}`",
        "",
        f"- Manifest: `{report['manifest']}`",
        f"- Rows: `{report['rows']}`",
        f"- Device: `{report['device']}`",
        f"- Decoder strategy: `{decoder_strategy}`",
        f"- Runtime: `{report['runtime_seconds']}s`",
        f"- WER: `{report['wer'] * 100:.2f}%`",
        f"- CER: `{report['cer'] * 100:.2f}%`",
        f"- Word errors: `{report.get('substitutions', 0)} substitutions + {report.get('deletions', 0)} deletions + {report.get('insertions', 0)} insertions / {report.get('reference_words', 0)} reference words`",
        "",
        "## Per-row metrics",
        "",
        "| Row | Speaker | Words | WER | CER | Sub | Del | Ins |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in predictions:
        lines.append(
            f"| {row['idx']} | `{row.get('speaker_id') or 'unknown'}` | {row.get('reference_words', 0)} | "
            f"{float(row.get('wer', 0)) * 100:.2f}% | {float(row.get('cer', 0)) * 100:.2f}% | "
            f"{row.get('substitutions', 0)} | {row.get('deletions', 0)} | {row.get('insertions', 0)} |"
        )
    lines.extend([
        "",
        "## Predictions",
        "",
    ])
    for row in predictions:
        lines.extend(
            [
                f"### Row {row['idx']} / speaker `{row.get('speaker_id')}`",
                "",
                f"**WER {float(row.get('wer', 0)) * 100:.2f}% · CER {float(row.get('cer', 0)) * 100:.2f}% · "
                f"{row.get('substitutions', 0)} substitutions · {row.get('deletions', 0)} deletions · {row.get('insertions', 0)} insertions**",
                "",
                "**Reference**",
                "",
                row.get("reference") or "",
                "",
                "**Prediction**",
                "",
                row.get("prediction") or "",
                "",
                "**Normalized Reference**",
                "",
                row.get("normalized_reference") or "",
                "",
                "**Normalized Prediction**",
                "",
                row.get("normalized_prediction") or "",
                "",
            ]
        )
    output = Path(args.output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

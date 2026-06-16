from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an ASR eval JSON report as Markdown.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    report = payload["report"]
    predictions = payload.get("predictions", [])
    lines = [
        f"# ASR Eval: `{report['model_id']}`",
        "",
        f"- Manifest: `{report['manifest']}`",
        f"- Rows: `{report['rows']}`",
        f"- Device: `{report['device']}`",
        f"- Runtime: `{report['runtime_seconds']}s`",
        f"- WER: `{report['wer']:.4f}`",
        f"- CER: `{report['cer']:.4f}`",
        "",
        "## Predictions",
        "",
    ]
    for row in predictions:
        lines.extend(
            [
                f"### Row {row['idx']} / speaker `{row.get('speaker_id')}`",
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


from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from akan_speech.eval.bootstrap import bootstrap_wer_interval
from akan_speech.eval.breakdown import grouped_error_rates
from akan_speech.eval.tokenization import tokenizer_fragmentation


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze three-strategy Whisper decoder comparison.")
    parser.add_argument("--input", default="evals/reports/waxal_decoder_comparison.json")
    parser.add_argument("--output-json", default="evals/reports/waxal_decoder_analysis.json")
    parser.add_argument("--output-md", default="evals/reports/waxal_decoder_analysis.md")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    references = []
    analysis = {}
    for strategy, run in payload["runs"].items():
        rows = run["predictions"]
        references = [str(row.get("reference") or "") for row in rows]
        analysis[strategy] = {
            "metrics": run["metrics"],
            "runtime_seconds": run["runtime_seconds"],
            "wer_95_percent_interval": bootstrap_wer_interval(
                rows, samples=args.bootstrap_samples, seed=20260618
            ),
            "by_speaker": grouped_error_rates(rows, "speaker_id"),
            "by_duration": grouped_error_rates(rows, "duration_bucket"),
        }
    tokenizer = AutoTokenizer.from_pretrained(payload["model_id"])
    fragmentation = tokenizer_fragmentation(tokenizer, references)
    selected = min(analysis, key=lambda strategy: analysis[strategy]["metrics"]["wer"])
    result = {
        "model_id": payload["model_id"],
        "benchmark_rows": payload["benchmark_rows"],
        "selected_strategy": selected,
        "selection_rule": "Lowest corpus WER; retain confidence interval and qualitative review.",
        "strategies": analysis,
        "tokenizer_fragmentation": fragmentation,
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Waxal decoder strategy analysis",
        "",
        f"Model: `{payload['model_id']}`  ",
        f"Frozen benchmark rows: **{payload['benchmark_rows']}**  ",
        f"Selected strategy: **{selected}**",
        "",
        "| Strategy | WER | 95% bootstrap interval | CER | Runtime |",
        "|---|---:|---:|---:|---:|",
    ]
    for strategy, details in sorted(analysis.items(), key=lambda item: item[1]["metrics"]["wer"]):
        metrics = details["metrics"]
        interval = details["wer_95_percent_interval"]
        lines.append(
            f"| `{strategy}` | {metrics['wer'] * 100:.2f}% | "
            f"{interval['low'] * 100:.2f}%–{interval['high'] * 100:.2f}% | "
            f"{metrics['cer'] * 100:.2f}% | {details['runtime_seconds']:.1f}s |"
        )
    lines.extend(
        [
            "",
            "## Tokenizer fragmentation",
            "",
            f"- Tokens per word: **{fragmentation['tokens_per_word']}**",
            f"- Characters per token: **{fragmentation['characters_per_token']}**",
            f"- Word occurrences split into 3+ tokens: **{fragmentation['words_split_into_3plus_tokens']}**",
            "",
            "The selected strategy is a training input, not proof of improvement. Promotion still requires the trained candidate to beat the published 34.28% WER floor on this frozen benchmark and pass Ghanaian qualitative review.",
        ]
    )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

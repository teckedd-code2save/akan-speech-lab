from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from akan_speech.data.manifest import read_jsonl
from akan_speech.data.normalize import normalize_akan_text
from akan_speech.eval.wer import speech_error_rates


def transcribe(pipe: Any, audio_path: str, generate_kwargs: dict[str, Any]) -> str:
    kwargs = {"generate_kwargs": generate_kwargs} if generate_kwargs else {}
    result = pipe(audio_path, **kwargs)
    if isinstance(result, dict):
        return str(result.get("text") or "").strip()
    return str(result).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a HF ASR model on a JSONL speech manifest.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--language", default="")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--output-json", default="evals/reports/asr_eval_report.json")
    parser.add_argument("--output-csv", default="evals/reports/asr_eval_predictions.csv")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest and print rows without loading a model.")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("Manifest has no rows to evaluate.")

    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else -1
    elif args.device == "cuda":
        device = 0
    elif args.device == "mps":
        device = "mps"
    else:
        device = -1

    generate_kwargs: dict[str, Any] = {}
    if args.language:
        generate_kwargs["language"] = args.language
    if args.task:
        generate_kwargs["task"] = args.task

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "manifest": args.manifest,
                    "rows": len(rows),
                    "generate_kwargs": generate_kwargs,
                    "first_rows": [
                        {
                            "audio_path": row.get("audio_path"),
                            "speaker_id": row.get("speaker_id"),
                            "text": row.get("text"),
                            "normalized_text": row.get("normalized_text"),
                        }
                        for row in rows[:3]
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    started = perf_counter()
    from transformers import pipeline

    print(f"Loading model: {args.model_id}", flush=True)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_id,
        device=device,
    )
    print(f"Loaded model on device={device}. Evaluating {len(rows)} row(s).", flush=True)

    predictions = []
    references = []
    result_rows = []
    for idx, row in enumerate(rows):
        print(f"Transcribing {idx + 1}/{len(rows)}: {row.get('audio_path')}", flush=True)
        reference = row.get("text") or row.get("normalized_text") or ""
        prediction = transcribe(pipe, row["audio_path"], generate_kwargs)
        predictions.append(prediction)
        references.append(reference)
        row_metrics = speech_error_rates([reference], [prediction])
        result_rows.append(
            {
                "idx": idx,
                "speaker_id": row.get("speaker_id"),
                "source": row.get("source"),
                "reference": reference,
                "prediction": prediction,
                "normalized_reference": normalize_akan_text(reference),
                "normalized_prediction": normalize_akan_text(prediction),
                **row_metrics,
            }
        )
        print(json.dumps(result_rows[-1], ensure_ascii=False), flush=True)

    metrics = speech_error_rates(references, predictions)
    report = {
        "model_id": args.model_id,
        "manifest": args.manifest,
        "rows": len(rows),
        "device": str(device),
        "generate_kwargs": generate_kwargs,
        "decoder_strategy": args.language or "no_forced_language",
        "runtime_seconds": round(perf_counter() - started, 3),
        **metrics,
    }

    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"report": report, "predictions": result_rows}, indent=2, ensure_ascii=False) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

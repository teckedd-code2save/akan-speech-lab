from __future__ import annotations

import json
from pathlib import Path

import modal


HOUR = 60 * 60
CACHE_DIR = "/cache"
CHECKPOINT_DIR = "/checkpoints"
OUTPUT_DIR = "/outputs"

app = modal.App("akan-speech-asr-eval")
cache_volume = modal.Volume.from_name("akan-speech-hf-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("akan-speech-checkpoints", create_if_missing=True)
output_volume = modal.Volume.from_name("akan-speech-eval-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "accelerate==1.8.1",
        "jiwer==4.0.0",
        "librosa==0.11.0",
        "numpy<2.3",
        "requests==2.32.4",
        "soundfile==0.13.1",
        "torch==2.7.1",
        "transformers==4.53.2",
    )
    .env({"HF_HOME": f"{CACHE_DIR}/huggingface", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir(
        "data/processed/waxal_benchmark_audio",
        remote_path="/benchmark_audio",
        copy=True,
    )
)


def normalize_akan_text(text: str) -> str:
    import re
    import unicodedata

    value = unicodedata.normalize("NFC", text or "").strip().lower()
    value = re.sub(r"[“”\"'‘’.,!?;:()\[\]{}<>/\\|*_+=~`]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def error_rates(references: list[str], predictions: list[str]) -> dict:
    from jiwer import process_characters, process_words

    refs = [normalize_akan_text(text) for text in references]
    preds = [normalize_akan_text(text) for text in predictions]
    words = process_words(refs, preds)
    chars = process_characters(refs, preds)
    return {
        "wer": float(words.wer),
        "cer": float(chars.cer),
        "reference_words": int(words.hits + words.substitutions + words.deletions),
        "hits": int(words.hits),
        "substitutions": int(words.substitutions),
        "deletions": int(words.deletions),
        "insertions": int(words.insertions),
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    volumes={
        CACHE_DIR: cache_volume,
        CHECKPOINT_DIR: checkpoint_volume,
        OUTPUT_DIR: output_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])],
    timeout=2 * HOUR,
    scaledown_window=30,
)
def compare_decoders(
    model_id: str,
    records: list[dict],
    strategies: list[str],
    batch_size: int = 4,
    report_name: str = "decoder-comparison.json",
) -> dict:
    import hashlib
    import time
    from concurrent.futures import ThreadPoolExecutor

    import requests
    import torch
    from transformers import pipeline

    audio_dir = Path("/tmp/akan-benchmark")
    audio_dir.mkdir(parents=True, exist_ok=True)

    def download(item: tuple[int, dict]) -> str:
        index, row = item
        local_name = Path(str(row.get("audio_path") or "")).name
        mounted_path = Path("/benchmark_audio") / local_name
        if mounted_path.exists():
            return str(mounted_path)
        url = str(row.get("remote_audio_path") or row.get("audio_path") or "")
        if not url.startswith("http"):
            raise ValueError(f"Row {index} has no remote audio URL")
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        path = audio_dir / f"{index:04d}_{digest}.mp3"
        if not path.exists():
            with requests.get(url, timeout=120, stream=True) as response:
                response.raise_for_status()
                with path.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
        return str(path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        audio_paths = list(executor.map(download, enumerate(records)))

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=0,
        torch_dtype=torch.float16,
    )
    asr.model.config.forced_decoder_ids = None
    asr.model.generation_config.forced_decoder_ids = None
    references = [str(row.get("text") or row.get("normalized_text") or "") for row in records]
    runs = {}
    for strategy in strategies:
        started = time.perf_counter()
        generate_kwargs = {"task": "transcribe"}
        if strategy == "no_forced_language":
            asr.model.generation_config.language = None
        else:
            asr.model.generation_config.language = strategy
            generate_kwargs["language"] = strategy
        outputs = asr(
            audio_paths,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs,
        )
        predictions = [str(output.get("text") or "").strip() for output in outputs]
        rows = []
        for index, (record, reference, prediction) in enumerate(
            zip(records, references, predictions, strict=True)
        ):
            rows.append(
                {
                    "idx": index,
                    "dataset_row": record.get("dataset_row"),
                    "speaker_id": record.get("speaker_id"),
                    "duration_seconds": record.get("duration_seconds"),
                    "reference": reference,
                    "prediction": prediction,
                    **error_rates([reference], [prediction]),
                }
            )
        runs[strategy] = {
            "strategy": strategy,
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "rows": len(rows),
            "metrics": error_rates(references, predictions),
            "predictions": rows,
        }
    result = {"model_id": model_id, "benchmark_rows": len(records), "runs": runs}
    result_path = Path(OUTPUT_DIR) / report_name
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    output_volume.commit()
    print(f"Persisted result to {result_path}")
    return result


@app.local_entrypoint()
def main(
    manifest: str = "evals/samples/waxal_aka_benchmark_local.jsonl",
    model_id: str = "teckedd/whisper_small-waxal_akan-asr-v1",
    output: str = "evals/reports/waxal_decoder_comparison.json",
    batch_size: int = 4,
    strategies: str = "no_forced_language,yoruba,english",
):
    records = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    selected_strategies = [item.strip() for item in strategies.split(",") if item.strip()]
    invalid = set(selected_strategies) - {"no_forced_language", "yoruba", "english"}
    if invalid:
        raise ValueError(f"Unsupported strategies: {sorted(invalid)}")
    print(
        f"Evaluating {len(records)} frozen rows across {len(selected_strategies)} strategies "
        "on one L4 container"
    )
    print("Conservative cost ceiling: <= $1.60 for the two-hour function timeout")
    result = compare_decoders.remote(
        model_id,
        records,
        selected_strategies,
        batch_size,
        Path(output).name,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    summary = {
        strategy: {
            **run["metrics"],
            "runtime_seconds": run["runtime_seconds"],
        }
        for strategy, run in result["runs"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path}")

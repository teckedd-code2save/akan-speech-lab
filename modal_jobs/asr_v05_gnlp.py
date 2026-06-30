from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import modal


HOUR = 60 * 60
CACHE_DIR = "/cache"
OUTPUT_DIR = "/outputs"
MANIFEST_PATH = Path("/manifests/v01.jsonl")
APP_NAME = "akan-speech-asr-v05-gnlp"
CODE_NAME = "serendepify-gsl-asr-ak-gnlp-whisper-small-only-lowlr-freezeenc-fullft-v0.5"
SOURCE_MANIFEST_CODE = "serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-replay-fullft-v0.1"
MODEL_REPO = f"teckedd/{CODE_NAME}"
GNLP_PARQUET = (
    "https://huggingface.co/datasets/ghananlpcommunity/"
    "twi-speech-text-multispeaker-16k/resolve/refs%2Fconvert%2Fparquet/default/train/"
    "0000.parquet",
    "https://huggingface.co/datasets/ghananlpcommunity/"
    "twi-speech-text-multispeaker-16k/resolve/refs%2Fconvert%2Fparquet/default/train/"
    "0001.parquet",
    "https://huggingface.co/datasets/ghananlpcommunity/"
    "twi-speech-text-multispeaker-16k/resolve/refs%2Fconvert%2Fparquet/default/train/"
    "0002.parquet",
)
WAXAL_PARQUET = {
    "train": (
        "https://huggingface.co/datasets/google/WaxalNLP/resolve/refs%2Fconvert%2Fparquet/"
        "aka_asr/train/0000.parquet",
        "https://huggingface.co/datasets/google/WaxalNLP/resolve/refs%2Fconvert%2Fparquet/"
        "aka_asr/train/0001.parquet",
        "https://huggingface.co/datasets/google/WaxalNLP/resolve/refs%2Fconvert%2Fparquet/"
        "aka_asr/train/0002.parquet",
        "https://huggingface.co/datasets/google/WaxalNLP/resolve/refs%2Fconvert%2Fparquet/"
        "aka_asr/train/0003.parquet",
        "https://huggingface.co/datasets/google/WaxalNLP/resolve/refs%2Fconvert%2Fparquet/"
        "aka_asr/train/0004.parquet",
        "https://huggingface.co/datasets/google/WaxalNLP/resolve/refs%2Fconvert%2Fparquet/"
        "aka_asr/train/0005.parquet",
    ),
    "validation": (
        "https://huggingface.co/datasets/google/WaxalNLP/resolve/refs%2Fconvert%2Fparquet/"
        "aka_asr/validation/0000.parquet",
    ),
    "test": (
        "https://huggingface.co/datasets/google/WaxalNLP/resolve/refs%2Fconvert%2Fparquet/"
        "aka_asr/test/0000.parquet",
    ),
}

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("akan-speech-hf-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("akan-speech-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "accelerate==1.8.1",
        "datasets[audio]==3.6.0",
        "huggingface_hub==0.33.4",
        "jiwer==4.0.0",
        "numpy<2.3",
        "soundfile==0.13.1",
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "transformers==4.53.2",
    )
    .env({"HF_HOME": f"{CACHE_DIR}/huggingface", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_file(
        f"data/manifests/{SOURCE_MANIFEST_CODE}.jsonl",
        remote_path=str(MANIFEST_PATH),
        copy=True,
    )
)


@dataclass
class V01TrainConfig:
    run_name: str = CODE_NAME
    base_model: str = "teckedd/whisper-small-waxal-round2-specaug-v1"
    model_repo: str = MODEL_REPO
    seed: int = 42
    max_steps: int = 800
    learning_rate: float = 3e-6
    weight_decay: float = 0.01
    warmup_steps: int = 50
    train_batch_size: int = 8
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    eval_steps: int = 200
    save_steps: int = 200
    generation_max_length: int = 225
    train_per_corpus: int = 10000
    eval_per_corpus: int = 602
    waxal_regression_eval_rows: int = 512
    apply_spec_augment: bool = True
    freeze_encoder: bool = True
    stop_on_first_regression: bool = True


def normalize_akan_text(text: str) -> str:
    import re
    import unicodedata

    value = unicodedata.normalize("NFC", text or "").strip().lower()
    value = re.sub(r"[“”\"'‘’.,!?;:()\[\]{}<>/\\|*_+=~`]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def read_manifest() -> list[dict]:
    return [json.loads(line) for line in MANIFEST_PATH.read_text().splitlines() if line.strip()]


def select_manifest_rows(config: V01TrainConfig) -> tuple[list[dict], list[dict]]:
    import hashlib

    rows = read_manifest()
    train = sorted(
        [row for row in rows if row["corpus"] == "gnlp" and row["split"] == "train"],
        key=lambda row: hashlib.sha256(
            f"{config.seed}:{config.run_name}:train:{row['record_id']}".encode()
        ).hexdigest(),
    )[: config.train_per_corpus]
    dev = sorted(
        [row for row in rows if row["corpus"] == "gnlp" and row["split"] == "validation"],
        key=lambda row: hashlib.sha256(
            f"{config.seed}:{config.run_name}:dev:{row['record_id']}".encode()
        ).hexdigest(),
    )[: config.eval_per_corpus]
    if not train or not dev:
        raise RuntimeError("v0.5 GhanaNLP-only train/dev row selection is empty")
    return train, dev


def select_waxal_regression_rows(config: V01TrainConfig) -> list[dict]:
    import hashlib

    rows = read_manifest()
    return sorted(
        [row for row in rows if row["corpus"] == "waxal" and row["split"] == "dev"],
        key=lambda row: hashlib.sha256(
            f"{config.seed}:{config.run_name}:waxal-regression:{row['record_id']}".encode()
        ).hexdigest(),
    )[: config.waxal_regression_eval_rows]


def source_split_key(record: dict) -> str:
    if record["corpus"] == "gnlp":
        return "train"
    return record["source_split"]


def load_source_dataset(corpus: str, split: str):
    from datasets import Audio, load_dataset

    if corpus == "gnlp":
        return load_dataset(
            "parquet",
            data_files={"train": list(GNLP_PARQUET)},
            split="train",
            cache_dir=CACHE_DIR,
        ).cast_column("audio", Audio(sampling_rate=16000))
    if corpus == "waxal":
        return load_dataset(
            "parquet",
            data_files={split: list(WAXAL_PARQUET[split])},
            split=split,
            cache_dir=CACHE_DIR,
        ).cast_column("audio", Audio(sampling_rate=16000))
    raise ValueError(f"Unsupported corpus: {corpus}")


def materialize_rows(records: list[dict]) -> list[dict]:
    loaded = {}
    materialized = []
    ordered_records = sorted(
        records,
        key=lambda row: (row["corpus"], source_split_key(row), int(row["dataset_row"])),
    )
    total = len(ordered_records)
    print(f"Materializing {total} audio rows", flush=True)
    for index, record in enumerate(ordered_records, start=1):
        corpus = record["corpus"]
        split = source_split_key(record)
        key = (corpus, split)
        if key not in loaded:
            print(f"Loading source dataset {corpus}/{split}", flush=True)
            loaded[key] = load_source_dataset(corpus, split)
        source = loaded[key][int(record["dataset_row"])]
        audio = source["audio"]
        materialized.append(
            {
                "record_id": record["record_id"],
                "corpus": corpus,
                "split": record["split"],
                "speaker_id": record.get("speaker_id") or "unknown",
                "sample_id": record.get("sample_id") or record["record_id"],
                "duration_seconds": record.get("duration_seconds"),
                "text": record["text_normalized"],
                "audio": audio,
            }
        )
        if index == 1 or index % 500 == 0 or index == total:
            print(f"Materialized {index}/{total} audio rows", flush=True)
    return materialized


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: object
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [
            {
                "input_features": feature["input_features"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": feature["labels"]} for feature in features],
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    volumes={CACHE_DIR: cache_volume, OUTPUT_DIR: output_volume},
    secrets=[modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])],
    timeout=4 * HOUR,
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0, initial_delay=10.0),
    scaledown_window=30,
    max_containers=1,
)
def train_and_publish(config_dict: dict) -> dict:
    import shutil

    import torch
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi
    from jiwer import wer
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
        set_seed,
    )

    config = V01TrainConfig(**config_dict)
    run_dir = Path(OUTPUT_DIR) / config.run_name
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        cached = json.loads(summary_path.read_text())
        if cached.get("status") == "complete":
            return cached

    set_seed(config.seed)
    train_rows, dev_rows = select_manifest_rows(config)
    waxal_regression_rows = select_waxal_regression_rows(config)
    dev_materialized = materialize_rows(dev_rows)
    waxal_regression_materialized = materialize_rows(waxal_regression_rows)
    raw = DatasetDict(
        {
            "train": Dataset.from_list(materialize_rows(train_rows)),
            "dev": Dataset.from_list(dev_materialized),
            "dev_gnlp": Dataset.from_list(
                [row for row in dev_materialized if row["corpus"] == "gnlp"]
            ),
            "dev_waxal_regression": Dataset.from_list(waxal_regression_materialized),
        }
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config.base_model,
        return_attention_mask=True,
    )
    tokenizer = WhisperTokenizer.from_pretrained(config.base_model, task="transcribe")
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare(row):
        import numpy as np

        audio = row["audio"]
        samples = np.asarray(audio["array"], dtype="<f4")
        extracted = feature_extractor(
            samples,
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=True,
        )
        return {
            "record_id": row["record_id"],
            "corpus": row["corpus"],
            "speaker_id": row["speaker_id"],
            "sample_id": row["sample_id"],
            "text": row["text"],
            "input_features": extracted.input_features[0],
            "attention_mask": extracted.attention_mask[0],
            "labels": tokenizer(row["text"]).input_ids,
        }

    dataset = raw.map(prepare, remove_columns=["audio"])
    dataset["train"] = dataset["train"].shuffle(seed=config.seed)
    model = WhisperForConditionalGeneration.from_pretrained(config.base_model)
    if config.freeze_encoder:
        model.freeze_encoder()
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = None
    model.generation_config.task = "transcribe"
    model.config.use_cache = False
    model.config.apply_spec_augment = config.apply_spec_augment
    model.config.mask_time_prob = 0.05
    model.config.mask_time_length = 10
    model.config.mask_feature_prob = 0.05
    model.config.mask_feature_length = 10

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def compute_metrics(prediction):
        label_ids = prediction.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        predictions = [
            normalize_akan_text(text)
            for text in tokenizer.batch_decode(prediction.predictions, skip_special_tokens=True)
        ]
        labels = [
            normalize_akan_text(text)
            for text in tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        ]
        return {"wer": wer(labels, predictions)}

    class CommitCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            output_volume.commit()
            return control

    stop_state = {"baseline_wer": None, "stopped_on_regression": False}

    class BaselineRegressionStopCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            baseline_wer = stop_state.get("baseline_wer")
            eval_wer = (metrics or {}).get("eval_wer")
            if (
                config.stop_on_first_regression
                and baseline_wer is not None
                and eval_wer is not None
                and state.global_step >= config.eval_steps
                and eval_wer > baseline_wer
            ):
                stop_state["stopped_on_regression"] = True
                control.should_training_stop = True
            return control

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        run_name=config.run_name,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type="linear",
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=1,
        logging_steps=2,
        predict_with_generate=True,
        generation_max_length=config.generation_max_length,
        generation_num_beams=1,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        seed=config.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        processing_class=feature_extractor,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            CommitCheckpointCallback(),
            EarlyStoppingCallback(early_stopping_patience=2),
            BaselineRegressionStopCallback(),
        ],
    )
    baseline_metrics = trainer.evaluate(metric_key_prefix="baseline")
    baseline_by_corpus = {
        "gnlp": trainer.evaluate(
            eval_dataset=dataset["dev_gnlp"],
            metric_key_prefix="baseline_gnlp",
        ),
        "waxal_regression": trainer.evaluate(
            eval_dataset=dataset["dev_waxal_regression"],
            metric_key_prefix="baseline_waxal_regression",
        ),
    }
    stop_state["baseline_wer"] = baseline_metrics["baseline_wer"]
    trainer.save_metrics("baseline", baseline_metrics)
    train_result = trainer.train()
    trainer.save_model(str(run_dir / "final"))
    processor.save_pretrained(str(run_dir / "final"))
    trainer.save_metrics("train", train_result.metrics)
    final_metrics = trainer.evaluate(metric_key_prefix="final")
    final_by_corpus = {
        "gnlp": trainer.evaluate(
            eval_dataset=dataset["dev_gnlp"],
            metric_key_prefix="final_gnlp",
        ),
        "waxal_regression": trainer.evaluate(
            eval_dataset=dataset["dev_waxal_regression"],
            metric_key_prefix="final_waxal_regression",
        ),
    }
    trainer.save_metrics("final", final_metrics)
    trainer.save_state()

    summary = {
        "status": "complete",
        "artifact_code": config.run_name,
        "repo_id": config.model_repo,
        "url": f"https://huggingface.co/{config.model_repo}",
        "run_dir": str(run_dir),
        "rows": {"train": len(dataset["train"]), "dev": len(dataset["dev"])},
        "baseline_metrics": baseline_metrics,
        "baseline_by_corpus": baseline_by_corpus,
        "train_metrics": train_result.metrics,
        "final_metrics": final_metrics,
        "final_by_corpus": final_by_corpus,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "stopped_on_regression": stop_state["stopped_on_regression"],
        "cuda": torch.cuda.get_device_name(0),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    shutil.copy2(summary_path, run_dir / "final" / "training_summary.json")

    readme = run_dir / "final" / "README.md"
    readme.write_text(
        f"""---
language:
- tw
- ak
license: cc-by-sa-4.0
tags:
- automatic-speech-recognition
- whisper
- akan
- twi
- ghanaian-speech-lab
- serendepify-gsl
pipeline_tag: automatic-speech-recognition
base_model: {config.base_model}
library_name: transformers
---

# {config.run_name}

This is a Ghanaian Speech Lab ASR artifact: a GhanaNLP-only low-learning-rate
full fine-tuning pass from the Round 2 checkpoint with the Whisper encoder frozen.

It is a real training run, not a smoke-test checkpoint, but it is still a
review candidate until the held-out evaluation and failure taxonomy are complete.

License is set conservatively because this pass is part of the Akan ASR lab's
traceable diagnostic sequence.

- Training rows: {len(dataset["train"])}
- Dev rows: {len(dataset["dev"])}
- Max steps: {config.max_steps}
- Method: full fine-tuning from `{config.base_model}`
- Dataset mix: GhanaNLP only for training; Waxal dev is used only as a regression check
- Frozen encoder: {config.freeze_encoder}
- Stop on first regression: {config.stop_on_first_regression}

Baseline metrics:

```json
{json.dumps(baseline_metrics, indent=2, ensure_ascii=False)}
```

Baseline by corpus:

```json
{json.dumps(baseline_by_corpus, indent=2, ensure_ascii=False)}
```

Final metrics:

```json
{json.dumps(final_metrics, indent=2, ensure_ascii=False)}
```

Final by corpus:

```json
{json.dumps(final_by_corpus, indent=2, ensure_ascii=False)}
```
""",
        encoding="utf-8",
    )

    api = HfApi()
    api.create_repo(config.model_repo, repo_type="model", exist_ok=True)
    commit = api.upload_folder(
        repo_id=config.model_repo,
        repo_type="model",
        folder_path=str(run_dir / "final"),
        commit_message="Publish GhanaNLP-only low-LR GSL ASR v0.5 checkpoint",
        ignore_patterns=["training_args.bin"],
    )
    info = api.model_info(config.model_repo, files_metadata=True)
    summary["commit"] = str(commit)
    summary["files"] = sorted(sibling.rfilename for sibling in info.siblings)
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    output_volume.commit()
    return summary

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import modal


HOUR = 60 * 60
CACHE_DIR = "/cache"
OUTPUT_DIR = "/outputs"
MANIFEST_PATH = Path("/manifests/v06.jsonl")
APP_NAME = "akan-speech-asr-v06-clean-replay"
CODE_NAME = "serendepify-gsl-asr-ak-waxal-gnlpclean-whisper-small-replay-fullft-v0.6"
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
        f"data/manifests/{CODE_NAME}.jsonl",
        remote_path=str(MANIFEST_PATH),
        copy=True,
    )
)


@dataclass
class V06TrainConfig:
    run_name: str = CODE_NAME
    base_model: str = "teckedd/whisper-small-waxal-round2-specaug-v1"
    model_repo: str = MODEL_REPO
    seed: int = 42
    max_steps: int = 1200
    learning_rate: float = 2e-6
    weight_decay: float = 0.01
    warmup_steps: int = 100
    train_batch_size: int = 8
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    eval_steps: int = 200
    save_steps: int = 200
    generation_max_length: int = 225
    waxal_train_rows: int = 9138
    gnlp_train_rows: int = 6033
    waxal_validation_rows: int = 1024
    gnlp_validation_rows: int = 326
    waxal_regression_test_rows: int = 512
    apply_spec_augment: bool = True
    freeze_encoder: bool = True
    stop_on_first_waxal_regression: bool = True


def normalize_akan_text(text: str) -> str:
    import re
    import unicodedata

    value = unicodedata.normalize("NFC", text or "").strip().lower()
    value = re.sub(r"[“”\"'‘’.,!?;:()\[\]{}<>/\\|*_+=~`]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def read_manifest() -> list[dict]:
    return [json.loads(line) for line in MANIFEST_PATH.read_text().splitlines() if line.strip()]


def deterministic_take(rows: list[dict], *, key: str, limit: int, seed: int) -> list[dict]:
    import hashlib

    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{key}:{row['record_id']}".encode()).hexdigest(),
    )[:limit]


def select_manifest_rows(config: V06TrainConfig) -> dict[str, list[dict]]:
    rows = read_manifest()
    waxal_train = [row for row in rows if row["training_bucket"] == "waxal_replay_train"]
    gnlp_train = [row for row in rows if row["training_bucket"] == "gnlp_clean_adaptation_train"]
    waxal_dev = [row for row in rows if row["training_bucket"] == "waxal_replay_validation"]
    gnlp_dev = [row for row in rows if row["training_bucket"] == "gnlp_clean_adaptation_validation"]
    waxal_test = [row for row in rows if row["training_bucket"] == "waxal_regression_test"]
    selected = {
        "train": deterministic_take(
            waxal_train,
            key=f"{config.run_name}:waxal-train",
            limit=config.waxal_train_rows,
            seed=config.seed,
        )
        + deterministic_take(
            gnlp_train,
            key=f"{config.run_name}:gnlp-train",
            limit=config.gnlp_train_rows,
            seed=config.seed,
        ),
        "dev_waxal": deterministic_take(
            waxal_dev,
            key=f"{config.run_name}:waxal-dev",
            limit=config.waxal_validation_rows,
            seed=config.seed,
        ),
        "dev_gnlp": deterministic_take(
            gnlp_dev,
            key=f"{config.run_name}:gnlp-dev",
            limit=config.gnlp_validation_rows,
            seed=config.seed,
        ),
        "test_waxal_regression": deterministic_take(
            waxal_test,
            key=f"{config.run_name}:waxal-test-regression",
            limit=config.waxal_regression_test_rows,
            seed=config.seed,
        ),
    }
    if not selected["train"] or not selected["dev_waxal"] or not selected["dev_gnlp"]:
        raise RuntimeError("v0.6 train/dev row selection is empty")
    return selected


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
        materialized.append(
            {
                "record_id": record["record_id"],
                "corpus": corpus,
                "split": record["split"],
                "training_bucket": record["training_bucket"],
                "speaker_id": record.get("speaker_id") or "unknown",
                "sample_id": record.get("sample_id") or record["record_id"],
                "duration_seconds": record.get("duration_seconds"),
                "text": record["text_normalized"],
                "audio": source["audio"],
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
    timeout=5 * HOUR,
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

    config = V06TrainConfig(**config_dict)
    run_dir = Path(OUTPUT_DIR) / config.run_name
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        cached = json.loads(summary_path.read_text())
        if cached.get("status") == "complete":
            return cached

    set_seed(config.seed)
    selected = select_manifest_rows(config)
    dev_waxal = materialize_rows(selected["dev_waxal"])
    dev_gnlp = materialize_rows(selected["dev_gnlp"])
    waxal_regression = materialize_rows(selected["test_waxal_regression"])
    raw = DatasetDict(
        {
            "train": Dataset.from_list(materialize_rows(selected["train"])),
            "dev": Dataset.from_list(dev_waxal + dev_gnlp),
            "dev_waxal": Dataset.from_list(dev_waxal),
            "dev_gnlp": Dataset.from_list(dev_gnlp),
            "test_waxal_regression": Dataset.from_list(waxal_regression),
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
            "training_bucket": row["training_bucket"],
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

    stop_state = {"baseline_waxal_wer": None, "stopped_on_waxal_regression": False}

    class WaxalRegressionStopCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            baseline_wer = stop_state.get("baseline_waxal_wer")
            eval_wer = (metrics or {}).get("eval_wer")
            if (
                config.stop_on_first_waxal_regression
                and baseline_wer is not None
                and eval_wer is not None
                and state.global_step >= config.eval_steps
                and eval_wer > baseline_wer
            ):
                stop_state["stopped_on_waxal_regression"] = True
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
        logging_steps=10,
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
        eval_dataset=dataset["dev_waxal"],
        processing_class=feature_extractor,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            CommitCheckpointCallback(),
            EarlyStoppingCallback(early_stopping_patience=2),
            WaxalRegressionStopCallback(),
        ],
    )
    baseline_by_bucket = {
        "waxal_validation": trainer.evaluate(
            eval_dataset=dataset["dev_waxal"],
            metric_key_prefix="baseline_waxal_validation",
        ),
        "gnlp_validation": trainer.evaluate(
            eval_dataset=dataset["dev_gnlp"],
            metric_key_prefix="baseline_gnlp_validation",
        ),
        "waxal_regression_test": trainer.evaluate(
            eval_dataset=dataset["test_waxal_regression"],
            metric_key_prefix="baseline_waxal_regression_test",
        ),
    }
    stop_state["baseline_waxal_wer"] = baseline_by_bucket["waxal_validation"][
        "baseline_waxal_validation_wer"
    ]
    train_result = trainer.train()
    trainer.save_model(str(run_dir / "final"))
    processor.save_pretrained(str(run_dir / "final"))
    trainer.save_metrics("train", train_result.metrics)
    final_by_bucket = {
        "waxal_validation": trainer.evaluate(
            eval_dataset=dataset["dev_waxal"],
            metric_key_prefix="final_waxal_validation",
        ),
        "gnlp_validation": trainer.evaluate(
            eval_dataset=dataset["dev_gnlp"],
            metric_key_prefix="final_gnlp_validation",
        ),
        "waxal_regression_test": trainer.evaluate(
            eval_dataset=dataset["test_waxal_regression"],
            metric_key_prefix="final_waxal_regression_test",
        ),
    }
    trainer.save_state()

    summary = {
        "status": "complete",
        "artifact_code": config.run_name,
        "repo_id": config.model_repo,
        "url": f"https://huggingface.co/{config.model_repo}",
        "run_dir": str(run_dir),
        "rows": {
            "train": len(dataset["train"]),
            "dev_waxal": len(dataset["dev_waxal"]),
            "dev_gnlp": len(dataset["dev_gnlp"]),
            "test_waxal_regression": len(dataset["test_waxal_regression"]),
        },
        "baseline_by_bucket": baseline_by_bucket,
        "train_metrics": train_result.metrics,
        "final_by_bucket": final_by_bucket,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "stopped_on_waxal_regression": stop_state["stopped_on_waxal_regression"],
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

This is a Ghanaian Speech Lab ASR artifact using cleaned GhanaNLP adaptation rows
with Waxal replay and Waxal regression gates.

It is an experimental review artifact. It is not production-promoted until the
held-out reports and Ghanaian listening review pass.

- Base model: `{config.base_model}`
- Training rows: {len(dataset["train"])}
- Waxal validation rows: {len(dataset["dev_waxal"])}
- GhanaNLP validation rows: {len(dataset["dev_gnlp"])}
- Waxal regression-test rows: {len(dataset["test_waxal_regression"])}
- Max steps: {config.max_steps}
- Frozen encoder: {config.freeze_encoder}
- Stop on Waxal regression: {config.stop_on_first_waxal_regression}

Baseline by bucket:

```json
{json.dumps(baseline_by_bucket, indent=2, ensure_ascii=False)}
```

Final by bucket:

```json
{json.dumps(final_by_bucket, indent=2, ensure_ascii=False)}
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
        commit_message="Publish cleaned GhanaNLP plus Waxal replay ASR v0.6 checkpoint",
        ignore_patterns=["training_args.bin"],
    )
    info = api.model_info(config.model_repo, files_metadata=True)
    summary["commit"] = str(commit)
    summary["files"] = sorted(sibling.rfilename for sibling in info.siblings)
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    output_volume.commit()
    return summary

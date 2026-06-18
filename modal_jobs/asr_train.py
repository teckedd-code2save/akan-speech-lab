from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import modal


MINUTE = 60
HOUR = 60 * MINUTE
CACHE_DIR = "/cache"
OUTPUT_DIR = "/outputs"

app = modal.App("akan-speech-asr-train")
cache_volume = modal.Volume.from_name("akan-speech-hf-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("akan-speech-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "accelerate==1.8.1",
        "datasets[audio]==3.6.0",
        "evaluate==0.4.5",
        "huggingface-hub==0.36.0",
        "jiwer==4.0.0",
        "librosa==0.11.0",
        "numpy<2.3",
        "soundfile==0.13.1",
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "transformers==4.53.2",
    )
    .env({"HF_HOME": f"{CACHE_DIR}/huggingface", "HF_XET_HIGH_PERFORMANCE": "1"})
)


@dataclass
class TrainConfig:
    run_name: str = "whisper-small-waxal-aka-no-language-v2"
    base_model: str = "openai/whisper-small"
    dataset_id: str = "google/WaxalNLP"
    dataset_config: str = "aka_asr"
    decoder_strategy: str = "no_forced_language"
    max_steps: int = 1200
    learning_rate: float = 1e-5
    warmup_steps: int = 200
    train_batch_size: int = 8
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    eval_steps: int = 200
    save_steps: int = 200
    generation_max_length: int = 225
    train_limit: int = 0
    eval_limit: int = 0
    seed: int = 42


def normalize_akan_text(text: str) -> str:
    import re
    import unicodedata

    value = unicodedata.normalize("NFC", text or "").strip().lower()
    value = re.sub(r"[“”\"'‘’.,!?;:()\[\]{}<>/\\|*_+=~`]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


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
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def prepared_dataset_path(config: TrainConfig) -> Path:
    model_name = config.base_model.rsplit("/", 1)[-1]
    return Path(CACHE_DIR) / "prepared" / (
        f"{config.dataset_config}-{model_name}-{config.decoder_strategy}-v1"
    )


def waxal_parquet_urls(split: str) -> list[str]:
    root = (
        "https://huggingface.co/datasets/google/WaxalNLP/resolve/"
        "refs%2Fconvert%2Fparquet/aka_asr"
    )
    shard_count = 6 if split == "train" else 1
    return [f"{root}/{split}/{index:04d}.parquet" for index in range(shard_count)]


def prepare_dataset(config: TrainConfig, feature_extractor, tokenizer):
    from datasets import Audio, Dataset, DatasetDict, load_dataset

    def load_split(split: str, limit: int):
        if not limit:
            if config.dataset_id == "google/WaxalNLP" and config.dataset_config == "aka_asr":
                return load_dataset(
                    "parquet",
                    data_files=waxal_parquet_urls(split),
                    split="train",
                    cache_dir=CACHE_DIR,
                )
            return load_dataset(
                config.dataset_id, config.dataset_config, split=split, cache_dir=CACHE_DIR
            )
        stream = load_dataset(
            config.dataset_id,
            config.dataset_config,
            split=split,
            streaming=True,
            cache_dir=CACHE_DIR,
        )
        rows = list(stream.take(limit))
        if len(rows) != limit:
            raise RuntimeError(f"Expected {limit} {split} rows, received {len(rows)}")
        return Dataset.from_list(rows)

    dataset = DatasetDict(
        {
            "train": load_split("train", config.train_limit),
            "validation": load_split("validation", config.eval_limit),
        }
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_row(row):
        audio = row["audio"]
        extracted = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_attention_mask=True
        )
        row["input_features"] = extracted.input_features[0]
        row["attention_mask"] = extracted.attention_mask[0]
        row["input_length"] = len(audio["array"])
        row["labels"] = tokenizer(row["transcription"]).input_ids
        return row

    dataset = dataset.map(
        prepare_row,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
        desc="Extract Whisper features and tokenize Akan transcripts",
    )
    return dataset.filter(
        lambda length: 0.4 * 16000 <= length <= 30.0 * 16000,
        input_columns=["input_length"],
        num_proc=4,
    )


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    volumes={CACHE_DIR: cache_volume},
    secrets=[modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])],
    timeout=6 * HOUR,
)
def prepare_full_training_data(config_dict: dict) -> dict:
    import shutil
    from transformers import WhisperFeatureExtractor, WhisperTokenizer

    config = TrainConfig(**config_dict)
    if config.train_limit or config.eval_limit:
        raise ValueError("Full preparation does not accept row limits.")
    target = prepared_dataset_path(config)
    marker = target.parent / f"{target.name}.complete.json"
    if target.exists() and marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    if target.exists():
        shutil.rmtree(target)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config.base_model, return_attention_mask=True
    )
    tokenizer_kwargs = {"task": "transcribe"}
    if config.decoder_strategy in {"yoruba", "english"}:
        tokenizer_kwargs["language"] = config.decoder_strategy
    tokenizer = WhisperTokenizer.from_pretrained(config.base_model, **tokenizer_kwargs)
    dataset = prepare_dataset(config, feature_extractor, tokenizer)
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(target), max_shard_size="1GB")
    summary = {
        "status": "complete",
        "prepared_dataset": str(target),
        "train_rows": len(dataset["train"]),
        "validation_rows": len(dataset["validation"]),
        "decoder_strategy": config.decoder_strategy,
    }
    marker.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    cache_volume.commit()
    return summary


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    volumes={CACHE_DIR: cache_volume, OUTPUT_DIR: output_volume},
    secrets=[modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])],
    timeout=4 * HOUR,
    scaledown_window=30,
)
def train_asr(config_dict: dict) -> dict:
    import torch
    from datasets import load_from_disk
    from jiwer import wer
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
        set_seed,
    )

    config = TrainConfig(**config_dict)
    set_seed(config.seed)
    run_dir = Path(OUTPUT_DIR) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config.base_model, return_attention_mask=True
    )
    tokenizer_kwargs = {"task": "transcribe"}
    if config.decoder_strategy in {"yoruba", "english"}:
        tokenizer_kwargs["language"] = config.decoder_strategy
    tokenizer = WhisperTokenizer.from_pretrained(config.base_model, **tokenizer_kwargs)
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = WhisperForConditionalGeneration.from_pretrained(config.base_model)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = (
        None if config.decoder_strategy == "no_forced_language" else config.decoder_strategy
    )
    model.generation_config.task = "transcribe"
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []
    model.config.use_cache = False

    if config.train_limit or config.eval_limit:
        dataset = prepare_dataset(config, feature_extractor, tokenizer)
    else:
        prepared_path = prepared_dataset_path(config)
        if not prepared_path.exists():
            raise RuntimeError(
                f"Prepared dataset missing at {prepared_path}; run CPU preparation first."
            )
        dataset = load_from_disk(str(prepared_path))

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def compute_metrics(prediction):
        pred_ids = prediction.predictions
        label_ids = prediction.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_text = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_text = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        normalized_predictions = [normalize_akan_text(text) for text in pred_text]
        normalized_labels = [normalize_akan_text(text) for text in label_text]
        return {"wer": wer(normalized_labels, normalized_predictions)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        run_name=config.run_name,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
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
        save_total_limit=2,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=config.generation_max_length,
        generation_num_beams=1,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=4,
        report_to=[],
        seed=config.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=feature_extractor,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    baseline_metrics = trainer.evaluate(metric_key_prefix="baseline")
    trainer.save_metrics("baseline", baseline_metrics)
    train_result = trainer.train()
    trainer.save_model(str(run_dir / "final"))
    processor.save_pretrained(str(run_dir / "final"))
    trainer.save_metrics("train", train_result.metrics)
    final_metrics = trainer.evaluate(metric_key_prefix="final")
    trainer.save_metrics("final", final_metrics)
    trainer.save_state()
    summary = {
        "status": "complete",
        "run_dir": str(run_dir),
        "config": asdict(config),
        "train_rows": len(dataset["train"]),
        "validation_rows": len(dataset["validation"]),
        "baseline_metrics": baseline_metrics,
        "train_metrics": train_result.metrics,
        "final_metrics": final_metrics,
        "cuda": torch.cuda.get_device_name(0),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    output_volume.commit()
    return summary


@app.local_entrypoint()
def main(smoke: bool = False, max_steps: int = 1200, decoder_strategy: str = "no_forced_language"):
    if decoder_strategy not in {"no_forced_language", "yoruba", "english"}:
        raise ValueError("decoder_strategy must be no_forced_language, yoruba, or english")
    config = TrainConfig(max_steps=max_steps, decoder_strategy=decoder_strategy)
    if smoke:
        config.run_name = "smoke-whisper-small-waxal-aka-no-language-v5"
        config.max_steps = 2
        config.warmup_steps = 0
        config.eval_steps = 1
        config.save_steps = 1
        config.train_limit = 32
        config.eval_limit = 16
        config.train_batch_size = 2
        config.eval_batch_size = 2
        config.gradient_accumulation_steps = 1
    estimated_l4_gpu_cost = 0.80 * (0.25 if smoke else 3.0)
    print(f"Launching {config.run_name}; conservative L4 GPU estimate <= ${estimated_l4_gpu_cost:.2f}")
    if not smoke:
        print("Preparing and caching the full dataset on CPU before allocating an L4...")
        preparation = prepare_full_training_data.remote(asdict(config))
        print(json.dumps(preparation, indent=2, default=str))
    result = train_asr.remote(asdict(config))
    print(json.dumps(result, indent=2, default=str))
    print(f"Result persisted under {OUTPUT_DIR}/{config.run_name}/summary.json")

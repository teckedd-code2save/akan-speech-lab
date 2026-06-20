from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import modal


HOUR = 60 * 60
CACHE_DIR = "/cache"
OUTPUT_DIR = "/outputs"
MANIFEST_DIR = Path("/round2_manifests")
APP_NAME = "akan-speech-asr-round2"
PARQUET_REVISION = "fe897206a41cad1b26f39f4c4088a45538ccfced"

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("akan-speech-hf-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("akan-speech-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "accelerate==1.8.1",
        "datasets[audio]==3.6.0",
        "jiwer==4.0.0",
        "librosa==0.11.0",
        "numpy<2.3",
        "soundfile==0.13.1",
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "transformers==4.53.2",
    )
    .env({"HF_HOME": f"{CACHE_DIR}/huggingface", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir("data/manifests/waxal_round2", remote_path=str(MANIFEST_DIR), copy=True)
)


@dataclass
class Round2Config:
    run_name: str = "whisper-small-waxal-round2-specaug-v1"
    base_model: str = "openai/whisper-small"
    decoder_strategy: str = "yoruba"
    seed: int = 42
    max_steps: int = 2000
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 200
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    eval_steps: int = 200
    save_steps: int = 200
    generation_max_length: int = 225
    early_stopping_patience: int = 3
    train_limit: int = 0
    eval_limit: int = 0
    apply_spec_augment: bool = True
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    mask_feature_prob: float = 0.05
    mask_feature_length: int = 10


def normalize_akan_text(text: str) -> str:
    import re
    import unicodedata

    value = unicodedata.normalize("NFC", text or "").strip().lower()
    value = re.sub(r"[“”\"'‘’.,!?;:()\[\]{}<>/\\|*_+=~`]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def prepared_dataset_path(config: Round2Config) -> Path:
    return Path(CACHE_DIR) / "prepared" / (
        f"waxal-round2-{config.base_model.rsplit('/', 1)[-1]}-"
        f"{config.decoder_strategy}-clean-v1"
    )


def parquet_urls(split: str) -> list[str]:
    root = (
        "https://huggingface.co/datasets/google/WaxalNLP/resolve/"
        f"{PARQUET_REVISION}/aka_asr/{split}"
    )
    shard_count = 6 if split == "train" else 1
    return [f"{root}/{index:04d}.parquet" for index in range(shard_count)]


def read_manifest(split: str) -> list[dict]:
    path = MANIFEST_DIR / f"{split}.jsonl"
    if not path.exists():
        raise RuntimeError(f"Missing deployed manifest: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_frozen_splits():
    from datasets import Audio, DatasetDict, load_dataset

    source = {}
    for split in ("train", "validation", "test"):
        source[split] = load_dataset(
            "parquet",
            data_files=parquet_urls(split),
            split="train",
            cache_dir=CACHE_DIR,
        )
    manifests = {split: read_manifest(split) for split in ("train", "dev", "test")}
    selected = {}
    for target_split, rows in manifests.items():
        indices_by_source: dict[str, list[int]] = {}
        for row in rows:
            indices_by_source.setdefault(row["source_split"], []).append(int(row["dataset_row"]))
        parts = [source[name].select(indices) for name, indices in sorted(indices_by_source.items())]
        if not parts:
            raise RuntimeError(f"Frozen {target_split} manifest is empty")
        if len(parts) == 1:
            selected[target_split] = parts[0]
        else:
            from datasets import concatenate_datasets

            selected[target_split] = concatenate_datasets(parts)

        expected_ids = {row["sample_id"] for row in rows}
        actual_ids = set(selected[target_split]["id"])
        if actual_ids != expected_ids:
            raise RuntimeError(
                f"{target_split} manifest mismatch: missing={len(expected_ids - actual_ids)}, "
                f"unexpected={len(actual_ids - expected_ids)}"
            )
    return DatasetDict(selected).cast_column("audio", Audio(sampling_rate=16000))


def prepare_round2_dataset(config: Round2Config):
    import hashlib

    import numpy as np
    from transformers import WhisperFeatureExtractor, WhisperTokenizer

    dataset = load_frozen_splits()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config.base_model, return_attention_mask=True
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        config.base_model,
        language=config.decoder_strategy,
        task="transcribe",
    )

    def inspect_and_prepare(row):
        audio = row["audio"]
        samples = np.asarray(audio["array"], dtype="<f4")
        duration = len(samples) / int(audio["sampling_rate"])
        absolute = np.abs(samples)
        rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
        clipping_ratio = float(np.mean(absolute >= 0.999)) if len(samples) else 1.0
        near_silence_ratio = float(np.mean(absolute < 1e-4)) if len(samples) else 1.0
        normalized_text = normalize_akan_text(row["transcription"])
        words_per_second = len(normalized_text.split()) / duration if duration else 0.0
        extracted = feature_extractor(
            samples,
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=True,
        )
        quality_flags = [
            name
            for name, flagged in (
                ("low_rms", rms < 0.005),
                ("clipping", clipping_ratio > 0.01),
                ("mostly_silence", near_silence_ratio > 0.8),
                ("text_audio_rate", words_per_second < 0.5 or words_per_second > 6.0),
            )
            if flagged
        ]
        return {
            "sample_id": row["id"],
            "speaker_id": row["speaker_id"],
            "normalized_text": normalized_text,
            "duration_seconds": duration,
            "audio_hash": hashlib.sha256(
                samples.tobytes() + int(audio["sampling_rate"]).to_bytes(4, "big")
            ).hexdigest(),
            "rms": rms,
            "clipping_ratio": clipping_ratio,
            "near_silence_ratio": near_silence_ratio,
            "words_per_second": words_per_second,
            "quality_flags": ",".join(quality_flags),
            "input_features": extracted.input_features[0],
            "attention_mask": extracted.attention_mask[0],
            "labels": tokenizer(normalized_text).input_ids,
        }

    prepared = dataset.map(
        inspect_and_prepare,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
        desc="Audit audio and extract Round 2 Whisper features",
    )

    invalid_counts = {}
    for split in prepared:
        rows = prepared[split]
        invalid_counts[split] = {
            "empty_text": sum(not text for text in rows["normalized_text"]),
            "duration_outside_0.4_30": sum(
                not 0.4 <= duration <= 30.0 for duration in rows["duration_seconds"]
            ),
        }
    if any(values["empty_text"] for values in invalid_counts.values()):
        raise RuntimeError(f"Empty normalized transcripts found: {invalid_counts}")

    prepared = prepared.filter(
        lambda duration: 0.4 <= duration <= 30.0,
        input_columns=["duration_seconds"],
        num_proc=4,
        desc="Apply fixed 0.4-30 second duration gate",
    )

    duplicate_audio_removed = {}
    for split in ("train", "dev"):
        seen = set()
        keep = []
        for index, audio_hash in enumerate(prepared[split]["audio_hash"]):
            if audio_hash not in seen:
                seen.add(audio_hash)
                keep.append(index)
        duplicate_audio_removed[split] = len(prepared[split]) - len(keep)
        prepared[split] = prepared[split].select(keep)

    # Dev is the immutable model-selection partition. If an identical decoded
    # waveform also occurs in train, quarantine the training copy and retain a
    # complete audit trail. Test is never repaired: any overlap with it aborts.
    hash_sets = {split: set(prepared[split]["audio_hash"]) for split in prepared}
    train_dev_hashes = hash_sets["train"] & hash_sets["dev"]
    train_dev_audio_removed = [
        {
            "sample_id": sample_id,
            "speaker_id": speaker_id,
            "audio_hash": audio_hash,
        }
        for sample_id, speaker_id, audio_hash in zip(
            prepared["train"]["sample_id"],
            prepared["train"]["speaker_id"],
            prepared["train"]["audio_hash"],
            strict=True,
        )
        if audio_hash in train_dev_hashes
    ]
    if train_dev_audio_removed:
        prepared["train"] = prepared["train"].filter(
            lambda audio_hash: audio_hash not in train_dev_hashes,
            input_columns=["audio_hash"],
            desc="Quarantine train audio duplicated in dev",
        )

    hash_sets = {split: set(prepared[split]["audio_hash"]) for split in prepared}
    immutable_test_overlap = {
        "train_test": len(hash_sets["train"] & hash_sets["test"]),
        "dev_test": len(hash_sets["dev"] & hash_sets["test"]),
    }
    if any(immutable_test_overlap.values()):
        raise RuntimeError(
            f"Decoded-audio overlap with immutable test partition: {immutable_test_overlap}"
        )

    text_sets = {split: set(prepared[split]["normalized_text"]) for split in prepared}
    speaker_sets = {split: set(prepared[split]["speaker_id"]) for split in prepared}
    assertions = {
        "train_dev_audio_overlap": len(hash_sets["train"] & hash_sets["dev"]),
        "train_test_audio_overlap": len(hash_sets["train"] & hash_sets["test"]),
        "dev_test_audio_overlap": len(hash_sets["dev"] & hash_sets["test"]),
        "train_dev_text_overlap": len(text_sets["train"] & text_sets["dev"]),
        "train_test_text_overlap": len(text_sets["train"] & text_sets["test"]),
        "dev_test_text_overlap": len(text_sets["dev"] & text_sets["test"]),
        "train_dev_speaker_overlap": len(speaker_sets["train"] & speaker_sets["dev"]),
        "train_test_speaker_overlap": len(speaker_sets["train"] & speaker_sets["test"]),
        "dev_test_speaker_overlap": len(speaker_sets["dev"] & speaker_sets["test"]),
    }
    if any(assertions.values()):
        raise RuntimeError(f"Round 2 contamination assertions failed: {assertions}")

    quality_flag_counts = {}
    for split in prepared:
        counts: dict[str, int] = {}
        for flags in prepared[split]["quality_flags"]:
            for flag in filter(None, flags.split(",")):
                counts[flag] = counts.get(flag, 0) + 1
        quality_flag_counts[split] = counts
    audit = {
        "status": "complete",
        "dataset": "google/WaxalNLP/aka_asr",
        "parquet_revision": PARQUET_REVISION,
        "prepared_dataset": str(prepared_dataset_path(config)),
        "rows": {split: len(prepared[split]) for split in prepared},
        "speakers": {split: len(speaker_sets[split]) for split in prepared},
        "invalid_before_filter": invalid_counts,
        "duplicate_audio_removed": duplicate_audio_removed,
        "train_dev_audio_quarantine": train_dev_audio_removed,
        "quality_flag_counts": quality_flag_counts,
        "assertions": assertions,
        "passed": not any(assertions.values()),
        "config": asdict(config),
    }
    return prepared, audit


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    volumes={CACHE_DIR: cache_volume},
    secrets=[modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])],
    timeout=6 * HOUR,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
    max_containers=1,
)
def prepare_round2_data(config_dict: dict) -> dict:
    import shutil

    from datasets import DatasetDict

    config = Round2Config(**config_dict)
    target = prepared_dataset_path(config)
    marker = target.parent / f"{target.name}.complete.json"
    if marker.exists() and target.exists():
        cached = json.loads(marker.read_text())
        if cached.get("passed"):
            return cached
    if target.exists():
        shutil.rmtree(target)
    prepared, audit = prepare_round2_dataset(config)
    if not isinstance(prepared, DatasetDict) or not audit["passed"]:
        raise RuntimeError("Round 2 preparation did not produce a passing DatasetDict")
    target.parent.mkdir(parents=True, exist_ok=True)
    prepared.save_to_disk(str(target), max_shard_size="1GB")
    marker.write_text(json.dumps(audit, indent=2) + "\n")
    cache_volume.commit()
    return audit


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


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = []
    for path in run_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[-1]), path))
        except ValueError:
            continue
    return max(checkpoints, default=(0, None))[1]


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    volumes={CACHE_DIR: cache_volume, OUTPUT_DIR: output_volume},
    secrets=[modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])],
    timeout=6 * HOUR,
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0, initial_delay=10.0),
    scaledown_window=30,
    max_containers=1,
)
def train_round2(config_dict: dict) -> dict:
    import torch
    from datasets import load_from_disk
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

    config = Round2Config(**config_dict)
    prepared_path = prepared_dataset_path(config)
    marker_path = prepared_path.parent / f"{prepared_path.name}.complete.json"
    if not prepared_path.exists() or not marker_path.exists():
        raise RuntimeError("Passing Round 2 preparation is required before GPU allocation")
    audit = json.loads(marker_path.read_text())
    if not audit.get("passed"):
        raise RuntimeError("Round 2 contamination audit did not pass")

    set_seed(config.seed)
    dataset = load_from_disk(str(prepared_path))
    if config.train_limit:
        dataset["train"] = dataset["train"].select(range(min(config.train_limit, len(dataset["train"]))))
    if config.eval_limit:
        dataset["dev"] = dataset["dev"].select(range(min(config.eval_limit, len(dataset["dev"]))))

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config.base_model, return_attention_mask=True
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        config.base_model,
        language=config.decoder_strategy,
        task="transcribe",
    )
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = WhisperForConditionalGeneration.from_pretrained(config.base_model)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = None
    model.generation_config.task = "transcribe"
    model.config.use_cache = False
    model.config.apply_spec_augment = config.apply_spec_augment
    model.config.mask_time_prob = config.mask_time_prob
    model.config.mask_time_length = config.mask_time_length
    model.config.mask_feature_prob = config.mask_feature_prob
    model.config.mask_feature_length = config.mask_feature_length

    run_dir = Path(OUTPUT_DIR) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        completed = json.loads(summary_path.read_text())
        if completed.get("status") == "complete":
            return completed
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def compute_metrics(prediction):
        label_ids = prediction.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        predictions = [normalize_akan_text(text) for text in tokenizer.batch_decode(
            prediction.predictions, skip_special_tokens=True
        )]
        labels = [normalize_akan_text(text) for text in tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )]
        return {"wer": wer(labels, predictions)}

    class CommitCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            output_volume.commit()
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
        save_total_limit=3,
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
        eval_dataset=dataset["dev"],
        processing_class=feature_extractor,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            CommitCheckpointCallback(),
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience),
        ],
    )
    checkpoint = latest_checkpoint(run_dir)
    baseline_path = run_dir / "baseline_results.json"
    if checkpoint and baseline_path.exists():
        baseline_metrics = json.loads(baseline_path.read_text())
        print(f"Resuming automatically from {checkpoint}")
    else:
        baseline_metrics = trainer.evaluate(metric_key_prefix="baseline")
        trainer.save_metrics("baseline", baseline_metrics)
        output_volume.commit()
    train_result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
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
        "rows": {"train": len(dataset["train"]), "dev": len(dataset["dev"])},
        "baseline_metrics": baseline_metrics,
        "train_metrics": train_result.metrics,
        "final_metrics": final_metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "cuda": torch.cuda.get_device_name(0),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    output_volume.commit()
    return summary

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import modal


HOUR = 60 * 60
CACHE_DIR = "/cache"
OUTPUT_DIR = "/outputs"
APP_NAME = "akan-speech-asr-round3-lora"
PREPARED_PATH = Path(CACHE_DIR) / "prepared/waxal-round2-whisper-small-yoruba-clean-v1"
PREPARED_MARKER = PREPARED_PATH.parent / f"{PREPARED_PATH.name}.complete.json"

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
        "numpy<2.3",
        "peft==0.15.2",
        "soundfile==0.13.1",
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "transformers==4.53.2",
    )
    .env({"HF_HOME": f"{CACHE_DIR}/huggingface", "HF_XET_HIGH_PERFORMANCE": "1"})
)


@dataclass
class Round3Config:
    run_name: str = "whisper-medium-waxal-round3-lora-v1"
    base_model: str = "openai/whisper-medium"
    decoder_strategy: str = "yoruba"
    seed: int = 42
    max_steps: int = 1200
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    train_batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    eval_steps: int = 200
    save_steps: int = 200
    generation_max_length: int = 225
    early_stopping_patience: int = 2
    train_limit: int = 0
    eval_limit: int = 0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


def normalize_akan_text(text: str) -> str:
    import re
    import unicodedata

    value = unicodedata.normalize("NFC", text or "").strip().lower()
    value = re.sub(r"[“”\"'‘’.,!?;:()\[\]{}<>/\\|*_+=~`]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = []
    for path in run_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[-1]), path))
        except ValueError:
            continue
    return max(checkpoints, default=(0, None))[1]


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
    timeout=6 * HOUR,
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0, initial_delay=10.0),
    scaledown_window=30,
    max_containers=1,
)
def train_lora(config_dict: dict) -> dict:
    import torch
    from datasets import load_from_disk
    from jiwer import wer
    from peft import LoraConfig, get_peft_model
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

    config = Round3Config(**config_dict)
    if not PREPARED_PATH.exists() or not PREPARED_MARKER.exists():
        raise RuntimeError("Passing Round 2 preparation is required before Round 3 GPU allocation")
    audit = json.loads(PREPARED_MARKER.read_text())
    if not audit.get("passed") or any(audit.get("assertions", {}).values()):
        raise RuntimeError("Frozen Waxal contamination audit is missing or failed")

    set_seed(config.seed)
    dataset = load_from_disk(str(PREPARED_PATH))
    if config.train_limit:
        dataset["train"] = dataset["train"].select(
            range(min(config.train_limit, len(dataset["train"])))
        )
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
    base_model = WhisperForConditionalGeneration.from_pretrained(config.base_model)
    base_model.config.forced_decoder_ids = None
    base_model.generation_config.forced_decoder_ids = None
    base_model.generation_config.language = None
    base_model.generation_config.task = "transcribe"
    base_model.config.use_cache = False
    base_model.config.apply_spec_augment = True
    base_model.config.mask_time_prob = 0.05
    base_model.config.mask_time_length = 10
    base_model.config.mask_feature_prob = 0.05
    base_model.config.mask_feature_length = 10
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=["q_proj", "v_proj"],
        ),
    )
    trainable, total = model.get_nb_trainable_parameters()
    if trainable <= 0 or trainable >= total * 0.1:
        raise RuntimeError(f"Unexpected LoRA parameter count: {trainable:,}/{total:,}")
    print(f"LoRA trainable parameters: {trainable:,}/{total:,} ({trainable / total:.4%})")

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
        decoder_start_token_id=base_model.config.decoder_start_token_id,
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
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience),
        ],
    )
    checkpoint = latest_checkpoint(run_dir)
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
        "trainable_parameters": trainable,
        "total_parameters": total,
        "train_metrics": train_result.metrics,
        "final_metrics": final_metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "cuda": torch.cuda.get_device_name(0),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    output_volume.commit()
    return summary

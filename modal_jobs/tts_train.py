from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import modal


HOUR = 60 * 60
APP_NAME = "akan-speech-tts"
CACHE_DIR = "/cache"
OUTPUT_DIR = "/outputs"
DATA_DIR = Path(CACHE_DIR) / "farmerline-akosua-diagnostic-v1"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"
AUDIT_PATH = DATA_DIR / "audit.json"
DATASET_ID = "Farmerline-DCS-HCI25/akan_tts_dataset"
DATASET_REVISION = "050b308e66155bb5e3d843b01ebca51f9d7056e2"

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("akan-speech-hf-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("akan-speech-tts-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "accelerate==1.8.1",
        "datasets[audio]==3.6.0",
        "librosa==0.11.0",
        "numpy<2.3",
        "safetensors==0.5.3",
        "soundfile==0.13.1",
        "speechbrain==1.0.3",
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "transformers==4.53.2",
    )
    .env({"HF_HOME": f"{CACHE_DIR}/huggingface", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir("src/akan_speech", remote_path="/root/akan_speech")
)


@dataclass
class TTSConfig:
    run_name: str = "speecht5-farmerline-akosua-diagnostic-v1"
    base_model: str = "microsoft/speecht5_tts"
    vocoder: str = "microsoft/speecht5_hifigan"
    speaker_encoder: str = "speechbrain/spkrec-xvect-voxceleb"
    seed: int = 42
    max_steps: int = 8000
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    train_batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    eval_steps: int = 200
    save_steps: int = 200
    early_stopping_patience: int = 3
    train_limit: int = 0
    eval_limit: int = 0


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = []
    for path in run_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[-1]), path))
        except ValueError:
            continue
    return max(checkpoints, default=(0, None))[1]


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    volumes={CACHE_DIR: cache_volume},
    secrets=[modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])],
    timeout=2 * HOUR,
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0, initial_delay=10.0),
    scaledown_window=30,
    max_containers=1,
)
def prepare_diagnostic_corpus() -> dict:
    import hashlib
    from collections import Counter

    import librosa
    import numpy as np
    import soundfile as sf
    from datasets import Audio, concatenate_datasets, load_dataset

    from akan_speech.tts.manifest import finalize_manifest
    from akan_speech.tts.quality import audit_audio
    from akan_speech.tts.text import normalize_tts_text

    if AUDIT_PATH.exists():
        existing = json.loads(AUDIT_PATH.read_text())
        if existing.get("passed"):
            return existing

    source = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    combined = concatenate_datasets([source[name] for name in sorted(source)]).cast_column(
        "audio", Audio(sampling_rate=16_000)
    )
    speaker_counts = Counter(str(row.get("speaker_id") or "unknown") for row in combined)
    akosua_ids = {speaker for speaker in speaker_counts if "akosua" in speaker.casefold()}
    if not akosua_ids:
        raise RuntimeError(f"Akosua speaker not found; observed speakers: {speaker_counts}")

    audio_dir = DATA_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, row in enumerate(combined):
        speaker_id = str(row.get("speaker_id") or "")
        if speaker_id not in akosua_ids:
            continue
        audio = row["audio"]
        values = np.asarray(audio["array"], dtype=np.float32)
        rate = int(audio["sampling_rate"])
        if values.ndim > 1:
            values = values.mean(axis=0)
        if rate != 16_000:
            values = librosa.resample(values, orig_sr=rate, target_sr=16_000)
            rate = 16_000
        values, _ = librosa.effects.trim(values, top_db=35)
        raw_audit = audit_audio(values, rate)
        if values.size:
            rms = float(np.sqrt(np.mean(np.square(values))))
            target = 10 ** (-23 / 20)
            if rms > 1e-8:
                values = np.clip(values * min(target / rms, 4.0), -1.0, 1.0)
        processed_audit = audit_audio(values, rate)
        sample_id = hashlib.sha256(
            f"{DATASET_REVISION}:{speaker_id}:{index}".encode()
        ).hexdigest()[:20]
        audio_path = audio_dir / f"{sample_id}.wav"
        sf.write(audio_path, values, rate, subtype="PCM_16")
        text = str(row.get("transcription") or row.get("text") or "")
        rows.append(
            {
                "sample_id": sample_id,
                "speaker_id": speaker_id,
                "source_row": index,
                "original_text": text,
                "normalized_text": normalize_tts_text(text),
                "audio_path": str(audio_path),
                **processed_audit.to_dict(),
                "raw_audio_flags": list(raw_audit.flags),
                "source_dataset": DATASET_ID,
                "source_revision": DATASET_REVISION,
                "source_license": "CC-BY-4.0",
                "consent_status": "public-diagnostic-no-explicit-speaker-consent",
            }
        )

    manifest, report = finalize_manifest(rows)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
        encoding="utf-8",
    )
    report.update(
        {
            "status": "complete" if report["passed"] else "failed",
            "dataset": DATASET_ID,
            "revision": DATASET_REVISION,
            "speakers": dict(speaker_counts),
            "selected_speakers": sorted(akosua_ids),
            "manifest_path": str(MANIFEST_PATH),
        }
    )
    AUDIT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    cache_volume.commit()
    return report


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
def train_speecht5(config_dict: dict) -> dict:
    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Dataset
    from speechbrain.inference.classifiers import EncoderClassifier
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        SpeechT5ForTextToSpeech,
        SpeechT5HifiGan,
        SpeechT5Processor,
        TrainerCallback,
        set_seed,
    )

    from akan_speech.tts.tokenizer import extend_character_tokenizer

    config = TTSConfig(**config_dict)
    if not AUDIT_PATH.exists() or not MANIFEST_PATH.exists():
        raise RuntimeError("Passing CPU corpus preparation is required before GPU allocation")
    audit = json.loads(AUDIT_PATH.read_text())
    if not audit.get("passed"):
        raise RuntimeError("Diagnostic corpus audit failed")
    rows = [json.loads(line) for line in MANIFEST_PATH.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["accepted"]]
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "validation"]
    if config.train_limit:
        train_rows = train_rows[: config.train_limit]
    if config.eval_limit:
        eval_rows = eval_rows[: config.eval_limit]
    if not train_rows or not eval_rows:
        raise RuntimeError("Prepared train and validation rows are required")

    set_seed(config.seed)
    processor = SpeechT5Processor.from_pretrained(config.base_model)
    tokenizer_audit = extend_character_tokenizer(
        processor.tokenizer, [row["normalized_text"] for row in rows]
    )
    if not tokenizer_audit["passed"]:
        raise RuntimeError(f"Tokenizer coverage failed: {tokenizer_audit}")
    model = SpeechT5ForTextToSpeech.from_pretrained(config.base_model)
    model.resize_token_embeddings(len(processor.tokenizer))
    model.config.use_cache = False
    vocoder = SpeechT5HifiGan.from_pretrained(config.vocoder).to("cuda")

    speaker_encoder = EncoderClassifier.from_hparams(
        source=config.speaker_encoder,
        run_opts={"device": "cuda"},
        savedir=f"{CACHE_DIR}/speechbrain/xvector",
    )
    embedding_rows = train_rows[: min(64, len(train_rows))]
    embeddings = []
    for row in embedding_rows:
        audio, rate = sf.read(row["audio_path"], dtype="float32")
        waveform = torch.from_numpy(np.asarray(audio)).unsqueeze(0).to("cuda")
        if rate != 16_000:
            raise RuntimeError("Prepared audio must be 16 kHz")
        with torch.inference_mode():
            embeddings.append(speaker_encoder.encode_batch(waveform).squeeze().cpu())
    speaker_embedding = torch.stack(embeddings).mean(dim=0)
    speaker_embedding = torch.nn.functional.normalize(speaker_embedding, dim=0).numpy()

    def prepare(row):
        audio, rate = sf.read(row["audio_path"], dtype="float32")
        encoded = processor(
            text=row["normalized_text"],
            audio_target=audio,
            sampling_rate=rate,
            return_attention_mask=False,
        )
        encoded["labels"] = encoded["labels"][0]
        encoded["speaker_embeddings"] = speaker_embedding
        return encoded

    train_dataset = Dataset.from_list(train_rows).map(prepare, remove_columns=list(train_rows[0]))
    eval_dataset = Dataset.from_list(eval_rows).map(prepare, remove_columns=list(eval_rows[0]))
    train_dataset = train_dataset.filter(lambda row: len(row["labels"]) < 600)
    eval_dataset = eval_dataset.filter(lambda row: len(row["labels"]) < 600)

    class Collator:
        def __call__(self, features):
            input_ids = [{"input_ids": item["input_ids"]} for item in features]
            labels = [{"input_values": item["labels"]} for item in features]
            batch = processor.pad(input_ids=input_ids, labels=labels, return_tensors="pt")
            batch["labels"] = batch["labels"].masked_fill(
                batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
            )
            del batch["decoder_attention_mask"]
            if model.config.reduction_factor > 1:
                target_lengths = torch.tensor([len(item["input_values"]) for item in labels])
                target_lengths = target_lengths.new(
                    [length - length % model.config.reduction_factor for length in target_lengths]
                )
                max_length = max(target_lengths)
                batch["labels"] = batch["labels"][:, :max_length]
            batch["speaker_embeddings"] = torch.tensor(
                np.array([item["speaker_embeddings"] for item in features]), dtype=torch.float32
            )
            return batch

    run_dir = Path(OUTPUT_DIR) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        completed = json.loads(summary_path.read_text())
        if completed.get("status") == "complete":
            return completed
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (run_dir / "tokenizer_audit.json").write_text(
        json.dumps(tokenizer_audit, indent=2, ensure_ascii=False) + "\n"
    )

    class CommitCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            output_volume.commit()
            return control

    args = Seq2SeqTrainingArguments(
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
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        seed=config.seed,
    )
    callbacks = [CommitCallback()]
    if config.max_steps >= config.eval_steps * 2:
        callbacks.append(EarlyStoppingCallback(config.early_stopping_patience))
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Collator(),
        processing_class=processor,
        callbacks=callbacks,
    )
    checkpoint = _latest_checkpoint(run_dir)
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    final_metrics = trainer.evaluate(metric_key_prefix="final")
    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    vocoder.save_pretrained(str(final_dir / "vocoder"))
    np.save(final_dir / "speaker_embedding.npy", speaker_embedding)

    prompt = eval_rows[0]["normalized_text"]
    inputs = processor(text=prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        speech = model.generate_speech(
            inputs["input_ids"],
            torch.tensor(speaker_embedding, device="cuda").unsqueeze(0),
            vocoder=vocoder,
        )
    sample_path = run_dir / "validation_sample.wav"
    sf.write(sample_path, speech.cpu().numpy(), 16_000)
    summary = {
        "status": "complete",
        "run_dir": str(run_dir),
        "config": asdict(config),
        "rows": {"train": len(train_dataset), "validation": len(eval_dataset)},
        "tokenizer_audit": tokenizer_audit,
        "train_metrics": result.metrics,
        "final_metrics": final_metrics,
        "loss_history": [
            {"step": item.get("step"), "loss": item.get("loss")}
            for item in trainer.state.log_history
            if item.get("loss") is not None
        ],
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "validation_prompt": prompt,
        "validation_audio": str(sample_path),
        "pilot_gate": {
            "passed": False,
            "status": "needs_human_review",
            "reason": "MOS and listener transcription accuracy have not been collected.",
        },
        "cuda": torch.cuda.get_device_name(0),
    }
    if config.run_name.startswith("overfit32-"):
        losses = [item["loss"] for item in summary["loss_history"]]
        summary["overfit_gate"] = {
            "loss_converged": bool(losses and losses[-1] < losses[0] * 0.8),
            "passed": False,
            "status": "needs_human_alignment_review",
            "reason": "Listen for intelligibility and inspect the saved loss history before pilot.",
        }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n")
    output_volume.commit()
    return summary


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    volumes={CACHE_DIR: cache_volume, OUTPUT_DIR: output_volume},
    timeout=30 * 60,
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0, initial_delay=10.0),
    scaledown_window=30,
    max_containers=1,
)
def synthesize_saved_checkpoint(run_name: str, split: str = "train", index: int = 0) -> dict:
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    final_dir = Path(OUTPUT_DIR) / run_name / "final"
    if not final_dir.exists():
        raise RuntimeError(f"Missing saved checkpoint: {final_dir}")
    rows = [json.loads(line) for line in MANIFEST_PATH.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("accepted") and row.get("split") == split]
    if not rows:
        raise RuntimeError(f"No accepted {split} rows")
    row = rows[index % len(rows)]

    processor = SpeechT5Processor.from_pretrained(str(final_dir))
    model = SpeechT5ForTextToSpeech.from_pretrained(str(final_dir)).to("cuda").eval()
    vocoder = SpeechT5HifiGan.from_pretrained(str(final_dir / "vocoder")).to("cuda").eval()
    embedding = torch.tensor(
        np.load(final_dir / "speaker_embedding.npy"), dtype=torch.float32, device="cuda"
    ).unsqueeze(0)
    inputs = processor(text=row["normalized_text"], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        speech = model.generate_speech(
            inputs["input_ids"], speaker_embeddings=embedding, vocoder=vocoder
        )
    diagnostic_dir = Path(OUTPUT_DIR) / run_name / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    output_path = diagnostic_dir / f"{split}-{index}.wav"
    sf.write(output_path, speech.cpu().numpy(), 16_000)
    result = {
        "status": "complete",
        "run_name": run_name,
        "split": split,
        "index": index,
        "sample_id": row["sample_id"],
        "text": row["normalized_text"],
        "reference_audio": row["audio_path"],
        "generated_audio": str(output_path),
        "duration_seconds": round(len(speech) / 16_000, 4),
    }
    (diagnostic_dir / f"{split}-{index}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    output_volume.commit()
    return result

from __future__ import annotations

import modal

app = modal.App("akan-speech-asr-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate",
        "datasets",
        "evaluate",
        "jiwer",
        "librosa",
        "soundfile",
        "torch",
        "transformers>=4.52.0",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 4,
    scaledown_window=60,
)
def train_asr(config_path: str = "configs/asr/whisper_small_ghana_nlp.yaml") -> dict:
    """Modal ASR training entrypoint.

    This is intentionally not auto-running. Wire the local package and a validated
    manifest before spending GPU time.
    """

    return {
        "status": "not_implemented",
        "config_path": config_path,
        "next": "Mount repo, load config, run Trainer, evaluate, push only if WER improves.",
    }


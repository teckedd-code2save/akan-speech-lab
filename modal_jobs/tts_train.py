from __future__ import annotations

import modal

app = modal.App("akan-speech-tts-train")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets",
    "soundfile",
    "torch",
    "transformers",
)


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 4, scaledown_window=60)
def train_tts(config_path: str = "configs/tts/mms_tts_akan.yaml") -> dict:
    return {
        "status": "gated",
        "config_path": config_path,
        "next": "Confirm speaker consent, transcript quality, and eval protocol before TTS fine-tuning.",
    }


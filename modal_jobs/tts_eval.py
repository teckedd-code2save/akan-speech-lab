from __future__ import annotations

import modal

app = modal.App("akan-speech-tts-eval")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "librosa",
    "numpy",
    "soundfile",
    "torch",
    "transformers",
)


@app.function(image=image, gpu="A10G", timeout=60 * 60, scaledown_window=60)
def eval_tts(model_id: str, prompt_manifest: str) -> dict:
    return {
        "status": "gated",
        "model_id": model_id,
        "prompt_manifest": prompt_manifest,
        "next": "Generate fixed prompts and collect intelligibility/naturalness review.",
    }


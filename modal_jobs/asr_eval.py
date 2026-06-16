from __future__ import annotations

import modal

app = modal.App("akan-speech-asr-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets", "evaluate", "jiwer", "librosa", "soundfile", "torch", "transformers")
)


@app.function(image=image, gpu="A10G", timeout=60 * 60, scaledown_window=60)
def eval_asr(model_id: str, manifest_path: str) -> dict:
    return {
        "status": "not_implemented",
        "model_id": model_id,
        "manifest_path": manifest_path,
        "next": "Run deterministic decode over manifest and write WER/CER report.",
    }


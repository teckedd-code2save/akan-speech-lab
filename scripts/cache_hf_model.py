from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache a Hugging Face model before evaluation.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--allow",
        nargs="*",
        default=[
            "*.json",
            "*.txt",
            "*.model",
            "*.safetensors",
            "*.bin",
            "merges.txt",
            "vocab.json",
            "tokenizer*",
            "preprocessor_config.json",
            "generation_config.json",
        ],
    )
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        allow_patterns=args.allow,
    )
    size = sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file())
    print({"model_id": args.model_id, "path": path, "size_mb": round(size / 1024 / 1024, 2)})


if __name__ == "__main__":
    main()


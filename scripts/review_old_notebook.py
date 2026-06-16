from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYWORDS = [
    "openai/whisper",
    "language",
    "yoruba",
    "forced_decoder_ids",
    "tokenizer",
    "wer",
    "learning_rate",
    "warmup",
    "max_steps",
    "gradient_accumulation",
    "push_to_hub",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract review clues from an old fine-tune notebook.")
    parser.add_argument("notebook")
    args = parser.parse_args()

    notebook = json.loads(Path(args.notebook).read_text(encoding="utf-8"))
    hits = []
    for idx, cell in enumerate(notebook.get("cells", [])):
        source = "".join(cell.get("source", []))
        lowered = source.lower()
        matched = [keyword for keyword in KEYWORDS if keyword.lower() in lowered]
        if matched:
            hits.append(
                {
                    "cell": idx,
                    "type": cell.get("cell_type"),
                    "matched": matched,
                    "preview": source[:1200],
                }
            )
    print(json.dumps({"notebook": args.notebook, "hits": hits}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


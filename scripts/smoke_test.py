from __future__ import annotations

from pathlib import Path

from akan_speech.data.manifest import make_record, validate_manifest, write_jsonl
from akan_speech.data.normalize import normalize_akan_text
from akan_speech.eval.wer import speech_error_rates


def main() -> None:
    assert normalize_akan_text("Ɛnnɛ, me ho yɛ!") == "ɛnnɛ me ho yɛ"
    report_dir = Path("evals/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = report_dir / "smoke_manifest.jsonl"
    write_jsonl(
        [
            make_record(
                audio_path="evals/samples/example.wav",
                text="Ɛnnɛ me ho yɛ",
                language="twi",
                source="smoke",
            )
        ],
        manifest,
    )
    validation = validate_manifest(manifest)
    assert validation["valid"], validation
    metrics = speech_error_rates(["Ɛnnɛ me ho yɛ"], ["ɛnnɛ me ho ye"])
    print({"validation": validation, "metrics": metrics})


if __name__ == "__main__":
    main()


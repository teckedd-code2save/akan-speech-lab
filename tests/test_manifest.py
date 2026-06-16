from pathlib import Path

from akan_speech.data.manifest import make_record, validate_manifest, write_jsonl


def test_manifest_roundtrip(tmp_path: Path):
    manifest = tmp_path / "sample.jsonl"
    count = write_jsonl(
        [make_record(audio_path="sample.wav", text="Ɛnnɛ me ho yɛ", language="twi")],
        manifest,
    )
    assert count == 1
    report = validate_manifest(manifest)
    assert report["valid"] is True
    assert report["languages"] == ["aka"]


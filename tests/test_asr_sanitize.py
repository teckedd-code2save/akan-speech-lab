import json
from pathlib import Path

from akan_speech.asr.sanitize import SanitizePaths, sanitize_v01


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_sanitize_writes_manifest_and_quarantines_short_rows(tmp_path: Path):
    waxal_meta = tmp_path / "waxal_meta.jsonl"
    waxal_dir = tmp_path / "waxal_round2"
    gnlp = tmp_path / "ghana.jsonl"
    write_jsonl(
        waxal_meta,
        [
            {
                "sample_id": "waxal-1",
                "audio_path": "audio://waxal-1",
                "text": "Ɛnnɛ yɛrekasa.",
                "speaker_id": "s1",
            }
        ],
    )
    write_jsonl(
        waxal_dir / "train.jsonl",
        [{"sample_id": "waxal-1", "text": "Ɛnnɛ yɛrekasa.", "speaker_id": "s1", "dataset_row": 1}],
    )
    write_jsonl(waxal_dir / "dev.jsonl", [])
    write_jsonl(waxal_dir / "test.jsonl", [])
    write_jsonl(
        gnlp,
        [
            {
                "audio_path": "audio://gnlp-1",
                "text": "Me ho yɛ.",
                "duration_seconds": 0.2,
                "split": "train",
                "sample_id": "g1",
                "dataset_row": 1,
            }
        ],
    )
    paths = SanitizePaths(
        waxal_metadata=waxal_meta,
        waxal_round2_dir=waxal_dir,
        ghana_nlp_manifest=gnlp,
        output_manifest=tmp_path / "out.jsonl",
        output_report=tmp_path / "report.json",
    )

    report = sanitize_v01(paths)

    assert report["accepted_rows"] == 1
    assert report["quarantined_rows"] == 1
    assert report["quarantine_reasons"] == {"duration_below_0.4s": 1}
    written = [json.loads(line) for line in paths.output_manifest.read_text().splitlines()]
    assert written[0]["text_punctuated"] == "Ɛnnɛ yɛrekasa."
    assert written[0]["text_normalized"] == "ɛnnɛ yɛrekasa"


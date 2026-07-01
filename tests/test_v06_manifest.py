from pathlib import Path

from akan_speech.asr.v06_manifest import V06ManifestPaths, build_v06_manifest


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(__import__("json").dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_v06_manifest_keeps_waxal_and_clean_gnlp_without_gnlp_test(tmp_path):
    source = tmp_path / "source.jsonl"
    clean = tmp_path / "clean.jsonl"
    output = tmp_path / "manifest.jsonl"
    report = tmp_path / "report.json"
    write_jsonl(
        source,
        [
            {"record_id": "w1", "corpus": "waxal", "split": "train"},
            {"record_id": "w2", "corpus": "waxal", "split": "test"},
            {"record_id": "g1", "corpus": "gnlp", "split": "train"},
            {"record_id": "g2", "corpus": "gnlp", "split": "validation"},
            {"record_id": "g3", "corpus": "gnlp", "split": "test"},
            {"record_id": "g4", "corpus": "gnlp", "split": "train"},
        ],
    )
    write_jsonl(
        clean,
        [
            {"record_id": "g1", "clean_candidate": True, "flags": ["missing_speaker_id"]},
            {"record_id": "g2", "clean_candidate": True, "tags": ["health_domain"]},
            {"record_id": "g3", "clean_candidate": True},
        ],
    )

    result = build_v06_manifest(
        V06ManifestPaths(
            source_manifest=source,
            clean_gnlp_candidates=clean,
            output_manifest=output,
            output_report=report,
        )
    )

    assert result["rows_by_bucket"]["waxal_replay_train"] == 1
    assert result["rows_by_bucket"]["waxal_regression_test"] == 1
    assert result["rows_by_bucket"]["gnlp_clean_adaptation_train"] == 1
    assert result["rows_by_bucket"]["gnlp_clean_adaptation_validation"] == 1
    assert result["excluded_reasons"]["gnlp_test_reserved"] == 1
    assert result["excluded_reasons"]["gnlp_not_clean_candidate"] == 1
    rows = output.read_text(encoding="utf-8")
    assert "g3" not in rows
    assert "serendepify-gsl-asr-ak-waxal-gnlpclean-whisper-small-replay-fullft-v0.6" in rows

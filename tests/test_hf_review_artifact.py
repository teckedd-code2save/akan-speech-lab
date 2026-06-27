import json
from pathlib import Path

from scripts.build_hf_review_artifact import build_package


def test_build_package_requires_pipeline_outputs(tmp_path: Path, monkeypatch):
    import scripts.build_hf_review_artifact as builder

    root = tmp_path
    code = builder.FIRST_ASR_REVIEW.code_name
    files = {
        root / "evals/reports" / f"{code}-sanitize.json": {
            "candidate_rows": 2,
            "accepted_rows": 1,
            "quarantined_rows": 1,
            "accepted_by_corpus": {"waxal": 1},
            "quarantine_reasons": {"duration_below_0.4s": 1},
        },
        root / "data/manifests" / f"{code}.jsonl": "{}\n",
        root / "outputs/pipeline_runs" / code / "latest.json": "{}\n",
        root / "configs/asr" / f"{code}.yaml": "code_name: test\n",
        root / "model_cards" / code / "README.md": "# draft\n",
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, dict):
            path.write_text(json.dumps(content), encoding="utf-8")
        else:
            path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(builder, "ROOT", root)

    result = build_package(root / "pkg")

    assert "README.md" in result["files"]
    assert "data/manifest.jsonl" in result["files"]
    assert "reports/sanitize.json" in result["files"]
    readme = (root / "pkg/README.md").read_text(encoding="utf-8")
    assert "not a trained model checkpoint" in readme


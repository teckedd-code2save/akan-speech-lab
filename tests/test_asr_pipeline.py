from pathlib import Path

from akan_speech.asr.artifacts import FIRST_ASR_REVIEW
from akan_speech.asr.pipeline import evaluate_pipeline, pipeline_next_action, render_pipeline_markdown


def test_pipeline_starts_at_pick_when_no_evidence(tmp_path: Path):
    statuses = evaluate_pipeline(tmp_path)

    assert statuses[0].stage.key == "pick"
    assert statuses[0].status == "active"
    assert statuses[1].status == "blocked"
    assert "Run scripts/build_asr_review_packet.py" in pipeline_next_action(tmp_path)


def test_pipeline_advances_to_prepare_when_pick_evidence_exists(tmp_path: Path):
    code = FIRST_ASR_REVIEW.code_name
    for relative in [
        f"configs/asr/{code}.yaml",
        f"model_cards/{code}/README.md",
        f"outputs/review_packets/{code}/review_packet.json",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")

    statuses = evaluate_pipeline(tmp_path)

    assert statuses[0].status == "complete"
    assert statuses[1].stage.key == "prepare"
    assert statuses[1].status == "active"
    assert "Run Waxal/GhanaNLP audits" in pipeline_next_action(tmp_path)


def test_pipeline_markdown_contains_missing_paths(tmp_path: Path):
    rendered = render_pipeline_markdown(tmp_path)

    assert FIRST_ASR_REVIEW.code_name in rendered
    assert "configs/asr/" in rendered
    assert "Missing evidence" in rendered


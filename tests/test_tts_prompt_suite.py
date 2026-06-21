from scripts.build_tts_prompt_suite import validate_prompt_suite


def test_prompt_suite_rejects_unreviewed_or_wrong_counts():
    blueprint = {
        "version": "test",
        "total_prompts": 2,
        "categories": {"health": 1, "questions": 1},
    }
    rows = [
        {"prompt_id": "1", "category": "health", "text": "Me ho yɛ", "review_status": "approved"},
        {"prompt_id": "2", "category": "questions", "text": "Wo ho te sɛn?", "review_status": "needs_review"},
    ]
    report = validate_prompt_suite(rows, blueprint)
    assert report["passed"] is False
    assert report["unapproved"] == ["2"]

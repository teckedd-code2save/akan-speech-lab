from akan_speech.asr.artifacts import FIRST_ASR_REVIEW, render_review_model_card


def test_first_asr_review_code_name_is_self_describing():
    spec = FIRST_ASR_REVIEW

    assert spec.code_name.startswith("serendepify-gsl-asr-ak-")
    assert "waxal-gnlp" in spec.code_name
    assert "whisper-small" in spec.code_name
    assert "replay-fullft" in spec.code_name
    assert spec.hf_repo.endswith(spec.code_name)


def test_first_asr_review_card_is_not_claiming_metrics():
    card = render_review_model_card()

    assert FIRST_ASR_REVIEW.code_name in card
    assert FIRST_ASR_REVIEW.hf_repo in card
    assert "No metrics yet" in card
    assert "Not trained" in card


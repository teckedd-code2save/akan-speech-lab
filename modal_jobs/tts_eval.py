"""TTS evaluation is intentionally kept in the stable training deployment.

Generated prompt audio and the human-review manifest are persisted beside each
checkpoint so evaluation cannot silently target a different model.
"""

from modal_jobs.tts_train import APP_NAME

__all__ = ["APP_NAME"]

import numpy as np

from akan_speech.tts.quality import audit_audio


def test_clean_audio_passes_basic_quality_gate():
    rate = 16_000
    time = np.arange(rate * 2) / rate
    audit = audit_audio(0.2 * np.sin(2 * np.pi * 220 * time), rate)
    assert audit.flags == ()
    assert audit.duration_seconds == 2.0


def test_silence_is_flagged():
    audit = audit_audio(np.zeros(32_000), 16_000)
    assert "excessive_silence" in audit.flags
    assert "low_loudness" in audit.flags

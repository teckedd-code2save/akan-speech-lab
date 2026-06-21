from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AudioAudit:
    audio_sha256: str
    duration_seconds: float
    sample_rate: int
    peak: float
    rms_dbfs: float
    clipping_ratio: float
    silence_ratio: float
    flags: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["flags"] = list(self.flags)
        return payload


def audit_audio(samples: np.ndarray, sample_rate: int) -> AudioAudit:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    flags: list[str] = []
    if sample_rate <= 0 or values.size == 0 or not np.isfinite(values).all():
        flags.append("invalid_audio")
        values = np.nan_to_num(values)
    duration = values.size / max(sample_rate, 1)
    peak = float(np.max(np.abs(values), initial=0.0))
    rms = float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0
    rms_dbfs = float(20 * np.log10(max(rms, 1e-9)))
    clipping_ratio = float(np.mean(np.abs(values) >= 0.999)) if values.size else 0.0
    silence_ratio = float(np.mean(np.abs(values) < 0.01)) if values.size else 1.0
    if duration < 1.0 or duration > 15.0:
        flags.append("duration_outside_1_15s")
    if clipping_ratio > 0.001:
        flags.append("clipping")
    if silence_ratio > 0.45:
        flags.append("excessive_silence")
    if rms_dbfs < -38.0:
        flags.append("low_loudness")
    digest = hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()
    return AudioAudit(
        audio_sha256=digest,
        duration_seconds=round(duration, 6),
        sample_rate=sample_rate,
        peak=round(peak, 6),
        rms_dbfs=round(rms_dbfs, 3),
        clipping_ratio=round(clipping_ratio, 8),
        silence_ratio=round(silence_ratio, 6),
        flags=tuple(flags),
    )

from __future__ import annotations

import hashlib


def stable_group_split(
    group: str,
    *,
    seed: int = 42,
    train_fraction: float = 0.9,
    validation_fraction: float = 0.05,
) -> str:
    """Assign a stable split while keeping equal group values together."""

    if not group:
        raise ValueError("group must not be empty")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave room for test")

    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < train_fraction:
        return "train"
    if bucket < train_fraction + validation_fraction:
        return "validation"
    return "test"

"""Application settings, read from environment variables with safe defaults.

Same code everywhere; only the environment differs (laptop, CI, Docker).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.core.detector import DEFAULT_MODEL_PATH


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


@dataclass(frozen=True)
class Settings:
    model_path: Path = DEFAULT_MODEL_PATH
    max_image_bytes: int = 10 * 1024 * 1024   # 10 MB
    max_image_pixels: int = 40_000_000         # ~40 MP; phones produce 12-50 MP

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_path=Path(os.getenv("TFR_MODEL_PATH", str(DEFAULT_MODEL_PATH))),
            max_image_bytes=_env_int("TFR_MAX_IMAGE_BYTES", cls.max_image_bytes),
            max_image_pixels=_env_int("TFR_MAX_IMAGE_PIXELS", cls.max_image_pixels),
        )
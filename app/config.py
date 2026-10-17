"""Application settings, read from environment variables with safe defaults.

Same code everywhere; only the environment differs (laptop, CI, Docker).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.core.detector import DEFAULT_MODEL_PATH


@dataclass(frozen=True)
class Settings:
    model_path: Path = DEFAULT_MODEL_PATH

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_path=Path(os.getenv("TFR_MODEL_PATH", str(DEFAULT_MODEL_PATH))),
        )
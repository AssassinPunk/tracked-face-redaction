"""Fixtures shared by every test file. pytest loads this automatically."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.detector import FaceDetector

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def detector() -> FaceDetector:
    # Loaded once for the whole test run; model loading is the slow part.
    return FaceDetector()


@pytest.fixture(scope="session")
def crew_image() -> np.ndarray:
    path = FIXTURES_DIR / "artemis2_crew.jpg"
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None, f"Missing fixture: {path}"
    return image
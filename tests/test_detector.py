"""Unit tests for FaceDetector that need no real face photos."""
import numpy as np
import pytest

from app.core.detector import FaceDetector


@pytest.fixture(scope="module")
def detector() -> FaceDetector:
    # Built once and shared by every test in this file (model loading is slow).
    return FaceDetector()


def test_missing_model_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_model.py"):
        FaceDetector(model_path=tmp_path / "missing.onnx")


def test_blank_image_has_no_faces(detector):
    assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []


def test_rejects_grayscale_image(detector):
    with pytest.raises(ValueError):
        detector.detect(np.zeros((480, 640), dtype=np.uint8))


def test_handles_changing_image_sizes(detector):
    # Exercises setInputSize() and the max_side resize on very different shapes.
    for shape in [(240, 320, 3), (1080, 1920, 3), (3000, 4000, 3), (500, 500, 3)]:
        assert detector.detect(np.zeros(shape, dtype=np.uint8)) == []
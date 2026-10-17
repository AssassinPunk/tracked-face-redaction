"""Fixtures shared by every test file. pytest loads this automatically."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.detector import FaceDetector
from collections import namedtuple

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

SyntheticVideo = namedtuple("SyntheticVideo", "path fps frames")


@pytest.fixture(scope="session")
def crew_video(tmp_path_factory, crew_image) -> SyntheticVideo:
    """A short video of the crew portrait sliding right 2 px per frame.

    Built from the public-domain fixture, so every frame has a known answer
    (4 faces), and the motion will exercise tracking in Milestone 6.
    """
    fps, frames = 10, 20
    path = tmp_path_factory.mktemp("video") / "crew.mp4"
    h, w = crew_image.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), "OpenCV cannot write mp4v video on this machine"
    for i in range(frames):
        shift = np.float32([[1, 0, 2 * i], [0, 1, 0]])  # move right by 2*i px
        writer.write(cv2.warpAffine(crew_image, shift, (w, h)))
    writer.release()
    return SyntheticVideo(path=path, fps=fps, frames=frames)
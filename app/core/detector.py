"""YuNet face detection behind a small, testable interface.

cv2.FaceDetectorYN returns one row of 15 numbers per face:
    [x, y, w, h,  5 x (landmark_x, landmark_y),  score]
This module turns those rows into Face objects, so the rest of the
project never has to remember column numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"

Point = tuple[float, float]


@dataclass(frozen=True)
class Face:
    """One detected face, in pixel coordinates of the ORIGINAL image.

    Note: x or y can be negative when a face is cut off by the image
    edge. Clipping is handled where the face is anonymised (Milestone 3).
    """

    x: int
    y: int
    w: int
    h: int
    score: float
    # right eye, left eye, nose tip, right mouth corner, left mouth corner
    landmarks: tuple[Point, ...]


class FaceDetector:
    """Wraps cv2.FaceDetectorYN and handles input sizing for any image."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        max_side: int = 640,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found at {model_path}. "
                "Run: python scripts/download_model.py"
            )
        if not 0.0 < score_threshold < 1.0:
            raise ValueError("score_threshold must be between 0 and 1.")

        self.max_side = max_side
        self._input_size = (320, 320)  # placeholder, replaced per image
        self._net = cv2.FaceDetectorYN.create(
            str(model_path), "", self._input_size,
            score_threshold, nms_threshold, 5000,
        )

    def detect(self, image: np.ndarray) -> list[Face]:
        """Return every face found in a BGR uint8 image."""
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError("Expected a BGR uint8 image of shape (height, width, 3).")

        h, w = image.shape[:2]

        # Shrink large images: faster, and keeps faces inside YuNet's
        # effective size range (~10-300 px).
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            small = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        else:
            small = image

        # YuNet must know the exact input size before every detection.
        sh, sw = small.shape[:2]
        if (sw, sh) != self._input_size:
            self._net.setInputSize((sw, sh))
            self._input_size = (sw, sh)

        _, rows = self._net.detect(small)
        if rows is None:  # OpenCV returns None, not an empty array
            return []
        return [self._row_to_face(row, scale) for row in rows]

    @staticmethod
    def _row_to_face(row: np.ndarray, scale: float) -> Face:
        coords = row[:14] / scale  # undo the resize; the score is NOT scaled
        x, y, w, h = (int(round(v)) for v in coords[:4])
        landmarks = tuple(
            (float(coords[i]), float(coords[i + 1])) for i in range(4, 14, 2)
        )
        return Face(x=x, y=y, w=w, h=h, score=float(row[14]), landmarks=landmarks)
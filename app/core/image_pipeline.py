"""Image redaction pipeline: bytes in -> detect -> anonymise -> bytes out.

Bytes (not file paths) are deliberate: the API in Milestone 4 receives
uploads as bytes, so scripts and the API share this exact code path.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.anonymize import DEFAULT_MODE, Mode, anonymize
from app.core.detector import Face, FaceDetector


@dataclass(frozen=True)
class RedactionResult:
    image: np.ndarray   # the redacted image
    faces: list[Face]   # what was detected (and therefore redacted)


def decode_image(data: bytes) -> np.ndarray:
    """Turn uploaded bytes into a BGR image, or raise ValueError."""
    if not data:
        raise ValueError("Empty file.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Data is not a supported image (JPEG, PNG, WebP, BMP).")
    return image


def encode_image(image: np.ndarray, ext: str = ".jpg", jpeg_quality: int = 92) -> bytes:
    """Encode an image to bytes. Pixels only: no EXIF/GPS metadata survives."""
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"Could not encode image as {ext}.")
    return buffer.tobytes()


def redact_image(
    image: np.ndarray,
    detector: FaceDetector,
    mode: Mode | str = DEFAULT_MODE,
    strength: float | None = None,
    expand: float = 1.3,
) -> RedactionResult:
    """Detect every face and anonymise it."""
    faces = detector.detect(image)
    redacted = anonymize(image, faces, mode=mode, strength=strength, expand=expand)
    return RedactionResult(image=redacted, faces=faces)

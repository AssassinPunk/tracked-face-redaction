"""Face anonymisation: box expansion, clipping, and three redaction modes.

anonymize() returns a NEW image; the caller's image is never modified.
"""
from __future__ import annotations

from enum import Enum
from typing import Iterable

import cv2
import numpy as np

from app.core.detector import Face

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1); x1 and y1 are exclusive


class Mode(str, Enum):
    BLUR = "blur"
    PIXELATE = "pixelate"
    MASK = "mask"


# strength = kernel size (blur) or block size (pixelate) as a fraction of
# the face's shorter side. Mask ignores strength.
#
# Chosen by measurement, not intuition (see docs/experiments.md, 2026-10-17):
# after redaction we re-run the detector; a face it can still find was not
# hidden well enough.
#   - pixelate 0.25 was the weakest strength leaving 0 faces detectable on
#     both test images; 0.3 adds a safety margin.
#   - blur left faces detectable even at 1.0 (its maximum), so blur is the
#     weaker, context-preserving option and defaults to its strongest setting.
DEFAULT_MODE = Mode.PIXELATE
DEFAULT_STRENGTH = {Mode.BLUR: 1.0, Mode.PIXELATE: 0.3, Mode.MASK: 1.0}

def expand_box(face: Face, factor: float = 1.3) -> Box:
    """Grow the face box around its centre so hair, ears and chin are covered."""
    cx, cy = face.x + face.w / 2, face.y + face.h / 2
    half_w, half_h = face.w * factor / 2, face.h * factor / 2
    return (round(cx - half_w), round(cy - half_h),
            round(cx + half_w), round(cy + half_h))


def clip_box(box: Box, width: int, height: int) -> Box | None:
    """Clamp a box to the image. Returns None if nothing is left inside.

    Essential: NumPy treats negative indices as 'count from the end',
    so an unclipped box at the left edge would silently hit the right edge.
    """
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _odd_at_least_3(n: float) -> int:
    k = max(3, int(n))
    return k if k % 2 == 1 else k + 1


def blur_roi(roi: np.ndarray, strength: float) -> np.ndarray:
    k = _odd_at_least_3(strength * min(roi.shape[:2]))
    return cv2.GaussianBlur(roi, (k, k), sigmaX=k / 6)  # k ≈ 6σ


def pixelate_roi(roi: np.ndarray, strength: float) -> np.ndarray:
    h, w = roi.shape[:2]
    block = max(2, round(strength * min(h, w)))
    grid = (max(1, w // block), max(1, h // block))
    small = cv2.resize(roi, grid, interpolation=cv2.INTER_AREA)       # block means
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)  # solid blocks


def anonymize(
    image: np.ndarray,
    faces: Iterable[Face],
    mode: Mode | str = DEFAULT_MODE,
    strength: float | None = None,
    expand: float = 1.3,
) -> np.ndarray:
    """Return a copy of `image` with every face redacted."""
    mode = Mode(mode)  # accepts "blur"; raises ValueError for unknown modes
    if strength is None:
        strength = DEFAULT_STRENGTH[mode]
    if not 0.0 < strength <= 1.0:
        raise ValueError("strength must be in (0, 1].")
    if expand < 1.0:
        raise ValueError("expand must be >= 1.0 (shrinking would expose the face).")

    out = image.copy()
    height, width = out.shape[:2]
    for face in faces:
        box = clip_box(expand_box(face, expand), width, height)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        roi = out[y0:y1, x0:x1]  # a VIEW: writing to it writes to `out`
        if mode is Mode.BLUR:
            roi[:] = blur_roi(roi, strength)
        elif mode is Mode.PIXELATE:
            roi[:] = pixelate_roi(roi, strength)
        else:
            roi[:] = 0  # solid black mask
    return out
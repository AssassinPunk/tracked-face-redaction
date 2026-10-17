"""Build the committed test fixture from the downloaded public-domain photo.

Usage (from the project root):
    python scripts/make_fixture.py samples/Artemis_2_Crew_Portrait.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "artemis2_crew.jpg"
MAX_SIDE = 640      # small enough to commit, large enough for clear faces
JPEG_QUALITY = 90   # visually lossless, much smaller file


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/make_fixture.py <path-to-original.jpg>")
        return 2

    source = Path(sys.argv[1])
    image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"ERROR: could not read {source} as an image.")
        return 1

    h, w = image.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        new_size = (round(w * scale), round(h * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # Re-encoding also drops the original's EXIF metadata.
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        print("ERROR: JPEG encoding failed.")
        return 1

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(FIXTURE_PATH)

    nh, nw = image.shape[:2]
    print(f"Original : {w}x{h}")
    print(f"Fixture  : {nw}x{nh}, {FIXTURE_PATH.stat().st_size:,} bytes")
    print(f"Saved    : {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""Download the YuNet face-detection model from the official OpenCV Zoo.

Run from the project root:  python scripts/download_model.py
YuNet is MIT-licensed: https://github.com/opencv/opencv_zoo
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

import cv2

MODEL_NAME = "face_detection_yunet_2023mar.onnx"

# Pinned to one opencv_zoo commit, so the file can never change under us.
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/"
    "fd2da740ebc6d0fe489d86cb55133148978cd82e/"
    "models/face_detection_yunet/" + MODEL_NAME
)

# Recorded after the first verified download (trust on first use).
# While None, the script prints the hash instead of enforcing it.
EXPECTED_SHA256: str | None = (
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
)

# A Git LFS pointer file is ~130 bytes; the real model is far larger.
MIN_BYTES = 100_000

# Build paths from this file's location, never from a hardcoded drive.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / MODEL_NAME


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """Download to a temporary file, then rename, so a failed or
    interrupted download never leaves a half-written model behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def opencv_can_load(path: Path) -> bool:
    """The real test: can OpenCV build a detector from this file?"""
    try:
        cv2.FaceDetectorYN.create(str(path), "", (320, 320))
        return True
    except cv2.error as exc:
        print(f"OpenCV failed to load the model: {exc}")
        return False


def main() -> int:
    if MODEL_PATH.exists():
        print(f"Model already present: {MODEL_PATH}")
    else:
        download(MODEL_URL, MODEL_PATH)

    size = MODEL_PATH.stat().st_size
    print(f"Size    : {size:,} bytes")
    if size < MIN_BYTES:
        print("ERROR: file is too small - probably a Git LFS pointer, not the model.")
        MODEL_PATH.unlink()  # delete it so the next run retries cleanly
        return 1

    actual = sha256_of(MODEL_PATH)
    print(f"SHA-256 : {actual}")
    if EXPECTED_SHA256 is None:
        print("NOTE: no expected checksum recorded yet - copy the SHA-256 above.")
    elif actual != EXPECTED_SHA256:
        print("ERROR: checksum mismatch - file is corrupt or not the pinned model.")
        return 1

    if not opencv_can_load(MODEL_PATH):
        return 1

    print("\nMODEL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
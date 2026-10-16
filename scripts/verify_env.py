"""Sanity-check the development environment.

Run from the project root:  python scripts/verify_env.py
Exits with code 0 if everything works, 1 otherwise.
"""
import sys
from importlib.metadata import version

import cv2
import numpy as np


def main() -> int:
    # 1. Report the versions pip actually installed.
    print(f"Python  : {sys.version.split()[0]}")
    for pkg in ("numpy", "opencv-python-headless", "norfair", "fastapi"):
        print(f"{pkg:<23}: {version(pkg)}")

    # 2. The YuNet detector API exists in this OpenCV build.
    #    (The model weights are downloaded separately in Milestone 2.)
    has_yunet = hasattr(cv2, "FaceDetectorYN")
    print(f"\ncv2.FaceDetectorYN available : {has_yunet}")

    # 3. OpenCV and NumPy cooperate: blur a synthetic image and
    #    confirm the pixels actually changed.
    img = np.zeros((100, 100, 3), dtype=np.uint8)  # black image, BGR
    img[40:60, 40:60] = 255                         # white square
    blurred = cv2.GaussianBlur(img, (15, 15), 0)    # 15x15 kernel
    blur_works = not np.array_equal(img, blurred)
    print(f"GaussianBlur changes pixels   : {blur_works}")

    # 4. Norfair can build a tracker and accept a detection.
    from norfair import Detection, Tracker

    tracker = Tracker(distance_function="euclidean", distance_threshold=30)
    tracker.update([Detection(points=np.array([[50.0, 50.0]]))])
    print("Norfair tracker update        : OK")

    # 5. NumPy must be 1.x for Norfair compatibility.
    numpy_ok = np.__version__.startswith("1.")
    print(f"NumPy is 1.x                  : {numpy_ok}")

    ok = has_yunet and blur_works and numpy_ok
    print("\nENVIRONMENT OK" if ok else "\nENVIRONMENT PROBLEM - see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
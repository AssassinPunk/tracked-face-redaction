"""Detect faces in one image and save an annotated copy.

Usage (from the project root):
    python scripts/detect_image.py samples/photo.jpg
    python scripts/detect_image.py samples/photo.jpg --threshold 0.8 --max-side 1280
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # lets this script `import app`

from app.core.detector import Face, FaceDetector  # noqa: E402


def load_image(path: Path) -> np.ndarray:
    # np.fromfile + imdecode works with any Windows path, including
    # non-English folder names, where cv2.imread silently returns None.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {path} as an image.")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"Could not encode image for {path}.")
    buffer.tofile(path)


def annotate(image: np.ndarray, faces: list[Face]) -> np.ndarray:
    out = image.copy()  # never draw on the original
    thickness = max(2, round(max(out.shape[:2]) / 500))  # visible on big photos
    for i, f in enumerate(faces, start=1):
        cv2.rectangle(out, (f.x, f.y), (f.x + f.w, f.y + f.h), (0, 255, 0), thickness)
        cv2.putText(out, f"#{i} {f.score:.2f}", (f.x, max(f.y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, thickness * 0.4, (0, 255, 0), thickness)
        for px, py in f.landmarks:
            cv2.circle(out, (int(px), int(py)), thickness + 1, (0, 0, 255), -1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect faces with YuNet.")
    parser.add_argument("image", type=Path, help="input image path")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--max-side", type=int, default=640)
    args = parser.parse_args()

    image = load_image(args.image)
    detector = FaceDetector(score_threshold=args.threshold, max_side=args.max_side)

    start = time.perf_counter()
    faces = detector.detect(image)
    elapsed_ms = (time.perf_counter() - start) * 1000

    h, w = image.shape[:2]
    print(f"Image     : {args.image.name} ({w}x{h})")
    print(f"Faces     : {len(faces)}   ({elapsed_ms:.1f} ms)")
    for i, f in enumerate(faces, start=1):
        print(f"  #{i}: x={f.x} y={f.y} w={f.w} h={f.h} score={f.score:.3f}")

    out_path = PROJECT_ROOT / "outputs" / f"{args.image.stem}_detected.jpg"
    save_image(out_path, annotate(image, faces))
    print(f"Saved     : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
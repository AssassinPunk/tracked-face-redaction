"""Detect and anonymise faces in one image.

Usage (from the project root):
    python scripts/redact_image.py samples/group.jpg
    python scripts/redact_image.py samples/group.jpg --mode pixelate --strength 0.2
    python scripts/redact_image.py samples/group.jpg --mode all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.anonymize import Mode  # noqa: E402
from app.core.detector import FaceDetector  # noqa: E402
from app.core.image_pipeline import decode_image, encode_image, redact_image  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def labelled(image: np.ndarray, text: str, height: int = 360) -> np.ndarray:
    """Resize to a common height and stamp a label, for side-by-side strips."""
    scale = height / image.shape[0]
    panel = cv2.resize(image, (round(image.shape[1] * scale), height),
                       interpolation=cv2.INTER_AREA)
    for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 2)):  # outlined text
        cv2.putText(panel, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, thick)
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect and anonymise faces.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--mode", choices=[m.value for m in Mode] + ["all"], default="blur")
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    if args.mode == "all" and args.strength is not None:
        parser.error("--strength means different things per mode; use it with one mode.")

    image = decode_image(args.image.read_bytes())
    detector = FaceDetector(score_threshold=args.threshold)
    modes = list(Mode) if args.mode == "all" else [Mode(args.mode)]
    label = "default" if args.strength is None else str(args.strength)
    OUTPUT_DIR.mkdir(exist_ok=True)

    panels = [labelled(image, "original")]
    for mode in modes:
        start = time.perf_counter()
        result = redact_image(image, detector, mode=mode, strength=args.strength)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Proxy for redaction strength: can the detector still find the faces?
        residual = len(detector.detect(result.image))

        out_path = OUTPUT_DIR / f"{args.image.stem}_{mode.value}_{label}.jpg"
        out_path.write_bytes(encode_image(result.image))
        print(f"{mode.value:<9} strength={label:<8} faces={len(result.faces):<3} "
              f"still detectable={residual:<3} ({elapsed_ms:.1f} ms) -> {out_path.name}")
        panels.append(labelled(result.image, mode.value))

    if len(panels) > 2:
        strip_path = OUTPUT_DIR / f"{args.image.stem}_comparison.jpg"
        strip_path.write_bytes(encode_image(np.hstack(panels)))
        print(f"Comparison strip -> {strip_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
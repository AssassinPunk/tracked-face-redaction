"""Redact faces in a video and report baseline speed statistics.

Usage (from the project root):
    python scripts/redact_video.py samples/clip.mp4
    python scripts/redact_video.py samples/clip.mp4 --mode blur --max-frames 150
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.anonymize import Mode  # noqa: E402
from app.core.detector import FaceDetector  # noqa: E402
from app.core.video_pipeline import probe_video, redact_video  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Redact faces in a video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--mode", choices=[m.value for m in Mode], default="pixelate")
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--detect-every", type=int, default=1,
                        help="run the detector every N frames")
    parser.add_argument("--strategy", choices=["none", "hold", "track", "union"], default=None,
                        help="boxes on skipped frames (default: none if N=1, else track)")
    args = parser.parse_args()
    strategy = args.strategy or ("none" if args.detect_every == 1 else "track")

    info = probe_video(args.video)
    print(f"Source     : {args.video.name}  {info.width}x{info.height} @ {info.fps:.2f} fps, "
          f"~{info.frame_count} frames (~{info.frame_count / info.fps:.1f} s)")

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.video.stem}_{args.mode}_n{args.detect_every}_{strategy}.mp4"

    result = redact_video(args.video, out_path, FaceDetector(),
                          mode=args.mode, strength=args.strength,
                          max_frames=args.max_frames,
                          detect_every=args.detect_every, strategy=strategy)

    counts = result.faces_per_frame
    print(f"Processed  : {result.frames_processed} frames in {result.seconds:.1f} s")
    print(f"Detection  : every {args.detect_every} frame(s), strategy={strategy}, "
          f"detector ran on {result.detection_frames} frames")
    print(f"Speed      : {result.processing_fps:.1f} fps  "
          f"(realtime factor {result.realtime_factor:.2f}x)")
    total = sum(result.stage_seconds.values())
    print("Time by stage:")
    for stage, secs in result.stage_seconds.items():
        print(f"  {stage:<10} {secs:6.1f} s  ({100 * secs / total:4.1f}%)")
    print(f"Faces/frame: min {min(counts)}, mean {statistics.mean(counts):.1f}, max {max(counts)}")
    def stamp(i: int) -> str:  # frame index plus its time in the video}
        return f"{i} ({i / info.fps:.2f}s)"

    zero = [i for i, c in enumerate(counts) if c == 0]
    # A one-frame dip: the count drops for a single frame, then recovers.
    # People don't vanish for one frame, so this almost always means a miss.
    dips = [i for i in range(1, len(counts) - 1)
            if counts[i] < counts[i - 1] and counts[i] < counts[i + 1]]
    print(f"Zero-face frames : {', '.join(map(stamp, zero)) or 'none'}")
    shown = ", ".join(map(stamp, dips[:15])) + (" ..." if len(dips) > 15 else "")
    print(f"One-frame dips   : {len(dips)} -> {shown or 'none'}")
    print(f"Saved      : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
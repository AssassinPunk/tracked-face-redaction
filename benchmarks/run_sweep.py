"""Sweep detection interval N and box strategy on one video; write a CSV.

Usage (from the project root):
    python benchmarks/run_sweep.py samples/clip.mp4
    python benchmarks/run_sweep.py samples/clip.mp4 --n 1 5 10 --repeats 3

For every (N, strategy) it redacts the whole video, records the timing per
stage and the per-frame box counts, and saves the output video so the runs
can be compared by eye. Speed only: recall needs hand-annotated frames
(Milestone 7), so nothing here claims that a setting is "safe".
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.detector import FaceDetector  # noqa: E402
from app.core.video_pipeline import STAGES, probe_video, redact_video  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"


def configurations(ns: list[int]) -> list[tuple[int, str]]:
    configs = [(1, "none")]                      # the every-frame baseline
    for n in ns:
        for strategy in ("hold", "track", "union"):
            if n == 1 and strategy != "track":
                continue  # at N=1, hold = none and union = track  
            configs.append((n, strategy))
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark detect-every-N strategies.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--n", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    parser.add_argument("--repeats", type=int, default=1,
                        help="runs per configuration; timings use the median")
    parser.add_argument("--mode", default="pixelate")
    args = parser.parse_args()

    info = probe_video(args.video)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_DIR / f"{args.video.stem}-{stamp}"
    run_dir.mkdir(parents=True)
    detector = FaceDetector()  # loaded once, shared by every run

    print(f"{args.video.name}: {info.width}x{info.height} @ {info.fps:.2f} fps, "
          f"~{info.frame_count} frames -> {run_dir}")
    rows, per_frame = [], {}
    for n, strategy in configurations(args.n):
        label = f"n{n}_{strategy}"
        results = []
        for _ in range(args.repeats):
            results.append(redact_video(args.video, run_dir / f"{label}.mp4", detector,
                                        mode=args.mode, detect_every=n, strategy=strategy))
        r = results[0]
        seconds = statistics.median(x.seconds for x in results)
        counts = r.faces_per_frame  # identical across repeats (deterministic)
        row = {
            "n": n, "strategy": strategy, "repeats": args.repeats,
            "seconds": round(seconds, 2),
            "fps": round(r.frames_processed / seconds, 2),
            "realtime_factor": round(r.frames_processed / seconds / info.fps, 3),
            "detection_frames": r.detection_frames,
            **{f"{s}_s": round(statistics.median(x.stage_seconds[s] for x in results), 2)
               for s in STAGES},
            "mean_boxes": round(statistics.mean(counts), 2),
            "zero_box_frames": counts.count(0),
        }
        rows.append(row)
        per_frame[label] = counts
        print(f"  {label:<9} {row['seconds']:7.1f} s  {row['fps']:5.1f} fps  "
              f"detector x{row['detection_frames']:<4} zero-box frames {row['zero_box_frames']}")

    with (run_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "video": args.video.name, "width": info.width, "height": info.height,
        "fps": info.fps, "frames": len(next(iter(per_frame.values()))),
        "mode": args.mode, "repeats": args.repeats,
        "machine": platform.processor() or platform.machine(),
        "python": platform.python_version(), "boxes_per_frame": per_frame,
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1))

    base = rows[0]["seconds"]
    print("\nSpeedup vs. detect-every-frame:")
    for row in rows:
        print(f"  n{row['n']:<3} {row['strategy']:<6} {base / row['seconds']:5.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
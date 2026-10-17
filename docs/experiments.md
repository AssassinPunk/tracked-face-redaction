# Experiment log

## 2026-10-17: Detection threshold and input size

**Setup:** one crowded group photo (2784×2333 px, ~25 visible faces; stock
photo kept locally, not committed). CPU: AMD Ryzen 7 4800H (laptop). An RTX 3050 Laptop GPU is present but unused;
the pip OpenCV build runs on the CPU only. Single runs, not
averaged, so timings are indicative only.

| Threshold | max_side | Faces | Time (ms) | Lowest score |
|---|---|---|---|---|
| 0.6 | 640  | 24 | 41.4 (median of 3; range 36.5–48.8) | 0.714 |
| 0.9 | 640  | 15 | 47.4      | 0.901 |
| 0.6 | 1280 | 24 | 129.4     | 0.630 |

**Findings**
- Raising the threshold from 0.6 to 0.9 lost 9 of 24 faces (37.5%). All were
  real faces with lower confidence (small or partly hidden), so for a privacy
  tool a lower threshold is safer.
- 640 and 1280 both found 24 faces, but not the same 24: each found one face
  the other missed (~25 distinct faces in total). Face count alone hides this;
  evaluation needs position-matched ground truth.
- 1280 raised most scores (e.g. 0.731 -> 0.881) but took ~3.1x longer
  (129.4 ms vs. a 41.4 ms median at 640).
- The threshold does not affect speed: the 0.9 run (47.4 ms) falls inside the
  0.6 run's range (36.5–48.8 ms). The CNN does the same work either way; the
  threshold only filters its finished output. Raising it costs recall and saves
  no time.

**Decisions**
- Keep `score_threshold = 0.6`.
- Keep `max_side = 640` as the default; re-evaluate in Milestone 7 with
  annotated frames.
- Future work: multi-scale detection (run at 640 and 1280, merge with NMS).
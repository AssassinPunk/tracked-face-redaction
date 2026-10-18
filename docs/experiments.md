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

## 2026-10-17: Redaction strength (re-detection test)

**Method:** redact all detected faces, then run the detector again on the
output. A face that is still detectable was not hidden well enough. This is a
minimum bar, not proof of anonymity: detectability is not identifiability.
Images: group photo (24 detected faces, 2784x2333) and the crew fixture
(4 faces, 640 px). CPU: AMD Ryzen 7 4800H. Single runs.

| Mode | Strength | Still detectable (group / crew) | Time, group (ms) |
|---|---|---|---|
| blur | 0.1 | 24 / - | 43.2 |
| blur | 0.25 | 25 / - | 83.6 |
| blur | 0.5 (old default) | 21 / 4 | 275.1 |
| blur | 0.75 | 4 / - | 664.5 |
| blur | 1.0 | 2 / 1 | 1385.7 |
| pixelate | 0.05 | 25 / - | 39.8 |
| pixelate | 0.08 | 25 / - | 56.7 |
| pixelate | 0.12 (old default) | 17 / 4 | 59.2 |
| pixelate | 0.2 | 2 / 0 | 43.2 |
| pixelate | 0.25 | 0 / 0 | 74.7 |
| pixelate | 0.3 | 0 / 0 | 53.5 |
| pixelate | 0.4 | 0 / 0 | 54.2 |
| mask | - | 0 / 0 | 54.6 |

**Findings**
- The original defaults (blur 0.5, pixelate 0.12) were chosen by intuition
  and failed: 21/24 and 17/24 faces remained detectable.
- Blur never reached 0, even at maximum strength. Gaussian blur is a low-pass
  filter: it removes fine detail but keeps the coarse layout (eye regions,
  face oval) that the detector relies on.
- Blur time grows with strength (43 -> 1386 ms) because work scales with
  kernel size; pixelation stays ~40-75 ms at every strength.
- Pixelation at 0.25+ breaks the spatial layout and left no face detectable.
- Metric flaw: "still detectable" counts every face found afterwards,
  including faces the first pass missed, hence 25 > 24 redacted.
  [your observation from group_blur_0.25_detected.jpg]
  Milestone 7 should count only re-detections overlapping a redacted box.

**Decisions**
- Default mode is now pixelate at 0.3 (0.25 measured + safety margin).
- Blur defaults to 1.0 and is documented as the weaker, slower,
  context-preserving option.
- A regression test now asserts that default redaction leaves no face in the
  crew fixture detectable.

## 2026-10-17: Video baseline (detect every frame)

**Clip:** own phone video (not committed), 3840x2160 @ 25 fps, 505 frames,
20.2 s, up to 6 people. CPU: AMD Ryzen 7 4800H. Mode: pixelate (default).

| Measurement | Value |
|---|---|
| Processing time | 66.6 s for 505 frames |
| Processing speed | 7.6 fps (realtime factor 0.30x) |
| Faces per frame | min 0, mean 4.3, max 6 |
| Frames with 0 faces | 18sec |
| One-frame dips | 3 |

**Findings**
- Detecting on every frame still leaks: faces were visibly unredacted in
  some frames. N = 1 is a baseline, not ground truth; recall must be measured
  against hand-annotated frames (Milestone 7).
- Misses observed (checked by eye, every 10 frames, plus an independent Haar
  scan): almost all leaks are faces cut off by the frame edge, with only the
  mouth/chin visible. Worst case: one person at the top edge from 10.4 to 13.6 s
  (~80 consecutive frames). Also 7.6-8.0 s, 10.4-12.4 s, 11.2-12.0 s,
  4.8-5.6 s and 16.8-18.8 s. Every face fully inside the frame was redacted.
- These are long runs, not one-frame dips, so the dip heuristic undercounts
  them, and tracking cannot recover faces that were never detected.
- 0.30x realtime on 4K. Per-frame time includes 4K decode, resize, detection,
  anonymisation and 4K encode. Tracking only removes detection on skipped
  frames, so its speedup is bounded by detection's share of the time
  (Amdahl's law). Stage timing must be measured before claiming a speedup.
- Transition measured frame by frame: the top-edge face is detected and
  pixelated up to frame 258 (10.32 s) while partly cut off, then lost
  abruptly at frames 259-260 once the eyes leave the frame, and stays exposed
  for ~80 frames. Test case for Milestone 6: does a track carried by the
  Kalman prediction keep covering it after frame 258, and for how long?

**Hypothesis for Milestone 6:** a tracker may carry a face through frames
where detection misses it, reducing leaks as well as cost. To be tested.

## 2026-10-18: Video stage timing (H.264 output)

Same 4K clip (3840x2160 @ 25 fps, 505 frames), Ryzen 7 4800H, pixelate, single run.

| Stage | Seconds | Share |
|---|---|---|
| decode | 14.4 | 32.3% |
| detect | 12.5 | 28.1% |
| anonymise | 12.2 | 27.3% |
| encode | 5.5 | 12.3% |
| total | 45.1, 11.2 fps, 0.45x realtime | |

- Piping frames to ffmpeg (separate, multi-threaded process) was faster than
  OpenCV's mp4v writer: 45.1 s vs 66.6 s (single runs, indicative).
- Amdahl: detection is 28% of the time, so skipping it entirely gives at most
  1.38x on 4K. Detection runs at 640 px regardless of input size while the
  other stages scale with pixels, so at 1080p tracking should matter more:
  benchmark a 1080p clip in Milestone 7.
- Anonymise (27%) likely dominated by the full-frame copy in anonymize();
  candidate optimisation, to be measured, not assumed.

## 2026-10-18: Detect-every-N sweep (speed only)

4K clip (3840x2160 @ 25 fps, 505 frames), Ryzen 7 4800H, pixelate,
median of 3 runs. Data: benchmarks/results/clip-20261018-161936/.

| N | hold (s) | track (s) | detector runs |
|---|---|---|---|
| 1 | 42.3 (none) | 44.5 | 505 |
| 2 | 38.0 | 37.7 | 253 |
| 5 | 33.5 | 35.5 | 101 |
| 10 | 33.3 | 34.4 | 51 |
| 20 | 32.9 | 32.3 | 26 |

- Speedup plateaus at ~1.3x (Amdahl prediction from stage timing: ~1.38x max).
  N=20 is ~1 s faster than N=5 but leaves new faces visible up to 19 frames
  instead of 4, so on 4K, speed cannot justify a large N; recall decides.
- Hold vs track timing differences (<= 2 s) are within run-to-run noise;
  tracker cost is ~0.2 s per video.
- Zero-box frames depend on where detection frames fall (hold: 5 at N=5,
  0 at N=10/20, because frame 295 is a detection frame only at N=5). Not a
  valid leak metric: Milestone 7 measures uncovered faces against annotations.
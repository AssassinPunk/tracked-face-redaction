"""Track face boxes between detection frames with Norfair (Kalman filter + IoU).

Call FaceTracker.update() once per frame:
  - on a detection frame, pass the detector's faces;
  - on a skipped frame, pass None: the tracker PREDICTS where each face moved.
It returns the boxes to redact on that frame.

Privacy-driven settings (see docs/experiments.md):
  - initialization_delay=0: a new face is redacted on the frame it is first
    detected. Norfair's default waits several detections before "confirming"
    an object, which would leave a new face visible for that long.
  - hit_counter_max = detect_every + coast_frames: a track survives the gap
    between detections, plus `coast_frames` more if the detector misses it.
    Norfair builds this budget up while a face keeps being matched, so a face
    tracked for a while coasts for the full budget, and a one-off false
    detection dies quickly.
"""
from __future__ import annotations

import numpy as np
from norfair import Detection, Tracker

from app.core.detector import Face


class FaceTracker:
    def __init__(self, detect_every: int = 1, coast_frames: int = 10,
                 min_iou: float = 0.1) -> None:
        if detect_every < 1:
            raise ValueError("detect_every must be >= 1.")
        if coast_frames < 0:
            raise ValueError("coast_frames must be >= 0.")
        self.detect_every = detect_every
        self._tracker = Tracker(
            distance_function="iou",           # distance = 1 - IoU of the two boxes
            distance_threshold=1.0 - min_iou,  # match only if IoU > min_iou
            hit_counter_max=detect_every + coast_frames,
            initialization_delay=0,            # redact new faces immediately
        )

    def update(self, faces: list[Face] | None) -> list[Face]:
        """Advance one frame. `faces` = detections, or None on a skipped frame."""
        if faces is None:
            tracked = self._tracker.update()  # predict only
        else:
            detections = [
                Detection(points=np.array([[f.x, f.y], [f.x + f.w, f.y + f.h]], dtype=float),
                          scores=np.array([f.score, f.score]))
                for f in faces
            ]
            # period tells Norfair how many frames pass between detections
            tracked = self._tracker.update(detections, period=self.detect_every)

        boxes = [b for b in (self._to_face(obj) for obj in tracked) if b is not None]
        # On detection frames, also redact the raw detections themselves:
        # they are the most accurate boxes we have.
        return (list(faces) + boxes) if faces else boxes

    @staticmethod
    def _to_face(obj) -> Face | None:
        (x0, y0), (x1, y1) = obj.estimate
        if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
            return None  # a degenerate prediction: nothing sensible to redact
        score = float(obj.last_detection.scores[0])
        return Face(x=round(x0), y=round(y0), w=round(x1 - x0), h=round(y1 - y0),
                    score=score, landmarks=())

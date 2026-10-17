"""Video redaction: decode frames -> detect -> anonymise -> encode.

Baseline: detection runs on EVERY frame. Milestone 6 adds detect-every-N
with tracking, and is measured against this baseline.

OpenCV reads and writes video through file paths, not bytes, so this module
takes paths. (The API will use temporary files.)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.core.anonymize import DEFAULT_MODE, Mode, anonymize
from app.core.detector import FaceDetector

FOURCC = "mp4v"  # MPEG-4 Part 2: writable by pip OpenCV builds on every platform


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int  # from the container header: an estimate, may be wrong


@dataclass(frozen=True)
class VideoResult:
    info: VideoInfo
    faces_per_frame: list[int]
    seconds: float

    @property
    def frames_processed(self) -> int:
        return len(self.faces_per_frame)

    @property
    def processing_fps(self) -> float:
        return self.frames_processed / self.seconds if self.seconds > 0 else 0.0

    @property
    def realtime_factor(self) -> float:
        """> 1.0 means faster than playback speed."""
        return self.processing_fps / self.info.fps


def _read_info(cap: cv2.VideoCapture) -> VideoInfo:
    info = VideoInfo(
        fps=cap.get(cv2.CAP_PROP_FPS),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if info.fps <= 0 or info.width <= 0 or info.height <= 0:
        raise ValueError("Could not read the video's frame rate or size.")
    return info


def probe_video(path: Path | str) -> VideoInfo:
    """Read a video's properties without processing it."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open {Path(path).name} as a video.")
        return _read_info(cap)
    finally:
        cap.release()


def redact_video(
    input_path: Path | str,
    output_path: Path | str,
    detector: FaceDetector,
    mode: Mode | str = DEFAULT_MODE,
    strength: float | None = None,
    max_frames: int | None = None,
) -> VideoResult:
    """Detect and anonymise faces in every frame; write a new video."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open {Path(input_path).name} as a video.")

    writer = None
    faces_per_frame: list[int] = []
    try:
        info = _read_info(cap)
        size = (info.width, info.height)  # VideoWriter wants (width, height)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*FOURCC),
                                 info.fps, size)
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video {Path(output_path).name}.")

        start = time.perf_counter()
        while max_frames is None or len(faces_per_frame) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break  # end of video (or an unreadable frame)
            if (frame.shape[1], frame.shape[0]) != size:
                # VideoWriter silently drops frames of the wrong size.
                frame = cv2.resize(frame, size)
            faces = detector.detect(frame)
            writer.write(anonymize(frame, faces, mode=mode, strength=strength))
            faces_per_frame.append(len(faces))
        seconds = time.perf_counter() - start
    finally:
        cap.release()  # always free the file handles, even after an error
        if writer is not None:
            writer.release()

    if not faces_per_frame:
        raise ValueError("The video contained no readable frames.")
    return VideoResult(info=info, faces_per_frame=faces_per_frame, seconds=seconds)
"""Video redaction: decode frames -> detect -> anonymise -> encode (H.264).

Baseline: detection runs on EVERY frame. Milestone 6 adds detect-every-N
with tracking, and is measured against this baseline.

OpenCV reads video through file paths, not bytes, so this module takes
paths. (The API will use temporary files.)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from app.core.anonymize import DEFAULT_MODE, Mode, anonymize
from app.core.detector import FaceDetector
from app.core.video_encoder import H264Writer

STAGES = ("decode", "detect", "anonymise", "encode")


class VideoLimitError(ValueError):
    """The video is longer or larger than the configured limits."""


@dataclass(frozen=True)
class VideoLimits:
    max_seconds: float
    max_pixels: int  # per frame, e.g. 3840 * 2160


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int  # from the container header: an estimate, may be wrong

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True)
class VideoResult:
    info: VideoInfo
    faces_per_frame: list[int]
    seconds: float
    stage_seconds: dict[str, float] = field(default_factory=dict)

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


def check_limits(info: VideoInfo, limits: VideoLimits) -> None:
    """Reject oversized videos BEFORE any expensive work."""
    if info.width * info.height > limits.max_pixels:
        raise VideoLimitError(f"Frame size {info.width}x{info.height} exceeds the limit.")
    if info.frame_count > 0 and info.duration > limits.max_seconds:
        raise VideoLimitError(f"Video is {info.duration:.1f} s; the limit is "
                              f"{limits.max_seconds:g} s.")


def redact_video(
    input_path: Path | str,
    output_path: Path | str,
    detector: FaceDetector,
    mode: Mode | str = DEFAULT_MODE,
    strength: float | None = None,
    max_frames: int | None = None,
    limits: VideoLimits | None = None,
) -> VideoResult:
    """Detect and anonymise faces in every frame; write a browser-playable MP4.

    max_frames: stop early (demos and tests).
    limits:     reject the video (VideoLimitError) if it is too large or too
                long. The header's frame count can lie, so the duration limit
                is also enforced while reading.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open {Path(input_path).name} as a video.")

    faces_per_frame: list[int] = []
    stages = dict.fromkeys(STAGES, 0.0)
    try:
        info = _read_info(cap)
        frame_cap = None
        if limits is not None:
            check_limits(info, limits)
            frame_cap = math.ceil(limits.max_seconds * info.fps)
        size = (info.width, info.height)  # (width, height)

        start = time.perf_counter()
        with H264Writer(output_path, info.fps, size) as writer:
            while max_frames is None or len(faces_per_frame) < max_frames:
                t0 = time.perf_counter()
                ok, frame = cap.read()
                t1 = time.perf_counter()
                if not ok:
                    break  # end of video (or an unreadable frame)
                if frame_cap is not None and len(faces_per_frame) >= frame_cap:
                    raise VideoLimitError(f"Video is longer than the "
                                          f"{limits.max_seconds:g} s limit.")
                if (frame.shape[1], frame.shape[0]) != size:
                    frame = cv2.resize(frame, size)
                faces = detector.detect(frame)
                t2 = time.perf_counter()
                redacted = anonymize(frame, faces, mode=mode, strength=strength)
                t3 = time.perf_counter()
                writer.write(redacted)
                t4 = time.perf_counter()

                stages["decode"] += t1 - t0
                stages["detect"] += t2 - t1
                stages["anonymise"] += t3 - t2
                stages["encode"] += t4 - t3
                faces_per_frame.append(len(faces))
        seconds = time.perf_counter() - start
    finally:
        cap.release()  # always free the file handle, even after an error

    if not faces_per_frame:
        Path(output_path).unlink(missing_ok=True)
        raise ValueError("The video contained no readable frames.")
    return VideoResult(info=info, faces_per_frame=faces_per_frame,
                       seconds=seconds, stage_seconds=stages)
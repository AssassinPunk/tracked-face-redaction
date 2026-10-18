"""Write frames straight into a browser-playable H.264 MP4 using ffmpeg.

Frames are piped to ffmpeg as raw pixels, so there is ONE lossy encode and
no intermediate video file on disk. The ffmpeg binary comes from the
imageio-ffmpeg package, so no system install is needed (laptop, CI, Docker).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np


class H264Writer:
    """Context manager: `with H264Writer(path, fps, (w, h)) as out: out.write(frame)`."""

    def __init__(self, path: Path | str, fps: float, size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.size = size  # (width, height), same order as cv2.VideoWriter
        width, height = size
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y", "-loglevel", "error",
            # input: raw BGR frames from OpenCV on stdin
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
            # H.264 needs even dimensions: pad by 1 px if the source is odd
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",      # the colour format every browser decodes
            "-movflags", "+faststart",  # index at the start: playback before full download
            "-an",                      # no audio track
            str(self.path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size or frame.dtype != np.uint8:
            raise ValueError(f"Frame must be uint8 of size {self.size}.")
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            raise RuntimeError(f"ffmpeg stopped early: {self._stderr()}") from None

    def close(self) -> None:
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()  # end of input: ffmpeg finishes the file
        if self._proc.wait(timeout=600) != 0:
            raise RuntimeError(f"ffmpeg failed: {self._stderr()}")

    def _stderr(self) -> str:
        return self._proc.stderr.read().decode(errors="replace").strip()[-500:]

    def __enter__(self) -> "H264Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:  # error while writing: stop ffmpeg and remove the partial file
            self._proc.kill()
            self._proc.wait()
            self.path.unlink(missing_ok=True)
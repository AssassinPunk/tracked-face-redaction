"""Video pipeline tests on a synthetic clip with a known answer (4 faces/frame)."""

import math

import cv2
import pytest

from app.core.video_pipeline import (Strategy, VideoLimitError, VideoLimits, check_limits,
                                     probe_video, redact_video)


def test_probe_reads_source_properties(crew_video, crew_image):
    info = probe_video(crew_video.path)
    h, w = crew_image.shape[:2]
    assert (info.width, info.height) == (w, h)
    assert info.fps == pytest.approx(crew_video.fps)


def test_output_keeps_fps_size_and_every_frame(crew_video, detector, tmp_path):
    out = tmp_path / "out.mp4"
    result = redact_video(crew_video.path, out, detector)
    assert result.frames_processed == crew_video.frames

    out_info = probe_video(out)
    assert out_info.fps == pytest.approx(crew_video.fps)
    assert (out_info.width, out_info.height) == (result.info.width, result.info.height)
    assert out_info.frame_count == crew_video.frames


def test_baseline_detects_all_four_faces_in_every_frame(crew_video, detector, tmp_path):
    result = redact_video(crew_video.path, tmp_path / "out.mp4", detector)
    assert result.faces_per_frame == [4] * crew_video.frames


def test_no_face_detectable_in_any_output_frame(crew_video, detector, tmp_path):
    out = tmp_path / "out.mp4"
    redact_video(crew_video.path, out, detector)  # default mode and strength
    cap = cv2.VideoCapture(str(out))
    try:
        frames_checked = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert detector.detect(frame) == [], f"face visible in frame {frames_checked}"
            frames_checked += 1
    finally:
        cap.release()
    assert frames_checked == crew_video.frames


def test_max_frames_limits_processing(crew_video, detector, tmp_path):
    result = redact_video(crew_video.path, tmp_path / "out.mp4", detector, max_frames=5)
    assert result.frames_processed == 5


def test_non_video_file_is_rejected(tmp_path, detector):
    fake = tmp_path / "fake.mp4"
    fake.write_text("this is not a video")
    with pytest.raises(ValueError):
        redact_video(fake, tmp_path / "out.mp4", detector)

def test_output_is_browser_playable_h264(crew_video, detector, tmp_path):
    out = tmp_path / "out.mp4"
    redact_video(crew_video.path, out, detector)
    cap = cv2.VideoCapture(str(out))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode().lower()
    cap.release()
    assert fourcc in {"avc1", "h264"}  # how OpenCV reports H.264 varies by platform


def test_too_long_video_is_rejected_without_output(crew_video, detector, tmp_path):
    out = tmp_path / "out.mp4"
    with pytest.raises(VideoLimitError):
        redact_video(crew_video.path, out, detector,
                     limits=VideoLimits(max_seconds=1, max_pixels=10**9))
    assert not out.exists()  # nothing half-written is left behind


def test_too_large_frame_is_rejected(crew_video):
    with pytest.raises(VideoLimitError):
        check_limits(probe_video(crew_video.path),
                     VideoLimits(max_seconds=60, max_pixels=100 * 100))


def test_stage_timings_are_recorded(crew_video, detector, tmp_path):
    result = redact_video(crew_video.path, tmp_path / "out.mp4", detector)
    assert set(result.stage_seconds) == {"decode", "detect", "track", "anonymise", "encode"}
    assert 0 < sum(result.stage_seconds.values()) <= result.seconds

class CountingDetector:
    """Wraps the real detector and counts how often it runs."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def detect(self, frame):
        self.calls += 1
        return self.inner.detect(frame)


def assert_no_face_in_video(path, detector, expected_frames):
    cap = cv2.VideoCapture(str(path))
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert detector.detect(frame) == [], f"face visible in frame {frames}"
            frames += 1
    finally:
        cap.release()
    assert frames == expected_frames


@pytest.mark.parametrize("n, strategy", [(5, "hold"), (5, "track"), (10, "track")])
def test_detector_runs_only_every_n_frames(crew_video, detector, tmp_path, n, strategy):
    counting = CountingDetector(detector)
    result = redact_video(crew_video.path, tmp_path / "out.mp4", counting,
                          detect_every=n, strategy=strategy)
    assert counting.calls == result.detection_frames == math.ceil(crew_video.frames / n)
    assert result.frames_processed == crew_video.frames


@pytest.mark.parametrize("strategy", [Strategy.HOLD, Strategy.TRACK])
def test_no_face_detectable_when_skipping_frames(crew_video, detector, tmp_path, strategy):
    # The crew faces move 2 px per frame, so skipped frames really need boxes.
    out = tmp_path / "out.mp4"
    redact_video(crew_video.path, out, detector, detect_every=5, strategy=strategy)
    assert_no_face_in_video(out, detector, crew_video.frames)


def test_skipping_frames_without_a_strategy_is_rejected(crew_video, detector, tmp_path):
    with pytest.raises(ValueError):
        redact_video(crew_video.path, tmp_path / "out.mp4", detector, detect_every=5)
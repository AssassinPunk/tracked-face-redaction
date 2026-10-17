"""Video pipeline tests on a synthetic clip with a known answer (4 faces/frame)."""
import cv2
import pytest

from app.core.video_pipeline import probe_video, redact_video


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
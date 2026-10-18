"""API tests for POST /api/redact/video, including temp-file cleanup."""
import cv2
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

URL = "/api/redact/video"


@pytest.fixture
def work_dir(tmp_path):
    d = tmp_path / "work"
    d.mkdir()
    return d


def make_client(work_dir, **overrides):
    return TestClient(create_app(Settings(tmp_dir=work_dir, **overrides)))


def upload(client, data, content_type="video/mp4", **params):
    return client.post(URL, files={"file": ("clip.mp4", data, content_type)}, params=params)


def test_redacts_every_frame_and_cleans_up(work_dir, crew_video, detector, tmp_path):
    with make_client(work_dir) as client:
        response = upload(client, crew_video.path.read_bytes())
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["x-frames-processed"] == str(crew_video.frames)
    assert response.headers["cache-control"] == "no-store"
    assert 'filename="redacted.mp4"' in response.headers["content-disposition"]

    # End to end: no face is detectable in any frame of the downloaded video.
    out = tmp_path / "downloaded.mp4"
    out.write_bytes(response.content)
    cap = cv2.VideoCapture(str(out))
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
    assert frames == crew_video.frames

    assert list(work_dir.iterdir()) == []  # upload and output both deleted


def test_unsupported_type_is_415(work_dir):
    with make_client(work_dir) as client:
        assert upload(client, b"hello", "text/plain").status_code == 415


def test_not_a_video_is_400_and_cleans_up(work_dir):
    with make_client(work_dir) as client:
        assert upload(client, b"definitely not a video").status_code == 400
    assert list(work_dir.iterdir()) == []


def test_oversized_upload_is_413_and_cleans_up(work_dir, crew_video):
    with make_client(work_dir, max_video_bytes=1_000) as client:
        assert upload(client, crew_video.path.read_bytes()).status_code == 413
    assert list(work_dir.iterdir()) == []


def test_too_long_video_is_413_and_cleans_up(work_dir, crew_video):
    with make_client(work_dir, max_video_seconds=1) as client:  # clip is 2 s
        assert upload(client, crew_video.path.read_bytes()).status_code == 413
    assert list(work_dir.iterdir()) == []


def test_invalid_strength_is_422(work_dir, crew_video):
    with make_client(work_dir) as client:
        assert upload(client, crew_video.path.read_bytes(), strength=2).status_code == 422


def test_missing_model_is_503(work_dir, tmp_path, crew_video):
    with make_client(work_dir, model_path=tmp_path / "missing.onnx") as client:
        assert upload(client, crew_video.path.read_bytes()).status_code == 503
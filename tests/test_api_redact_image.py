"""API tests for POST /api/redact/image."""
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.core.anonymize import Mode
from app.core.image_pipeline import decode_image, encode_image
from app.main import create_app

URL = "/api/redact/image"


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture(scope="module")
def crew_jpeg(crew_image):
    return encode_image(crew_image, ".jpg")


def upload(client, data, content_type="image/jpeg", filename="photo.jpg", **params):
    return client.post(URL, files={"file": (filename, data, content_type)}, params=params)


# ---- success --------------------------------------------------------------

def test_returns_redacted_image_with_no_detectable_faces(client, crew_jpeg, crew_image, detector):
    response = upload(client, crew_jpeg)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-faces-detected"] == "4"
    output = decode_image(response.content)
    assert output.shape == crew_image.shape
    assert detector.detect(output) == []  # end-to-end privacy check


@pytest.mark.parametrize("mode", [m.value for m in Mode])
def test_every_mode_is_accepted(client, crew_jpeg, mode):
    assert upload(client, crew_jpeg, mode=mode).status_code == 200


def test_png_in_png_out(client, crew_image):
    response = upload(client, encode_image(crew_image, ".png"), "image/png", "photo.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_privacy_headers_and_safe_filename(client, crew_jpeg):
    response = upload(client, crew_jpeg, filename="../../etc/passwd.jpg")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-disposition"] == 'attachment; filename="redacted.jpg"'


# ---- rejections -----------------------------------------------------------

def test_unsupported_type_is_415(client):
    assert upload(client, b"hello", "text/plain", "notes.txt").status_code == 415


def test_lying_content_type_is_400(client):
    # Claims to be a JPEG, but the bytes are text.
    assert upload(client, b"definitely not an image", "image/jpeg").status_code == 400


def test_oversized_file_is_413(crew_jpeg):
    with TestClient(create_app(Settings(max_image_bytes=1_000))) as small_client:
        assert upload(small_client, crew_jpeg).status_code == 413


def test_too_many_pixels_is_413(crew_jpeg):
    with TestClient(create_app(Settings(max_image_pixels=10_000))) as small_client:
        assert upload(small_client, crew_jpeg).status_code == 413


def test_invalid_strength_is_422(client, crew_jpeg):
    assert upload(client, crew_jpeg, strength=1.5).status_code == 422


def test_unknown_mode_is_422(client, crew_jpeg):
    assert upload(client, crew_jpeg, mode="blurr").status_code == 422


def test_missing_model_is_503(tmp_path, crew_jpeg):
    settings = Settings(model_path=tmp_path / "missing.onnx")
    with TestClient(create_app(settings)) as no_model_client:
        assert upload(no_model_client, crew_jpeg).status_code == 503
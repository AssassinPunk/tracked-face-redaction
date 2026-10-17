"""Pipeline tests on the public-domain crew fixture (4 faces)."""
import numpy as np
import pytest

from app.core.anonymize import Mode
from app.core.image_pipeline import decode_image, encode_image, redact_image


def test_encode_decode_round_trip_keeps_shape(crew_image):
    assert decode_image(encode_image(crew_image)).shape == crew_image.shape


@pytest.mark.parametrize("bad", [b"", b"not an image", b"\x00" * 100])
def test_decode_rejects_non_images(bad):
    with pytest.raises(ValueError):
        decode_image(bad)


def test_redaction_reports_all_detected_faces(detector, crew_image):
    assert len(redact_image(crew_image, detector).faces) == 4


@pytest.mark.parametrize("mode", list(Mode))
def test_every_detected_face_is_changed(detector, crew_image, mode):
    result = redact_image(crew_image, detector, mode=mode)
    for f in result.faces:
        before = crew_image[f.y:f.y + f.h, f.x:f.x + f.w]
        after = result.image[f.y:f.y + f.h, f.x:f.x + f.w]
        assert not np.array_equal(before, after)


def test_masked_faces_are_no_longer_detectable(detector, crew_image):
    result = redact_image(crew_image, detector, mode=Mode.MASK)
    assert detector.detect(result.image) == []


def test_encoded_output_carries_no_exif_metadata(crew_image):
    assert b"Exif" not in encode_image(crew_image)
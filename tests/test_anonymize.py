"""Unit tests for anonymisation, using synthetic images (no faces needed)."""
import numpy as np
import pytest

from app.core.anonymize import Mode, anonymize, clip_box, expand_box
from app.core.detector import Face


def make_face(x, y, w, h):
    return Face(x=x, y=y, w=w, h=h, score=0.9, landmarks=())


@pytest.fixture
def noise():
    # Random pixels: high detail, so any blur or pixelation is measurable.
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)


def outside(image, box):
    """All pixels NOT inside the box, as a flat array."""
    x0, y0, x1, y1 = box
    keep = np.ones(image.shape[:2], dtype=bool)
    keep[y0:y1, x0:x1] = False
    return image[keep]


# ---- geometry -----------------------------------------------------------

def test_expand_box_grows_around_centre():
    assert expand_box(make_face(100, 100, 100, 100), 1.3) == (85, 85, 215, 215)


def test_clip_box_clamps_to_image():
    assert clip_box((-20, -10, 50, 60), width=40, height=30) == (0, 0, 40, 30)


def test_clip_box_returns_none_when_fully_outside():
    assert clip_box((500, 500, 600, 600), width=400, height=300) is None


# ---- the three modes ----------------------------------------------------

@pytest.mark.parametrize("mode", list(Mode))
def test_pixels_outside_the_face_are_untouched(noise, mode):
    out = anonymize(noise, [make_face(100, 100, 100, 100)], mode=mode)
    box = (85, 85, 215, 215)
    assert np.array_equal(outside(out, box), outside(noise, box))


@pytest.mark.parametrize("mode", list(Mode))
def test_input_image_is_not_modified(noise, mode):
    before = noise.copy()
    anonymize(noise, [make_face(100, 100, 100, 100)], mode=mode)
    assert np.array_equal(noise, before)


def test_blur_removes_detail(noise):
    out = anonymize(noise, [make_face(100, 100, 100, 100)], mode="blur")
    before, after = noise[85:215, 85:215], out[85:215, 85:215]
    # Blur averages neighbours, so pixel-to-pixel variation collapses.
    assert after.std() < before.std() / 3


def test_pixelate_leaves_only_a_few_solid_blocks(noise):
    out = anonymize(noise, [make_face(100, 100, 100, 100)], mode="pixelate")
    roi = out[85:215, 85:215].reshape(-1, 3)
    # 130 px ROI, block = round(0.12 * 130) = 16 -> an 8 x 8 grid = 64 colours max.
    assert len(np.unique(roi, axis=0)) <= 64


def test_mask_makes_face_black(noise):
    out = anonymize(noise, [make_face(100, 100, 100, 100)], mode="mask")
    assert not out[85:215, 85:215].any()


# ---- edge cases ---------------------------------------------------------

def test_face_cut_off_at_left_edge_does_not_wrap_around(noise):
    # Expanded box is (-65, 35, 65, 165); without clipping, NumPy's negative
    # indexing would hit the RIGHT side of the image.
    out = anonymize(noise, [make_face(-50, 50, 100, 100)], mode="mask")
    assert not out[35:165, 0:65].any()                      # left part masked
    assert np.array_equal(out[:, 300:], noise[:, 300:])     # right side untouched


def test_face_completely_outside_image_is_ignored(noise):
    out = anonymize(noise, [make_face(1000, 1000, 50, 50)], mode="mask")
    assert np.array_equal(out, noise)


def test_no_faces_returns_identical_copy(noise):
    out = anonymize(noise, [], mode="blur")
    assert np.array_equal(out, noise) and out is not noise


@pytest.mark.parametrize("bad", [0, -0.5, 1.5])
def test_invalid_strength_rejected(noise, bad):
    with pytest.raises(ValueError):
        anonymize(noise, [], strength=bad)


def test_unknown_mode_rejected(noise):
    with pytest.raises(ValueError):
        anonymize(noise, [], mode="blurr")


def test_shrinking_box_rejected(noise):
    with pytest.raises(ValueError):
        anonymize(noise, [], expand=0.8)
"""Detector tests on a real photo with a known answer: 4 faces."""
from concurrent.futures import ThreadPoolExecutor

import cv2

EXPECTED_FACES = 4
SCALE = 4


def test_finds_all_four_crew_faces(detector, crew_image):
    assert len(detector.detect(crew_image)) == EXPECTED_FACES


def test_scores_respect_threshold(detector, crew_image):
    assert all(f.score >= 0.6 for f in detector.detect(crew_image))


def test_boxes_lie_inside_image(detector, crew_image):
    h, w = crew_image.shape[:2]
    for f in detector.detect(crew_image):
        assert 0 <= f.x and f.x + f.w <= w
        assert 0 <= f.y and f.y + f.h <= h


def test_landmarks_lie_inside_their_face(detector, crew_image):
    # Eyes, nose and mouth must sit inside the box of the face they belong to.
    for f in detector.detect(crew_image):
        for px, py in f.landmarks:
            assert f.x <= px <= f.x + f.w
            assert f.y <= py <= f.y + f.h


def test_same_faces_found_on_enlarged_image(detector, crew_image):
    # A 4x larger copy exercises the max_side resize path. Face centres,
    # scaled back down, should land within a few pixels of the originals.
    big = cv2.resize(crew_image, None, fx=SCALE, fy=SCALE,
                     interpolation=cv2.INTER_CUBIC)
    small_faces = detector.detect(crew_image)
    big_faces = detector.detect(big)
    assert len(big_faces) == EXPECTED_FACES

    small_centres = sorted((f.x + f.w / 2, f.y + f.h / 2) for f in small_faces)
    big_centres = sorted(((f.x + f.w / 2) / SCALE, (f.y + f.h / 2) / SCALE)
                         for f in big_faces)
    for (ax, ay), (bx, by) in zip(small_centres, big_centres):
        assert abs(ax - bx) < 10 and abs(ay - by) < 10


def test_detect_is_correct_under_concurrent_calls(detector, crew_image):
    # Two different input sizes force setInputSize() on alternating calls,
    # which is exactly what concurrent API requests do. Without the lock in
    # FaceDetector.detect(), threads could detect with the wrong input size.
    padded = cv2.copyMakeBorder(crew_image, 0, 0, 0, 200, cv2.BORDER_CONSTANT)
    images = [crew_image, padded] * 20
    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(lambda img: len(detector.detect(img)), images))
    assert counts == [EXPECTED_FACES] * len(images)
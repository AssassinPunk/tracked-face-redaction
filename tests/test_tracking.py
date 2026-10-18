"""FaceTracker tests on synthetic detections (no model needed)."""
import pytest

from app.core.detector import Face
from app.core.tracking import FaceTracker


def face(x, y, w=80, h=100):
    return Face(x=x, y=y, w=w, h=h, score=0.9, landmarks=())


def centre(f):
    return f.x + f.w / 2, f.y + f.h / 2


def covers(boxes, target, tol=15):
    """True if some box's centre is within `tol` px of the target's centre."""
    tx, ty = centre(target)
    return any(abs(cx - tx) <= tol and abs(cy - ty) <= tol
               for cx, cy in map(centre, boxes))


def run(tracker, truth, detect_every):
    """Feed one frame at a time; detect only every `detect_every` frames."""
    outputs = []
    for i, faces in enumerate(truth):
        outputs.append(tracker.update(faces if i % detect_every == 0 else None))
    return outputs


def test_new_face_is_redacted_on_its_first_detection():
    assert covers(FaceTracker(detect_every=5).update([face(100, 100)]), face(100, 100))


@pytest.mark.parametrize("n", [2, 5, 10])
def test_moving_face_stays_covered_between_detections(n):
    # A face moving right 4 px per frame; the detector runs every n frames.
    truth = [[face(100 + 4 * i, 200)] for i in range(60)]
    outputs = run(FaceTracker(detect_every=n), truth, n)
    # After the Kalman filter has seen two detections it knows the velocity.
    for i in range(2 * n, 60):
        assert covers(outputs[i], truth[i][0]), f"frame {i} not covered (n={n})"


def test_two_faces_are_tracked_independently():
    truth = [[face(100 + 3 * i, 100), face(500 - 3 * i, 300)] for i in range(40)]
    outputs = run(FaceTracker(detect_every=5), truth, 5)
    for i in range(10, 40):
        assert all(covers(outputs[i], f) for f in truth[i])


def test_track_coasts_after_detector_misses_then_expires():
    tracker = FaceTracker(detect_every=1, coast_frames=10)
    # Norfair grows a track's remaining life by 1 per matched frame, up to
    # detect_every + coast_frames, so an established face coasts the longest.
    for _ in range(30):
        tracker.update([face(100, 100)])
    # The detector now misses the (still present) face on every frame.
    alive = [bool(tracker.update([])) for _ in range(20)]
    assert all(alive[:10])        # covered for coast_frames after the last hit
    assert not any(alive[12:])    # then the track is dropped


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        FaceTracker(detect_every=0)
    with pytest.raises(ValueError):
        FaceTracker(coast_frames=-1)
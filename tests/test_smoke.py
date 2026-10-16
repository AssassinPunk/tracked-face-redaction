"""Smoke tests: the cheapest possible proof that the project is wired up."""
import cv2
import numpy as np

import app


def test_app_package_imports():
    assert app.__version__ == "0.1.0"


def test_opencv_has_yunet_api():
    # The YuNet detector class must exist in this OpenCV build.
    assert hasattr(cv2, "FaceDetectorYN")


def test_numpy_is_1x():
    # Norfair 2.3.0 is incompatible with NumPy 2.
    assert np.__version__.startswith("1.")
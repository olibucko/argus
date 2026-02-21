import numpy as np
import pytest

from security_dashboard.motion_detector import MotionDetector, MotionDetectorConfig


def _static_frame(height=100, width=100):
    """A consistent grey frame with no motion."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


def _shifted_frame(base, shift_x=20, shift_y=20):
    """Shift a frame to simulate camera-wide motion."""
    shifted = np.zeros_like(base)
    h, w = base.shape[:2]
    shifted[shift_y:, shift_x:] = base[: h - shift_y, : w - shift_x]
    return shifted


class TestMotionDetector:

    def test_first_frame_returns_false(self):
        detector = MotionDetector(MotionDetectorConfig(), viewport_id=(0, 0))
        frame = _static_frame()
        # First frame always returns False (no previous frame to compare)
        assert detector.check(frame) is False

    def test_static_scene_no_motion(self):
        detector = MotionDetector(
            MotionDetectorConfig(sensitivity=0.5, motion_aggressiveness=15.0),
            viewport_id=(0, 0),
        )
        frame = _static_frame()
        detector.check(frame)  # init
        # Feed the same frame again — no motion expected
        assert detector.check(frame) is False

    def test_large_shift_triggers_motion(self):
        config = MotionDetectorConfig(sensitivity=0.9, motion_aggressiveness=1.0)
        detector = MotionDetector(config, viewport_id=(0, 0))

        # Use a textured frame so feature detection finds trackable points
        np.random.seed(42)
        base = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        detector.check(base)  # init

        shifted = _shifted_frame(base, shift_x=30, shift_y=30)
        assert detector.check(shifted) is True

    def test_none_frame_returns_false(self):
        detector = MotionDetector(MotionDetectorConfig(), viewport_id=(0, 0))
        assert detector.check(None) is False

    def test_empty_frame_returns_false(self):
        detector = MotionDetector(MotionDetectorConfig(), viewport_id=(0, 0))
        assert detector.check(np.array([])) is False

    def test_reset_clears_state(self):
        detector = MotionDetector(MotionDetectorConfig(), viewport_id=(0, 0))
        detector.check(_static_frame())
        assert detector.prev_gray is not None
        detector._reset_state()
        assert detector.prev_gray is None
        assert detector.prev_points is None

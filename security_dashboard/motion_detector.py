import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

# --- CONSTANTS ---
OPTICAL_FLOW_MAX_CORNERS = 1000
OPTICAL_FLOW_MIN_DISTANCE = 10
OPTICAL_FLOW_BLOCK_SIZE = 7
OPTICAL_FLOW_WIN_SIZE = (15, 15)
OPTICAL_FLOW_MAX_LEVEL = 2
OPTICAL_FLOW_CRITERIA_EPS = 0.03
OPTICAL_FLOW_CRITERIA_COUNT = 10
MIN_MOTION_POINTS_BASE = 5
YOLO_INFERENCE_SIZE = 640


@dataclass
class MotionDetectorConfig:
    """Configuration for the motion detector."""
    sensitivity: float = 0.5
    motion_aggressiveness: float = 15.0

class MotionDetector:
    """
    Handles optical flow motion detection for a single video stream.
    This class isolates the logic previously found in `_check_motion`.
    """
    def __init__(self, config: MotionDetectorConfig, viewport_id: tuple):
        self.config = config
        self.viewport_id = viewport_id
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None

    def check(self, frame: np.ndarray) -> bool:
        """
        Analyzes a frame for motion and returns True if significant motion is detected.
        
        Args:
            frame: The input frame for motion analysis.

        Returns:
            A boolean indicating if motion was detected.
        """
        try:
            if frame is None or frame.size == 0:
                return False

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.prev_gray is None or self.prev_points is None or len(self.prev_points) == 0:
                self.prev_gray = gray
                self.prev_points = self._find_features(gray)
                return False

            curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None,
                winSize=OPTICAL_FLOW_WIN_SIZE,
                maxLevel=OPTICAL_FLOW_MAX_LEVEL,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, OPTICAL_FLOW_CRITERIA_COUNT, OPTICAL_FLOW_CRITERIA_EPS)
            )
            self.prev_gray = gray

            motion_detected = False

            # Validate all arrays exist and have matching lengths
            if (curr_points is not None and status is not None and self.prev_points is not None and
                len(curr_points) > 0 and len(status) > 0 and len(self.prev_points) > 0):

                try:
                    # Flatten status to ensure it's 1D
                    status_flat = status.flatten()

                    # Find minimum safe length across all arrays
                    min_len = min(len(curr_points), len(status_flat), len(self.prev_points))

                    if min_len > 0:
                        # Trim all arrays to same length first
                        curr_trimmed = curr_points[:min_len]
                        prev_trimmed = self.prev_points[:min_len]
                        status_trimmed = status_flat[:min_len]

                        # Create boolean mask for good points
                        good_mask = (status_trimmed == 1)

                        # Apply mask to get good points
                        good_curr = curr_trimmed[good_mask]
                        good_prev = prev_trimmed[good_mask]

                        if len(good_curr) > 0 and len(good_prev) > 0:
                            velocities = np.sqrt(np.sum((good_curr - good_prev) ** 2, axis=1))
                            velocity_threshold = max(0.1, 2.0 * (1.0 - self.config.sensitivity))
                            min_motion_points = int(MIN_MOTION_POINTS_BASE + (self.config.motion_aggressiveness * self.config.sensitivity**2))

                            if np.sum(velocities > velocity_threshold) >= min_motion_points:
                                motion_detected = True

                            # Refresh feature points if sparse
                            if len(good_curr) < (OPTICAL_FLOW_MAX_CORNERS / 2):
                                self.prev_points = self._find_features(gray)
                            else:
                                self.prev_points = good_curr.reshape(-1, 1, 2)
                        else:
                            # No good points, find new features
                            self.prev_points = self._find_features(gray)
                    else:
                        self.prev_points = self._find_features(gray)

                except Exception as e:
                    # On any array operation error, reset and find new features
                    print(f"Motion detection array error in viewport {self.viewport_id}: {e}")
                    self.prev_points = self._find_features(gray)
            else:
                self.prev_points = self._find_features(gray)
                    
            return motion_detected

        except Exception as e:
            print(f"Motion detection error in viewport {self.viewport_id}: {e}")
            self._reset_state()
            return False

    def _find_features(self, gray_frame: np.ndarray) -> Optional[np.ndarray]:
        """Finds good features to track in a grayscale frame."""
        return cv2.goodFeaturesToTrack(
            gray_frame, 
            OPTICAL_FLOW_MAX_CORNERS, 
            max(0.01, 0.3 * (1.0 - self.config.sensitivity)), 
            OPTICAL_FLOW_MIN_DISTANCE, 
            blockSize=OPTICAL_FLOW_BLOCK_SIZE
        )

    def _reset_state(self):
        """Resets the internal state of the detector."""
        self.prev_gray = None
        self.prev_points = None

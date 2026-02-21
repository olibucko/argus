"""
Motion Detector - Optical flow based motion detection for security cameras.

This module provides real-time motion detection using the Lucas-Kanade optical flow
algorithm. It is designed to be lightweight and responsive, allowing the system
to selectively trigger more computationally expensive processes like object detection.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# --- CONSTANTS ---
OPTICAL_FLOW_MAX_CORNERS = 1000
OPTICAL_FLOW_MIN_DISTANCE = 10
OPTICAL_FLOW_BLOCK_SIZE = 7
OPTICAL_FLOW_WIN_SIZE = (15, 15)
OPTICAL_FLOW_MAX_LEVEL = 2
OPTICAL_FLOW_CRITERIA_EPS = 0.03
OPTICAL_FLOW_CRITERIA_COUNT = 10
MIN_MOTION_POINTS_BASE = 5


@dataclass
class MotionDetectorConfig:
    """
    Configuration parameters for tuning motion detection sensitivity.

    Attributes:
        sensitivity (float): Threshold for pixel movement (0.0 to 1.0).
        motion_aggressiveness (float): Threshold for the number of moving points.
    """
    sensitivity: float = 0.5
    motion_aggressiveness: float = 15.0


class MotionDetector:
    """
    Analyzes successive video frames to detect significant motion.

    The detector uses sparse optical flow (Lucas-Kanade) to track feature points
    between frames. It determines motion based on the velocity and quantity
    of these tracked points.

    Attributes:
        config (MotionDetectorConfig): Sensitivity and aggressiveness settings.
        viewport_id (tuple): Identifier for the viewport being analyzed.
        prev_gray (Optional[np.ndarray]): Grayscale version of the previous frame.
        prev_points (Optional[np.ndarray]): Tracked feature points from the previous frame.
    """

    def __init__(self, config: MotionDetectorConfig, viewport_id: tuple) -> None:
        """
        Initializes the MotionDetector.

        Args:
            config (MotionDetectorConfig): Tuning parameters.
            viewport_id (tuple): Viewport identifier.
        """
        self.config: MotionDetectorConfig = config
        self.viewport_id: tuple = viewport_id
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None

    def check(self, frame: np.ndarray) -> bool:
        """
        Analyzes the current frame for motion relative to the previous frame.

        Args:
            frame (np.ndarray): The current BGR video frame.

        Returns:
            bool: True if significant motion is detected, False otherwise.
        """
        try:
            if frame is None or frame.size == 0:
                return False

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Initialize state on the first frame or after a reset
            if self.prev_gray is None or self.prev_points is None or len(self.prev_points) == 0:
                self.prev_gray = gray
                self.prev_points = self._find_features(gray)
                return False

            # Calculate sparse optical flow
            curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None,
                winSize=OPTICAL_FLOW_WIN_SIZE,
                maxLevel=OPTICAL_FLOW_MAX_LEVEL,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                          OPTICAL_FLOW_CRITERIA_COUNT, OPTICAL_FLOW_CRITERIA_EPS)
            )
            self.prev_gray = gray

            motion_detected = False

            # Analyze flow results
            if curr_points is not None and status is not None:
                status_flat = status.flatten()
                min_len = min(len(curr_points), len(status_flat), len(self.prev_points))

                if min_len > 0:
                    good_mask = (status_flat[:min_len] == 1)
                    good_curr = curr_points[:min_len][good_mask]
                    good_prev = self.prev_points[:min_len][good_mask]

                    if len(good_curr) > 0:
                        # Calculate Euclidean distance between point pairs
                        velocities = np.sqrt(np.sum((good_curr - good_prev) ** 2, axis=1))
                        
                        # Apply sensitivity-based thresholding
                        vel_threshold = max(0.1, 2.0 * (1.0 - self.config.sensitivity))
                        points_threshold = int(MIN_MOTION_POINTS_BASE + 
                                              (self.config.motion_aggressiveness * self.config.sensitivity**2))

                        if np.sum(velocities > vel_threshold) >= points_threshold:
                            motion_detected = True

                        # Update points for the next iteration
                        # Re-detect features if the point cloud becomes too sparse
                        if len(good_curr) < (OPTICAL_FLOW_MAX_CORNERS / 2):
                            self.prev_points = self._find_features(gray)
                        else:
                            self.prev_points = good_curr.reshape(-1, 1, 2)
                    else:
                        self.prev_points = self._find_features(gray)
                else:
                    self.prev_points = self._find_features(gray)
            else:
                self.prev_points = self._find_features(gray)
                    
            return motion_detected

        except Exception as e:
            logger.error(f"Motion detection error in viewport {self.viewport_id}: {e}", exc_info=True)
            self._reset_state()
            return False

    def _find_features(self, gray_frame: np.ndarray) -> Optional[np.ndarray]:
        """Detects high-quality feature points to track in the next frame."""
        quality_level = max(0.01, 0.3 * (1.0 - self.config.sensitivity))
        return cv2.goodFeaturesToTrack(
            gray_frame, 
            OPTICAL_FLOW_MAX_CORNERS, 
            quality_level, 
            OPTICAL_FLOW_MIN_DISTANCE, 
            blockSize=OPTICAL_FLOW_BLOCK_SIZE
        )

    def _reset_state(self) -> None:
        """Resets the internal tracking state."""
        self.prev_gray = None
        self.prev_points = None

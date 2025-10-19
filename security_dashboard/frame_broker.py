"""
Frame Broker - Centralized frame distribution with memory-awareness.

"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Callable, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class FrameMetrics:
    """Metrics for frame processing performance."""
    frames_received: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    memory_usage_mb: float = 0.0
    last_update: float = field(default_factory=time.time)


@dataclass
class CameraState:
    """State tracking for a single camera."""
    name: str
    base_fps: float = 30.0
    current_fps: float = 30.0
    last_frame_time: float = 0.0
    frame_interval: float = 1.0/30.0  # Calculated from current_fps
    priority_level: int = 0  # 0=normal, 1=motion detected, 2=active alert
    consecutive_drops: int = 0
    total_frames: int = 0
    dropped_frames: int = 0


class FrameBroker:
    """
    Centralized frame distribution.
    """

    def __init__(self, max_memory_mb: int = 512, memory_manager=None):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.memory_manager = memory_manager

        # Camera state tracking
        self.cameras: Dict[str, CameraState] = {}
        self.camera_lock = threading.RLock()

        # Frame distribution
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.priority_cameras: Set[str] = set()

        # Performance tracking
        self.frame_metrics: Dict[str, FrameMetrics] = defaultdict(FrameMetrics)
        self.global_metrics = FrameMetrics()

        logger.info(f"FrameBroker initialized with {max_memory_mb}MB memory limit")

    def register_camera(self, camera_name: str, base_fps: float = 30.0) -> None:
        """Register a camera with the broker."""
        with self.camera_lock:
            self.cameras[camera_name] = CameraState(
                name=camera_name,
                base_fps=base_fps,
                current_fps=base_fps,
                frame_interval=1.0/base_fps
            )
        logger.info(f"Registered camera '{camera_name}' with {base_fps} FPS")

    def subscribe_to_camera(self, camera_name: str, callback: Callable[[str, np.ndarray, float], None]) -> None:
        """Subscribe to frames from a specific camera."""
        self.subscribers[camera_name].append(callback)
        logger.debug(f"Added subscriber for camera '{camera_name}'")

    def set_camera_priority(self, camera_name: str, priority: int) -> None:
        """Set priority level for a camera (0=normal, 1=motion, 2=alert)."""
        with self.camera_lock:
            if camera_name in self.cameras:
                self.cameras[camera_name].priority_level = priority
                if priority > 0:
                    self.priority_cameras.add(camera_name)
                else:
                    self.priority_cameras.discard(camera_name)

    def submit_frame(self, camera_name: str, frame: np.ndarray, timestamp: float) -> bool:
        """
        Submit a frame for distribution.

        Returns:
            bool: True if frame was accepted, False if dropped
        """
        # Check if this camera is registered
        with self.camera_lock:
            if camera_name not in self.cameras:
                logger.error(f"Frame submitted for unregistered camera: {camera_name}. Registered cameras: {list(self.cameras.keys())}")
                return False

            camera_state = self.cameras[camera_name]
            camera_state.total_frames += 1

        # Check frame rate limiting
        if not self._should_process_frame(camera_name, timestamp):
            # Log at info level to track drop reasons
            logger.info(f"Frame from {camera_name} dropped due to rate limiting")
            self._record_dropped_frame(camera_name, "rate_limited")
            return False

        # Check global memory pressure via memory manager
        # Note: Individual buffers have their own limits and eviction policies,
        # so we only drop frames if memory is CRITICALLY low
        if self.memory_manager and self.memory_manager.is_memory_pressure_critical():
            # Only drop if we can't handle the pressure - buffers will evict as needed
            pressure = self.memory_manager.get_memory_pressure_level()
            if pressure > 0.95:  # Only drop at 95%+ (vs 90% critical threshold)
                logger.warning(f"Extreme memory pressure ({pressure:.1%}) - dropping frame from {camera_name}")
                self._record_dropped_frame(camera_name, "memory_pressure")
                return False

        # Distribute frame to subscribers
        try:
            self._distribute_frame(camera_name, frame, timestamp)
            self._update_camera_metrics(camera_name, frame.nbytes, timestamp)
            return True
        except Exception as e:
            logger.error(f"Error distributing frame from {camera_name}: {e}", exc_info=True)
            return False

    def _should_process_frame(self, camera_name: str, timestamp: float) -> bool:
        """Determine if frame should be processed based on adaptive FPS."""
        # No rate limiting - camera loop already controls FPS via sleep intervals.
        # The CameraManager loop is the authoritative rate controller.
        return True

    def _distribute_frame(self, camera_name: str, frame: np.ndarray, timestamp: float) -> None:
        """Distribute frame to all subscribers for this camera.

        NOTE: Callbacks are called synchronously. If callbacks are slow, this will
        block frame submission and cause drops. Callbacks should be fast or use
        async processing internally.
        """
        subscribers = self.subscribers.get(camera_name, [])

        for callback in subscribers:
            try:
                callback(camera_name, frame, timestamp)
            except Exception as e:
                logger.error(f"Error in frame callback for {camera_name}: {e}", exc_info=True)

    def _record_dropped_frame(self, camera_name: str, reason: str) -> None:
        """Record a dropped frame and update metrics."""
        with self.camera_lock:
            if camera_name in self.cameras:
                self.cameras[camera_name].dropped_frames += 1
                self.cameras[camera_name].consecutive_drops += 1

        self.frame_metrics[camera_name].frames_dropped += 1
        self.global_metrics.frames_dropped += 1

        # Log drop reason at info level for visibility
        logger.info(f"Dropped frame from {camera_name}: {reason}")

    def _update_camera_metrics(self, camera_name: str, frame_size: int, timestamp: float) -> None:
        """Update performance metrics for a camera."""
        metrics = self.frame_metrics[camera_name]
        metrics.frames_received += 1
        metrics.frames_processed += 1
        metrics.memory_usage_mb = frame_size / (1024 * 1024)
        metrics.last_update = timestamp

        # Update global metrics
        self.global_metrics.frames_received += 1
        self.global_metrics.frames_processed += 1

        # Memory usage is tracked by MemoryBoundedBuffers
        self.global_metrics.last_update = timestamp

        # Reset consecutive drops on successful processing
        # Update last_frame_time AFTER callback completes to account for processing time
        with self.camera_lock:
            if camera_name in self.cameras:
                camera_state = self.cameras[camera_name]
                camera_state.consecutive_drops = 0
                # Update timestamp after processing to properly account for callback duration
                old_time = camera_state.last_frame_time
                camera_state.last_frame_time = timestamp

                # Debug log every 100 frames to verify timing
                if camera_state.total_frames % 100 == 0:
                    logger.info(f"Updated {camera_name} timestamp: {old_time:.3f} -> {timestamp:.3f} (delta: {timestamp-old_time:.4f}s)")

    def get_camera_metrics(self, camera_name: str) -> Optional[FrameMetrics]:
        """Get metrics for a specific camera."""
        return self.frame_metrics.get(camera_name)

    def get_global_metrics(self) -> FrameMetrics:
        """Get global frame processing metrics."""
        return self.global_metrics

    def get_camera_status(self) -> Dict[str, Dict]:
        """Get status of all registered cameras."""
        status = {}
        with self.camera_lock:
            for camera_name, camera_state in self.cameras.items():
                metrics = self.frame_metrics[camera_name]

                # Calculate drop rate
                drop_rate = 0.0
                if camera_state.total_frames > 0:
                    drop_rate = camera_state.dropped_frames / camera_state.total_frames

                status[camera_name] = {
                    'base_fps': camera_state.base_fps,
                    'current_fps': camera_state.current_fps,
                    'priority_level': camera_state.priority_level,
                    'total_frames': camera_state.total_frames,
                    'dropped_frames': camera_state.dropped_frames,
                    'drop_rate': drop_rate,
                    'consecutive_drops': camera_state.consecutive_drops,
                    'frames_processed': metrics.frames_processed,
                    'memory_usage_mb': metrics.memory_usage_mb,
                    'last_update': metrics.last_update
                }

        return status

    def shutdown(self) -> None:
        """Shutdown the frame broker and clean up resources."""
        logger.info("Shutting down FrameBroker")
        with self.camera_lock:
            self.cameras.clear()
        self.subscribers.clear()
        self.priority_cameras.clear()
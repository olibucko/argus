"""
Frame Broker - Centralized frame distribution with memory-awareness.

This module provides a centralized mechanism for distributing frames from cameras
to various processing components while monitoring system memory pressure.
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Callable, List, Any
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
    """State tracking for a single camera stream."""
    name: str
    base_fps: float = 30.0
    current_fps: float = 30.0
    last_frame_time: float = 0.0
    frame_interval: float = 1.0/30.0
    priority_level: int = 0  # 0=normal, 1=motion detected, 2=active alert
    consecutive_drops: int = 0
    total_frames: int = 0
    dropped_frames: int = 0


class FrameBroker:
    """
    Orchestrates the distribution of camera frames to multiple subscribers.

    The FrameBroker ensures that frames are distributed efficiently and that
    the system responds gracefully to memory pressure by dropping frames from
    unregistered or low-priority sources.

    Attributes:
        max_memory_bytes (int): Maximum memory threshold for the broker.
        memory_manager (GlobalMemoryManager): Global manager for memory pressure.
        cameras (Dict[str, CameraState]): Map of camera names to their state.
        subscribers (Dict[str, List[Callable]]): Map of camera names to callbacks.
        priority_cameras (Set[str]): Set of cameras currently in a high-priority state.
    """

    def __init__(self, max_memory_mb: int = 512, memory_manager: Optional[Any] = None) -> None:
        """
        Initializes the FrameBroker.

        Args:
            max_memory_mb (int): Memory limit in megabytes.
            memory_manager (GlobalMemoryManager): Optional memory manager instance.
        """
        self.max_memory_bytes: int = max_memory_mb * 1024 * 1024
        self.memory_manager: Optional[Any] = memory_manager
        self.cameras: Dict[str, CameraState] = {}
        self.camera_lock: threading.RLock = threading.RLock()
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.priority_cameras: Set[str] = set()
        self.frame_metrics: Dict[str, FrameMetrics] = defaultdict(FrameMetrics)
        self.global_metrics: FrameMetrics = FrameMetrics()

        logger.info(f"FrameBroker initialized with {max_memory_mb}MB memory limit")

    def register_camera(self, camera_name: str, base_fps: float = 30.0) -> None:
        """
        Registers a new camera with the broker.

        Args:
            camera_name (str): The unique name of the camera.
            base_fps (float): The expected frame rate of the camera.
        """
        with self.camera_lock:
            self.cameras[camera_name] = CameraState(
                name=camera_name,
                base_fps=base_fps,
                current_fps=base_fps,
                frame_interval=1.0/base_fps
            )
        logger.info(f"Registered camera '{camera_name}' with {base_fps} FPS")

    def subscribe_to_camera(self, camera_name: str, callback: Callable[[str, np.ndarray, float], None]) -> None:
        """
        Subscribes a callback function to a camera's frame stream.

        Args:
            camera_name (str): The name of the camera to subscribe to.
            callback (Callable): Function called with (camera_name, frame, timestamp).
        """
        self.subscribers[camera_name].append(callback)
        logger.debug(f"Added subscriber for camera '{camera_name}'")

    def set_camera_priority(self, camera_name: str, priority: int) -> None:
        """
        Sets the priority level for a camera.

        Args:
            camera_name (str): The name of the camera.
            priority (int): The priority level (0=normal, >0=high).
        """
        with self.camera_lock:
            if camera_name in self.cameras:
                self.cameras[camera_name].priority_level = priority
                if priority > 0:
                    self.priority_cameras.add(camera_name)
                else:
                    self.priority_cameras.discard(camera_name)

    def submit_frame(self, camera_name: str, frame: np.ndarray, timestamp: float) -> bool:
        """
        Submits a frame from a camera for distribution to subscribers.

        Args:
            camera_name (str): The source camera's name.
            frame (np.ndarray): The raw frame data.
            timestamp (float): The timestamp when the frame was captured.

        Returns:
            bool: True if the frame was accepted and distributed, False if dropped.
        """
        with self.camera_lock:
            if camera_name not in self.cameras:
                logger.error(f"Unregistered camera: {camera_name}")
                return False
            self.cameras[camera_name].total_frames += 1

        if self.memory_manager and self.memory_manager.is_memory_pressure_critical():
            pressure = self.memory_manager.get_memory_pressure_level()
            if pressure > 0.95:
                logger.warning(f"Extreme memory pressure ({pressure:.1%}) - dropping frame from {camera_name}")
                self._record_dropped_frame(camera_name, "memory_pressure")
                return False

        try:
            self._distribute_frame(camera_name, frame, timestamp)
            self._update_camera_metrics(camera_name, frame.nbytes, timestamp)
            return True
        except Exception as e:
            logger.error(f"Error distributing frame from {camera_name}: {e}", exc_info=True)
            return False

    def _distribute_frame(self, camera_name: str, frame: np.ndarray, timestamp: float) -> None:
        """Calls all subscriber callbacks for the given camera."""
        for callback in self.subscribers.get(camera_name, []):
            try:
                callback(camera_name, frame, timestamp)
            except Exception as e:
                logger.error(f"Callback error for {camera_name}: {e}", exc_info=True)

    def _record_dropped_frame(self, camera_name: str, reason: str) -> None:
        """Updates internal metrics to reflect a dropped frame."""
        with self.camera_lock:
            if camera_name in self.cameras:
                self.cameras[camera_name].dropped_frames += 1
                self.cameras[camera_name].consecutive_drops += 1
        self.frame_metrics[camera_name].frames_dropped += 1
        self.global_metrics.frames_dropped += 1
        logger.info(f"Dropped frame from {camera_name}: {reason}")

    def _update_camera_metrics(self, camera_name: str, frame_size: int, timestamp: float) -> None:
        """Updates performance metrics after successful frame processing."""
        metrics = self.frame_metrics[camera_name]
        metrics.frames_received += 1
        metrics.frames_processed += 1
        metrics.memory_usage_mb = frame_size / (1024 * 1024)
        metrics.last_update = timestamp

        self.global_metrics.frames_received += 1
        self.global_metrics.frames_processed += 1
        self.global_metrics.last_update = timestamp

        with self.camera_lock:
            if camera_name in self.cameras:
                state = self.cameras[camera_name]
                state.consecutive_drops = 0
                state.last_frame_time = timestamp

    def get_camera_status(self) -> Dict[str, Dict]:
        """Returns a snapshot of the current status of all registered cameras."""
        status = {}
        with self.camera_lock:
            for name, state in self.cameras.items():
                metrics = self.frame_metrics[name]
                drop_rate = state.dropped_frames / state.total_frames if state.total_frames > 0 else 0.0
                status[name] = {
                    'base_fps': state.base_fps, 'current_fps': state.current_fps,
                    'priority_level': state.priority_level, 'total_frames': state.total_frames,
                    'dropped_frames': state.dropped_frames, 'drop_rate': drop_rate,
                    'consecutive_drops': state.consecutive_drops, 'frames_processed': metrics.frames_processed,
                    'memory_usage_mb': metrics.memory_usage_mb, 'last_update': metrics.last_update
                }
        return status

    def shutdown(self) -> None:
        """Shutdown the frame broker and clean up resources."""
        logger.info("Shutting down FrameBroker")
        with self.camera_lock:
            self.cameras.clear()
        self.subscribers.clear()
        self.priority_cameras.clear()

"""
Shared Frame Pool - Memory-efficient frame storage with reference counting.

This module provides a mechanism to store video frames once in memory and distribute
lightweight references to multiple consumers, eliminating redundant copies and
reducing overall memory usage.
"""

import threading
import time
import uuid
from typing import Dict, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FrameReference:
    """
    A lightweight handle to a frame stored in the SharedFramePool.

    The FrameReference ensures that the associated frame's reference count is
    managed automatically using Python's garbage collection. When the last
    reference to a frame is destroyed, the frame data is freed from the pool.

    Attributes:
        frame_id (str): The unique identifier of the frame in the pool.
        frame_pool (SharedFramePool): The pool where the frame data is stored.
    """

    def __init__(self, frame_id: str, frame_pool: 'SharedFramePool') -> None:
        """
        Initializes a FrameReference.

        Args:
            frame_id (str): The ID of the frame.
            frame_pool (SharedFramePool): The parent pool instance.
        """
        self.frame_id: str = frame_id
        self.frame_pool: 'SharedFramePool' = frame_pool
        self._released: bool = False

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Retrieves the raw frame data from the pool.

        Returns:
            Optional[np.ndarray]: The frame data, or None if it has been released.
        """
        if self._released:
            return None
        return self.frame_pool.get_frame(self.frame_id)

    def get_frame_size(self) -> int:
        """
        Returns the size of the referenced frame in bytes.

        Returns:
            int: The size of the frame data.
        """
        return self.frame_pool.get_frame_size(self.frame_id)

    def release(self) -> None:
        """
        Manually decrements the reference count for this frame in the pool.
        """
        if not self._released:
            self.frame_pool.release_reference(self.frame_id)
            self._released = True

    def __del__(self) -> None:
        """Automatically releases the reference when the object is garbage collected."""
        self.release()

    def __repr__(self) -> str:
        return f"FrameReference(id={self.frame_id[:8]}..., released={self._released})"


class SharedFramePool:
    """
    Manages a thread-safe pool of video frames with reference counting.

    The pool allows multiple components (e.g., motion detectors, recorders,
    dashboard displays) to share the same underlying image data without
    performing expensive copies.

    Attributes:
        max_memory_bytes (int): Maximum allowed memory usage for frame storage.
        current_memory_bytes (int): Current memory usage.
        frames (Dict[str, Dict]): Map of frame IDs to data and reference counts.
    """

    def __init__(self, max_memory_mb: int = 512) -> None:
        """
        Initializes the SharedFramePool.

        Args:
            max_memory_mb (int): Memory limit in megabytes.
        """
        self.max_memory_bytes: int = max_memory_mb * 1024 * 1024
        self.current_memory_bytes: int = 0
        self.frames: Dict[str, Dict[str, Any]] = {}
        self.lock: threading.RLock = threading.RLock()

        # Statistics
        self.total_frames_added: int = 0
        self.total_frames_released: int = 0
        self.total_references_created: int = 0
        self.total_references_released: int = 0

        logger.info(f"SharedFramePool initialized with {max_memory_mb}MB limit")

    def add_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Optional[FrameReference]:
        """
        Adds a new frame to the pool and returns a lightweight reference.

        Args:
            frame (np.ndarray): The raw image data.
            timestamp (float, optional): The capture timestamp.

        Returns:
            Optional[FrameReference]: A reference to the stored frame, or None if full.
        """
        if timestamp is None:
            timestamp = time.time()

        frame_size = frame.nbytes
        frame_id = str(uuid.uuid4())

        with self.lock:
            if self.current_memory_bytes + frame_size > self.max_memory_bytes:
                logger.warning(f"Frame pool limit reached ({self.current_memory_bytes/1024/1024:.1f}MB). Dropping frame.")
                return None

            self.frames[frame_id] = {
                'frame': frame.copy(),  # Internal storage copy
                'ref_count': 1,
                'timestamp': timestamp,
                'size': frame_size
            }

            self.current_memory_bytes += frame_size
            self.total_frames_added += 1
            self.total_references_created += 1

            logger.debug(f"Added frame {frame_id[:8]}... to pool.")
            return FrameReference(frame_id, self)

    def acquire_reference(self, frame_id: str) -> Optional[FrameReference]:
        """
        Increments the reference count for an existing frame and returns a new reference.

        Args:
            frame_id (str): The unique ID of the frame.

        Returns:
            Optional[FrameReference]: A new reference to the frame, or None if not found.
        """
        with self.lock:
            if frame_id not in self.frames:
                return None

            self.frames[frame_id]['ref_count'] += 1
            self.total_references_created += 1
            return FrameReference(frame_id, self)

    def get_frame(self, frame_id: str) -> Optional[np.ndarray]:
        """Retrieves the underlying frame data for a given ID."""
        with self.lock:
            return self.frames.get(frame_id, {}).get('frame')

    def get_frame_size(self, frame_id: str) -> int:
        """Retrieves the size of a frame in bytes."""
        with self.lock:
            return self.frames.get(frame_id, {}).get('size', 0)

    def release_reference(self, frame_id: str) -> None:
        """
        Decrements the reference count for a frame and frees data if it reaches zero.

        Args:
            frame_id (str): The ID of the frame to release.
        """
        with self.lock:
            if frame_id not in self.frames:
                return

            self.frames[frame_id]['ref_count'] -= 1
            self.total_references_released += 1

            if self.frames[frame_id]['ref_count'] <= 0:
                self.current_memory_bytes -= self.frames[frame_id]['size']
                del self.frames[frame_id]
                self.total_frames_released += 1
                logger.debug(f"Freed frame {frame_id[:8]}... from pool.")

    def get_stats(self) -> Dict[str, Any]:
        """Returns internal performance and memory usage statistics for the pool."""
        with self.lock:
            total_refs = sum(f['ref_count'] for f in self.frames.values())
            return {
                'current_memory_mb': self.current_memory_bytes / 1024 / 1024,
                'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
                'active_frames': len(self.frames),
                'total_references': total_refs,
                'avg_refs_per_frame': total_refs / len(self.frames) if self.frames else 0
            }

    def shutdown(self) -> None:
        """Clears all frames from the pool and resets memory tracking."""
        with self.lock:
            logger.info(f"Shutting down pool, releasing {len(self.frames)} frames.")
            self.frames.clear()
            self.current_memory_bytes = 0

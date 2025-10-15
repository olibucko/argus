"""
Shared Frame Pool - Memory-efficient frame storage with reference counting.

Eliminates redundant frame copying by storing frames once and distributing
references to multiple consumers (viewport, recording, display buffers).
"""

import threading
import time
import uuid
from typing import Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FrameReference:
    """
    Lightweight reference to a frame in the shared pool.
    Automatically releases the frame when the reference is destroyed.
    """

    def __init__(self, frame_id: str, frame_pool: 'SharedFramePool'):
        self.frame_id = frame_id
        self.frame_pool = frame_pool
        self._released = False

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the actual frame data from the pool."""
        if self._released:
            return None
        return self.frame_pool.get_frame(self.frame_id)

    def get_frame_size(self) -> int:
        """Get the size of the referenced frame in bytes."""
        return self.frame_pool.get_frame_size(self.frame_id)

    def release(self):
        """Release this reference (decrement ref count)."""
        if not self._released:
            self.frame_pool.release_reference(self.frame_id)
            self._released = True

    def __del__(self):
        """Automatically release when reference is garbage collected."""
        self.release()

    def __repr__(self):
        return f"FrameReference(id={self.frame_id[:8]}..., released={self._released})"


class SharedFramePool:
    """
    Central pool for frame storage with reference counting.

    Features:
    - Stores frames once, distributes references
    - Automatic cleanup when references drop to zero
    - Thread-safe operations
    - Memory tracking and limits
    """

    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_bytes = 0

        # Frame storage: frame_id -> {'frame': ndarray, 'ref_count': int, 'timestamp': float, 'size': int}
        self.frames: Dict[str, Dict] = {}
        self.lock = threading.RLock()

        # Statistics
        self.total_frames_added = 0
        self.total_frames_released = 0
        self.total_references_created = 0
        self.total_references_released = 0

        logger.info(f"SharedFramePool initialized with {max_memory_mb}MB limit")

    def add_frame(self, frame: np.ndarray, timestamp: float = None) -> Optional[FrameReference]:
        """
        Add a frame to the pool and return a reference.

        Args:
            frame: Frame to store
            timestamp: Frame timestamp

        Returns:
            FrameReference if successful, None if memory limit exceeded
        """
        if timestamp is None:
            timestamp = time.time()

        frame_size = frame.nbytes
        frame_id = str(uuid.uuid4())

        with self.lock:
            # Check memory limit
            if self.current_memory_bytes + frame_size > self.max_memory_bytes:
                logger.warning(f"Frame pool memory limit exceeded: {self.current_memory_bytes/1024/1024:.1f}MB")
                # Try to cleanup old frames with 0 references
                self._cleanup_orphaned_frames()

                # Check again after cleanup
                if self.current_memory_bytes + frame_size > self.max_memory_bytes:
                    return None

            # Store frame with initial ref count of 1
            self.frames[frame_id] = {
                'frame': frame.copy(),  # Single copy stored here
                'ref_count': 1,
                'timestamp': timestamp,
                'size': frame_size
            }

            self.current_memory_bytes += frame_size
            self.total_frames_added += 1
            self.total_references_created += 1

            logger.debug(f"Added frame {frame_id[:8]}... ({frame_size/1024:.1f}KB) to pool, "
                        f"pool size: {self.current_memory_bytes/1024/1024:.1f}MB")

            return FrameReference(frame_id, self)

    def acquire_reference(self, frame_id: str) -> Optional[FrameReference]:
        """
        Create a new reference to an existing frame.

        Args:
            frame_id: ID of the frame to reference

        Returns:
            FrameReference if frame exists, None otherwise
        """
        with self.lock:
            if frame_id not in self.frames:
                return None

            self.frames[frame_id]['ref_count'] += 1
            self.total_references_created += 1

            logger.debug(f"Acquired reference to {frame_id[:8]}..., ref_count: {self.frames[frame_id]['ref_count']}")

            return FrameReference(frame_id, self)

    def get_frame(self, frame_id: str) -> Optional[np.ndarray]:
        """
        Get the actual frame data.

        Args:
            frame_id: ID of the frame

        Returns:
            Frame array or None if not found
        """
        with self.lock:
            if frame_id in self.frames:
                return self.frames[frame_id]['frame']
            return None

    def get_frame_size(self, frame_id: str) -> int:
        """
        Get the size of a frame in bytes.

        Args:
            frame_id: ID of the frame

        Returns:
            Frame size in bytes, or 0 if not found
        """
        with self.lock:
            if frame_id in self.frames:
                return self.frames[frame_id]['size']
            return 0

    def release_reference(self, frame_id: str):
        """
        Release a reference to a frame. When ref_count reaches 0, the frame is freed.

        Args:
            frame_id: ID of the frame
        """
        with self.lock:
            if frame_id not in self.frames:
                return

            self.frames[frame_id]['ref_count'] -= 1
            self.total_references_released += 1

            ref_count = self.frames[frame_id]['ref_count']

            logger.debug(f"Released reference to {frame_id[:8]}..., ref_count: {ref_count}")

            # Free frame when no more references
            if ref_count <= 0:
                frame_size = self.frames[frame_id]['size']
                del self.frames[frame_id]
                self.current_memory_bytes -= frame_size
                self.total_frames_released += 1

                logger.debug(f"Freed frame {frame_id[:8]}... ({frame_size/1024:.1f}KB), "
                           f"pool size: {self.current_memory_bytes/1024/1024:.1f}MB")

    def _cleanup_orphaned_frames(self):
        """Remove frames with 0 references (shouldn't happen, but safety check)."""
        with self.lock:
            orphaned = [fid for fid, data in self.frames.items() if data['ref_count'] <= 0]

            for frame_id in orphaned:
                frame_size = self.frames[frame_id]['size']
                del self.frames[frame_id]
                self.current_memory_bytes -= frame_size
                logger.warning(f"Cleaned up orphaned frame {frame_id[:8]}...")

            if orphaned:
                logger.info(f"Cleaned up {len(orphaned)} orphaned frames")

    def get_stats(self) -> Dict:
        """Get pool statistics."""
        with self.lock:
            total_ref_count = sum(data['ref_count'] for data in self.frames.values())

            return {
                'current_memory_mb': self.current_memory_bytes / 1024 / 1024,
                'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
                'memory_usage_percent': (self.current_memory_bytes / self.max_memory_bytes) * 100,
                'active_frames': len(self.frames),
                'total_references': total_ref_count,
                'total_frames_added': self.total_frames_added,
                'total_frames_released': self.total_frames_released,
                'total_references_created': self.total_references_created,
                'total_references_released': self.total_references_released,
                'avg_refs_per_frame': total_ref_count / len(self.frames) if self.frames else 0
            }

    def shutdown(self):
        """Cleanup all frames."""
        with self.lock:
            logger.info(f"Shutting down SharedFramePool, releasing {len(self.frames)} frames")
            self.frames.clear()
            self.current_memory_bytes = 0

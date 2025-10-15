"""
Memory Manager - Memory-bounded buffer system for frame processing.

This module provides memory-aware data structures that prevent memory exhaustion
by implementing intelligent eviction policies and memory tracking.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any, Callable
from enum import Enum
import numpy as np
import logging

# Delayed import of FrameReference to avoid circular dependency
# Done in __post_init__ of FrameEntry

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    """Frame eviction policies for memory management."""
    FIFO = "fifo"  # First In, First Out
    LRU = "lru"   # Least Recently Used
    PRIORITY = "priority"  # Priority-based (keep motion frames longer)


@dataclass
class FrameEntry:
    """Entry in memory-bounded buffer."""
    frame: Any  # Can be np.ndarray or FrameReference
    timestamp: float
    priority: int = 0  # 0=normal, 1=motion detected, 2=active alert
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0
    is_reference: bool = False  # True if frame is a FrameReference

    def __post_init__(self):
        # Check if this is a FrameReference
        from .frame_pool import FrameReference
        if isinstance(self.frame, FrameReference):
            self.is_reference = True
            # For references, use the actual frame size from the pool
            if self.size_bytes == 0:
                self.size_bytes = self.frame.get_frame_size()
        else:
            # Regular numpy array
            if self.size_bytes == 0:
                self.size_bytes = self.frame.nbytes

        if self.last_access == 0.0:
            self.last_access = self.timestamp

    def get_frame_data(self) -> Optional[np.ndarray]:
        """Get the actual frame data (handles both references and direct frames)."""
        if self.is_reference:
            return self.frame.get_frame()
        else:
            return self.frame

    def release(self):
        """Release the frame if it's a reference."""
        if self.is_reference:
            self.frame.release()

    def __eq__(self, other):
        """Custom equality for FrameEntry to avoid numpy array comparison issues."""
        if not isinstance(other, FrameEntry):
            return False
        return (self.timestamp == other.timestamp and
                self.priority == other.priority and
                self.size_bytes == other.size_bytes)

    def __hash__(self):
        """Make FrameEntry hashable."""
        return hash((self.timestamp, self.priority, self.size_bytes))


class MemoryBoundedBuffer:
    """
    Memory-aware buffer that automatically evicts frames when memory limits are reached.

    Features:
    - Configurable eviction policies
    - Priority-based frame retention
    - Automatic memory tracking
    - Thread-safe operations
    """

    def __init__(self,
                 max_memory_bytes: int,
                 eviction_policy: EvictionPolicy = EvictionPolicy.PRIORITY,
                 name: str = "UnnamedBuffer"):
        self.max_memory_bytes = max_memory_bytes
        self.eviction_policy = eviction_policy
        self.name = name

        # Frame storage
        self.frames: deque[FrameEntry] = deque()
        self.current_memory_usage = 0
        self.lock = threading.RLock()

        # Event-driven frame availability notification
        self.frame_available = threading.Condition(self.lock)

        # Statistics
        self.total_frames_added = 0
        self.total_frames_evicted = 0
        self.memory_pressure_events = 0

        logger.info(f"Created MemoryBoundedBuffer '{name}' with {max_memory_bytes/1024/1024:.1f}MB limit")

    def add_frame(self, frame: np.ndarray, timestamp: float = None, priority: int = 0) -> bool:
        """
        Add a frame to the buffer.

        Args:
            frame: Frame to add
            timestamp: Frame timestamp (defaults to current time)
            priority: Frame priority (0=normal, 1=motion, 2=alert)

        Returns:
            bool: True if frame was added, False if rejected
        """
        if timestamp is None:
            timestamp = time.time()

        # Check if this is a FrameReference or a numpy array
        from .frame_pool import FrameReference
        if isinstance(frame, FrameReference):
            frame_size = frame.get_frame_size()  # Actual size of referenced frame
        else:
            frame_size = frame.nbytes

        if frame_size > self.max_memory_bytes:
            logger.warning(f"Frame size ({frame_size/1024/1024:.1f}MB) exceeds buffer limit "
                         f"({self.max_memory_bytes/1024/1024:.1f}MB)")
            return False

        with self.lock:
            # Make space if needed
            if self.current_memory_usage + frame_size > self.max_memory_bytes:
                if not self._evict_frames_for_space(frame_size):
                    logger.debug(f"Could not free enough space for frame in buffer '{self.name}'")
                    return False

            # Add frame
            entry = FrameEntry(
                frame=frame,
                timestamp=timestamp,
                priority=priority,
                size_bytes=frame_size
            )

            self.frames.append(entry)
            self.current_memory_usage += frame_size
            self.total_frames_added += 1

            # Notify waiting consumers that a frame is available
            self.frame_available.notify()

            logger.debug(f"Added frame to buffer '{self.name}' "
                        f"({len(self.frames)} frames, {self.current_memory_usage/1024/1024:.1f}MB)")
            return True

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame without removing it."""
        with self.lock:
            if not self.frames:
                return None

            entry = self.frames[-1]
            entry.access_count += 1
            entry.last_access = time.time()
            return entry.get_frame_data()  # Handles both references and direct frames

    def pop_latest_frame(self) -> Optional[np.ndarray]:
        """Get and remove the most recent frame from the buffer."""
        with self.lock:
            if not self.frames:
                return None

            entry = self.frames.pop()
            self.current_memory_usage -= entry.size_bytes

            # Get frame data before releasing reference
            frame_data = entry.get_frame_data()

            # Release reference if applicable
            entry.release()

            return frame_data

    def wait_and_pop_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Wait for a frame to become available and consume it (blocking with timeout).
        This is the event-driven alternative to polling pop_latest_frame().

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Frame if available, None if timeout
        """
        with self.frame_available:
            # Wait for notification or timeout
            if not self.frames:
                self.frame_available.wait(timeout=timeout)

            # Check again after waiting
            if not self.frames:
                return None

            entry = self.frames.pop()
            self.current_memory_usage -= entry.size_bytes

            # Get frame data before releasing reference
            frame_data = entry.get_frame_data()

            # Release reference if applicable
            entry.release()

            return frame_data

    def wait_and_pop_entry(self, timeout: float = 1.0) -> Optional[FrameEntry]:
        """
        Wait for a frame entry to become available and consume it.
        This returns the full FrameEntry object, not just the frame data.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            FrameEntry if available, None if timeout
        """
        with self.frame_available:
            if not self.frames:
                self.frame_available.wait(timeout=timeout)

            if not self.frames:
                return None

            entry = self.frames.popleft()  # Use popleft for FIFO processing
            self.current_memory_usage -= entry.size_bytes

            return entry

    def get_frame_history(self, max_frames: int = None) -> List[Tuple[np.ndarray, float]]:
        """Get frame history as list of (frame, timestamp) tuples."""
        with self.lock:
            frames = list(self.frames)
            if max_frames:
                frames = frames[-max_frames:]

            result = []
            current_time = time.time()
            for entry in frames:
                entry.access_count += 1
                entry.last_access = current_time
                frame_data = entry.get_frame_data()
                if frame_data is not None:
                    result.append((frame_data, entry.timestamp))

            return result

    def get_all_frames(self) -> List[np.ndarray]:
        """Get all frames in the buffer as a list (for video recording)."""
        with self.lock:
            result = []
            current_time = time.time()
            for entry in self.frames:
                entry.access_count += 1
                entry.last_access = current_time
                frame_data = entry.get_frame_data()
                if frame_data is not None:
                    result.append(frame_data)
            return result

    def get_frames_since(self, since_timestamp: float) -> List[Tuple[np.ndarray, float]]:
        """Get all frames added since the specified timestamp."""
        with self.lock:
            result = []
            current_time = time.time()

            for entry in self.frames:
                if entry.timestamp >= since_timestamp:
                    entry.access_count += 1
                    entry.last_access = current_time
                    frame_data = entry.get_frame_data()
                    if frame_data is not None:
                        result.append((frame_data, entry.timestamp))

            return result

    def _evict_frames_for_space(self, required_bytes: int) -> bool:
        """Evict frames to make space for new frame."""
        if not self.frames:
            return False

        self.memory_pressure_events += 1
        initial_count = len(self.frames)
        freed_bytes = 0

        while freed_bytes < required_bytes and self.frames:
            if self.eviction_policy == EvictionPolicy.FIFO:
                entry = self.frames.popleft()
            elif self.eviction_policy == EvictionPolicy.LRU:
                entry = self._find_lru_frame()
            elif self.eviction_policy == EvictionPolicy.PRIORITY:
                entry = self._find_priority_eviction_candidate()
            else:
                entry = self.frames.popleft()  # Fallback to FIFO

            if entry:
                freed_bytes += entry.size_bytes
                self.current_memory_usage -= entry.size_bytes
                self.total_frames_evicted += 1

                # Release frame reference if applicable
                entry.release()

        evicted_count = initial_count - len(self.frames)
        logger.debug(f"Evicted {evicted_count} frames from buffer '{self.name}' "
                    f"(freed {freed_bytes/1024/1024:.1f}MB)")

        return freed_bytes >= required_bytes

    def _find_lru_frame(self) -> Optional[FrameEntry]:
        """Find the least recently used frame for eviction."""
        if not self.frames:
            return None

        # Find frame with oldest last_access time
        lru_entry = min(self.frames, key=lambda e: e.last_access)
        self.frames.remove(lru_entry)
        return lru_entry

    def _find_priority_eviction_candidate(self) -> Optional[FrameEntry]:
        """Find the best frame to evict based on priority and age."""
        if not self.frames:
            return None

        # Sort by priority (ascending) then by age (descending)
        # This will evict old, low-priority frames first
        candidate = min(self.frames, key=lambda e: (e.priority, -e.timestamp))
        self.frames.remove(candidate)
        return candidate

    def clear(self) -> None:
        """Clear all frames from the buffer."""
        with self.lock:
            # Release all frame references before clearing
            for entry in self.frames:
                entry.release()

            self.frames.clear()
            self.current_memory_usage = 0
            logger.debug(f"Cleared buffer '{self.name}'")

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self.lock:
            return {
                'name': self.name,
                'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
                'current_memory_mb': self.current_memory_usage / 1024 / 1024,
                'memory_usage_percent': (self.current_memory_usage / self.max_memory_bytes) * 100,
                'frame_count': len(self.frames),
                'total_frames_added': self.total_frames_added,
                'total_frames_evicted': self.total_frames_evicted,
                'memory_pressure_events': self.memory_pressure_events,
                'eviction_policy': self.eviction_policy.value,
                'oldest_frame_age': time.time() - self.frames[0].timestamp if self.frames else 0,
                'newest_frame_age': time.time() - self.frames[-1].timestamp if self.frames else 0
            }


class GlobalMemoryManager:
    """
    Global memory manager that coordinates memory usage across all buffers.

    Features:
    - Total system memory limit enforcement
    - Cross-buffer memory rebalancing
    - Memory pressure detection and response
    - Centralized memory statistics
    """

    def __init__(self, total_memory_limit_mb: int = 1024):
        self.total_memory_limit_bytes = total_memory_limit_mb * 1024 * 1024
        self.buffers: Dict[str, MemoryBoundedBuffer] = {}
        self.lock = threading.RLock()

        # Memory pressure thresholds
        self.warning_threshold = 0.8  # 80%
        self.critical_threshold = 0.9  # 90%

        logger.info(f"GlobalMemoryManager initialized with {total_memory_limit_mb}MB limit")

    def register_buffer(self, buffer: MemoryBoundedBuffer) -> None:
        """Register a buffer for global memory management."""
        with self.lock:
            self.buffers[buffer.name] = buffer
            logger.debug(f"Registered buffer '{buffer.name}' with memory manager")

    def unregister_buffer(self, buffer_name: str) -> None:
        """Unregister a buffer from global memory management."""
        with self.lock:
            if buffer_name in self.buffers:
                del self.buffers[buffer_name]
                logger.debug(f"Unregistered buffer '{buffer_name}' from memory manager")

    def get_total_memory_usage(self) -> int:
        """Get total memory usage across all registered buffers."""
        with self.lock:
            return sum(buffer.current_memory_usage for buffer in self.buffers.values())

    def get_memory_pressure_level(self) -> float:
        """Get current memory pressure as a percentage (0.0 to 1.0)."""
        total_usage = self.get_total_memory_usage()
        pressure_level = total_usage / self.total_memory_limit_bytes

        # Debug logging to understand false pressure reports
        if pressure_level > 0.5:  # Log when pressure exceeds 50%
            logger.warning(f"Memory pressure: {pressure_level:.2%} ({total_usage/1024/1024:.1f}MB/{self.total_memory_limit_bytes/1024/1024:.1f}MB)")
            for name, buffer in self.buffers.items():
                logger.warning(f"  Buffer '{name}': {buffer.current_memory_usage/1024/1024:.1f}MB ({len(buffer.frames)} frames)")

        return pressure_level

    def is_memory_pressure_critical(self) -> bool:
        """Check if memory pressure is at critical levels."""
        return self.get_memory_pressure_level() >= self.critical_threshold

    def handle_memory_pressure(self) -> bool:
        """Handle memory pressure by evicting frames from low-priority buffers."""
        pressure_level = self.get_memory_pressure_level()

        if pressure_level < self.warning_threshold:
            return True  # No pressure

        logger.warning(f"Memory pressure detected: {pressure_level:.1%}")

        with self.lock:
            # Sort buffers by priority (viewport buffers have lower priority than recording buffers)
            buffer_priority = []
            for name, buffer in self.buffers.items():
                if 'recording' in name.lower():
                    priority = 0  # High priority - keep recording buffers
                elif 'display' in name.lower():
                    priority = 2  # Low priority - evict display buffers first
                else:
                    priority = 1  # Medium priority

                buffer_priority.append((priority, name, buffer))

            # Sort by priority (higher priority last)
            buffer_priority.sort(key=lambda x: x[0])

            # Evict frames from low-priority buffers first
            target_usage = self.total_memory_limit_bytes * self.warning_threshold
            current_usage = self.get_total_memory_usage()

            for priority, name, buffer in buffer_priority:
                if current_usage <= target_usage:
                    break

                # Calculate how much to evict from this buffer
                buffer_usage = buffer.current_memory_usage
                evict_fraction = min(0.5, (current_usage - target_usage) / buffer_usage)
                frames_to_evict = max(1, int(len(buffer.frames) * evict_fraction))

                logger.debug(f"Evicting {frames_to_evict} frames from buffer '{name}'")

                # Evict frames
                for _ in range(frames_to_evict):
                    if not buffer.frames:
                        break
                    buffer._evict_frames_for_space(0)  # Evict one frame

                current_usage = self.get_total_memory_usage()

        return self.get_memory_pressure_level() < self.critical_threshold

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global memory statistics."""
        with self.lock:
            total_usage = self.get_total_memory_usage()
            pressure_level = self.get_memory_pressure_level()

            buffer_stats = {}
            for name, buffer in self.buffers.items():
                buffer_stats[name] = buffer.get_stats()

            return {
                'total_memory_limit_mb': self.total_memory_limit_bytes / 1024 / 1024,
                'total_memory_usage_mb': total_usage / 1024 / 1024,
                'memory_pressure_level': pressure_level,
                'is_warning_level': pressure_level >= self.warning_threshold,
                'is_critical_level': pressure_level >= self.critical_threshold,
                'buffer_count': len(self.buffers),
                'buffer_stats': buffer_stats
            }
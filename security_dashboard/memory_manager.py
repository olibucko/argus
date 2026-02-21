"""
Memory Manager - Memory-bounded buffer system for frame processing.

This module provides specialized data structures and managers to prevent application
memory exhaustion. It implements various eviction policies (FIFO, LRU, Priority)
to ensure that the system remains stable even under high frame rates or resolution.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any, Callable
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    """Supported strategies for removing frames when a buffer is full."""
    FIFO = "fifo"      # First In, First Out (oldest added is removed first)
    LRU = "lru"       # Least Recently Used (oldest accessed is removed first)
    PRIORITY = "priority"  # Removes low-priority frames (e.g., non-motion) first


@dataclass
class FrameEntry:
    """
    Wraps a frame (or a reference to one) with metadata for memory management.

    Attributes:
        frame (Any): Either a raw np.ndarray or a FrameReference object.
        timestamp (float): The time the frame was added to the buffer.
        priority (int): Importance level (0=normal, 1=motion, 2=alert).
        access_count (int): How many times this entry has been retrieved.
        last_access (float): The last time this entry was retrieved.
        size_bytes (int): Total size of the frame data in bytes.
        is_reference (bool): True if the frame is managed by SharedFramePool.
    """
    frame: Any
    timestamp: float
    priority: int = 0
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0
    is_reference: bool = False

    def __post_init__(self) -> None:
        from .frame_pool import FrameReference
        if isinstance(self.frame, FrameReference):
            self.is_reference = True
            if self.size_bytes == 0:
                self.size_bytes = self.frame.get_frame_size()
        else:
            if self.size_bytes == 0:
                self.size_bytes = self.frame.nbytes

        if self.last_access == 0.0:
            self.last_access = self.timestamp

    def get_frame_data(self) -> Optional[np.ndarray]:
        """Returns the raw numpy array, resolving pool references if necessary."""
        return self.frame.get_frame() if self.is_reference else self.frame

    def release(self) -> None:
        """Frees the entry's underlying resource if it is a managed reference."""
        if self.is_reference:
            self.frame.release()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FrameEntry):
            return False
        return (self.timestamp == other.timestamp and
                self.priority == other.priority and
                self.size_bytes == other.size_bytes)


class MemoryBoundedBuffer:
    """
    A thread-safe, size-limited queue for storing video frame entries.

    When the buffer reaches its maximum memory capacity, it automatically evicts
    frames based on its configured EvictionPolicy.

    Attributes:
        name (str): Human-readable name for logging and monitoring.
        max_memory_bytes (int): Total bytes allowed in this buffer.
        eviction_policy (EvictionPolicy): Strategy for frame removal.
        frames (deque): The internal queue of FrameEntry objects.
    """

    def __init__(self, max_memory_bytes: int, 
                 eviction_policy: EvictionPolicy = EvictionPolicy.PRIORITY, 
                 name: str = "UnnamedBuffer") -> None:
        """
        Initializes the MemoryBoundedBuffer.

        Args:
            max_memory_bytes (int): Memory limit in bytes.
            eviction_policy (EvictionPolicy): Policy for removing old data.
            name (str): Identifier for this buffer.
        """
        self.max_memory_bytes: int = max_memory_bytes
        self.eviction_policy: EvictionPolicy = eviction_policy
        self.name: str = name
        self.frames: deque[FrameEntry] = deque()
        self.current_memory_usage: int = 0
        self.lock: threading.RLock = threading.RLock()
        self.frame_available: threading.Condition = threading.Condition(self.lock)

        # Statistics
        self.total_frames_added: int = 0
        self.total_frames_evicted: int = 0
        self.memory_pressure_events: int = 0

        logger.info(f"MemoryBoundedBuffer '{name}' created ({max_memory_bytes/1024/1024:.1f}MB limit)")

    def add_frame(self, frame: Any, timestamp: Optional[float] = None, priority: int = 0) -> bool:
        """
        Adds a frame to the buffer, potentially triggering eviction.

        Args:
            frame (Any): The frame data or reference.
            timestamp (float, optional): Capture time.
            priority (int): Frame importance level.

        Returns:
            bool: True if added, False if the single frame is larger than the entire buffer.
        """
        if timestamp is None:
            timestamp = time.time()

        from .frame_pool import FrameReference
        size = frame.get_frame_size() if isinstance(frame, FrameReference) else frame.nbytes

        if size > self.max_memory_bytes:
            logger.warning(f"Frame ({size}B) too large for buffer '{self.name}' ({self.max_memory_bytes}B).")
            return False

        with self.lock:
            while self.current_memory_usage + size > self.max_memory_bytes:
                if not self._evict_single_frame():
                    return False

            entry = FrameEntry(frame=frame, timestamp=timestamp, priority=priority, size_bytes=size)
            self.frames.append(entry)
            self.current_memory_usage += size
            self.total_frames_added += 1
            self.frame_available.notify()
            return True

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Retrieves data for the most recently added frame without removing it."""
        with self.lock:
            if not self.frames:
                return None
            entry = self.frames[-1]
            entry.access_count += 1
            entry.last_access = time.time()
            return entry.get_frame_data()

    def wait_and_pop_entry(self, timeout: float = 1.0) -> Optional[FrameEntry]:
        """Blocks until a frame is available, then removes and returns it."""
        with self.frame_available:
            if not self.frames:
                if not self.frame_available.wait(timeout=timeout):
                    return None
            if not self.frames:
                return None
            entry = self.frames.popleft()
            self.current_memory_usage -= entry.size_bytes
            return entry

    def get_all_frames(self) -> List[np.ndarray]:
        """Returns raw data for every frame in the buffer (ordered oldest to newest)."""
        with self.lock:
            now = time.time()
            results = []
            for e in self.frames:
                e.access_count += 1
                e.last_access = now
                if (data := e.get_frame_data()) is not None:
                    results.append(data)
            return results

    def _evict_single_frame(self) -> bool:
        """Removes one frame from the buffer based on the configured policy."""
        if not self.frames: return False
        self.memory_pressure_events += 1

        if self.eviction_policy == EvictionPolicy.FIFO:
            entry = self.frames.popleft()
        elif self.eviction_policy == EvictionPolicy.LRU:
            entry = min(self.frames, key=lambda e: e.last_access)
            self.frames.remove(entry)
        elif self.eviction_policy == EvictionPolicy.PRIORITY:
            entry = min(self.frames, key=lambda e: (e.priority, -e.timestamp))
            self.frames.remove(entry)
        else:
            entry = self.frames.popleft()

        self.current_memory_usage -= entry.size_bytes
        self.total_frames_evicted += 1
        entry.release()
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics for this buffer."""
        with self.lock:
            usage_pct = (self.current_memory_usage / self.max_memory_bytes) * 100 if self.max_memory_bytes > 0 else 0
            return {
                'name': self.name, 'current_memory_mb': self.current_memory_usage / 1024 / 1024,
                'usage_percent': usage_pct, 'frame_count': len(self.frames),
                'total_evicted': self.total_frames_evicted
            }


class GlobalMemoryManager:
    """
    Coordinates and monitors memory usage across multiple MemoryBoundedBuffer instances.

    Attributes:
        total_limit_bytes (int): Maximum combined memory for all registered buffers.
        buffers (Dict[str, MemoryBoundedBuffer]): Map of buffer names to instances.
    """

    def __init__(self, total_memory_limit_mb: int = 1024) -> None:
        """
        Initializes the GlobalMemoryManager.

        Args:
            total_memory_limit_mb (int): Memory limit in megabytes.
        """
        self.total_limit_bytes: int = total_memory_limit_mb * 1024 * 1024
        self.buffers: Dict[str, MemoryBoundedBuffer] = {}
        self.lock: threading.RLock = threading.RLock()
        self.warning_threshold: float = 0.8
        self.critical_threshold: float = 0.9

        logger.info(f"GlobalMemoryManager initialized with {total_memory_limit_mb}MB limit")

    def register_buffer(self, buffer: MemoryBoundedBuffer) -> None:
        """Adds a buffer to the global monitoring and management list."""
        with self.lock:
            self.buffers[buffer.name] = buffer

    def unregister_buffer(self, name: str) -> None:
        """Removes a buffer from global management."""
        with self.lock:
            self.buffers.pop(name, None)

    def get_total_memory_usage(self) -> int:
        """Calculates the sum of memory used by all registered buffers."""
        with self.lock:
            return sum(b.current_memory_usage for b in self.buffers.values())

    def get_memory_pressure_level(self) -> float:
        """Returns the current usage ratio (0.0 to 1.0)."""
        return self.get_total_memory_usage() / self.total_limit_bytes if self.total_limit_bytes > 0 else 0.0

    def is_memory_pressure_critical(self) -> bool:
        """True if current usage exceeds the critical threshold."""
        return self.get_memory_pressure_level() >= self.critical_threshold

    def handle_memory_pressure(self) -> bool:
        """
        Attempts to alleviate memory pressure by forcing eviction from low-priority buffers.

        Returns:
            bool: True if pressure was reduced below critical levels.
        """
        if self.get_memory_pressure_level() < self.warning_threshold:
            return True

        with self.lock:
            # Group buffers by implicit priority based on their name/purpose
            sorted_buffers = sorted(self.buffers.values(), key=lambda b: self._get_buffer_rank(b.name))
            target = self.total_limit_bytes * self.warning_threshold

            for buffer in sorted_buffers:
                if self.get_total_memory_usage() <= target: break
                
                # Evict up to 50% of the frames in this buffer
                evict_count = len(buffer.frames) // 2
                for _ in range(evict_count):
                    if not buffer._evict_single_frame(): break

        return self.get_memory_pressure_level() < self.critical_threshold

    def _get_buffer_rank(self, name: str) -> int:
        """Ranks buffers for eviction priority (higher rank = evicted first)."""
        name = name.lower()
        if 'display' in name: return 2 # Low priority
        if 'viewport' in name: return 1 # Medium priority
        return 0 # High priority (e.g., recording)

    def get_global_stats(self) -> Dict[str, Any]:
        """Returns a snapshot of system-wide memory metrics."""
        with self.lock:
            usage = self.get_total_memory_usage()
            pressure = usage / self.total_limit_bytes if self.total_limit_bytes > 0 else 0
            return {
                'total_memory_limit_mb': self.total_limit_bytes / 1024 / 1024,
                'total_memory_usage_mb': usage / 1024 / 1024,
                'memory_pressure_level': pressure,
                'buffer_stats': {name: b.get_stats() for name, b in self.buffers.items()}
            }

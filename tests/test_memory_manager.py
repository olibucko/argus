import numpy as np
import pytest

from security_dashboard.memory_manager import (
    MemoryBoundedBuffer,
    GlobalMemoryManager,
    EvictionPolicy,
    FrameEntry,
)


def _make_frame(height=10, width=10, channels=3):
    """Create a small test frame with a known byte size."""
    return np.zeros((height, width, channels), dtype=np.uint8)


class TestMemoryBoundedBuffer:

    def test_add_and_retrieve_frame(self):
        buf = MemoryBoundedBuffer(max_memory_bytes=1_000_000, name="test")
        frame = _make_frame()
        assert buf.add_frame(frame, timestamp=1.0) is True
        assert buf.total_frames_added == 1
        result = buf.get_latest_frame()
        assert result is not None
        assert np.array_equal(result, frame)

    def test_rejects_frame_larger_than_buffer(self):
        buf = MemoryBoundedBuffer(max_memory_bytes=100, name="tiny")
        big_frame = _make_frame(100, 100)  # 30,000 bytes
        assert buf.add_frame(big_frame) is False
        assert buf.total_frames_added == 0

    def test_fifo_eviction(self):
        frame = _make_frame()
        frame_size = frame.nbytes
        # Buffer fits exactly 2 frames
        buf = MemoryBoundedBuffer(
            max_memory_bytes=frame_size * 2,
            eviction_policy=EvictionPolicy.FIFO,
            name="fifo_test",
        )
        buf.add_frame(frame, timestamp=1.0)
        buf.add_frame(frame, timestamp=2.0)
        # Adding a third should evict the oldest (timestamp=1.0)
        buf.add_frame(frame, timestamp=3.0)
        assert buf.total_frames_evicted == 1
        assert len(buf.frames) == 2
        # Oldest remaining should be timestamp=2.0
        assert buf.frames[0].timestamp == 2.0

    def test_lru_eviction(self):
        frame = _make_frame()
        frame_size = frame.nbytes
        buf = MemoryBoundedBuffer(
            max_memory_bytes=frame_size * 3,
            eviction_policy=EvictionPolicy.LRU,
            name="lru_test",
        )
        buf.add_frame(frame, timestamp=1.0)
        buf.add_frame(frame, timestamp=2.0)
        buf.add_frame(frame, timestamp=3.0)

        # Access the first frame to make it recently used
        buf.frames[0].last_access = 999.0

        # Adding a fourth should evict the least recently accessed (timestamp=2.0)
        buf.add_frame(frame, timestamp=4.0)
        assert buf.total_frames_evicted == 1
        timestamps = [e.timestamp for e in buf.frames]
        assert 2.0 not in timestamps

    def test_priority_eviction(self):
        frame = _make_frame()
        frame_size = frame.nbytes
        buf = MemoryBoundedBuffer(
            max_memory_bytes=frame_size * 2,
            eviction_policy=EvictionPolicy.PRIORITY,
            name="priority_test",
        )
        buf.add_frame(frame, timestamp=1.0, priority=2)  # high priority
        buf.add_frame(frame, timestamp=2.0, priority=0)  # low priority

        # Adding a third should evict the low-priority frame
        buf.add_frame(frame, timestamp=3.0, priority=1)
        assert buf.total_frames_evicted == 1
        priorities = [e.priority for e in buf.frames]
        assert 0 not in priorities

    def test_wait_and_pop_returns_oldest(self):
        buf = MemoryBoundedBuffer(max_memory_bytes=1_000_000, name="pop_test")
        buf.add_frame(_make_frame(), timestamp=1.0)
        buf.add_frame(_make_frame(), timestamp=2.0)
        entry = buf.wait_and_pop_entry(timeout=0.1)
        assert entry is not None
        assert entry.timestamp == 1.0
        assert len(buf.frames) == 1

    def test_wait_and_pop_timeout(self):
        buf = MemoryBoundedBuffer(max_memory_bytes=1_000_000, name="empty")
        entry = buf.wait_and_pop_entry(timeout=0.05)
        assert entry is None

    def test_get_all_frames(self):
        buf = MemoryBoundedBuffer(max_memory_bytes=1_000_000, name="all_test")
        for i in range(5):
            buf.add_frame(_make_frame(), timestamp=float(i))
        frames = buf.get_all_frames()
        assert len(frames) == 5

    def test_get_stats(self):
        buf = MemoryBoundedBuffer(max_memory_bytes=1_000_000, name="stats_test")
        buf.add_frame(_make_frame(), timestamp=1.0)
        stats = buf.get_stats()
        assert stats["name"] == "stats_test"
        assert stats["frame_count"] == 1
        assert stats["current_memory_mb"] > 0


class TestGlobalMemoryManager:

    def test_register_and_track_memory(self):
        mgr = GlobalMemoryManager(total_memory_limit_mb=1)
        buf = MemoryBoundedBuffer(max_memory_bytes=500_000, name="buf1")
        mgr.register_buffer(buf)
        buf.add_frame(_make_frame(100, 100))  # 30,000 bytes
        assert mgr.get_total_memory_usage() == buf.current_memory_usage

    def test_pressure_level(self):
        mgr = GlobalMemoryManager(total_memory_limit_mb=1)  # 1 MB
        buf = MemoryBoundedBuffer(max_memory_bytes=1_000_000, name="buf1")
        mgr.register_buffer(buf)
        assert mgr.get_memory_pressure_level() == 0.0

    def test_critical_threshold(self):
        mgr = GlobalMemoryManager(total_memory_limit_mb=1)
        assert mgr.is_memory_pressure_critical() is False

    def test_unregister_buffer(self):
        mgr = GlobalMemoryManager(total_memory_limit_mb=1)
        buf = MemoryBoundedBuffer(max_memory_bytes=500_000, name="temp")
        mgr.register_buffer(buf)
        mgr.unregister_buffer("temp")
        assert "temp" not in mgr.buffers

    def test_buffer_rank_ordering(self):
        mgr = GlobalMemoryManager()
        # Display buffers should be evicted first (highest rank)
        assert mgr._get_buffer_rank("display_0_0") > mgr._get_buffer_rank("viewport_0_0")
        assert mgr._get_buffer_rank("viewport_0_0") > mgr._get_buffer_rank("recording_0_0")

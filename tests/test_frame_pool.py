import numpy as np
import pytest

from security_dashboard.frame_pool import SharedFramePool, FrameReference


def _make_frame(height=10, width=10):
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


class TestSharedFramePool:

    def test_add_and_retrieve(self):
        pool = SharedFramePool(max_memory_mb=1)
        frame = _make_frame()
        ref = pool.add_frame(frame, timestamp=1.0)
        assert ref is not None
        assert isinstance(ref, FrameReference)
        retrieved = ref.get_frame()
        assert retrieved is not None
        assert np.array_equal(retrieved, frame)

    def test_reference_counting(self):
        pool = SharedFramePool(max_memory_mb=1)
        frame = _make_frame()
        ref1 = pool.add_frame(frame)
        frame_id = ref1.frame_id

        # Acquire a second reference
        ref2 = pool.acquire_reference(frame_id)
        assert ref2 is not None
        assert pool.frames[frame_id]["ref_count"] == 2

        # Release one — frame should still exist
        ref1.release()
        assert frame_id in pool.frames
        assert pool.frames[frame_id]["ref_count"] == 1

        # Release second — frame should be freed
        ref2.release()
        assert frame_id not in pool.frames

    def test_memory_tracking(self):
        pool = SharedFramePool(max_memory_mb=1)
        frame = _make_frame()
        ref = pool.add_frame(frame)
        assert pool.current_memory_bytes == frame.nbytes

        ref.release()
        assert pool.current_memory_bytes == 0

    def test_pool_limit_rejects_when_full(self):
        # Pool with tiny limit
        pool = SharedFramePool(max_memory_mb=0)  # 0 MB = 0 bytes
        frame = _make_frame()
        ref = pool.add_frame(frame)
        assert ref is None

    def test_acquire_nonexistent_reference(self):
        pool = SharedFramePool(max_memory_mb=1)
        ref = pool.acquire_reference("nonexistent-id")
        assert ref is None

    def test_double_release_is_safe(self):
        pool = SharedFramePool(max_memory_mb=1)
        ref = pool.add_frame(_make_frame())
        ref.release()
        # Second release should not raise
        ref.release()
        assert pool.current_memory_bytes == 0

    def test_get_frame_after_release_returns_none(self):
        pool = SharedFramePool(max_memory_mb=1)
        ref = pool.add_frame(_make_frame())
        ref.release()
        assert ref.get_frame() is None

    def test_get_frame_size(self):
        pool = SharedFramePool(max_memory_mb=1)
        frame = _make_frame(20, 20)
        ref = pool.add_frame(frame)
        assert ref.get_frame_size() == frame.nbytes

    def test_shutdown_clears_all(self):
        pool = SharedFramePool(max_memory_mb=10)
        for _ in range(5):
            pool.add_frame(_make_frame())
        pool.shutdown()
        assert len(pool.frames) == 0
        assert pool.current_memory_bytes == 0

    def test_stats(self):
        pool = SharedFramePool(max_memory_mb=10)
        ref = pool.add_frame(_make_frame())  # hold reference to prevent GC
        stats = pool.get_stats()
        assert stats["active_frames"] == 1
        assert stats["current_memory_mb"] > 0
        ref.release()

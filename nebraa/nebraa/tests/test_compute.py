"""
Tests for compute backend.
"""

import pytest
import os


class TestBackend:
    """Tests for Backend class."""
    
    def test_init_cpu(self):
        """Test CPU initialization."""
        from nebraa.utils.compute import Backend
        
        # Force CPU
        backend = Backend()
        backend.init(compute_mode="CPU")
        
        assert backend.xp.__name__ == 'numpy'
        assert backend.gpu is False
    
    def test_to_numpy(self, backend):
        """Test conversion to numpy."""
        import numpy as np
        
        arr = backend.xp.array([1, 2, 3])
        result = backend.to_numpy(arr)
        
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])
    
    def test_to_device(self, backend):
        """Test conversion to device."""
        import numpy as np
        
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        device_arr = backend.to_device(arr)
        
        # Should be same type as backend's xp arrays
        assert type(device_arr).__module__ == backend.xp.__name__


class TestDistributed:
    """Tests for distributed utilities."""
    
    def test_split_work(self):
        """Test work splitting."""
        from nebraa.utils.compute import split_work
        
        # 10 items across 3 workers
        items = list(range(10))
        
        splits = split_work(items, n_workers=3)
        
        # Should have 3 sublists
        assert len(splits) == 3
        
        # All items should be covered
        all_items = []
        for s in splits:
            all_items.extend(s)
        assert sorted(all_items) == items
    
    def test_split_work_single(self):
        """Test work splitting with single worker."""
        from nebraa.utils.compute import split_work
        
        items = list(range(5))
        splits = split_work(items, n_workers=1)
        
        assert len(splits) == 1
        assert splits[0] == items
    
    def test_get_worker_items(self):
        """Test get_worker_items function."""
        from nebraa.utils.compute import get_worker_items
        
        items = list(range(10))
        
        # Worker 0 of 3
        w0 = get_worker_items(items, rank=0, world_size=3)
        w1 = get_worker_items(items, rank=1, world_size=3)
        w2 = get_worker_items(items, rank=2, world_size=3)
        
        # All items should be covered
        all_items = w0 + w1 + w2
        assert sorted(all_items) == items
        
        # No duplicates
        assert len(all_items) == len(set(all_items))

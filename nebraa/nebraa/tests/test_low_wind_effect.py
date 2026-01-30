"""
Tests for Low Wind Effect (LWE) model improvements.

Tests configurable threshold, NaN guards, caching, and edge cases.
"""

import numpy as np
import pytest
from nebraa.physics.low_wind_effect import LowWindEffect, LowWindEffectConfig
from nebraa.utils.compute import get_backend


class TestLowWindEffectConfig:
    """Test LowWindEffectConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LowWindEffectConfig()
        assert config.piston_rms_rad == 0.5
        assert config.tilt_rms_rad == 0.3
        assert config.ar_coeff == 0.95
        assert config.n_realizations == 10
        assert config.seed is None
        assert config.pupil_threshold == 0.99
        assert config.connectivity == 1
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = LowWindEffectConfig(
            piston_rms_rad=1.0,
            tilt_rms_rad=0.5,
            pupil_threshold=0.5,
            connectivity=2,
            seed=42
        )
        assert config.piston_rms_rad == 1.0
        assert config.tilt_rms_rad == 0.5
        assert config.pupil_threshold == 0.5
        assert config.connectivity == 2
        assert config.seed == 42


class TestConfigurableThreshold:
    """Test configurable pupil threshold for island detection."""
    
    def test_high_threshold_few_islands(self):
        """High threshold (0.99) should detect fewer islands for apodized pupils."""
        n_pix = 64
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        # Create apodized edges
        y, x = np.ogrid[:n_pix, :n_pix]
        center = (n_pix - 1) / 2.0
        r = np.sqrt((x - center)**2 + (y - center)**2)
        pupil = np.clip(1.0 - (r - 20) / 5, 0, 1)  # Smooth edge
        
        # Add spider (narrow obstruction)
        pupil[n_pix//2 - 1:n_pix//2 + 1, :] = 0.5  # Semi-transparent spider
        
        # High threshold should not detect split (spider too faint)
        lwe_high = LowWindEffect(pupil, pupil_threshold=0.99)
        assert lwe_high.n_islands <= 2  # May detect 1 or 2 depending on apodization
    
    def test_low_threshold_more_islands(self):
        """Low threshold (0.4) should detect more islands for apodized pupils."""
        n_pix = 64
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        # Create apodized edges
        y, x = np.ogrid[:n_pix, :n_pix]
        center = (n_pix - 1) / 2.0
        r = np.sqrt((x - center)**2 + (y - center)**2)
        pupil = np.clip(1.0 - (r - 20) / 5, 0, 1)  # Smooth edge
        
        # Add spider (narrow obstruction)
        pupil[n_pix//2 - 1:n_pix//2 + 1, :] = 0.3  # Semi-transparent spider
        
        # Low threshold should detect split
        lwe_low = LowWindEffect(pupil, pupil_threshold=0.4)
        assert lwe_low.n_islands >= 2  # Should detect split by spider
    
    def test_threshold_in_info(self):
        """Threshold should be reported in info dict."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        lwe = LowWindEffect(pupil, pupil_threshold=0.75)
        info = lwe.info
        
        assert 'pupil_threshold' in info
        assert info['pupil_threshold'] == 0.75


class TestNoIslandsGuard:
    """Test early return guard for zero islands case (no NaNs)."""
    
    def test_zero_islands_no_error(self):
        """Empty pupil (threshold too strict) should not raise error."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32) * 0.5  # All below threshold
        
        # Should not raise error
        lwe = LowWindEffect(pupil, pupil_threshold=0.99)
        assert lwe.n_islands == 0
    
    def test_zero_islands_returns_zeros(self):
        """Zero islands should return zero phase screens (not NaN)."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32) * 0.5  # All below threshold
        
        lwe = LowWindEffect(pupil, pupil_threshold=0.99)
        phase = lwe.generate(n_screens=5)
        
        # Should return zeros, not NaNs
        assert phase.shape == (5, n_pix, n_pix)
        assert not np.any(np.isnan(phase))
        assert np.allclose(phase, 0.0)
    
    def test_zero_islands_info(self):
        """Zero islands should have empty island_sizes list."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32) * 0.5
        
        lwe = LowWindEffect(pupil, pupil_threshold=0.99)
        info = lwe.info
        
        assert info['n_islands'] == 0
        assert info['island_sizes'] == []
    
    def test_empty_pupil_no_nan(self):
        """Completely empty pupil should not produce NaNs."""
        n_pix = 32
        pupil = np.zeros((n_pix, n_pix), dtype=np.float32)
        
        lwe = LowWindEffect(pupil)
        phase = lwe.generate(n_screens=3)
        
        assert not np.any(np.isnan(phase))
        assert np.allclose(phase, 0.0)


class TestIslandMaskCaching:
    """Test island mask caching functionality."""
    
    def teardown_method(self):
        """Clear cache after each test."""
        LowWindEffect.clear_cache()
    
    def test_cache_enabled_by_default(self):
        """Cache should be enabled by default."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        # Clear cache first
        LowWindEffect.clear_cache()
        assert LowWindEffect.get_cache_size() == 0
        
        # Create LWE instance
        lwe = LowWindEffect(pupil)
        
        # Cache should now have one entry
        assert LowWindEffect.get_cache_size() == 1
        assert lwe.info['cache_enabled'] is True
    
    def test_cache_hit_avoids_recomputation(self):
        """Second instance with same pupil should hit cache."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        # Clear cache
        LowWindEffect.clear_cache()
        
        # First instance (cache miss)
        lwe1 = LowWindEffect(pupil)
        cache_size_1 = LowWindEffect.get_cache_size()
        
        # Second instance with same pupil (cache hit)
        lwe2 = LowWindEffect(pupil)
        cache_size_2 = LowWindEffect.get_cache_size()
        
        # Cache size should not increase
        assert cache_size_1 == cache_size_2 == 1
        
        # Results should be identical
        assert lwe1.n_islands == lwe2.n_islands
        np.testing.assert_array_equal(lwe1.island_masks, lwe2.island_masks)
    
    def test_cache_miss_different_threshold(self):
        """Different threshold should cause cache miss."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        # Clear cache
        LowWindEffect.clear_cache()
        
        # First instance
        lwe1 = LowWindEffect(pupil, pupil_threshold=0.99)
        cache_size_1 = LowWindEffect.get_cache_size()
        
        # Second instance with different threshold
        lwe2 = LowWindEffect(pupil, pupil_threshold=0.5)
        cache_size_2 = LowWindEffect.get_cache_size()
        
        # Cache size should increase (different key)
        assert cache_size_2 == cache_size_1 + 1
    
    def test_cache_miss_different_connectivity(self):
        """Different connectivity should cause cache miss."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        # Clear cache
        LowWindEffect.clear_cache()
        
        # First instance
        lwe1 = LowWindEffect(pupil, connectivity=1)
        cache_size_1 = LowWindEffect.get_cache_size()
        
        # Second instance with different connectivity
        lwe2 = LowWindEffect(pupil, connectivity=2)
        cache_size_2 = LowWindEffect.get_cache_size()
        
        # Cache size should increase
        assert cache_size_2 == cache_size_1 + 1
    
    def test_cache_disabled(self):
        """Cache can be disabled via enable_cache=False."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        # Clear cache
        LowWindEffect.clear_cache()
        
        # Create instance with cache disabled
        lwe = LowWindEffect(pupil, enable_cache=False)
        
        # Cache should remain empty
        assert LowWindEffect.get_cache_size() == 0
        assert lwe.info['cache_enabled'] is False
    
    def test_clear_cache(self):
        """clear_cache() should empty the cache."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        # Create instance (populates cache)
        lwe = LowWindEffect(pupil)
        assert LowWindEffect.get_cache_size() > 0
        
        # Clear cache
        LowWindEffect.clear_cache()
        assert LowWindEffect.get_cache_size() == 0


class TestPrecomputedMasks:
    """Test using precomputed island masks."""
    
    def test_precomputed_masks_basic(self):
        """Precomputed masks should skip island detection."""
        n_pix = 32
        n_islands = 2
        
        # Create dummy masks
        masks = np.zeros((n_islands, n_pix, n_pix), dtype=np.float32)
        masks[0, :n_pix//2, :] = 1.0  # Top half
        masks[1, n_pix//2:, :] = 1.0  # Bottom half
        
        # Create dummy pupil (will be ignored)
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        # Create LWE with precomputed masks
        lwe = LowWindEffect(pupil, precomputed_island_masks=masks)
        
        assert lwe.n_islands == n_islands
        np.testing.assert_array_equal(lwe.island_masks, masks)
    
    def test_precomputed_masks_skip_cache(self):
        """Precomputed masks should not populate cache."""
        n_pix = 32
        masks = np.zeros((2, n_pix, n_pix), dtype=np.float32)
        masks[0, :n_pix//2, :] = 1.0
        masks[1, n_pix//2:, :] = 1.0
        
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        # Clear cache
        LowWindEffect.clear_cache()
        
        # Create with precomputed masks
        lwe = LowWindEffect(pupil, precomputed_island_masks=masks, enable_cache=True)
        
        # Cache should remain empty (detection was skipped)
        assert LowWindEffect.get_cache_size() == 0
    
    def test_precomputed_masks_generate(self):
        """Generate should work correctly with precomputed masks."""
        n_pix = 32
        masks = np.zeros((2, n_pix, n_pix), dtype=np.float32)
        masks[0, :n_pix//2, :] = 1.0
        masks[1, n_pix//2:, :] = 1.0
        
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        lwe = LowWindEffect(pupil, precomputed_island_masks=masks)
        phase = lwe.generate(n_screens=5, seed=42)
        
        # Check output shape
        assert phase.shape == (5, n_pix, n_pix)
        
        # Phase should be different between islands
        # Check mean phase in top vs bottom half
        phase_top_mean = phase[:, :n_pix//2, :].mean(axis=(1, 2))
        phase_bottom_mean = phase[:, n_pix//2:, :].mean(axis=(1, 2))
        
        # At least one screen should have different mean values (differential piston)
        assert not np.allclose(phase_top_mean, phase_bottom_mean)


class TestConnectivity:
    """Test connectivity parameter for island detection."""
    
    def test_connectivity_1_vs_2(self):
        """Different connectivity values should affect island detection."""
        n_pix = 32
        pupil = np.zeros((n_pix, n_pix), dtype=np.float32)
        
        # Create diagonal pattern (only diagonally connected)
        for i in range(n_pix // 2):
            pupil[i, i] = 1.0
            pupil[n_pix - 1 - i, i] = 1.0
        
        # 4-connected (connectivity=1): diagonals are separate islands
        lwe_4 = LowWindEffect(pupil, connectivity=1)
        
        # 8-connected (connectivity=2): diagonals are one island
        lwe_8 = LowWindEffect(pupil, connectivity=2)
        
        # With 8-connectivity, should have fewer islands
        assert lwe_8.n_islands <= lwe_4.n_islands
    
    def test_connectivity_invalid(self):
        """Invalid connectivity should raise error."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        with pytest.raises(ValueError, match="connectivity must be 1 or 2"):
            LowWindEffect(pupil, connectivity=3)
    
    def test_connectivity_in_info(self):
        """Connectivity should be reported in info dict."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        lwe = LowWindEffect(pupil, connectivity=2)
        info = lwe.info
        
        assert 'connectivity' in info
        assert info['connectivity'] == 2


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_old_style_init(self):
        """Old-style init (no threshold/connectivity) should work."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        # Old-style call (positional args only)
        lwe = LowWindEffect(pupil, 0.5, 0.3, 0.95)
        
        assert lwe.piston_rms == 0.5
        assert lwe.tilt_rms == 0.3
        assert lwe.ar_coeff == 0.95
        assert lwe.pupil_threshold == 0.99  # Default
        assert lwe.connectivity == 1  # Default
    
    def test_generate_unchanged(self):
        """Generate method signature unchanged."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        lwe = LowWindEffect(pupil)
        
        # Old-style generate calls
        phase1 = lwe.generate(10)
        phase2 = lwe.generate(5, seed=42)
        
        assert phase1.shape == (10, n_pix, n_pix)
        assert phase2.shape == (5, n_pix, n_pix)


class TestGPUSupport:
    """Test GPU support (if CuPy available)."""
    
    def test_gpu_basic(self):
        """GPU arrays should work correctly."""
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not available")
        
        n_pix = 32
        pupil = cp.ones((n_pix, n_pix), dtype=cp.float32)
        
        lwe = LowWindEffect(pupil)
        phase = lwe.generate(5)
        
        # Output should be CuPy array
        assert isinstance(phase, cp.ndarray)
        assert phase.shape == (5, n_pix, n_pix)
    
    def test_gpu_zero_islands(self):
        """GPU should handle zero islands correctly."""
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not available")
        
        n_pix = 32
        pupil = cp.ones((n_pix, n_pix), dtype=cp.float32) * 0.5
        
        lwe = LowWindEffect(pupil, pupil_threshold=0.99)
        phase = lwe.generate(3)
        
        # Should return zeros on GPU
        assert isinstance(phase, cp.ndarray)
        assert cp.allclose(phase, 0.0)
        assert not cp.any(cp.isnan(phase))


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_pixel_island(self):
        """Single-pixel islands should work correctly."""
        n_pix = 16
        pupil = np.zeros((n_pix, n_pix), dtype=np.float32)
        pupil[n_pix//2, n_pix//2] = 1.0  # Single pixel
        
        lwe = LowWindEffect(pupil, pupil_threshold=0.5)
        phase = lwe.generate(3)
        
        assert phase.shape == (3, n_pix, n_pix)
        assert not np.any(np.isnan(phase))
    
    def test_large_n_islands(self):
        """Many islands should work correctly."""
        n_pix = 32
        pupil = np.zeros((n_pix, n_pix), dtype=np.float32)
        
        # Create grid of isolated pixels (many islands)
        pupil[::4, ::4] = 1.0
        
        lwe = LowWindEffect(pupil, pupil_threshold=0.5, connectivity=1)
        
        # Should detect many islands
        assert lwe.n_islands >= 4
        
        # Generate should work
        phase = lwe.generate(2)
        assert phase.shape == (2, n_pix, n_pix)
    
    def test_seed_reproducibility(self):
        """Same seed should produce identical results."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        lwe = LowWindEffect(pupil)
        
        phase1 = lwe.generate(5, seed=42)
        phase2 = lwe.generate(5, seed=42)
        
        np.testing.assert_array_almost_equal(phase1, phase2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

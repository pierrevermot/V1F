"""
Tests for PSF engine batch normalization bug.

Tests that the normalize parameter is correctly respected in compute_psf_batch.
"""

import numpy as np
import pytest
from nebraa.physics.psf_engine import PSFEngine


class TestBatchNormalizationBug:
    """Test that normalize flag is correctly respected in batch API."""
    
    def test_normalize_false_returns_unnormalized(self):
        """With normalize=False, PSFs should NOT sum to 1."""
        n_pix = 64
        
        # Create simple circular pupil
        y, x = np.ogrid[:n_pix, :n_pix]
        center = (n_pix - 1) / 2.0
        r = np.sqrt((x - center)**2 + (y - center)**2)
        pupil = (r <= 20).astype(np.float32)
        
        # Create random phase screens
        np.random.seed(42)
        phases = np.random.randn(5, n_pix, n_pix).astype(np.float32) * 0.5
        
        # Compute with normalize=False
        engine = PSFEngine(n_pix, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to="sum")
        psf_avg = engine.compute_psf_batch(pupil, phases, normalize=False)
        
        # Should NOT sum to 1 (unnormalized)
        total = float(np.sum(psf_avg))
        print(f"PSF sum with normalize=False: {total}")
        
        # If normalized, sum would be ~1.0
        # If unnormalized, sum should be much larger (raw FFT power)
        assert not np.isclose(total, 1.0, rtol=0.01), \
            f"PSF sum={total} is close to 1.0, but normalize=False was passed!"
        assert total > 10.0, \
            f"Expected unnormalized PSF to have sum >> 1, got {total}"
    
    def test_normalize_true_returns_normalized(self):
        """With normalize=True, PSFs SHOULD sum to 1."""
        n_pix = 64
        
        # Create simple circular pupil
        y, x = np.ogrid[:n_pix, :n_pix]
        center = (n_pix - 1) / 2.0
        r = np.sqrt((x - center)**2 + (y - center)**2)
        pupil = (r <= 20).astype(np.float32)
        
        # Create random phase screens
        np.random.seed(42)
        phases = np.random.randn(5, n_pix, n_pix).astype(np.float32) * 0.5
        
        # Compute with normalize=True (default)
        engine = PSFEngine(n_pix, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to="sum")
        psf_avg = engine.compute_psf_batch(pupil, phases, normalize=True)
        
        # Should sum to 1 (normalized)
        total = float(np.sum(psf_avg))
        print(f"PSF sum with normalize=True: {total}")
        
        assert np.isclose(total, 1.0, rtol=0.01), \
            f"Expected normalized PSF to sum to 1.0, got {total}"
    
    def test_normalize_to_none_returns_unnormalized(self):
        """Engine with normalize_to='none' should return unnormalized PSFs."""
        n_pix = 64
        
        # Create simple circular pupil
        y, x = np.ogrid[:n_pix, :n_pix]
        center = (n_pix - 1) / 2.0
        r = np.sqrt((x - center)**2 + (y - center)**2)
        pupil = (r <= 20).astype(np.float32)
        
        # Create random phase screens
        np.random.seed(42)
        phases = np.random.randn(5, n_pix, n_pix).astype(np.float32) * 0.5
        
        # Engine with normalize_to="none"
        engine = PSFEngine(n_pix, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to="none")
        psf_avg = engine.compute_psf_batch(pupil, phases, normalize=True)
        
        # Should NOT sum to 1 (unnormalized even with normalize=True)
        total = float(np.sum(psf_avg))
        print(f"PSF sum with normalize_to='none': {total}")
        
        assert not np.isclose(total, 1.0, rtol=0.01), \
            f"PSF sum={total} is close to 1.0, but normalize_to='none' was set!"
        assert total > 10.0, \
            f"Expected unnormalized PSF to have sum >> 1, got {total}"
    
    def test_individual_psfs_normalized_correctly(self):
        """Individual PSFs in batch should respect normalize flag."""
        n_pix = 64
        
        # Create simple circular pupil
        y, x = np.ogrid[:n_pix, :n_pix]
        center = (n_pix - 1) / 2.0
        r = np.sqrt((x - center)**2 + (y - center)**2)
        pupil = (r <= 20).astype(np.float32)
        
        # Create random phase screens
        np.random.seed(42)
        phases = np.random.randn(5, n_pix, n_pix).astype(np.float32) * 0.5
        
        # Compute with normalize=False and return_individual=True
        engine = PSFEngine(n_pix, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to="sum")
        psf_avg, psfs_individual = engine.compute_psf_batch(
            pupil, phases, normalize=False, return_individual=True
        )
        
        # Check individual PSFs are unnormalized
        for i, psf in enumerate(psfs_individual):
            total = float(np.sum(psf))
            print(f"Individual PSF {i} sum: {total}")
            assert not np.isclose(total, 1.0, rtol=0.01), \
                f"Individual PSF {i} sum={total} close to 1.0, but normalize=False!"
            assert total > 10.0, \
                f"Individual PSF {i} should be unnormalized, got sum={total}"
        
        # Check average is also unnormalized
        avg_total = float(np.sum(psf_avg))
        assert not np.isclose(avg_total, 1.0, rtol=0.01), \
            f"Average PSF sum={avg_total} close to 1.0, but normalize=False!"
    
    def test_normalize_to_peak(self):
        """Test normalize_to='peak' mode."""
        n_pix = 64
        
        # Create simple circular pupil
        y, x = np.ogrid[:n_pix, :n_pix]
        center = (n_pix - 1) / 2.0
        r = np.sqrt((x - center)**2 + (y - center)**2)
        pupil = (r <= 20).astype(np.float32)
        
        # Create random phase screens
        np.random.seed(42)
        phases = np.random.randn(3, n_pix, n_pix).astype(np.float32) * 0.5
        
        # Engine with normalize_to="peak"
        engine = PSFEngine(n_pix, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to="peak")
        psf_avg = engine.compute_psf_batch(pupil, phases, normalize=True)
        
        # Peak should be 1.0
        peak = float(np.max(psf_avg))
        print(f"PSF peak with normalize_to='peak': {peak}")
        
        assert np.isclose(peak, 1.0, rtol=0.01), \
            f"Expected peak-normalized PSF to have peak=1.0, got {peak}"
        
        # Sum should NOT be 1.0 (different normalization)
        total = float(np.sum(psf_avg))
        assert not np.isclose(total, 1.0, rtol=0.01), \
            f"Peak-normalized PSF should not sum to 1.0, got sum={total}"
    
    def test_consistency_with_single_psf(self):
        """Batch of 1 should match single PSF computation."""
        n_pix = 64
        
        # Create simple circular pupil
        y, x = np.ogrid[:n_pix, :n_pix]
        center = (n_pix - 1) / 2.0
        r = np.sqrt((x - center)**2 + (y - center)**2)
        pupil = (r <= 20).astype(np.float32)
        
        # Single phase screen
        np.random.seed(42)
        phase = np.random.randn(n_pix, n_pix).astype(np.float32) * 0.5
        phases = phase[None, :, :]  # Batch of 1
        
        engine = PSFEngine(n_pix, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to="sum")
        
        # Single PSF with normalize=False
        psf_single = engine.compute_psf(pupil, phase, normalize=False)
        
        # Batch of 1 with normalize=False
        psf_batch = engine.compute_psf_batch(pupil, phases, normalize=False)
        
        # Convert to numpy for comparison (handle both CPU and GPU)
        try:
            import cupy as cp
            if isinstance(psf_single, cp.ndarray):
                psf_single = cp.asnumpy(psf_single)
            if isinstance(psf_batch, cp.ndarray):
                psf_batch = cp.asnumpy(psf_batch)
        except ImportError:
            pass
        
        psf_single = np.asarray(psf_single)
        psf_batch = np.asarray(psf_batch)
        
        # Should be identical
        np.testing.assert_allclose(psf_single, psf_batch, rtol=1e-5)
        
        # Both should be unnormalized
        total_single = float(np.sum(psf_single))
        total_batch = float(np.sum(psf_batch))
        
        assert total_single > 10.0, f"Single PSF should be unnormalized, got sum={total_single}"
        assert total_batch > 10.0, f"Batch PSF should be unnormalized, got sum={total_batch}"


class TestBatchNormalizationEdgeCases:
    """Test edge cases in batch normalization."""
    
    def test_long_exposure_psf_normalize_false(self):
        """Test compute_long_exposure_psf with normalize=False."""
        n_pix = 64
        
        y, x = np.ogrid[:n_pix, :n_pix]
        center = (n_pix - 1) / 2.0
        r = np.sqrt((x - center)**2 + (y - center)**2)
        pupil = (r <= 20).astype(np.float32)
        
        # Random phase screens
        np.random.seed(42)
        phases = np.random.randn(5, n_pix, n_pix).astype(np.float32) * 0.5
        
        engine = PSFEngine(n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        
        # With normalize=False (NEW - was hardcoded to True before)
        psf_unnorm = engine.compute_long_exposure_psf(pupil, phases, normalize=False)
        total = float(np.sum(psf_unnorm))
        print(f"Long-exposure PSF sum with normalize=False: {total}")
        
        assert not np.isclose(total, 1.0, rtol=0.01), \
            f"Long-exposure PSF sum={total} close to 1.0, but normalize=False!"
        assert total > 10.0, \
            f"Expected unnormalized long-exposure PSF to have sum >> 1, got {total}"
        
        # With normalize=True (default)
        psf_norm = engine.compute_long_exposure_psf(pupil, phases, normalize=True)
        total_norm = float(np.sum(psf_norm))
        assert np.isclose(total_norm, 1.0, rtol=0.01), \
            f"Expected normalized PSF to sum to 1.0, got {total_norm}"
    
    def test_diffraction_limited_psf_normalize_false(self):
        """Test compute_diffraction_limited_psf with normalize=False."""
        n_pix = 64
        
        y, x = np.ogrid[:n_pix, :n_pix]
        center = (n_pix - 1) / 2.0
        r = np.sqrt((x - center)**2 + (y - center)**2)
        pupil = (r <= 20).astype(np.float32)
        
        engine = PSFEngine(n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        
        # With normalize=False (NEW - was hardcoded to True before)
        psf_unnorm = engine.compute_diffraction_limited_psf(pupil, normalize=False)
        total = float(np.sum(psf_unnorm))
        print(f"DL PSF sum with normalize=False: {total}")
        
        assert not np.isclose(total, 1.0, rtol=0.01), \
            f"DL PSF sum={total} close to 1.0, but normalize=False!"
        assert total > 10.0, \
            f"Expected unnormalized DL PSF to have sum >> 1, got {total}"
        
        # With normalize=True (default)
        psf_norm = engine.compute_diffraction_limited_psf(pupil, normalize=True)
        total_norm = float(np.sum(psf_norm))
        assert np.isclose(total_norm, 1.0, rtol=0.01), \
            f"Expected normalized DL PSF to sum to 1.0, got {total_norm}"
    
    def test_long_exposure_with_lwe_normalize(self):
        """Test compute_long_exposure_psf with LWE phases respects normalize flag."""
        n_pix = 64
        
        y, x = np.ogrid[:n_pix, :n_pix]
        center = (n_pix - 1) / 2.0
        r = np.sqrt((x - center)**2 + (y - center)**2)
        pupil = (r <= 20).astype(np.float32)
        
        # Random phase screens
        np.random.seed(42)
        phases = np.random.randn(5, n_pix, n_pix).astype(np.float32) * 0.5
        phases_lwe = np.random.randn(5, n_pix, n_pix).astype(np.float32) * 0.2
        
        engine = PSFEngine(n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        
        # With normalize=False
        psf_unnorm = engine.compute_long_exposure_psf(
            pupil, phases, phases_lwe=phases_lwe, normalize=False
        )
        total = float(np.sum(psf_unnorm))
        
        assert not np.isclose(total, 1.0, rtol=0.01), \
            f"LWE PSF sum={total} close to 1.0, but normalize=False!"
        
        # With normalize=True
        psf_norm = engine.compute_long_exposure_psf(
            pupil, phases, phases_lwe=phases_lwe, normalize=True
        )
        total_norm = float(np.sum(psf_norm))
        assert np.isclose(total_norm, 1.0, rtol=0.01)
    
    def test_zero_phase_screens(self):
        """Test with zero phase (diffraction-limited)."""
        n_pix = 64
        
        y, x = np.ogrid[:n_pix, :n_pix]
        center = (n_pix - 1) / 2.0
        r = np.sqrt((x - center)**2 + (y - center)**2)
        pupil = (r <= 20).astype(np.float32)
        
        # Zero phases (diffraction-limited)
        phases = np.zeros((3, n_pix, n_pix), dtype=np.float32)
        
        engine = PSFEngine(n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        
        # With normalize=True
        psf_norm = engine.compute_psf_batch(pupil, phases, normalize=True)
        assert np.isclose(np.sum(psf_norm), 1.0, rtol=0.01)
        
        # With normalize=False
        psf_unnorm = engine.compute_psf_batch(pupil, phases, normalize=False)
        assert not np.isclose(np.sum(psf_unnorm), 1.0, rtol=0.01)
        assert np.sum(psf_unnorm) > 10.0
    
    def test_large_phase_rms(self):
        """Test with large phase RMS (heavily aberrated)."""
        n_pix = 64
        
        y, x = np.ogrid[:n_pix, :n_pix]
        center = (n_pix - 1) / 2.0
        r = np.sqrt((x - center)**2 + (y - center)**2)
        pupil = (r <= 20).astype(np.float32)
        
        # Large phase RMS
        np.random.seed(42)
        phases = np.random.randn(5, n_pix, n_pix).astype(np.float32) * 5.0  # ~5 rad RMS
        
        engine = PSFEngine(n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        
        # With normalize=True, should still sum to 1
        psf_norm = engine.compute_psf_batch(pupil, phases, normalize=True)
        assert np.isclose(np.sum(psf_norm), 1.0, rtol=0.01)
        
        # With normalize=False, should be unnormalized
        psf_unnorm = engine.compute_psf_batch(pupil, phases, normalize=False)
        assert not np.isclose(np.sum(psf_unnorm), 1.0, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

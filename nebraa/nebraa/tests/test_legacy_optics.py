"""
Tests for legacy optics module consistency with PSFEngine.

Verifies that legacy optics.compute_psf* functions produce results
consistent with PSFEngine implementations.
"""

import numpy as np
import pytest
import warnings
from nebraa.physics.optics import (
    compute_psf,
    compute_psf_batch,
    compute_reference_psf,
)
from nebraa.physics.psf_engine import PSFEngine
from nebraa.utils.compute import init_backend


@pytest.fixture(autouse=True, scope="module")
def force_cpu_backend():
    """Force CPU backend for reproducibility."""
    backend = init_backend(compute_mode="CPU")
    yield backend


class TestLegacyOpticsDeprecation:
    """Test that deprecation warnings are emitted."""
    
    def test_compute_psf_deprecation_warning(self):
        """compute_psf should emit deprecation warning."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        phase = np.zeros((n_pix, n_pix), dtype=np.float32)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = compute_psf(pupil, phase)
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "PSFEngine" in str(w[0].message)
    
    def test_compute_psf_batch_deprecation_warning(self):
        """compute_psf_batch should emit deprecation warning."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        phases = np.zeros((3, n_pix, n_pix), dtype=np.float32)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = compute_psf_batch(pupil, phases)
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "PSFEngine" in str(w[0].message)
    
    def test_compute_reference_psf_deprecation_warning(self):
        """compute_reference_psf should emit deprecation warning."""
        n_pix = 32
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = compute_reference_psf(pupil)
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "PSFEngine" in str(w[0].message)


class TestLegacyOpticsConsistency:
    """Test that legacy optics functions match PSFEngine results."""
    
    def test_compute_psf_matches_psf_engine_normalize_sum(self):
        """Legacy compute_psf should match PSFEngine with normalize_to='sum'."""
        n_pix = 64
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        phase = np.random.randn(n_pix, n_pix).astype(np.float32)
        
        # Legacy path with normalize_to='sum'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psf_legacy = compute_psf(pupil, phase, normalize=True, normalize_to="sum")
        
        # PSFEngine path (always normalizes to sum=1)
        engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        psf_engine = engine.compute_psf(pupil, phase, normalize=True)
        
        # Should be identical
        np.testing.assert_allclose(psf_legacy, psf_engine, rtol=1e-5, atol=1e-7)
        
        # Check normalization
        assert np.isclose(np.sum(psf_legacy), 1.0, rtol=1e-5)
        assert np.isclose(np.sum(psf_engine), 1.0, rtol=1e-5)
    
    def test_compute_psf_matches_psf_engine_normalize_peak(self):
        """Legacy compute_psf with normalize_to='peak' should work correctly."""
        n_pix = 64
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        phase = np.random.randn(n_pix, n_pix).astype(np.float32)
        
        # Legacy path with normalize_to='peak'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psf_legacy = compute_psf(pupil, phase, normalize=True, normalize_to="peak")
        
        # Check peak normalization
        assert np.isclose(np.max(psf_legacy), 1.0, rtol=1e-5)
    
    def test_compute_psf_no_normalize(self):
        """Legacy compute_psf with normalize=False should work."""
        n_pix = 64
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        phase = np.random.randn(n_pix, n_pix).astype(np.float32)
        
        # Legacy path without normalization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psf_legacy = compute_psf(pupil, phase, normalize=False)
        
        # PSFEngine path without normalization
        engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        psf_engine = engine.compute_psf(pupil, phase, normalize=False)
        
        # Should be identical
        np.testing.assert_allclose(psf_legacy, psf_engine, rtol=1e-5, atol=1e-7)
    
    def test_compute_psf_batch_matches_psf_engine(self):
        """Legacy compute_psf_batch should match PSFEngine individual PSFs."""
        n_pix = 64
        n_screens = 5
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        phases = np.random.randn(n_screens, n_pix, n_pix).astype(np.float32)
        
        # Legacy path (returns individual PSFs, normalized to sum)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psfs_legacy = compute_psf_batch(pupil, phases, normalize=True, normalize_to="sum")
        
        # PSFEngine path (get individual PSFs)
        engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        _, psfs_engine = engine.compute_psf_batch(
            pupil, phases, normalize=True, return_individual=True
        )
        
        # Should be very close
        np.testing.assert_allclose(psfs_legacy, psfs_engine, rtol=1e-5, atol=1e-7)
        
        # Check normalization
        for i in range(n_screens):
            assert np.isclose(np.sum(psfs_legacy[i]), 1.0, rtol=1e-5)
            assert np.isclose(np.sum(psfs_engine[i]), 1.0, rtol=1e-5)
    
    def test_compute_psf_batch_peak_normalization(self):
        """Legacy compute_psf_batch with peak normalization."""
        n_pix = 64
        n_screens = 5
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        phases = np.random.randn(n_screens, n_pix, n_pix).astype(np.float32)
        
        # Legacy path with peak normalization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psfs_legacy = compute_psf_batch(pupil, phases, normalize=True, normalize_to="peak")
        
        # Check each PSF has peak=1
        for i in range(n_screens):
            assert np.isclose(np.max(psfs_legacy[i]), 1.0, rtol=1e-5)
    
    def test_compute_reference_psf_matches_psf_engine(self):
        """Legacy compute_reference_psf should match PSFEngine DL PSF."""
        n_pix = 64
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        # Legacy path
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psf_legacy = compute_reference_psf(pupil)
        
        # PSFEngine path (phase=None gives DL PSF)
        engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        psf_engine = engine.compute_psf(pupil, phase=None, normalize=True)
        
        # Should be identical
        np.testing.assert_allclose(psf_legacy, psf_engine, rtol=1e-5, atol=1e-7)
        
        # Check normalization
        assert np.isclose(np.sum(psf_legacy), 1.0, rtol=1e-5)


class TestLegacyOpticsShapes:
    """Test that legacy functions handle various input shapes correctly."""
    
    def test_compute_psf_different_sizes(self):
        """Test different pupil sizes."""
        for n_pix in [32, 64, 128]:
            pupil = np.ones((n_pix, n_pix), dtype=np.float32)
            phase = np.zeros((n_pix, n_pix), dtype=np.float32)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                psf = compute_psf(pupil, phase)
            
            assert psf.shape == (n_pix, n_pix)
    
    def test_compute_psf_batch_different_batch_sizes(self):
        """Test different batch sizes."""
        n_pix = 64
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        for n_screens in [1, 5, 10, 20]:
            phases = np.zeros((n_screens, n_pix, n_pix), dtype=np.float32)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                psfs = compute_psf_batch(pupil, phases)
            
            assert psfs.shape == (n_screens, n_pix, n_pix)


class TestLegacyOpticsNumericalAccuracy:
    """Test numerical accuracy of legacy implementations."""
    
    def test_compute_psf_energy_conservation(self):
        """PSF should conserve energy with sum normalization."""
        n_pix = 64
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        phase = np.random.randn(n_pix, n_pix).astype(np.float32) * 0.5
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psf = compute_psf(pupil, phase, normalize=True, normalize_to="sum")
        
        # Total energy should be 1.0
        assert np.isclose(np.sum(psf), 1.0, rtol=1e-6)
    
    def test_compute_psf_batch_energy_conservation(self):
        """Each PSF in batch should conserve energy."""
        n_pix = 64
        n_screens = 5
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        phases = np.random.randn(n_screens, n_pix, n_pix).astype(np.float32) * 0.5
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psfs = compute_psf_batch(pupil, phases, normalize=True, normalize_to="sum")
        
        # Each PSF should have sum=1
        for i in range(n_screens):
            assert np.isclose(np.sum(psfs[i]), 1.0, rtol=1e-6)
    
    def test_diffraction_limited_psf_peak_location(self):
        """DL PSF should have peak at center."""
        n_pix = 64
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psf = compute_reference_psf(pupil)
        
        # Peak should be at or very near center
        peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
        center = n_pix // 2
        
        # Peak should be within 2 pixels of center
        assert abs(peak_idx[0] - center) <= 2
        assert abs(peak_idx[1] - center) <= 2



class TestBackwardCompatibility:
    """Test backward compatibility of legacy API."""
    
    def test_normalize_false_sets_normalize_to_none(self):
        """normalize=False should result in no normalization."""
        n_pix = 64
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        phase = np.random.randn(n_pix, n_pix).astype(np.float32)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psf_no_norm = compute_psf(pupil, phase, normalize=False)
            psf_none = compute_psf(pupil, phase, normalize_to="none")
        
        # Should be identical
        np.testing.assert_array_equal(psf_no_norm, psf_none)
    
    def test_batch_size_parameter_ignored(self):
        """batch_size parameter should be ignored (deprecated)."""
        n_pix = 64
        n_screens = 5
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        phases = np.zeros((n_screens, n_pix, n_pix), dtype=np.float32)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psfs_no_batch = compute_psf_batch(pupil, phases)
            psfs_with_batch = compute_psf_batch(pupil, phases, batch_size=2)
        
        # Should be identical (batch_size ignored)
        np.testing.assert_array_equal(psfs_no_batch, psfs_with_batch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

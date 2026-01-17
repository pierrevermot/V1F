"""
Tests for optics module.
"""

import numpy as np
import pytest


class TestPSF:
    """Tests for PSF computation."""
    
    def test_compute_psf_shape(self, backend):
        """Test PSF output shape."""
        from nebraa.physics.optics import compute_psf
        
        xp = backend.xp
        n_pix = 64
        
        # Simple circular pupil
        x = xp.linspace(-1, 1, n_pix)
        xx, yy = xp.meshgrid(x, x)
        pupil = ((xx**2 + yy**2) < 0.8**2).astype(xp.float32)
        phase = xp.zeros((n_pix, n_pix), dtype=xp.float32)
        
        psf = compute_psf(pupil, phase)
        
        assert psf.shape == (n_pix, n_pix)
    
    def test_psf_is_positive(self, backend):
        """Test PSF is positive."""
        from nebraa.physics.optics import compute_psf
        
        xp = backend.xp
        n_pix = 64
        
        x = xp.linspace(-1, 1, n_pix)
        xx, yy = xp.meshgrid(x, x)
        pupil = ((xx**2 + yy**2) < 1).astype(xp.float32)
        phase = xp.zeros((n_pix, n_pix), dtype=xp.float32)
        
        psf = compute_psf(pupil, phase)
        
        assert xp.all(psf >= 0)
    
    def test_psf_normalization(self, backend):
        """Test PSF peak normalization."""
        from nebraa.physics.optics import compute_psf
        
        xp = backend.xp
        n_pix = 64
        
        x = xp.linspace(-1, 1, n_pix)
        xx, yy = xp.meshgrid(x, x)
        pupil = ((xx**2 + yy**2) < 1).astype(xp.float32)
        phase = xp.zeros((n_pix, n_pix), dtype=xp.float32)
        
        psf = compute_psf(pupil, phase, normalize=True)
        
        max_val = backend.to_numpy(xp.max(psf))
        np.testing.assert_almost_equal(max_val, 1.0, decimal=5)
    
    def test_psf_batch(self, backend):
        """Test batch PSF computation."""
        from nebraa.physics.optics import compute_psf_batch
        
        xp = backend.xp
        n_pix = 64
        n_screens = 5
        
        x = xp.linspace(-1, 1, n_pix)
        xx, yy = xp.meshgrid(x, x)
        pupil = ((xx**2 + yy**2) < 1).astype(xp.float32)
        
        # Random phase screens
        phase_screens = xp.random.randn(n_screens, n_pix, n_pix).astype(xp.float32) * 0.5
        
        psfs = compute_psf_batch(pupil, phase_screens)
        
        assert psfs.shape == (n_screens, n_pix, n_pix)


class TestStrehl:
    """Tests for Strehl ratio computation."""
    
    def test_perfect_strehl(self, backend):
        """Test Strehl=1 for flat wavefront."""
        from nebraa.physics.optics import compute_strehl
        
        xp = backend.xp
        n_pix = 64
        
        x = xp.linspace(-1, 1, n_pix)
        xx, yy = xp.meshgrid(x, x)
        pupil = ((xx**2 + yy**2) < 1).astype(xp.float32)
        phase = xp.zeros((n_pix, n_pix), dtype=xp.float32)
        
        strehl = compute_strehl(phase, pupil)
        
        np.testing.assert_almost_equal(strehl, 1.0, decimal=3)
    
    def test_aberrated_strehl(self, backend):
        """Test Strehl < 1 for aberrated wavefront."""
        from nebraa.physics.optics import compute_strehl
        
        xp = backend.xp
        n_pix = 64
        
        x = xp.linspace(-1, 1, n_pix)
        xx, yy = xp.meshgrid(x, x)
        pupil = ((xx**2 + yy**2) < 1).astype(xp.float32)
        
        # Add some wavefront error
        phase = (0.5 * (xx**2 + yy**2)).astype(xp.float32)
        
        strehl = compute_strehl(phase, pupil)
        
        # Strehl should be reduced but positive
        assert 0 < strehl < 1


class TestRMSPhase:
    """Tests for RMS phase computation."""
    
    def test_zero_rms(self, backend):
        """Test RMS=0 for constant phase."""
        from nebraa.physics.optics import compute_rms_phase
        
        xp = backend.xp
        n_pix = 64
        
        x = xp.linspace(-1, 1, n_pix)
        xx, yy = xp.meshgrid(x, x)
        pupil = ((xx**2 + yy**2) < 1).astype(xp.float32)
        phase = xp.ones((n_pix, n_pix), dtype=xp.float32) * 0.5  # Constant
        
        rms = compute_rms_phase(phase, pupil)
        
        np.testing.assert_almost_equal(rms, 0.0, decimal=5)
    
    def test_nonzero_rms(self, backend):
        """Test RMS > 0 for varying phase."""
        from nebraa.physics.optics import compute_rms_phase
        
        xp = backend.xp
        n_pix = 64
        
        x = xp.linspace(-1, 1, n_pix)
        xx, yy = xp.meshgrid(x, x)
        pupil = ((xx**2 + yy**2) < 1).astype(xp.float32)
        
        # Varying phase
        phase = (xx + yy).astype(xp.float32)
        
        rms = compute_rms_phase(phase, pupil)
        
        assert rms > 0

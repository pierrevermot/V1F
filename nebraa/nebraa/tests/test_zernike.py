"""
Tests for Zernike polynomials.
"""

import numpy as np
import pytest


class TestZernikeFunctions:
    """Tests for Zernike conversion functions."""
    
    def test_noll_to_nm(self):
        """Test Noll index conversion."""
        from nebraa.physics.zernike import noll_to_nm
        
        # Test known values
        # Noll 1 -> (0, 0) piston
        n, m = noll_to_nm(1)
        assert (n, m) == (0, 0)
        
        # Noll 2 -> (1, -1) or (1, 1) depending on convention - test that |m|=1
        n, m = noll_to_nm(2)
        assert n == 1 and abs(m) == 1
        
        # Noll 3 -> the other (1, ±1) mode
        n, m = noll_to_nm(3)
        assert n == 1 and abs(m) == 1
        
        # Noll 4 -> (2, 0) defocus
        n, m = noll_to_nm(4)
        assert (n, m) == (2, 0)
    
    def test_nm_to_noll(self):
        """Test (n,m) to Noll conversion."""
        from nebraa.physics.zernike import nm_to_noll, noll_to_nm
        
        # Round-trip test for first 15 modes
        for j in range(1, 16):
            n, m = noll_to_nm(j)
            j_back = nm_to_noll(n, m)
            assert j_back == j, f"Round-trip failed for j={j}: got {j_back}"
    
    def test_count_modes(self):
        """Test mode counting."""
        from nebraa.physics.zernike import count_modes
        
        # n=2: 3 modes, n=3: 4 modes, n=4: 5 modes
        assert count_modes(2, 5) == 3 + 4 + 5  # 12 modes
    
    def test_get_nm_list(self):
        """Test (n,m) list generation."""
        from nebraa.physics.zernike import get_nm_list
        
        nm_list = get_nm_list(2, 4)  # Orders 2 and 3
        
        # Order 2: (2,-2), (2,0), (2,2)
        # Order 3: (3,-3), (3,-1), (3,1), (3,3)
        assert len(nm_list) == 7


class TestZernikePolynomials:
    """Tests for Zernike polynomial computation."""
    
    def test_zernike_nm_shape(self, backend):
        """Test zernike_nm output shape."""
        from nebraa.physics.zernike import zernike_nm
        
        xp = backend.xp
        n_pix = 64
        
        # Create coordinate arrays
        c = (n_pix - 1) / 2.0
        idx = xp.arange(n_pix, dtype=xp.float32)
        X, Y = xp.meshgrid(idx - c, idx - c)
        rho = xp.sqrt(X**2 + Y**2) / (n_pix / 2.0)
        theta = xp.arctan2(Y, X)
        
        Z = zernike_nm(2, 0, rho, theta)  # Defocus
        
        assert Z.shape == (n_pix, n_pix)
    
    def test_piston_is_constant(self, backend):
        """Test that piston mode (0,0) is constant."""
        from nebraa.physics.zernike import zernike_nm
        
        xp = backend.xp
        n_pix = 64
        
        c = (n_pix - 1) / 2.0
        idx = xp.arange(n_pix, dtype=xp.float32)
        X, Y = xp.meshgrid(idx - c, idx - c)
        rho = xp.sqrt(X**2 + Y**2) / (n_pix / 2.0)
        theta = xp.arctan2(Y, X)
        
        # Piston (n=0, m=0)
        Z = zernike_nm(0, 0, rho, theta)
        
        # Within unit circle, should be constant
        mask = rho <= 1.0
        Z_in_pupil = Z[mask]
        
        # Check variance is near zero
        var = backend.to_numpy(xp.var(Z_in_pupil))
        assert var < 1e-6


class TestZernikeCache:
    """Tests for Zernike mode caching."""
    
    def test_build_zernike_modes_shape(self, backend):
        """Test building Zernike modes."""
        from nebraa.physics.zernike import build_zernike_modes
        
        modes = build_zernike_modes(n_pix=64, radius=1.0, n_range=(2, 5))
        
        # Orders 2, 3, 4 have 3+4+5=12 modes
        assert modes.shape[0] == 12
        assert modes.shape[1] == 64
        assert modes.shape[2] == 64
    
    def test_cache_reuse(self, backend):
        """Test that cache returns same object."""
        from nebraa.physics.zernike import build_zernike_modes, _cache
        
        # Clear cache first
        _cache.clear()
        
        modes1 = build_zernike_modes(n_pix=64, radius=1.0, n_range=(2, 4))
        modes2 = build_zernike_modes(n_pix=64, radius=1.0, n_range=(2, 4))
        
        # Should be the same object (cached)
        assert modes1 is modes2


class TestPhaseGeneration:
    """Tests for phase screen generation."""
    
    def test_generate_zernike_phase_shape(self, backend):
        """Test phase generation shape."""
        from nebraa.physics.zernike import generate_zernike_phase
        
        phase = generate_zernike_phase(
            n_screens=5,
            n_pix=64,
            radius=1.0,
            n_range=(2, 5),
            seed=42,
        )
        
        assert phase.shape == (5, 64, 64)
    
    def test_phase_statistics(self, backend):
        """Test that generated phases have expected statistics."""
        from nebraa.physics.zernike import generate_zernike_phase
        
        # Generate multiple screens
        phase = generate_zernike_phase(
            n_screens=100,
            n_pix=64,
            radius=1.0,
            n_range=(2, 5),
            seed=42,
        )
        
        phase_np = backend.to_numpy(phase)
        
        # Should have non-zero variance
        assert np.var(phase_np) > 0
    
    def test_normalize_phase_rms(self, backend):
        """Test phase RMS normalization."""
        from nebraa.physics.zernike import generate_zernike_phase, normalize_phase_rms
        
        xp = backend.xp
        n_pix = 64
        
        # Create simple pupil mask
        c = (n_pix - 1) / 2.0
        idx = xp.arange(n_pix, dtype=xp.float32)
        X, Y = xp.meshgrid(idx - c, idx - c)
        rho = xp.sqrt(X**2 + Y**2) / (n_pix / 2.0)
        pupil = (rho <= 1.0).astype(xp.float32)
        
        # Generate phase
        phase = generate_zernike_phase(
            n_screens=1,
            n_pix=n_pix,
            radius=1.0,
            n_range=(2, 5),
        )[0]
        
        # Normalize to target RMS
        target_rms = 0.5
        phase_norm = normalize_phase_rms(phase, pupil, target_rms)
        
        # Check RMS
        pupil_sum = backend.to_numpy(xp.sum(pupil))
        actual_rms = np.sqrt(backend.to_numpy(xp.sum(phase_norm**2 * pupil)) / pupil_sum)
        
        np.testing.assert_almost_equal(actual_rms, target_rms, decimal=3)

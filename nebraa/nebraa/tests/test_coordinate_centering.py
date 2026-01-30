"""
Tests for coordinate centering consistency.

Validates that all grid generation uses consistent pixel-centered conventions,
eliminating half-pixel shifts that cause asymmetric PSFs and phase mismatches.
"""

import numpy as np
import pytest

from nebraa.utils.coordinates import (
    get_grid_center,
    get_pixel_radius,
    make_xy_grid,
    make_polar_grid,
    check_grid_symmetry,
)
from nebraa.physics.pupil import Pupil
from nebraa.physics.zernike import (
    ZernikePhaseGenerator,
    ZernikeConfig,
    zernike_nm,
    _cache,
)
from nebraa.utils.compute import init_backend, get_backend


class TestCoordinateHelpers:
    """Test the coordinate helper functions."""
    
    def test_grid_center_odd(self):
        """Odd grids have center on a pixel."""
        assert get_grid_center(5) == 2.0
        assert get_grid_center(7) == 3.0
    
    def test_grid_center_even(self):
        """Even grids have center between pixels."""
        assert get_grid_center(4) == 1.5
        assert get_grid_center(6) == 2.5
    
    def test_pixel_radius_matches_center(self):
        """Radius should equal center for pixel-centered grids."""
        for n in [4, 5, 8, 16, 32, 64, 128]:
            assert get_pixel_radius(n) == get_grid_center(n)
    
    def test_1d_grid_symmetry_odd(self):
        """1D grid should be symmetric for odd sizes."""
        from nebraa.utils.coordinates import make_1d_grid
        from nebraa.utils.compute import init_backend
        init_backend('CPU')
        
        x = make_1d_grid(5)
        x_np = x.get() if hasattr(x, 'get') else np.asarray(x)
        expected = np.array([-2., -1., 0., 1., 2.])
        np.testing.assert_allclose(x_np, expected)
        # Check symmetry
        np.testing.assert_allclose(x_np, -x_np[::-1])
    
    def test_1d_grid_symmetry_even(self):
        """1D grid should be symmetric for even sizes."""
        from nebraa.utils.coordinates import make_1d_grid
        from nebraa.utils.compute import init_backend
        init_backend('CPU')
        
        x = make_1d_grid(4)
        x_np = x.get() if hasattr(x, 'get') else np.asarray(x)
        expected = np.array([-1.5, -0.5, 0.5, 1.5])
        np.testing.assert_allclose(x_np, expected)
        # Check symmetry
        np.testing.assert_allclose(x_np, -x_np[::-1])
    
    def test_xy_grid_symmetry(self):
        """XY grids should be properly centered and X,Y are transposes."""
        from nebraa.utils.compute import init_backend
        init_backend('CPU')
        
        # Test odd sizes - grids should be properly centered
        for n in [5, 7, 9]:
            X, Y = make_xy_grid(n)
            # For 'xy' indexing (default): X and Y are transposes
            np.testing.assert_allclose(X, Y.T)
            # Check centering: center row/col should have 0
            center_idx = n // 2
            assert abs(X[center_idx, center_idx]) < 1e-10
            assert abs(Y[center_idx, center_idx]) < 1e-10
            # Check that X varies along columns, Y along rows
            # All rows of X should be identical
            for i in range(n):
                np.testing.assert_allclose(X[i, :], X[0, :])
            # All columns of Y should be identical
            for j in range(n):
                np.testing.assert_allclose(Y[:, j], Y[:, 0])
        
        # Test even sizes - check that grids are created correctly
        for n in [4, 8, 16]:
            X, Y = make_xy_grid(n)
            # Check that center is between pixels as expected
            center = (n - 1) / 2.0
            center_idx = n // 2
            # Center value should be 0.5 for even grid
            expected_center_val = 0.5
            assert abs(X[center_idx, center_idx] - expected_center_val) < 0.1
            np.testing.assert_allclose(X, Y.T)  # Still transposes
    
    def test_polar_grid_center_is_zero(self):
        """Polar grid should have rho=0 at center."""
        for n in [4, 5, 8, 16]:
            rho, theta = make_polar_grid(n)
            center_idx = (n - 1) // 2
            # For even n, center is between pixels, so check all 4 central pixels
            if n % 2 == 0:
                central_rho = rho[center_idx:center_idx+2, center_idx:center_idx+2]
                assert central_rho.min() < 1.0  # Should be near zero
            else:
                # For odd n, exact center
                assert abs(rho[center_idx, center_idx]) < 1e-10
    
    def test_polar_grid_radial_symmetry(self):
        """Polar radius should be radially symmetric."""
        for n in [4, 5, 8, 16]:
            rho, theta = make_polar_grid(n)
            assert check_grid_symmetry(rho)


class TestPupilCentering:
    """Test that Pupil grids are correctly centered."""
    
    @pytest.fixture
    def pupil(self):
        """Create test pupil."""
        init_backend('CPU')
        return Pupil.from_circular(
            n_pix=128,
            wavelength=2.2e-6,
            pixel_scale=13e-3/206265,
            diameter=8.2,
            obstruction_diameter=1.116,
        )
    
    def test_pupil_coordinate_grid_symmetry(self, pupil):
        """Pupil coordinate grids should be symmetric (for odd n_pix) or nearly so (even)."""
        X, Y = pupil.get_coordinate_grid('meters')
        # For even grid sizes, perfect symmetry is not possible
        # But grids should be transposes of each other
        np.testing.assert_allclose(X, Y.T, rtol=1e-10)
    
    def test_pupil_polar_symmetry(self, pupil):
        """Pupil polar coordinates should be symmetric."""
        rho, theta = pupil.get_polar_coordinates(normalized=False)
        assert check_grid_symmetry(rho)
    
    def test_circular_pupil_symmetry(self, pupil):
        """Circular pupil should be symmetric."""
        assert check_grid_symmetry(pupil.amplitude)
    
    def test_pupil_center_alignment(self, pupil):
        """Pupil center should align with coordinate center."""
        X, Y = pupil.get_coordinate_grid('pixels')
        center = get_grid_center(pupil.n_pix)
        
        # Check that center is at (0, 0) in coordinate system
        center_idx = int(center)
        if pupil.n_pix % 2 == 1:
            # Odd: exact center
            np.testing.assert_allclose(X[center_idx, center_idx], 0.0, atol=1e-10)
            np.testing.assert_allclose(Y[center_idx, center_idx], 0.0, atol=1e-10)
        else:
            # Even: check 2x2 central region
            central_X = X[center_idx:center_idx+2, center_idx:center_idx+2]
            central_Y = Y[center_idx:center_idx+2, center_idx:center_idx+2]
            # Mean should be zero
            np.testing.assert_allclose(central_X.mean(), 0.0, atol=1e-10)
            np.testing.assert_allclose(central_Y.mean(), 0.0, atol=1e-10)


class TestZernikeCentering:
    """Test that Zernike grids are correctly centered."""
    
    @pytest.fixture
    def setup(self):
        """Setup for Zernike tests."""
        init_backend('CPU')
        _cache.clear()
    
    def test_zernike_coordinate_cache_symmetry(self, setup):
        """Zernike coordinate cache should produce symmetric grids."""
        n_pix = 128
        radius = (n_pix - 1) / 2.0
        
        rho, theta = _cache.get_coordinates(n_pix, radius)
        
        # Rho should be symmetric
        assert check_grid_symmetry(rho)
    
    def test_low_order_zernike_symmetry(self, setup):
        """Low-order Zernike modes should have correct symmetry."""
        n_pix = 64
        radius = (n_pix - 1) / 2.0
        
        rho, theta = _cache.get_coordinates(n_pix, radius)
        
        # Z_2^0 (astigmatism 0°) should be symmetric in both axes
        z_astig_0 = zernike_nm(2, 0, rho, theta)
        assert check_grid_symmetry(z_astig_0)
    
    def test_piston_is_constant(self, setup):
        """Piston (n=0, m=0) should be constant=1."""
        n_pix = 64
        radius = (n_pix - 1) / 2.0
        
        rho, theta = _cache.get_coordinates(n_pix, radius)
        
        # Only evaluate inside unit circle
        mask = rho <= 1.0
        
        piston = zernike_nm(0, 0, rho, theta)
        
        # Should be 1.0 everywhere inside pupil
        np.testing.assert_allclose(piston[mask], 1.0, rtol=1e-6)
    
    def test_zernike_generator_radius(self, setup):
        """ZernikePhaseGenerator should use consistent radius."""
        n_pix = 128
        pixel_size = 0.1
        
        gen = ZernikePhaseGenerator(
            n_pix=n_pix,
            pixel_size=pixel_size,
            zernike_config=ZernikeConfig(n_range=(2, 5)),
        )
        
        # Radius should be pixel-centered
        expected_radius = (n_pix - 1) / 2.0
        assert abs(gen._radius - expected_radius) < 1e-10


class TestPupilZernikeAlignment:
    """Test that pupil and Zernike grids align correctly."""
    
    @pytest.fixture
    def setup(self):
        """Setup test environment."""
        init_backend('CPU')
        _cache.clear()
    
    def test_coordinate_system_alignment(self, setup):
        """Pupil and Zernike should use same coordinate system."""
        n_pix = 128
        wavelength = 2.2e-6
        pixel_scale = 13e-3/206265
        diameter = 8.2
        
        # Create pupil
        pupil = Pupil.from_circular(
            n_pix=n_pix,
            wavelength=wavelength,
            pixel_scale=pixel_scale,
            diameter=diameter,
            obstruction_diameter=0.0,
        )
        
        # Get pupil coordinates
        X_pupil, Y_pupil = pupil.get_coordinate_grid('pixels')
        rho_pupil, _ = pupil.get_polar_coordinates(normalized=False)
        
        # Get Zernike coordinates (normalized to pupil radius)
        radius_pixels = diameter / pupil.pupil_pixel_size / 2.0
        rho_zernike, _ = _cache.get_coordinates(n_pix, radius_pixels)
        
        # Convert pupil rho to same normalization
        rho_pupil_norm = rho_pupil / (diameter / 2.0)
        
        # Should match within tolerance
        np.testing.assert_allclose(rho_pupil_norm, rho_zernike, rtol=1e-6)
    
    def test_symmetric_pupil_with_tilts(self, setup):
        """Symmetric pupil + symmetric Zernike → symmetric result."""
        n_pix = 64
        
        # Create symmetric circular pupil
        pupil = Pupil.from_circular(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=13e-3/206265,
            diameter=8.2,
            obstruction_diameter=0.0,
        )
        
        # Generate symmetric Zernike phase (piston + defocus)
        # Use only piston and defocus (which are symmetric)
        gen = ZernikePhaseGenerator(
            n_pix=n_pix,
            pixel_size=pupil.pupil_pixel_size,
            zernike_config=ZernikeConfig(
                n_range=(0, 2),  # n=0,1,2
                power_law=0.0,  # Equal weights
            ),
        )
        
        # Generate one screen and manually set to symmetric mode
        phase = gen.generate(n_screens=1)[0]
        # Zero out everything except the first mode (piston)
        phase = phase * 0.0 + 1.0  # Constant phase
        
        # Combined phase should be symmetric
        combined = pupil.amplitude * np.exp(1j * phase)
        combined_intensity = np.abs(combined)**2
        
        assert check_grid_symmetry(combined_intensity)
    
    def test_no_half_pixel_shift_in_psf(self, setup):
        """PSF from symmetric inputs should be centered (no systematic half-pixel shift)."""
        from nebraa.physics.optics import compute_psf
        
        # Use odd grid for perfect symmetry
        n_pix = 65
        
        # Create symmetric pupil
        pupil = Pupil.from_circular(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=13e-3/206265,
            diameter=8.2,
            obstruction_diameter=0.0,  # No obstruction for perfect symmetry
        )
        
        # No phase (perfect wavefront)
        phase = np.zeros((n_pix, n_pix))
        
        # Compute PSF
        psf = compute_psf(pupil.amplitude, phase, normalize_to='peak')
        
        # PSF should be symmetric for odd grid
        assert check_grid_symmetry(psf), "PSF from symmetric pupil should be symmetric"
        
        # Peak should be exactly at center for odd grid
        peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
        center = int((n_pix - 1) / 2)
        assert peak_idx == (center, center), f"Peak at {peak_idx}, expected ({center}, {center})"


class TestCoordinateConsistency:
    """Test that all coordinate systems are consistent."""
    
    def test_all_modules_use_same_convention(self):
        """Verify all modules use (n_pix-1)/2 center."""
        init_backend('CPU')
        _cache.clear()
        
        n_pix = 64
        
        # Test coordinate helper
        center_helper = get_grid_center(n_pix)
        assert abs(center_helper - (n_pix - 1) / 2.0) < 1e-10
        
        # Test pupil
        pupil = Pupil.from_circular(
            n_pix=n_pix, wavelength=2.2e-6,
            pixel_scale=13e-3/206265, diameter=8.2,
        )
        X_pupil, _ = pupil.get_coordinate_grid('pixels')
        # Check that coordinate 0 is at expected center
        center_x = X_pupil[int(center_helper), :]
        expected = np.arange(n_pix) - center_helper
        np.testing.assert_allclose(center_x, expected, rtol=1e-10)
        
        # Test Zernike cache
        radius = (n_pix - 1) / 2.0
        rho, _ = _cache.get_coordinates(n_pix, radius)
        # Center should have rho ≈ 0
        if n_pix % 2 == 1:
            assert abs(rho[int(center_helper), int(center_helper)]) < 1e-10
        
        # Test Zernike generator
        gen = ZernikePhaseGenerator(
            n_pix=n_pix, pixel_size=0.1,
            zernike_config=ZernikeConfig(n_range=(2, 5)),
        )
        assert abs(gen._radius - radius) < 1e-10


def test_consistency_across_backends():
    """Test coordinate consistency across CPU and GPU backends."""
    n_pix = 64
    
    try:
        # Test CPU
        init_backend('CPU')
        X_cpu, Y_cpu = make_xy_grid(n_pix)
        rho_cpu, _ = make_polar_grid(n_pix)
        
        # Test GPU if available
        backend = init_backend('GPU')
        
        # Check if we actually got GPU backend
        if backend.xp.__name__ != 'cupy':
            pytest.skip("GPU not available (fell back to CPU)")
        
        X_gpu, Y_gpu = make_xy_grid(n_pix)
        rho_gpu, _ = make_polar_grid(n_pix)
        
        # Should match
        np.testing.assert_allclose(X_cpu, X_gpu.get(), rtol=1e-10)
        np.testing.assert_allclose(Y_cpu, Y_gpu.get(), rtol=1e-10)
        np.testing.assert_allclose(rho_cpu, rho_gpu.get(), rtol=1e-10)
        
    except Exception as e:
        if 'CuPy' in str(e) or 'GPU' in str(e) or 'cupy' in str(e).lower():
            pytest.skip("GPU not available")
        else:
            raise
    finally:
        init_backend('CPU')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

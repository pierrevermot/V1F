"""
Tests for FFT centering convention in PSF computation.

Verifies that:
1. Centered circular pupils produce centered PSFs (peak at array center)
2. PSF centering is consistent across modules (PSFEngine, optics, powerlaw_psd)
3. ifftshift/fftshift convention is applied correctly
4. No phase ramps introduced by incorrect FFT centering

Author: NEBRAA
"""

import pytest
import numpy as np
from nebraa.physics.psf_engine import PSFEngine
from nebraa.physics.pupil import Pupil
from nebraa.physics.optics import compute_psf
from nebraa.utils.compute import init_backend

# Force CPU for reproducibility
init_backend('CPU')


class TestFFTCenteringConvention:
    """Test that FFT centering convention is applied correctly."""
    
    def test_centered_pupil_produces_centered_psf_odd_grid(self):
        """Verify centered circular pupil produces PSF peak at array center (odd grid)."""
        n_pix = 65  # Odd grid for perfect symmetry
        pupil_obj = Pupil.from_circular(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            diameter=8.0,
        )
        
        engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to='peak')
        psf = engine.compute_psf(pupil_obj.amplitude, phase=None)
        
        # Find peak location
        peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
        center = (n_pix - 1) // 2
        
        assert peak_idx == (center, center), \
            f"PSF peak at {peak_idx}, expected ({center}, {center})"
    
    def test_centered_pupil_produces_centered_psf_even_grid(self):
        """Verify centered circular pupil produces PSF peak at array center (even grid)."""
        n_pix = 64  # Even grid
        pupil_obj = Pupil.from_circular(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            diameter=8.0,
        )
        
        engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to='peak')
        psf = engine.compute_psf(pupil_obj.amplitude, phase=None)
        
        # Find peak location
        peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
        center = n_pix // 2
        
        # For even grids, peak should be within 1 pixel of center
        distance = np.abs(peak_idx[0] - center) + np.abs(peak_idx[1] - center)
        assert distance <= 1, \
            f"PSF peak at {peak_idx}, too far from center ({center}, {center})"
    
    def test_psf_with_zero_phase_is_centered(self):
        """PSF with zero phase screen should be centered (no phase ramp)."""
        n_pix = 128
        pupil_obj = Pupil.from_circular(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            diameter=8.0,
        )
        
        phase = np.zeros((n_pix, n_pix))
        
        engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        psf = engine.compute_psf(pupil_obj.amplitude, phase=phase)
        
        # Peak should be near center
        peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
        center = n_pix // 2
        distance = np.abs(peak_idx[0] - center) + np.abs(peak_idx[1] - center)
        
        assert distance <= 2, \
            f"PSF peak offset by {distance} pixels, expected centered"
    
    def test_psf_with_constant_phase_is_centered(self):
        """PSF with constant (piston) phase should be centered."""
        n_pix = 128
        pupil_obj = Pupil.from_circular(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            diameter=8.0,
        )
        
        # Constant phase (piston only)
        phase = np.ones((n_pix, n_pix)) * 2.5  # 2.5 radians piston
        
        engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        psf = engine.compute_psf(pupil_obj.amplitude, phase=phase)
        
        # Peak should be near center
        peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
        center = n_pix // 2
        distance = np.abs(peak_idx[0] - center) + np.abs(peak_idx[1] - center)
        
        assert distance <= 2, \
            f"PSF with piston offset by {distance} pixels, expected centered"
    
    def test_batch_psfs_are_centered(self):
        """Batch processing should produce centered PSFs."""
        n_pix = 64
        n_screens = 5
        
        pupil_obj = Pupil.from_circular(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            diameter=8.0,
        )
        
        # Zero phases (all diffraction-limited)
        phases = np.zeros((n_screens, n_pix, n_pix))
        
        engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        psf_avg, psfs = engine.compute_psf_batch(pupil_obj.amplitude, phases, return_individual=True)
        
        # Check average PSF
        peak_avg = np.unravel_index(np.argmax(psf_avg), psf_avg.shape)
        center = n_pix // 2
        distance_avg = np.abs(peak_avg[0] - center) + np.abs(peak_avg[1] - center)
        assert distance_avg <= 1, f"Average PSF offset by {distance_avg} pixels"
        
        # Check individual PSFs
        for i, psf in enumerate(psfs):
            peak = np.unravel_index(np.argmax(psf), psf.shape)
            distance = np.abs(peak[0] - center) + np.abs(peak[1] - center)
            assert distance <= 1, f"Individual PSF {i} offset by {distance} pixels"
    
    def test_psf_with_zero_pad_is_centered(self):
        """PSF with zero-padding should still be centered."""
        n_pix = 64
        pupil_obj = Pupil.from_circular(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            diameter=8.0,
        )
        
        for zero_pad_factor in [1, 2, 4]:
            engine = PSFEngine(
                n_pix=n_pix,
                wavelength=2.2e-6,
                pixel_scale=1e-5,
                zero_pad_factor=zero_pad_factor,
            )
            psf = engine.compute_psf(pupil_obj.amplitude, phase=None)
            
            # Peak should be at center of padded array
            peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
            center = psf.shape[0] // 2
            distance = np.abs(peak_idx[0] - center) + np.abs(peak_idx[1] - center)
            
            assert distance <= 2, \
                f"PSF with zero_pad_factor={zero_pad_factor} offset by {distance} pixels"


class TestCrossModuleConsistency:
    """Test FFT convention consistency across modules."""
    
    def test_psf_engine_vs_legacy_optics_centering(self):
        """PSFEngine and legacy optics should produce identically centered PSFs."""
        n_pix = 128
        pupil_obj = Pupil.from_circular(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            diameter=8.0,
        )
        
        phase = np.zeros((n_pix, n_pix))
        
        # PSFEngine
        engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        psf_engine = engine.compute_psf(pupil_obj.amplitude, phase=phase)
        
        # Legacy optics (wraps PSFEngine, so should be identical)
        with pytest.warns(DeprecationWarning):
            psf_legacy = compute_psf(pupil_obj.amplitude, phase, normalize_to='sum')
        
        # Should be identical (legacy wraps PSFEngine)
        np.testing.assert_array_equal(psf_engine, psf_legacy)
        
        # Peaks should be at exact same location
        peak_engine = np.unravel_index(np.argmax(psf_engine), psf_engine.shape)
        peak_legacy = np.unravel_index(np.argmax(psf_legacy), psf_legacy.shape)
        assert peak_engine == peak_legacy


class TestPSFSymmetry:
    """Test PSF symmetry for symmetric inputs."""
    
    def test_symmetric_pupil_produces_symmetric_psf(self):
        """Circular pupil with no phase should produce symmetric PSF."""
        n_pix = 128
        pupil_obj = Pupil.from_circular(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            diameter=8.0,
            obstruction_diameter=0.0,  # No obstruction for perfect symmetry
        )
        
        engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        psf = engine.compute_psf(pupil_obj.amplitude, phase=None)
        
        # PSF should be symmetric about center
        center = n_pix // 2
        
        # Check horizontal symmetry
        left = psf[center, :center]
        right = psf[center, center+1:][::-1]
        min_len = min(len(left), len(right))
        np.testing.assert_allclose(left[:min_len], right[:min_len], rtol=1e-5)
        
        # Check vertical symmetry
        top = psf[:center, center]
        bottom = psf[center+1:, center][::-1]
        min_len = min(len(top), len(bottom))
        np.testing.assert_allclose(top[:min_len], bottom[:min_len], rtol=1e-5)
    
    def test_psf_radial_profile_is_symmetric(self):
        """PSF from circular pupil should have radially symmetric profile."""
        n_pix = 128
        pupil_obj = Pupil.from_circular(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            diameter=8.0,
        )
        
        engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        psf = engine.compute_psf(pupil_obj.amplitude, phase=None)
        
        # Create radial profile
        center = n_pix // 2
        y, x = np.ogrid[:n_pix, :n_pix]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Sample PSF at various radii
        radii = [0, 5, 10, 15, 20]
        for radius in radii:
            # Get all pixels at approximately this radius
            mask = (r >= radius - 0.5) & (r < radius + 0.5)
            if not np.any(mask):
                continue
            
            values = psf[mask]
            # Values should be similar (within 10% variation for radial symmetry)
            if len(values) > 1:
                rel_std = np.std(values) / (np.mean(values) + 1e-10)
                assert rel_std < 0.2, \
                    f"Radial profile at r={radius} has high variation: {rel_std:.2%}"


class TestNoPhaseRamps:
    """Test that FFT centering doesn't introduce spurious phase ramps."""
    
    def test_no_phase_ramp_without_phase(self):
        """Diffraction-limited PSF should not show phase ramp effects."""
        n_pix = 128
        pupil_obj = Pupil.from_circular(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            diameter=8.0,
        )
        
        engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        psf = engine.compute_psf(pupil_obj.amplitude, phase=None)
        
        # Peak should be near center (no systematic shift)
        peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
        center = n_pix // 2
        
        # Check no systematic bias in either direction
        x_offset = peak_idx[1] - center
        y_offset = peak_idx[0] - center
        
        assert abs(x_offset) <= 1, f"Systematic X shift: {x_offset} pixels"
        assert abs(y_offset) <= 1, f"Systematic Y shift: {y_offset} pixels"
    
    def test_constant_phase_does_not_shift_psf(self):
        """Constant phase (piston) should not shift PSF location."""
        n_pix = 128
        pupil_obj = Pupil.from_circular(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            diameter=8.0,
        )
        
        engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
        
        # PSF without phase
        psf_no_phase = engine.compute_psf(pupil_obj.amplitude, phase=None)
        peak_no_phase = np.unravel_index(np.argmax(psf_no_phase), psf_no_phase.shape)
        
        # PSF with constant phase
        phase_piston = np.ones((n_pix, n_pix)) * 3.0
        psf_with_piston = engine.compute_psf(pupil_obj.amplitude, phase=phase_piston)
        peak_with_piston = np.unravel_index(np.argmax(psf_with_piston), psf_with_piston.shape)
        
        # Peaks should be at same location (piston doesn't shift PSF)
        assert peak_no_phase == peak_with_piston, \
            f"Piston shifted PSF: {peak_no_phase} -> {peak_with_piston}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

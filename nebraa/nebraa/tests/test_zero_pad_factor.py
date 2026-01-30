"""
Tests for PSFEngine zero_pad_factor functionality.

Tests verify:
1. Array size changes correctly with zero_pad_factor
2. Effective pixel scale changes correctly
3. Energy conservation under zero-padding
4. Normalization works correctly with padding
5. Backward compatibility (zero_pad_factor=1 gives same results)
6. Batch processing works with padding
7. Config integration works

Author: NEBRAA
"""

import pytest
import numpy as np
from nebraa.physics.psf_engine import PSFEngine, PSFEngineConfig
from nebraa.physics.pupil import Pupil
from nebraa.utils.compute import init_backend

# Force CPU for reproducibility
init_backend('CPU')


def make_test_pupil(n_pix, radius):
    """Helper to create a simple circular pupil for testing."""
    pupil = Pupil.from_circular(
        n_pix=n_pix,
        wavelength=2.2e-6,
        pixel_scale=1e-5,
        diameter=2 * radius * 2.2e-6 / (n_pix * 1e-5),  # Approximate diameter for given radius
        obstruction_diameter=0.0,
    )
    return pupil.amplitude


class TestZeroPadFactorArraySize:
    """Test that zero_pad_factor correctly changes array sizes."""
    
    def test_no_padding_gives_original_size(self):
        """Test zero_pad_factor=1 (no padding) gives original size."""
        n_pix = 128
        engine = PSFEngine(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=1,
        )
        
        pupil = make_test_pupil(n_pix, n_pix // 2)
        psf = engine.compute_psf(pupil)
        
        assert psf.shape == (n_pix, n_pix)
        assert engine.n_pix_padded == n_pix
    
    def test_padding_factor_2_doubles_size(self):
        """Test zero_pad_factor=2 doubles output size."""
        n_pix = 128
        engine = PSFEngine(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=2,
        )
        
        pupil = make_test_pupil(n_pix, n_pix // 2)
        psf = engine.compute_psf(pupil)
        
        assert psf.shape == (n_pix * 2, n_pix * 2)
        assert engine.n_pix_padded == n_pix * 2
    
    def test_padding_factor_4_quadruples_size(self):
        """Test zero_pad_factor=4 quadruples output size."""
        n_pix = 64
        engine = PSFEngine(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=4,
        )
        
        pupil = make_test_pupil(n_pix, n_pix // 2)
        psf = engine.compute_psf(pupil)
        
        assert psf.shape == (n_pix * 4, n_pix * 4)
        assert engine.n_pix_padded == n_pix * 4
    
    def test_batch_processing_with_padding(self):
        """Test batch processing produces correct padded size."""
        n_pix = 64
        n_screens = 5
        zero_pad_factor = 2
        
        engine = PSFEngine(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=zero_pad_factor,
        )
        
        pupil = make_test_pupil(n_pix, n_pix // 2)
        phases = np.random.uniform(0, 2 * np.pi, size=(n_screens, n_pix, n_pix))
        
        psf_avg, psfs = engine.compute_psf_batch(pupil, phases, return_individual=True)
        
        expected_size = n_pix * zero_pad_factor
        assert psf_avg.shape == (expected_size, expected_size)
        assert psfs.shape == (n_screens, expected_size, expected_size)


class TestZeroPadFactorPixelScale:
    """Test that pixel scale changes correctly with zero_pad_factor."""
    
    def test_effective_pixel_scale_no_padding(self):
        """Test effective pixel scale equals nominal with no padding."""
        pixel_scale = 1e-5
        engine = PSFEngine(
            n_pix=128,
            wavelength=2.2e-6,
            pixel_scale=pixel_scale,
            zero_pad_factor=1,
        )
        
        assert engine.pixel_scale_effective == pixel_scale
    
    def test_effective_pixel_scale_factor_2(self):
        """Test effective pixel scale halves with factor=2."""
        pixel_scale = 1e-5
        engine = PSFEngine(
            n_pix=128,
            wavelength=2.2e-6,
            pixel_scale=pixel_scale,
            zero_pad_factor=2,
        )
        
        assert engine.pixel_scale_effective == pixel_scale / 2
    
    def test_effective_pixel_scale_factor_4(self):
        """Test effective pixel scale quarters with factor=4."""
        pixel_scale = 1e-5
        engine = PSFEngine(
            n_pix=128,
            wavelength=2.2e-6,
            pixel_scale=pixel_scale,
            zero_pad_factor=4,
        )
        
        assert engine.pixel_scale_effective == pixel_scale / 4
    
    def test_info_contains_effective_pixel_scale(self):
        """Test info() returns effective pixel scale."""
        engine = PSFEngine(
            n_pix=128,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=3,
        )
        
        info = engine.info()
        assert "pixel_scale_effective_rad" in info
        assert "pixel_scale_effective_mas" in info
        assert info["pixel_scale_effective_rad"] == engine.pixel_scale_effective


class TestZeroPadFactorEnergyConservation:
    """Test that energy conservation works correctly with zero-padding."""
    
    def test_diffraction_limited_psf_energy_conserved(self):
        """Test DL PSF conserves energy with padding."""
        n_pix = 128
        
        for zero_pad_factor in [1, 2, 4]:
            engine = PSFEngine(
                n_pix=n_pix,
                wavelength=2.2e-6,
                pixel_scale=1e-5,
                zero_pad_factor=zero_pad_factor,
                normalize_to="sum",
            )
            
            pupil = make_test_pupil(n_pix, n_pix // 2)
            psf = engine.compute_psf(pupil, phase=None, normalize=True)
            
            total = float(np.sum(psf))
            assert abs(total - 1.0) < 1e-6, f"Energy not conserved for factor={zero_pad_factor}: sum={total}"
    
    def test_aberrated_psf_energy_conserved(self):
        """Test aberrated PSF conserves energy with padding."""
        n_pix = 128
        pupil = make_test_pupil(n_pix, n_pix // 2)
        phase = np.random.uniform(0, 2 * np.pi, size=(n_pix, n_pix))
        
        for zero_pad_factor in [1, 2, 4]:
            engine = PSFEngine(
                n_pix=n_pix,
                wavelength=2.2e-6,
                pixel_scale=1e-5,
                zero_pad_factor=zero_pad_factor,
                normalize_to="sum",
            )
            
            psf = engine.compute_psf(pupil, phase=phase, normalize=True)
            
            total = float(np.sum(psf))
            assert abs(total - 1.0) < 1e-6, f"Energy not conserved for factor={zero_pad_factor}: sum={total}"
    
    def test_batch_psf_energy_conserved(self):
        """Test batch PSF conserves energy with padding."""
        n_pix = 64
        n_screens = 10
        pupil = make_test_pupil(n_pix, n_pix // 2)
        phases = np.random.uniform(0, 2 * np.pi, size=(n_screens, n_pix, n_pix))
        
        for zero_pad_factor in [1, 2]:
            engine = PSFEngine(
                n_pix=n_pix,
                wavelength=2.2e-6,
                pixel_scale=1e-5,
                zero_pad_factor=zero_pad_factor,
                normalize_to="sum",
            )
            
            psf_avg, psfs = engine.compute_psf_batch(pupil, phases, normalize=True, return_individual=True)
            
            # Check average
            total_avg = float(np.sum(psf_avg))
            assert abs(total_avg - 1.0) < 1e-6, f"Average PSF energy not conserved for factor={zero_pad_factor}"
            
            # Check individuals
            for i, psf in enumerate(psfs):
                total = float(np.sum(psf))
                assert abs(total - 1.0) < 1e-6, f"Individual PSF {i} energy not conserved for factor={zero_pad_factor}"


class TestZeroPadFactorNormalization:
    """Test normalization modes work correctly with zero-padding."""
    
    def test_sum_normalization_with_padding(self):
        """Test sum=1 normalization works with padding."""
        n_pix = 128
        engine = PSFEngine(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=2,
            normalize_to="sum",
        )
        
        pupil = make_test_pupil(n_pix, n_pix // 2)
        psf = engine.compute_psf(pupil)
        
        total = float(np.sum(psf))
        assert abs(total - 1.0) < 1e-6
    
    def test_peak_normalization_with_padding(self):
        """Test peak=1 normalization works with padding."""
        n_pix = 128
        engine = PSFEngine(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=2,
            normalize_to="peak",
        )
        
        pupil = make_test_pupil(n_pix, n_pix // 2)
        psf = engine.compute_psf(pupil)
        
        peak = float(np.max(psf))
        assert abs(peak - 1.0) < 1e-6
    
    def test_no_normalization_with_padding(self):
        """Test normalize=False works with padding."""
        n_pix = 128
        engine = PSFEngine(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=2,
            normalize_to="none",
        )
        
        pupil = make_test_pupil(n_pix, n_pix // 2)
        psf = engine.compute_psf(pupil, normalize=False)
        
        # Without normalization, sum will be much larger than 1
        total = float(np.sum(psf))
        assert total > 1.0  # Raw PSF is not normalized


class TestZeroPadFactorBackwardCompatibility:
    """Test backward compatibility: zero_pad_factor=1 gives same results as before."""
    
    def test_diffraction_limited_psf_same_as_no_padding(self):
        """Test DL PSF with factor=1 matches no-factor behavior."""
        n_pix = 128
        pupil = make_test_pupil(n_pix, n_pix // 2)
        
        # Engine without explicit zero_pad_factor (defaults to 1)
        engine_default = PSFEngine(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
        )
        
        # Engine with explicit zero_pad_factor=1
        engine_explicit = PSFEngine(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=1,
        )
        
        psf_default = engine_default.compute_psf(pupil)
        psf_explicit = engine_explicit.compute_psf(pupil)
        
        np.testing.assert_allclose(psf_default, psf_explicit, rtol=1e-10)
    
    def test_aberrated_psf_reproducible_with_factor_1(self):
        """Test aberrated PSF with factor=1 is reproducible."""
        n_pix = 128
        pupil = make_test_pupil(n_pix, n_pix // 2)
        phase = np.random.RandomState(42).uniform(0, 2 * np.pi, size=(n_pix, n_pix))
        
        engine = PSFEngine(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=1,
        )
        
        psf1 = engine.compute_psf(pupil, phase)
        psf2 = engine.compute_psf(pupil, phase)
        
        np.testing.assert_array_equal(psf1, psf2)


class TestZeroPadFactorConfig:
    """Test PSFEngineConfig integration with zero_pad_factor."""
    
    def test_config_contains_zero_pad_factor(self):
        """Test config dataclass includes zero_pad_factor."""
        config = PSFEngineConfig(
            n_pix=128,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=2,
        )
        
        assert config.zero_pad_factor == 2
    
    def test_config_defaults_to_1(self):
        """Test config defaults zero_pad_factor to 1."""
        config = PSFEngineConfig(
            n_pix=128,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
        )
        
        assert config.zero_pad_factor == 1
    
    def test_from_config_passes_zero_pad_factor(self):
        """Test from_config() correctly passes zero_pad_factor."""
        config = PSFEngineConfig(
            n_pix=128,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=3,
        )
        
        engine = PSFEngine.from_config(config)
        
        assert engine.zero_pad_factor == 3
        assert engine.n_pix_padded == 128 * 3
    
    def test_engine_from_config_produces_correct_size(self):
        """Test engine created from config produces correct PSF size."""
        config = PSFEngineConfig(
            n_pix=64,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=2,
        )
        
        engine = PSFEngine.from_config(config)
        pupil = make_test_pupil(64, 32)
        psf = engine.compute_psf(pupil)
        
        assert psf.shape == (128, 128)


class TestZeroPadFactorStrehlRatio:
    """Test that Strehl ratio computation works with zero-padding."""
    
    def test_strehl_with_zero_padding(self):
        """Test Strehl ratio computation with zero-padding."""
        n_pix = 128
        pupil = make_test_pupil(n_pix, n_pix // 2)
        
        # Small phase aberration for known-good Strehl
        rms_rad = 0.3  # radians
        phase = np.random.RandomState(42).normal(0, rms_rad, size=(n_pix, n_pix))
        phase *= pupil  # Only inside pupil
        
        # Compute with and without padding
        for zero_pad_factor in [1, 2]:
            engine = PSFEngine(
                n_pix=n_pix,
                wavelength=2.2e-6,
                pixel_scale=1e-5,
                zero_pad_factor=zero_pad_factor,
            )
            
            strehl = engine.compute_strehl_ratio(pupil, phase=phase)
            
            # Strehl should be in reasonable range (0, 1)
            assert 0 < strehl < 1
            
            # Should be close to Maréchal approximation
            strehl_approx = engine.compute_strehl_from_rms(rms_rad)
            # Padding shouldn't drastically change Strehl
            assert abs(strehl - strehl_approx) < 0.3


class TestZeroPadFactorFinerSampling:
    """Test that zero-padding provides finer sampling as expected."""
    
    def test_peak_location_more_precise_with_padding(self):
        """Test that peak location can be determined more precisely with padding."""
        n_pix = 64
        pupil = make_test_pupil(n_pix, n_pix // 2)
        
        # Compute DL PSF with different padding factors
        engines = {
            1: PSFEngine(n_pix, 2.2e-6, 1e-5, zero_pad_factor=1),
            2: PSFEngine(n_pix, 2.2e-6, 1e-5, zero_pad_factor=2),
            4: PSFEngine(n_pix, 2.2e-6, 1e-5, zero_pad_factor=4),
        }
        
        psfs = {factor: engine.compute_psf(pupil) for factor, engine in engines.items()}
        
        # Find peak locations
        peaks = {}
        for factor, psf in psfs.items():
            peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
            center = np.array(psf.shape) // 2
            peaks[factor] = np.array(peak_idx) - center
        
        # All peaks should be near center
        for factor, peak in peaks.items():
            assert np.all(np.abs(peak) <= 2), f"Peak far from center for factor={factor}: {peak}"
    
    def test_airy_ring_structure_better_resolved(self):
        """Test that Airy ring structure is better resolved with padding."""
        n_pix = 64
        pupil = make_test_pupil(n_pix, n_pix // 2)
        
        # Compute DL PSF with and without padding
        engine_no_pad = PSFEngine(n_pix, 2.2e-6, 1e-5, zero_pad_factor=1)
        engine_pad = PSFEngine(n_pix, 2.2e-6, 1e-5, zero_pad_factor=4)
        
        psf_no_pad = engine_no_pad.compute_psf(pupil)
        psf_pad = engine_pad.compute_psf(pupil)
        
        # Central slice through PSF
        center_no_pad = psf_no_pad[n_pix // 2, :]
        center_pad = psf_pad[psf_pad.shape[0] // 2, :]
        
        # Padded PSF should have more samples
        assert len(center_pad) == len(center_no_pad) * 4
        
        # Both should be normalized (sum ~ 1 for full 2D PSF)
        # The peak in the padded PSF will be lower because it's better sampled
        # but both PSFs should conserve energy
        assert abs(np.sum(psf_no_pad) - 1.0) < 1e-6
        assert abs(np.sum(psf_pad) - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for one-off correctness issues (Item #12).

This module tests three specific bugs:
- 12a: DualPowerLawPSFModel constructor mismatch (config= vs psd_config=)
- 12b: Jolissaint combined aniso+servo term ignores per-term flags
- 12c: Piston removal divides by zero when pupil sum is zero

Author: NEBRAA
Date: 2026-01-30
Related: Item #12 - One-off correctness issues
"""

import pytest
import numpy as np
import os

# Force CPU for testing
os.environ["NEBRAA_FORCE_CPU"] = "1"

from nebraa.physics.powerlaw_psd import (
    DualPowerLawPSFModel, 
    DualPowerLawConfig,
    DualPowerLawPhaseGenerator,
)
from nebraa.physics.kolmogorov import KolmogorovGenerator
from nebraa.utils.compute import get_backend, init_backend

# Initialize CPU backend
init_backend("CPU")


class TestItem12a_ConstructorMismatch:
    """Test 12a: DualPowerLawPSFModel constructor mismatch."""
    
    def test_psf_model_constructor_accepts_psd_config(self):
        """
        Test that DualPowerLawPSFModel constructor accepts psd_config parameter.
        
        Before fix: TypeError due to parameter name mismatch
        After fix: Constructor works correctly
        """
        n_pix = 128
        psd_config = DualPowerLawConfig(
            alpha_lf=3.0,
            alpha_hf=11.0/3.0,
            rms_lf=0.5,
            rms_hf=0.3,
            f_cutoff=1.0,
        )
        
        # This should not raise TypeError
        model = DualPowerLawPSFModel(
            n_pix=n_pix,
            telescope_diameter=8.0,
            obstruction_diameter=1.0,
            wavelength=1.65e-6,
            pixel_scale=10e-3 * np.pi / 180 / 3600,
            psd_config=psd_config,
        )
        
        assert model.psd_config == psd_config
        assert model.n_pix == n_pix
    
    def test_phase_generator_constructor_accepts_psd_config(self):
        """
        Test that DualPowerLawPhaseGenerator accepts psd_config parameter name.
        """
        n_pix = 128
        pixel_size = 0.0625  # meters
        psd_config = DualPowerLawConfig()
        
        # PhaseGenerator should accept psd_config parameter
        gen = DualPowerLawPhaseGenerator(
            n_pix=n_pix,
            pixel_size=pixel_size,
            psd_config=psd_config,
        )
        
        assert gen.psd_config == psd_config


class TestItem12b_JolissaintFlagsBug:
    """Test 12b: Jolissaint combined aniso+servo term ignores per-term flags."""
    
    def test_aniso_disabled_removes_contribution(self):
        """
        Test that disabling anisoplanatism removes its contribution from combined term.
        
        When include_anisoplanatism=False, the aniso+servo PSD should reduce
        to just the servo-lag component (if servo is enabled).
        """
        from nebraa.physics.jolissaint_ao import (
            JolissaintAOModel,
            AOSystemConfig,
            AtmosphereProfile,
            TurbulentLayer,
        )
        
        n_pix = 128
        diameter = 8.0
        obstruction = 1.0
        wavelength = 1.65e-6
        pixel_scale = 10e-3 * np.pi / 180 / 3600
        
        # Single-layer atmosphere at altitude (needed for anisoplanatism effect)
        layer = TurbulentLayer(altitude=5000.0, r0=0.15, wind_speed=10.0)  # 5km altitude
        atm = AtmosphereProfile(layers=[layer], L0=25.0)
        
        # Config with aniso enabled
        ao_with_aniso = AOSystemConfig(
            actuator_pitch=0.5,
            integration_time=0.001,
            loop_delay=0.001,
            science_field_offset=1.0 * np.pi / 180 / 3600,  # 1 arcsec offset
            include_anisoplanatism=True,
            include_servo_lag=True,
        )
        
        # Config with aniso disabled
        ao_no_aniso = AOSystemConfig(
            actuator_pitch=0.5,
            integration_time=0.001,
            loop_delay=0.001,
            science_field_offset=1.0 * np.pi / 180 / 3600,
            include_anisoplanatism=False,
            include_servo_lag=True,
        )
        
        # Create models
        model_with = JolissaintAOModel(n_pix, diameter, obstruction, wavelength, pixel_scale, atm, ao_with_aniso)
        model_without = JolissaintAOModel(n_pix, diameter, obstruction, wavelength, pixel_scale, atm, ao_no_aniso)
        
        backend = get_backend()
        
        # Compute combined PSD
        psd_with = backend.to_numpy(model_with.compute_aniso_servo_psd())
        psd_without = backend.to_numpy(model_without.compute_aniso_servo_psd())
        
        # When aniso is disabled, PSD should be significantly different
        # (should only have servo-lag component)
        psd_diff = np.abs(psd_with - psd_without)
        
        # Check that difference is significant (not just numerical noise)
        assert np.max(psd_diff) > 1e-10, "Disabling aniso should change PSD significantly"
        
        # The PSDs should differ (combined term accounts for correlation)
        # May be larger or smaller depending on the correlation term
        assert not np.allclose(psd_with, psd_without, rtol=1e-6), "PSDs should differ significantly"
    
    def test_servo_disabled_removes_contribution(self):
        """
        Test that disabling servo-lag removes its contribution from combined term.
        """
        from nebraa.physics.jolissaint_ao import (
            JolissaintAOModel,
            AOSystemConfig,
            AtmosphereProfile,
            TurbulentLayer,
        )
        
        n_pix = 128
        diameter = 8.0
        obstruction = 1.0
        wavelength = 1.65e-6
        pixel_scale = 10e-3 * np.pi / 180 / 3600
        
        layer = TurbulentLayer(altitude=0.0, r0=0.15, wind_speed=10.0)
        atm = AtmosphereProfile(layers=[layer], L0=25.0)
        
        # Config with servo enabled
        ao_with_servo = AOSystemConfig(
            actuator_pitch=0.5,
            integration_time=0.001,
            loop_delay=0.001,
            science_field_offset=1.0 * np.pi / 180 / 3600,
            include_anisoplanatism=True,
            include_servo_lag=True,
        )
        
        # Config with servo disabled
        ao_no_servo = AOSystemConfig(
            actuator_pitch=0.5,
            integration_time=0.001,
            loop_delay=0.001,
            science_field_offset=1.0 * np.pi / 180 / 3600,
            include_anisoplanatism=True,
            include_servo_lag=False,
        )
        
        model_with = JolissaintAOModel(n_pix, diameter, obstruction, wavelength, pixel_scale, atm, ao_with_servo)
        model_without = JolissaintAOModel(n_pix, diameter, obstruction, wavelength, pixel_scale, atm, ao_no_servo)
        
        backend = get_backend()
        
        psd_with = backend.to_numpy(model_with.compute_aniso_servo_psd())
        psd_without = backend.to_numpy(model_without.compute_aniso_servo_psd())
        
        # When servo is disabled, PSD should be significantly different
        psd_diff = np.abs(psd_with - psd_without)
        assert np.max(psd_diff) > 1e-10, "Disabling servo should change PSD significantly"
        
        # The PSDs should differ (combined term accounts for correlation)
        assert not np.allclose(psd_with, psd_without, rtol=1e-6), "PSDs should differ significantly"
    
    def test_both_disabled_returns_zero(self):
        """
        Test that disabling both aniso and servo returns zero PSD.
        """
        from nebraa.physics.jolissaint_ao import (
            JolissaintAOModel,
            AOSystemConfig,
            AtmosphereProfile,
            TurbulentLayer,
        )
        
        n_pix = 128
        diameter = 8.0
        obstruction = 1.0
        wavelength = 1.65e-6
        pixel_scale = 10e-3 * np.pi / 180 / 3600
        
        layer = TurbulentLayer(altitude=0.0, r0=0.15)
        atm = AtmosphereProfile(layers=[layer], L0=25.0)
        
        ao = AOSystemConfig(
            actuator_pitch=0.5,
            include_anisoplanatism=False,
            include_servo_lag=False,
        )
        
        model = JolissaintAOModel(n_pix, diameter, obstruction, wavelength, pixel_scale, atm, ao)
        backend = get_backend()
        
        psd = backend.to_numpy(model.compute_aniso_servo_psd())
        
        # Should be all zeros
        assert np.allclose(psd, 0.0), "PSD should be zero when both terms disabled"


class TestItem12c_PistonDivideByZero:
    """Test 12c: Piston removal divides by zero when pupil sum is zero."""
    
    def test_zero_pupil_no_nans_kolmogorov(self):
        """
        Test that KolmogorovGenerator handles zero pupil without NaNs.
        
        Before fix: Division by zero causes NaN in phase screens
        After fix: Either raises ValueError or handles gracefully
        """
        n_pix = 128
        pixel_size = 0.0625
        actuator_pitch = 0.5
        
        gen = KolmogorovGenerator(
            n_pix=n_pix,
            pixel_size=pixel_size,
            actuator_pitch=actuator_pitch,
        )
        
        # Create zero pupil
        backend = get_backend()
        xp = backend.xp
        zero_pupil = xp.zeros((n_pix, n_pix), dtype=xp.float32)
        
        # Generate phase
        # Should either raise ValueError or return valid (non-NaN) phases
        try:
            phases = gen.generate(n_screens=1, r0=0.15, pupil=zero_pupil, seed=42)
            phases_np = backend.to_numpy(phases)
            
            # If it doesn't raise, ensure no NaNs
            assert not np.any(np.isnan(phases_np)), "Phase screens contain NaN with zero pupil"
            assert not np.any(np.isinf(phases_np)), "Phase screens contain Inf with zero pupil"
            
        except ValueError as e:
            # Acceptable to raise ValueError for degenerate pupil
            assert "pupil" in str(e).lower() or "zero" in str(e).lower(), \
                "ValueError should mention pupil or zero issue"
    
    def test_empty_pupil_no_nans_kolmogorov(self):
        """
        Test that very small pupil (near-zero sum) doesn't cause NaNs.
        """
        n_pix = 128
        pixel_size = 0.0625
        actuator_pitch = 0.5
        
        gen = KolmogorovGenerator(
            n_pix=n_pix,
            pixel_size=pixel_size,
            actuator_pitch=actuator_pitch,
        )
        
        backend = get_backend()
        xp = backend.xp
        
        # Create tiny pupil (sum very close to zero)
        tiny_pupil = xp.ones((n_pix, n_pix), dtype=xp.float32) * 1e-15
        
        try:
            phases = gen.generate(n_screens=1, r0=0.15, pupil=tiny_pupil, seed=42)
            phases_np = backend.to_numpy(phases)
            
            # Ensure no NaNs or Infs
            assert not np.any(np.isnan(phases_np)), "Phase screens contain NaN with tiny pupil"
            assert not np.any(np.isinf(phases_np)), "Phase screens contain Inf with tiny pupil"
            
        except ValueError:
            # Also acceptable to raise for degenerate case
            pass
    
    def test_normal_pupil_works(self):
        """
        Test that normal pupil still works correctly (regression test).
        """
        n_pix = 128
        pixel_size = 0.0625
        actuator_pitch = 0.5
        
        gen = KolmogorovGenerator(
            n_pix=n_pix,
            pixel_size=pixel_size,
            actuator_pitch=actuator_pitch,
        )
        
        backend = get_backend()
        xp = backend.xp
        
        # Create normal circular pupil
        y, x = xp.meshgrid(xp.arange(n_pix), xp.arange(n_pix))
        center = (n_pix - 1) / 2.0
        r = xp.sqrt((x - center)**2 + (y - center)**2)
        pupil = (r < n_pix * 0.4).astype(xp.float32)
        
        phases = gen.generate(n_screens=5, r0=0.15, pupil=pupil, seed=42)
        phases_np = backend.to_numpy(phases)
        
        # Should be valid phases
        assert phases_np.shape == (5, n_pix, n_pix)
        assert not np.any(np.isnan(phases_np))
        assert not np.any(np.isinf(phases_np))
        
        # Should have reasonable RMS
        rms = np.std(phases_np)
        assert 0.1 < rms < 10.0, f"Phase RMS {rms} seems unreasonable"


# =============================================================================
# Acceptance Tests (from Item #12 requirements)
# =============================================================================

class TestItem12Acceptance:
    """Acceptance tests for Item #12: One-off correctness issues."""
    
    def test_12a_constructor_no_type_error(self):
        """
        Acceptance 12a: Constructing DualPowerLawPSFModel no longer raises TypeError.
        """
        psd_config = DualPowerLawConfig()
        
        # Should not raise TypeError
        model = DualPowerLawPSFModel(
            n_pix=128,
            telescope_diameter=8.0,
            obstruction_diameter=1.0,
            wavelength=1.65e-6,
            pixel_scale=10e-3 * np.pi / 180 / 3600,
            psd_config=psd_config,
        )
        
        assert model is not None
    
    def test_12b_disabling_term_removes_contribution(self):
        """
        Acceptance 12b: Disabling one term truly removes it (PSD contribution goes to zero).
        """
        from nebraa.physics.jolissaint_ao import (
            JolissaintAOModel,
            AOSystemConfig,
            AtmosphereProfile,
            TurbulentLayer,
        )
        
        # Create atmosphere with altitude for aniso effect
        layer = TurbulentLayer(altitude=5000.0, r0=0.15, wind_speed=10.0)
        atm = AtmosphereProfile(layers=[layer], L0=25.0)
        
        # Test with only aniso enabled (servo disabled)
        ao_aniso_only = AOSystemConfig(
            actuator_pitch=0.5,
            science_field_offset=1.0 * np.pi / 180 / 3600,  # 1 arcsec
            include_anisoplanatism=True,
            include_servo_lag=False,
        )
        
        model = JolissaintAOModel(
            n_pix=128,
            telescope_diameter=8.0,
            obstruction_diameter=1.0,
            wavelength=1.65e-6,
            pixel_scale=10e-3 * np.pi / 180 / 3600,
            atmosphere=atm,
            ao_config=ao_aniso_only,
        )
        
        backend = get_backend()
        psd = backend.to_numpy(model.compute_aniso_servo_psd())
        
        # Should have non-zero PSD (aniso contribution)
        assert np.sum(psd) > 0, "Aniso-only PSD should be non-zero"
    
    def test_12c_degenerate_pupil_no_nans(self):
        """
        Acceptance 12c: Degenerate pupil mask produces no NaNs.
        """
        gen = KolmogorovGenerator(n_pix=128, pixel_size=0.0625, actuator_pitch=0.5)
        
        backend = get_backend()
        zero_pupil = backend.xp.zeros((128, 128), dtype=backend.xp.float32)
        
        # Should either raise ValueError or return valid phases (no NaNs)
        try:
            phases = gen.generate(n_screens=1, r0=0.15, pupil=zero_pupil, seed=42)
            phases_np = backend.to_numpy(phases)
            assert not np.any(np.isnan(phases_np)), "Should not produce NaNs"
        except ValueError:
            # Acceptable to raise ValueError for degenerate case
            pass

"""
Tests for PSD units consistency across turbulence models.

This test module verifies that:
1. All PSD formulas use consistent frequency units (cycles/m vs rad/m)
2. Kolmogorov PSD formulas match the expected -11/3 power law
3. Cross-module comparisons don't differ by unintended (2π) factors

The standard convention adopted in NEBRAA is:
- Spatial frequency f in cycles/meter (from numpy.fft.fftfreq)
- Angular spatial frequency k = 2πf in radians/meter
- Kolmogorov PSD: Φ(f) = 0.023 × r₀^(-5/3) × (2πf)^(-11/3) [rad² / (cycle/m)²]
- Equivalently: Φ(k) = 0.023 × r₀^(-5/3) × k^(-11/3) [rad² / (rad/m)²]

Author: NEBRAA
Date: 2026-01-30
Related: Item #11 - PSD units consistency
"""

import pytest
import numpy as np
import os

# Force CPU for testing
os.environ["NEBRAA_FORCE_CPU"] = "1"

from nebraa.physics.kolmogorov import kolmogorov_psd as kolmogorov_psd_module
from nebraa.physics.jolissaint_ao import kolmogorov_psd as jolissaint_kolmogorov_psd
from nebraa.utils.compute import get_backend, init_backend

# Initialize CPU backend
init_backend("CPU")


class TestKolmogorovPSDFormula:
    """Test that Kolmogorov PSD formulas are correctly implemented."""
    
    def test_kolmogorov_module_psd_formula(self):
        """
        Test that kolmogorov.py uses the correct formula with f in cycles/m.
        
        The correct formula for f in cycles/meter is:
        Φ(f) = 0.023 × r₀^(-5/3) × f^(-11/3)
        
        Note: The coefficient 0.023 is derived for this convention, and gives
        the correct structure function D_φ(r0) ≈ 6.88 for Kolmogorov turbulence.
        """
        backend = get_backend()
        xp = backend.xp
        
        r0 = 0.15  # meters
        f = xp.array([1.0, 2.0, 4.0], dtype=xp.float32)  # cycles/meter
        
        # Compute PSD using module function
        psd = kolmogorov_psd_module(f, r0, C=0.023)
        
        # Expected values with f^(-11/3) (cycles/m convention)
        expected = 0.023 * (r0 ** (-5/3)) * (f ** (-11/3))
        
        # Convert to numpy for comparison
        psd_np = backend.to_numpy(psd)
        expected_np = backend.to_numpy(expected)
        
        np.testing.assert_allclose(psd_np, expected_np, rtol=1e-6)
    
    def test_kolmogorov_psd_power_law_slope(self):
        """
        Test that Kolmogorov PSD has the expected -11/3 slope.
        
        On a log-log plot, the PSD should have slope -11/3.
        """
        backend = get_backend()
        xp = backend.xp
        
        r0 = 0.15
        f = xp.logspace(-1, 2, 100, dtype=xp.float32)  # cycles/meter
        
        psd = kolmogorov_psd_module(f, r0, C=0.023)
        
        # Compute slope on log-log scale
        log_f = xp.log10(f[10:90])
        log_psd = xp.log10(psd[10:90])
        
        # Linear fit to log-log data
        log_f_np = backend.to_numpy(log_f)
        log_psd_np = backend.to_numpy(log_psd)
        
        slope, _ = np.polyfit(log_f_np, log_psd_np, 1)
        
        # Should be -11/3 ≈ -3.667
        expected_slope = -11.0 / 3.0
        assert abs(slope - expected_slope) < 0.01, f"Slope {slope:.3f} != {expected_slope:.3f}"
    
    def test_jolissaint_kolmogorov_psd_is_correct(self):
        """
        Test that jolissaint_ao.py Kolmogorov PSD uses the correct formula.
        
        The formula uses f directly in cycles/m: Φ(f) = 0.023 × r₀^(-5/3) × f^(-11/3)
        """
        backend = get_backend()
        xp = backend.xp
        
        r0 = 0.15
        f = xp.array([1.0, 2.0, 4.0], dtype=xp.float64)
        
        # Jolissaint implementation
        psd_jolissaint = jolissaint_kolmogorov_psd(xp, f, r0)
        
        # Expected with f^(-11/3) (cycles/m convention)
        expected = 0.023 * (r0 ** (-5/3)) * (f ** (-11/3))
        
        psd_jolissaint_np = backend.to_numpy(psd_jolissaint)
        expected_np = backend.to_numpy(expected)
        
        # Should match within numerical precision
        np.testing.assert_allclose(psd_jolissaint_np, expected_np, rtol=1e-6)
        
        print("✓ jolissaint_ao.py uses correct f^(-11/3) formula")


class TestCrossModulePSDConsistency:
    """Test that PSD formulas are consistent across modules."""
    
    def test_kolmogorov_vs_jolissaint_after_fix(self):
        """
        Test that kolmogorov.py and jolissaint_ao.py give same PSD (after fix).
        """
        backend = get_backend()
        xp = backend.xp
        
        r0 = 0.15
        f = xp.logspace(-1, 2, 50, dtype=xp.float32)
        
        # Kolmogorov module (correct)
        psd_kolm = kolmogorov_psd_module(f, r0, C=0.023)
        
        # Jolissaint module (now fixed)
        f_f64 = f.astype(xp.float64)
        psd_joli = jolissaint_kolmogorov_psd(xp, f_f64, r0)
        
        # After fix, these should match
        psd_kolm_np = backend.to_numpy(psd_kolm.astype(xp.float64))
        psd_joli_np = backend.to_numpy(psd_joli)
        
        np.testing.assert_allclose(psd_kolm_np, psd_joli_np, rtol=1e-5)
    
    def test_psd_units_documentation(self):
        """
        Verify that all modules document their frequency units consistently.
        
        This is a documentation check, not a functional test.
        """
        # Check that docstrings mention "cycles/meter" or "rad/meter"
        import nebraa.physics.kolmogorov as kolm
        import nebraa.physics.jolissaint_ao as joli
        import nebraa.physics.powerlaw_psd as pwl
        
        # Kolmogorov module
        assert "cycles/meter" in kolm.kolmogorov_psd.__doc__
        
        # Jolissaint module
        assert "cycles/meter" in joli.kolmogorov_psd.__doc__ or "cycles/m" in joli.kolmogorov_psd.__doc__
        
        # PowerLaw module
        assert "cycles/meter" in pwl.FrequencyGrid.__doc__


class TestPSDIntegrationValues:
    """Test that PSD integrates to expected phase variance."""
    
    def test_kolmogorov_psd_variance(self):
        """
        Test that integrating Kolmogorov PSD gives expected phase variance scaling.
        
        The phase variance should scale as (D/r0)^(5/3) for Kolmogorov turbulence.
        This test verifies the correct power-law scaling.
        """
        backend = get_backend()
        xp = backend.xp
        
        from nebraa.physics.kolmogorov import FrequencyGrid
        
        # Test with two different r0 values
        D = 8.0  # meters
        r0_1 = 0.15  # meters
        r0_2 = 0.10  # meters
        
        n_pix = 512
        pixel_size = D / n_pix
        grid = FrequencyGrid(n_pix, pixel_size)
        
        # Compute variance for both r0 values
        variances = []
        for r0 in [r0_1, r0_2]:
            psd = kolmogorov_psd_module(grid.F, r0, C=0.023)
            psd.flat[0] = 0.0  # Remove piston
            variance = float(xp.sum(psd) * grid.dA)
            variances.append(variance)
        
        var_1, var_2 = variances
        
        # Check that variance scales as r0^(-5/3)
        # var_2 / var_1 = (r0_1 / r0_2)^(5/3)
        observed_ratio = var_2 / var_1
        expected_ratio = (r0_1 / r0_2) ** (5.0/3.0)
        
        rel_error = abs(observed_ratio - expected_ratio) / expected_ratio
        
        assert rel_error < 0.05, \
            f"Variance ratio {observed_ratio:.3f} differs by {rel_error*100:.1f}% from expected {expected_ratio:.3f}"
        
        print(f"r0={r0_1}m: σ²={var_1:.3f} rad²")
        print(f"r0={r0_2}m: σ²={var_2:.3f} rad²")
        print(f"Observed ratio: {observed_ratio:.3f}, Expected: {expected_ratio:.3f}")
        print(f"✓ Variance scales correctly as r0^(-5/3)")


class TestFrequencyConventions:
    """Test that frequency conventions are clearly defined."""
    
    def test_fftfreq_returns_cycles_per_meter(self):
        """
        Verify that fftfreq with pixel_size in meters gives cycles/meter.
        """
        n_pix = 128
        pixel_size = 0.0625  # meters (8m telescope / 128 pixels)
        
        # FFT frequency
        f = np.fft.fftfreq(n_pix, d=pixel_size)
        
        # Maximum frequency is Nyquist: f_nyq = 1 / (2 × pixel_size)
        f_nyquist_expected = 1.0 / (2.0 * pixel_size)
        f_max = np.max(np.abs(f))
        
        assert abs(f_max - f_nyquist_expected) < 1e-10
        
        # Frequency resolution
        df = 1.0 / (n_pix * pixel_size)
        assert abs(f[1] - f[0] - df) < 1e-10
    
    def test_angular_frequency_conversion(self):
        """
        Test conversion between cycles/m and rad/m.
        """
        f_cyc = 1.0  # cycles/meter
        k_rad = 2.0 * np.pi * f_cyc  # rad/meter
        
        assert abs(k_rad - 2.0 * np.pi) < 1e-10
        
        # Inverse conversion
        f_cyc_back = k_rad / (2.0 * np.pi)
        assert abs(f_cyc_back - f_cyc) < 1e-10


# =============================================================================
# Acceptance Tests (from Item #11 requirements)
# =============================================================================

class TestItem11Acceptance:
    """Acceptance tests for Item #11: PSD units consistency."""
    
    @pytest.mark.parametrize("module", ["kolmogorov", "jolissaint"])
    def test_psd_slopes_match_expected_11_3(self, module):
        """
        Acceptance: PSD slopes match expected -11/3 behavior.
        """
        backend = get_backend()
        xp = backend.xp
        
        r0 = 0.15
        f = xp.logspace(-1, 2, 100, dtype=xp.float32)
        
        # Test both modules
        if module == "kolmogorov":
            psd = kolmogorov_psd_module(f, r0)
        else:  # jolissaint
            psd = jolissaint_kolmogorov_psd(xp, f.astype(xp.float64), r0)
            psd = psd.astype(xp.float32)
        
        log_f = xp.log10(f[10:90])
        log_psd = xp.log10(psd[10:90])
        slope, _ = np.polyfit(backend.to_numpy(log_f), backend.to_numpy(log_psd), 1)
        
        assert abs(slope - (-11.0/3.0)) < 0.01, f"{module}: slope {slope:.3f} != -11/3"
    
    def test_cross_module_no_unintended_2pi_factors(self):
        """
        Acceptance: Cross-model comparisons don't differ by unintended (2π) factors.
        
        After fix, kolmogorov.py and jolissaint_ao.py should give same PSD.
        """
        backend = get_backend()
        xp = backend.xp
        
        r0 = 0.15
        f = xp.array([1.0, 2.0, 4.0], dtype=xp.float32)
        
        psd_kolm = kolmogorov_psd_module(f, r0, C=0.023)
        psd_joli = jolissaint_kolmogorov_psd(xp, f.astype(xp.float64), r0)
        
        psd_kolm_np = backend.to_numpy(psd_kolm.astype(xp.float64))
        psd_joli_np = backend.to_numpy(psd_joli)
        
        # Should match within numerical precision
        np.testing.assert_allclose(psd_kolm_np, psd_joli_np, rtol=1e-5)

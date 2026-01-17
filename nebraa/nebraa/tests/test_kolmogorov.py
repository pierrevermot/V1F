"""
Tests for Kolmogorov turbulence module.
"""

import numpy as np
import pytest


class TestKolmogorovPSD:
    """Tests for Kolmogorov PSD function."""
    
    def test_psd_positive(self, backend):
        """Test PSD is always positive."""
        from nebraa.physics.kolmogorov import kolmogorov_psd
        
        xp = backend.xp
        
        # Create frequency grid
        f = xp.linspace(0.1, 10.0, 100).astype(xp.float32)
        
        psd = kolmogorov_psd(f, r0=0.15)
        
        assert xp.all(psd > 0)
    
    def test_psd_r0_scaling(self, backend):
        """Test PSD scales correctly with r0."""
        from nebraa.physics.kolmogorov import kolmogorov_psd
        
        xp = backend.xp
        
        f = xp.linspace(0.1, 10.0, 100).astype(xp.float32)
        
        psd1 = kolmogorov_psd(f, r0=0.1)
        psd2 = kolmogorov_psd(f, r0=0.2)
        
        # Better seeing (larger r0) -> smaller PSD
        total1 = backend.to_numpy(xp.sum(psd1))
        total2 = backend.to_numpy(xp.sum(psd2))
        
        assert total2 < total1


class TestFrequencyGrid:
    """Tests for FrequencyGrid class."""
    
    def test_grid_shape(self, backend):
        """Test frequency grid shape."""
        from nebraa.physics.kolmogorov import FrequencyGrid
        
        grid = FrequencyGrid(n_pix=64, pixel_size=0.01)
        
        assert grid.FX.shape == (64, 64)
        assert grid.FY.shape == (64, 64)
        assert grid.F.shape == (64, 64)
    
    def test_frequency_range(self, backend):
        """Test frequency range is correct."""
        from nebraa.physics.kolmogorov import FrequencyGrid
        
        n_pix = 64
        pixel_size = 0.01
        
        grid = FrequencyGrid(n_pix=n_pix, pixel_size=pixel_size)
        
        # Nyquist frequency
        f_nyquist = 1.0 / (2.0 * pixel_size)
        
        F_np = backend.to_numpy(grid.F)
        
        # Max frequency should be close to Nyquist * sqrt(2)
        assert F_np.max() <= f_nyquist * np.sqrt(2) * 1.01


class TestRaisedCosineFilter:
    """Tests for raised cosine filter."""
    
    def test_highpass_values(self, backend):
        """Test high-pass filter values."""
        from nebraa.physics.kolmogorov import raised_cosine_filter, FrequencyGrid
        
        grid = FrequencyGrid(n_pix=64, pixel_size=0.01)
        fc = 2.0  # cutoff frequency
        
        hp = raised_cosine_filter(grid.F, fc=fc, transition_frac=0.15, highpass=True)
        
        hp_np = backend.to_numpy(hp)
        F_np = backend.to_numpy(grid.F)
        
        # Below cutoff should be ~0
        assert np.mean(hp_np[F_np < fc * 0.7]) < 0.1
        
        # Above cutoff should be ~1
        assert np.mean(hp_np[F_np > fc * 1.3]) > 0.9


class TestKolmogorovGenerator:
    """Tests for KolmogorovGenerator class."""
    
    def test_init(self, backend):
        """Test initialization."""
        from nebraa.physics.kolmogorov import KolmogorovGenerator
        
        gen = KolmogorovGenerator(
            n_pix=128,
            pixel_size=0.01,
            actuator_pitch=0.20,
        )
        
        assert gen.n_pix == 128
        assert gen.fc == 1.0 / (2.0 * 0.20)  # = 2.5 cycles/m
    
    def test_generate_shape(self, backend):
        """Test phase generation shape."""
        from nebraa.physics.kolmogorov import KolmogorovGenerator
        
        gen = KolmogorovGenerator(
            n_pix=64,
            pixel_size=0.01,
            actuator_pitch=0.20,
        )
        
        phase = gen.generate(n_screens=5, r0=0.15, seed=42)
        
        assert phase.shape == (5, 64, 64)
    
    def test_phase_variability(self, backend):
        """Test that phases vary between realizations."""
        from nebraa.physics.kolmogorov import KolmogorovGenerator
        
        gen = KolmogorovGenerator(
            n_pix=64,
            pixel_size=0.01,
            actuator_pitch=0.20,
        )
        
        phase = gen.generate(n_screens=10, r0=0.15)
        phase_np = backend.to_numpy(phase)
        
        # Different screens should be different
        diff = np.abs(phase_np[0] - phase_np[1])
        assert np.max(diff) > 0
    
    def test_r0_effect(self, backend):
        """Test that r0 affects phase variance."""
        from nebraa.physics.kolmogorov import KolmogorovGenerator
        
        gen = KolmogorovGenerator(
            n_pix=64,
            pixel_size=0.01,
            actuator_pitch=0.20,
        )
        
        # Same seed for comparison
        phase_small_r0 = gen.generate(n_screens=10, r0=0.05, seed=42)
        phase_large_r0 = gen.generate(n_screens=10, r0=0.30, seed=42)
        
        var_small = backend.to_numpy(backend.xp.var(phase_small_r0))
        var_large = backend.to_numpy(backend.xp.var(phase_large_r0))
        
        # Smaller r0 (worse seeing) -> more phase variance
        assert var_small > var_large

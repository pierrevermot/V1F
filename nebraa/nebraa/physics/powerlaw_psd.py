"""
Dual Power-Law PSD Phase Generator

Generates residual phase screens from a power spectral density (PSD) defined
by two power-law components:
- Low-frequency (LF) component: PSD_LF(f) ∝ f^(-α_LF)
- High-frequency (HF) component: PSD_HF(f) ∝ f^(-α_HF)

The components are blended with a smooth transition at a configurable cutoff
frequency. Each component is scaled independently to achieve target RMS values.

This module supports:
- Short-exposure PSF computation (single phase screen realization)
- Long-exposure PSF computation (average over multiple realizations)
- Optional Low Wind Effect (LWE) integration

Author: NEBRAA
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional, Tuple, Dict, Union, Any
from dataclasses import dataclass

from ..utils.compute import get_backend
from .low_wind_effect import LowWindEffect, LowWindEffectConfig
from .phase_generator import PhaseGeneratorBase

# Type alias for array-like objects (numpy or cupy arrays)
ArrayLike = Any


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class DualPowerLawConfig:
    """
    Configuration for Dual Power-Law PSD model.
    
    The PSD is constructed as:
        PSD(f) = A_LF * f^(-α_LF) * W_LF(f) + A_HF * f^(-α_HF) * W_HF(f)
    
    where W_LF and W_HF are complementary smooth transition windows, and
    A_LF, A_HF are amplitude coefficients computed to achieve target RMS values.
    
    Attributes:
        alpha_lf: Power-law exponent for low-frequency component (typically 2-4)
        alpha_hf: Power-law exponent for high-frequency component (typically 2-4)
        rms_lf: Target RMS of low-frequency phase component (radians)
        rms_hf: Target RMS of high-frequency phase component (radians)
        f_cutoff: Cutoff frequency separating LF and HF regimes (cycles/meter)
        transition_width: Width of smooth transition as fraction of f_cutoff (0-1)
        n_realizations: Number of realizations for long-exposure averaging
        seed: Random seed for reproducibility. If None, random each time.
    """
    alpha_lf: float = 3.0
    alpha_hf: float = 11.0 / 3.0  # Kolmogorov default
    rms_lf: float = 0.5  # radians
    rms_hf: float = 0.3  # radians
    f_cutoff: float = 1.0  # cycles/meter
    transition_width: float = 0.2  # fraction of f_cutoff
    n_realizations: int = 100
    seed: Optional[int] = None


# =============================================================================
# Frequency Grid
# =============================================================================

@dataclass
class FrequencyGrid:
    """
    Spatial frequency grid for Fourier-domain operations.
    
    Provides frequency coordinates in cycles/meter using FFT conventions
    (unshifted, DC at corner).
    
    Attributes:
        n_pix: Grid size in pixels
        pixel_size: Physical pixel size in pupil plane (meters)
        FX, FY: 2D frequency coordinate grids (cycles/meter)
        F: Frequency magnitude grid
        df: Frequency resolution (cycles/meter)
        dA: Frequency area element (df²)
    """
    n_pix: int
    pixel_size: float
    
    def __post_init__(self):
        backend = get_backend()
        xp = backend.xp
        
        # Frequency arrays (unshifted FFT convention)
        fx = xp.fft.fftfreq(self.n_pix, d=float(self.pixel_size)).astype(xp.float64)
        fy = xp.fft.fftfreq(self.n_pix, d=float(self.pixel_size)).astype(xp.float64)
        self.FX, self.FY = xp.meshgrid(fx, fy)
        
        # Frequency magnitude (avoid zero for power-law evaluation)
        self.F = xp.sqrt(self.FX**2 + self.FY**2)
        self.F_safe = xp.maximum(self.F, xp.float64(1e-12))
        
        # Frequency resolution and area element
        self.df = 1.0 / (self.n_pix * float(self.pixel_size))
        self.dA = self.df ** 2
        
        # Nyquist frequency
        self.f_nyquist = 0.5 / float(self.pixel_size)


# =============================================================================
# Dual Power-Law PSD
# =============================================================================

class DualPowerLawPSD:
    """
    Dual Power-Law Power Spectral Density model.
    
    Constructs a PSD from two power-law components with independent RMS scaling
    and a smooth transition at the cutoff frequency.
    
    The PSD has units of rad² per (cycles/meter)².
    """
    
    def __init__(
        self,
        freq_grid: FrequencyGrid,
        config: DualPowerLawConfig,
    ):
        """
        Initialize dual power-law PSD.
        
        Args:
            freq_grid: FrequencyGrid instance
            config: DualPowerLawConfig with PSD parameters
        """
        self.freq_grid = freq_grid
        self.config = config
        
        backend = get_backend()
        self._xp = backend.xp
        
        # Build transition windows
        self._build_windows()
        
        # Build and scale PSD components
        self._build_psd()
    
    def _build_windows(self):
        """Build smooth transition windows for LF and HF components."""
        xp = self._xp
        F = self.freq_grid.F_safe
        fc = self.config.f_cutoff
        w = fc * self.config.transition_width
        
        # Transition boundaries
        f1 = max(fc - w, 1e-12)
        f2 = fc + w
        
        # Low-pass window (1 below f1, 0 above f2, smooth transition)
        W_LF = xp.ones_like(F, dtype=xp.float64)
        
        # Transition region
        mid_mask = (F >= f1) & (F <= f2)
        if xp.any(mid_mask):
            x = (F[mid_mask] - f1) / (f2 - f1)
            W_LF[mid_mask] = 0.5 * (1.0 + xp.cos(xp.pi * x))
        
        # Above transition
        W_LF[F > f2] = 0.0
        
        # High-pass window (complement)
        W_HF = 1.0 - W_LF
        
        self.W_LF = W_LF
        self.W_HF = W_HF
    
    def _build_psd(self):
        """Build and scale PSD components to target RMS values."""
        xp = self._xp
        F = self.freq_grid.F_safe
        dA = self.freq_grid.dA
        
        # Raw power-law shapes (unscaled)
        # PSD(f) = f^(-α)
        psd_lf_raw = F ** (-self.config.alpha_lf) * self.W_LF
        psd_hf_raw = F ** (-self.config.alpha_hf) * self.W_HF
        
        # Zero DC component to remove piston
        psd_lf_raw.flat[0] = 0.0
        psd_hf_raw.flat[0] = 0.0
        
        # Compute variance of raw components (Parseval's theorem)
        # var = ∫∫ PSD(f) d²f ≈ Σ PSD(f) × dA
        var_lf_raw = float(xp.sum(psd_lf_raw) * dA)
        var_hf_raw = float(xp.sum(psd_hf_raw) * dA)
        
        # Target variances from RMS values
        var_lf_target = self.config.rms_lf ** 2
        var_hf_target = self.config.rms_hf ** 2
        
        # Scale factors to achieve target RMS
        # A² × var_raw = var_target => A = sqrt(var_target / var_raw)
        self.scale_lf = math.sqrt(var_lf_target / var_lf_raw) if var_lf_raw > 0 else 0.0
        self.scale_hf = math.sqrt(var_hf_target / var_hf_raw) if var_hf_raw > 0 else 0.0
        
        # Scaled PSD components
        self.psd_lf = psd_lf_raw * (self.scale_lf ** 2)
        self.psd_hf = psd_hf_raw * (self.scale_hf ** 2)
        
        # Total PSD
        self.psd_total = self.psd_lf + self.psd_hf
        
        # Store variances for verification
        self.var_lf = float(xp.sum(self.psd_lf) * dA)
        self.var_hf = float(xp.sum(self.psd_hf) * dA)
        self.var_total = float(xp.sum(self.psd_total) * dA)
    
    @property
    def rms_lf_actual(self) -> float:
        """Actual RMS of LF component (should match config.rms_lf)."""
        return math.sqrt(self.var_lf)
    
    @property
    def rms_hf_actual(self) -> float:
        """Actual RMS of HF component (should match config.rms_hf)."""
        return math.sqrt(self.var_hf)
    
    @property
    def rms_total(self) -> float:
        """Total phase RMS (radians)."""
        return math.sqrt(self.var_total)
    
    @property
    def strehl_marechal(self) -> float:
        """Maréchal approximation for Strehl ratio: exp(-σ²)."""
        return math.exp(-self.var_total)


# =============================================================================
# Phase Screen Generator
# =============================================================================

class DualPowerLawPhaseGenerator(PhaseGeneratorBase):
    """
    Phase screen generator using dual power-law PSD.
    
    Generates phase screens by:
    1. Creating complex Gaussian white noise in Fourier domain
    2. Multiplying by sqrt(PSD) to color the spectrum
    3. Inverse FFT to get spatial phase screen
    
    The amplitude scaling follows NumPy FFT conventions to ensure
    the generated phase screens have the correct variance.
    
    This implements the unified PhaseGeneratorBase interface.
    
    Example:
        ```python
        # Create configuration
        config = DualPowerLawConfig(
            alpha_lf=3.0, alpha_hf=11/3,
            rms_lf=0.3, rms_hf=0.2,
            f_cutoff=0.5,
        )
        
        # Create generator
        gen = DualPowerLawPhaseGenerator(
            n_pix=256,
            pixel_size=0.032,  # 32mm per pixel
            psd_config=config,
            lwe_config=LowWindEffectConfig(piston_rms_rad=0.3),
        )
        
        # Generate phase screens
        phases = gen.generate(100, pupil)
        
        # With LWE
        phases_res, phases_lwe, phases_total = gen.generate_with_lwe(100, pupil)
        ```
    """
    
    def __init__(
        self,
        n_pix: int,
        pixel_size: float,
        psd_config: DualPowerLawConfig,
        seed: Optional[int] = None,
        lwe_config: Optional[LowWindEffectConfig] = None,
    ):
        """
        Initialize phase generator.
        
        Args:
            n_pix: Grid size in pixels
            pixel_size: Physical pixel size in pupil plane (meters)
            psd_config: DualPowerLawConfig with PSD parameters
            seed: Random seed (overrides psd_config.seed)
            lwe_config: Optional LWE configuration
        """
        # Use config seed unless explicitly overridden
        if seed is None:
            seed = psd_config.seed
        
        super().__init__(
            n_pix=n_pix,
            pixel_size=pixel_size,
            seed=seed,
            lwe_config=lwe_config,
        )
        
        self.psd_config = psd_config
        # Keep backward compat alias
        self.config = psd_config
        
        # Build frequency grid
        self.freq_grid = FrequencyGrid(n_pix, pixel_size)
        
        # Build PSD
        self.psd = DualPowerLawPSD(self.freq_grid, psd_config)
        
        # Precompute amplitude array for phase generation
        # Following kolmogorov.py convention:
        # A = sqrt(N^4 × PSD × df²) for real output from complex noise
        xp = self._xp
        self.amplitude = xp.sqrt(
            (n_pix ** 4) * self.psd.psd_total * self.freq_grid.dA
        ).astype(xp.float64)
        
        # Zero DC to ensure zero mean phase
        self.amplitude.flat[0] = 0.0
    
    @classmethod
    def from_telescope(
        cls,
        n_pix: int,
        telescope_diameter: float,
        psd_config: DualPowerLawConfig,
        seed: Optional[int] = None,
        lwe_config: Optional[LowWindEffectConfig] = None,
    ) -> "DualPowerLawPhaseGenerator":
        """
        Create generator from telescope parameters.
        
        Args:
            n_pix: Grid size in pixels
            telescope_diameter: Primary mirror diameter (meters)
            psd_config: PSD configuration
            seed: Random seed
            lwe_config: Optional LWE configuration
            
        Returns:
            DualPowerLawPhaseGenerator
        """
        pixel_size = telescope_diameter / n_pix
        return cls(
            n_pix=n_pix,
            pixel_size=pixel_size,
            psd_config=psd_config,
            seed=seed,
            lwe_config=lwe_config,
        )
    
    @classmethod
    def from_fourier_sampling(
        cls,
        n_pix: int,
        wavelength: float,
        pixel_scale_rad: float,
        psd_config: DualPowerLawConfig,
        seed: Optional[int] = None,
        lwe_config: Optional[LowWindEffectConfig] = None,
    ) -> "DualPowerLawPhaseGenerator":
        """
        Create generator from Fourier sampling requirements.
        
        Args:
            n_pix: Grid size in pixels
            wavelength: Observation wavelength (meters)
            pixel_scale_rad: PSF pixel scale (radians/pixel)
            psd_config: PSD configuration
            seed: Random seed
            lwe_config: Optional LWE configuration
            
        Returns:
            DualPowerLawPhaseGenerator
        """
        pixel_size = wavelength / (n_pix * pixel_scale_rad)
        return cls(
            n_pix=n_pix,
            pixel_size=pixel_size,
            psd_config=psd_config,
            seed=seed,
            lwe_config=lwe_config,
        )
    
    def generate(
        self,
        n_screens: int,
        pupil: Optional[ArrayLike] = None,
        seed: Optional[int] = None,
    ) -> ArrayLike:
        """
        Generate phase screens.
        
        Args:
            n_screens: Number of phase screens to generate
            pupil: Optional pupil mask for piston removal
            seed: Random seed (overrides config.seed if provided)
            
        Returns:
            Phase screens array of shape (n_screens, n_pix, n_pix) in radians
        """
        xp = self._xp
        n_pix = self.n_pix
        
        # Set random seed
        actual_seed = seed if seed is not None else self.config.seed
        if actual_seed is not None:
            if hasattr(xp, 'random'):
                xp.random.seed(actual_seed)
            else:
                np.random.seed(actual_seed)
        
        # Generate complex white noise
        noise_real = xp.random.randn(n_screens, n_pix, n_pix).astype(xp.float64)
        noise_imag = xp.random.randn(n_screens, n_pix, n_pix).astype(xp.float64)
        W = noise_real + 1j * noise_imag
        
        # Apply PSD coloring
        Phi_f = W * self.amplitude[None, :, :]
        
        # Transform to spatial domain (take real part)
        phi = xp.real(xp.fft.ifft2(Phi_f)).astype(xp.float64)
        
        # Remove piston over pupil if provided
        if pupil is not None:
            pupil = xp.asarray(pupil, dtype=xp.float64)
            pupil_sum = xp.sum(pupil)
            if pupil_sum > 0:
                mean_phi = xp.sum(phi * pupil[None, :, :], axis=(1, 2)) / pupil_sum
                phi = (phi - mean_phi[:, None, None]) * pupil[None, :, :]
        
        return phi
    
    @property
    def rms_expected(self) -> float:
        """Expected RMS of generated phase screens (radians)."""
        return self.psd.rms_total
    
    @property
    def info(self) -> Dict[str, Any]:
        """Return information about the PSD and generator."""
        base_info = super().info()
        base_info.update({
            'f_cutoff': self.psd_config.f_cutoff,
            'f_nyquist': self.freq_grid.f_nyquist,
            'alpha_lf': self.psd_config.alpha_lf,
            'alpha_hf': self.psd_config.alpha_hf,
            'rms_lf_target': self.psd_config.rms_lf,
            'rms_hf_target': self.psd_config.rms_hf,
            'rms_lf_actual': self.psd.rms_lf_actual,
            'rms_hf_actual': self.psd.rms_hf_actual,
            'rms_total': self.psd.rms_total,
            'strehl_marechal': self.psd.strehl_marechal,
        })
        return base_info


# =============================================================================
# PSF Model
# =============================================================================

class DualPowerLawPSFModel:
    """
    PSF model using dual power-law PSD phase screens.
    
    Supports:
    - Short-exposure PSF: Single phase screen realization
    - Long-exposure PSF: Average over multiple realizations
    - Optional LWE integration
    
    The PSF is computed using the direct method:
        OTF = |FT(pupil × exp(i × φ))|² normalized by |FT(pupil)|²
        PSF = IFT(OTF)
    """
    
    def __init__(
        self,
        n_pix: int,
        telescope_diameter: float,
        obstruction_diameter: float,
        wavelength: float,
        pixel_scale: float,
        psd_config: DualPowerLawConfig,
        lwe_config: Optional[LowWindEffectConfig] = None,
    ):
        """
        Initialize PSF model.
        
        Args:
            n_pix: Grid size in pixels
            telescope_diameter: Primary mirror diameter (meters)
            obstruction_diameter: Central obstruction diameter (meters)
            wavelength: Wavelength (meters)
            pixel_scale: Angular pixel scale (radians/pixel)
            psd_config: DualPowerLawConfig for phase generation
            lwe_config: Optional LowWindEffectConfig for LWE
        """
        self.n_pix = n_pix
        self.telescope_diameter = telescope_diameter
        self.obstruction_diameter = obstruction_diameter
        self.wavelength = wavelength
        self.pixel_scale = pixel_scale
        self.psd_config = psd_config
        self.lwe_config = lwe_config
        
        backend = get_backend()
        self._xp = backend.xp
        
        # Pupil plane pixel size
        # pixel_scale [rad/pix] = λ / (N × d_pupil) => d_pupil = λ / (N × pixel_scale)
        self.pupil_pixel_size = wavelength / (n_pix * pixel_scale)
        
        # Initialize phase generator
        self.phase_generator = DualPowerLawPhaseGenerator(
            n_pix=n_pix,
            pixel_size=self.pupil_pixel_size,
            psd_config=psd_config,
        )
        
        # LWE model (lazy initialization)
        self._lwe_model: Optional[LowWindEffect] = None
    
    def compute_telescope_otf(self, pupil: ArrayLike) -> ArrayLike:
        """
        Compute telescope OTF from pupil function.
        
        OTF_tsc = autocorr(pupil) / max(autocorr)
        
        Args:
            pupil: 2D pupil amplitude mask
            
        Returns:
            Normalized telescope OTF (centered)
        """
        xp = self._xp
        pupil = xp.asarray(pupil, dtype=xp.float64)
        
        # Autocorrelation via FFT
        Pupil_f = xp.fft.fft2(xp.fft.ifftshift(pupil))
        autocorr = xp.fft.ifft2(xp.abs(Pupil_f) ** 2)
        autocorr = xp.fft.fftshift(xp.real(autocorr))
        
        # Normalize by peak
        OTF_tsc = autocorr / xp.max(autocorr)
        
        return OTF_tsc
    
    def compute_short_exposure_psf(
        self,
        pupil: ArrayLike,
        phase_screen: Optional[ArrayLike] = None,
        seed: Optional[int] = None,
    ) -> ArrayLike:
        """
        Compute short-exposure PSF from a single phase screen realization.
        
        Uses direct OTF computation:
            OTF = |FT(pupil × exp(i × φ))|²
        
        Args:
            pupil: 2D pupil amplitude mask
            phase_screen: Optional pre-computed phase screen. If None, generates one.
            seed: Random seed for phase generation (if phase_screen is None)
            
        Returns:
            Normalized PSF (sum = 1)
        """
        xp = self._xp
        pupil = xp.asarray(pupil, dtype=xp.float64)
        
        # Generate phase screen if not provided
        if phase_screen is None:
            phase_screen = self.phase_generator.generate(1, pupil=pupil, seed=seed)[0]
        else:
            phase_screen = xp.asarray(phase_screen, dtype=xp.float64)
        
        # Complex pupil function with phase
        pupil_complex = pupil * xp.exp(1j * phase_screen)
        
        # Fourier transform
        Pupil_f = xp.fft.fft2(xp.fft.ifftshift(pupil_complex))
        
        # PSF is squared magnitude of FT (intensity)
        PSF = xp.abs(Pupil_f) ** 2
        PSF = xp.fft.fftshift(PSF)
        
        # Normalize
        PSF = PSF / xp.sum(PSF)
        
        return PSF
    
    def compute_long_exposure_psf(
        self,
        pupil: ArrayLike,
        n_realizations: Optional[int] = None,
        include_lwe: bool = True,
        seed: Optional[int] = None,
        return_components: bool = False,
    ) -> Union[ArrayLike, Dict[str, ArrayLike]]:
        """
        Compute long-exposure PSF by averaging over multiple realizations.
        
        Args:
            pupil: 2D pupil amplitude mask
            n_realizations: Number of realizations (default: from psd_config)
            include_lwe: If True and lwe_config is set, include LWE
            seed: Random seed for reproducibility
            return_components: If True, return dict with intermediate results
            
        Returns:
            If return_components=False: Normalized PSF array
            If return_components=True: Dict with 'psf', 'psf_screens', 'phase_screens'
        """
        xp = self._xp
        pupil = xp.asarray(pupil, dtype=xp.float64)
        
        # Number of realizations
        n_real = n_realizations if n_realizations is not None else self.psd_config.n_realizations
        
        # Use seed from config if not provided
        actual_seed = seed if seed is not None else self.psd_config.seed
        
        # Generate all phase screens at once
        phase_screens = self.phase_generator.generate(n_real, pupil=pupil, seed=actual_seed)
        
        # Generate LWE phase screens if configured
        lwe_phases = None
        if include_lwe and self.lwe_config is not None:
            # Initialize LWE model if needed
            if self._lwe_model is None:
                self._lwe_model = LowWindEffect(
                    pupil=pupil,
                    piston_rms_rad=self.lwe_config.piston_rms_rad,
                    tilt_rms_rad=self.lwe_config.tilt_rms_rad,
                    ar_coeff=self.lwe_config.ar_coeff,
                )
            
            # Generate LWE phases with appropriate seed
            lwe_seed = self.lwe_config.seed
            lwe_phases = self._lwe_model.generate(n_real, seed=lwe_seed)
        
        # Accumulate PSFs
        PSF_sum = xp.zeros((self.n_pix, self.n_pix), dtype=xp.float64)
        
        if return_components:
            psf_screens = []
        
        for i in range(n_real):
            # Get phase screen for this realization
            phase = phase_screens[i]
            
            # Add LWE if available
            if lwe_phases is not None:
                phase = phase + lwe_phases[i]
            
            # Compute short-exposure PSF
            pupil_complex = pupil * xp.exp(1j * phase)
            Pupil_f = xp.fft.fft2(xp.fft.ifftshift(pupil_complex))
            PSF_i = xp.abs(Pupil_f) ** 2
            PSF_i = xp.fft.fftshift(PSF_i)
            PSF_i = PSF_i / xp.sum(PSF_i)
            
            PSF_sum += PSF_i
            
            if return_components:
                psf_screens.append(PSF_i)
        
        # Average
        PSF_avg = PSF_sum / n_real
        
        # Ensure normalization
        PSF_avg = PSF_avg / xp.sum(PSF_avg)
        
        if return_components:
            return {
                'psf': PSF_avg,
                'psf_screens': xp.stack(psf_screens, axis=0),
                'phase_screens': phase_screens,
                'lwe_phases': lwe_phases,
            }
        
        return PSF_avg
    
    def compute_diffraction_limited_psf(self, pupil: ArrayLike) -> ArrayLike:
        """
        Compute diffraction-limited PSF (no aberrations).
        
        Args:
            pupil: 2D pupil amplitude mask
            
        Returns:
            Normalized diffraction-limited PSF
        """
        xp = self._xp
        pupil = xp.asarray(pupil, dtype=xp.float64)
        
        # PSF is |FT(pupil)|²
        Pupil_f = xp.fft.fft2(xp.fft.ifftshift(pupil))
        PSF_dl = xp.abs(Pupil_f) ** 2
        PSF_dl = xp.fft.fftshift(PSF_dl)
        PSF_dl = PSF_dl / xp.sum(PSF_dl)
        
        return PSF_dl
    
    def compute_strehl_ratio(
        self,
        pupil: ArrayLike,
        psf: Optional[ArrayLike] = None,
        n_realizations: Optional[int] = None,
    ) -> float:
        """
        Compute Strehl ratio.
        
        Strehl = peak(PSF) / peak(PSF_DL)
        
        Args:
            pupil: 2D pupil amplitude mask
            psf: Pre-computed PSF. If None, computes long-exposure PSF.
            n_realizations: Number of realizations if computing PSF
            
        Returns:
            Strehl ratio
        """
        xp = self._xp
        
        # Get PSF if not provided
        if psf is None:
            psf = self.compute_long_exposure_psf(
                pupil, n_realizations=n_realizations, include_lwe=False
            )
        
        # Compute DL PSF
        psf_dl = self.compute_diffraction_limited_psf(pupil)
        
        # Strehl is ratio of peaks
        strehl = float(xp.max(psf) / xp.max(psf_dl))
        
        return strehl
    
    @property
    def info(self) -> Dict[str, Any]:
        """Return model information."""
        info = {
            'n_pix': self.n_pix,
            'telescope_diameter': self.telescope_diameter,
            'obstruction_diameter': self.obstruction_diameter,
            'wavelength': self.wavelength,
            'pixel_scale_rad': self.pixel_scale,
            'pixel_scale_mas': self.pixel_scale * 180 * 3600 * 1000 / math.pi,
            'pupil_pixel_size': self.pupil_pixel_size,
            'has_lwe': self.lwe_config is not None,
        }
        info.update(self.phase_generator.info)
        return info


# =============================================================================
# Convenience Functions
# =============================================================================

def create_simple_psd_config(
    rms_total: float = 1.0,
    lf_fraction: float = 0.7,
    alpha_lf: float = 3.0,
    alpha_hf: float = 11.0 / 3.0,
    f_cutoff: float = 1.0,
    transition_width: float = 0.2,
    n_realizations: int = 100,
    seed: Optional[int] = None,
) -> DualPowerLawConfig:
    """
    Create a DualPowerLawConfig with simplified parameters.
    
    Instead of specifying separate LF and HF RMS values, specify total RMS
    and the fraction allocated to low frequencies.
    
    Args:
        rms_total: Total phase RMS (radians)
        lf_fraction: Fraction of variance in LF component (0-1)
        alpha_lf: LF power-law exponent
        alpha_hf: HF power-law exponent (default: Kolmogorov 11/3)
        f_cutoff: Cutoff frequency (cycles/meter)
        transition_width: Transition width as fraction of f_cutoff
        n_realizations: Number of realizations for long-exposure
        seed: Random seed
        
    Returns:
        DualPowerLawConfig
    """
    # Total variance
    var_total = rms_total ** 2
    
    # Split variance between LF and HF
    var_lf = var_total * lf_fraction
    var_hf = var_total * (1 - lf_fraction)
    
    # Convert to RMS
    rms_lf = math.sqrt(var_lf)
    rms_hf = math.sqrt(var_hf)
    
    return DualPowerLawConfig(
        alpha_lf=alpha_lf,
        alpha_hf=alpha_hf,
        rms_lf=rms_lf,
        rms_hf=rms_hf,
        f_cutoff=f_cutoff,
        transition_width=transition_width,
        n_realizations=n_realizations,
        seed=seed,
    )

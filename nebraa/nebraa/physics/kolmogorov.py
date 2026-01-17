"""
Kolmogorov atmospheric turbulence models.

Provides high-frequency phase screen generation for spatial frequencies
above the AO correction cutoff using the Kolmogorov power spectral density.
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

from ..utils.compute import get_backend


# =============================================================================
# Kolmogorov Power Spectral Density
# =============================================================================

def kolmogorov_psd(f, r0: float, C: float = 0.023):
    """
    Compute Kolmogorov phase power spectral density.
    
    W_phi(f) = C * r0^(-5/3) * (2*pi*f)^(-11/3)
    
    Args:
        f: Spatial frequency array (cycles/meter)
        r0: Fried parameter (meters)
        C: Kolmogorov constant (default 0.023)
    
    Returns:
        Power spectral density values
    """
    backend = get_backend()
    xp = backend.xp
    
    # Avoid division by zero
    f = xp.maximum(f, xp.float32(1e-10))
    
    return C * (r0 ** (-5/3)) * ((2 * xp.pi * f) ** (-11/3))


# =============================================================================
# Frequency Grid
# =============================================================================

class FrequencyGrid:
    """
    Spatial frequency grid for pupil-plane operations.
    
    Provides frequency coordinates in cycles/meter for Fourier-domain
    atmospheric modeling.
    """
    
    def __init__(self, n_pix: int, pixel_size: float):
        """
        Initialize frequency grid.
        
        Args:
            n_pix: Grid size in pixels
            pixel_size: Physical pixel size in pupil plane (meters)
        """
        backend = get_backend()
        xp = backend.xp
        
        self.n_pix = n_pix
        self.pixel_size = pixel_size
        
        # Frequency grid in cycles/meter
        fx = xp.fft.fftfreq(n_pix, d=float(pixel_size)).astype(xp.float32)
        fy = xp.fft.fftfreq(n_pix, d=float(pixel_size)).astype(xp.float32)
        self.FX, self.FY = xp.meshgrid(fx, fy)
        
        # Frequency magnitude
        self.F = xp.sqrt(self.FX**2 + self.FY**2)
        self.F = xp.maximum(self.F, xp.float32(1e-10))
        
        # Frequency resolution and area element
        self.df = 1.0 / (n_pix * float(pixel_size))
        self.dA = self.df ** 2


# =============================================================================
# Frequency Domain Filtering
# =============================================================================

def raised_cosine_filter(F, fc: float, transition_frac: float = 0.15, highpass: bool = True):
    """
    Create raised cosine frequency filter.
    
    Args:
        F: Frequency magnitude array
        fc: Cutoff frequency (cycles/meter)
        transition_frac: Transition width as fraction of fc
        highpass: If True, return high-pass; else low-pass
    
    Returns:
        Filter array
    """
    backend = get_backend()
    xp = backend.xp
    
    if transition_frac <= 0:
        # Hard cutoff
        if highpass:
            return (F >= fc).astype(xp.float32)
        else:
            return (F < fc).astype(xp.float32)
    
    # Soft cutoff
    w = fc * transition_frac
    f1 = fc - w
    f2 = fc + w
    
    low_pass = xp.ones_like(F, dtype=xp.float32)
    
    # Transition region
    mid_mask = (F >= f1) & (F <= f2)
    x = (F[mid_mask] - f1) / (f2 - f1)
    low_pass[mid_mask] = 0.5 * (1.0 + xp.cos(xp.pi * x))
    
    # Above transition
    low_pass[F > f2] = 0.0
    
    if highpass:
        return xp.float32(1.0) - low_pass
    return low_pass


# =============================================================================
# Kolmogorov Phase Generator
# =============================================================================

class KolmogorovGenerator:
    """
    High-frequency Kolmogorov phase screen generator.
    
    Generates atmospheric phase screens for spatial frequencies above
    the AO correction cutoff, scaled by the Fried parameter r0.
    """
    
    def __init__(
        self,
        n_pix: int,
        pixel_size: float,
        actuator_pitch: float,
        transition_frac: float = 0.15,
    ):
        """
        Initialize generator.
        
        Args:
            n_pix: Grid size
            pixel_size: Pupil plane pixel size (meters)
            actuator_pitch: DM actuator spacing (meters)
            transition_frac: Smooth transition width
        """
        backend = get_backend()
        xp = backend.xp
        
        self.n_pix = n_pix
        self.freq_grid = FrequencyGrid(n_pix, pixel_size)
        
        # Cutoff frequency from Nyquist criterion
        self.fc = 1.0 / (2.0 * actuator_pitch)
        
        # High-pass filter
        self.HP = raised_cosine_filter(
            self.freq_grid.F, self.fc, transition_frac, highpass=True
        )
        
        # Kolmogorov constant for PSD in rad^2/(cy/m)^2
        # W_phi(f) = C * r0^(-5/3) * f^(-11/3) with C = 0.023
        self.C = 0.023
        
        # FFT normalization for phase screen generation:
        # 
        # NumPy's ifft2 includes a 1/N^2 factor: ifft2(X) = (1/N^2) * sum(X * exp)
        # From Parseval: var(x) = (1/N^4) * E[sum|X|^2]
        # 
        # For complex Gaussian noise Z with E[|Z|^2] = 2 and spectrum X = A * Z:
        #   E[|X|^2] = A^2 * 2
        # 
        # We want: var(phi) = sum(W_phi * df^2) where df = 1/(N*pixel_size)
        # So: (1/N^4) * sum(A^2 * 2) = sum(W_phi * df^2)
        # => A = sqrt(N^4 * W_phi * df^2 / 2)
        #
        # However, since we take only the real part of ifft2(X), we lose half the
        # variance (the imaginary part). To compensate, we use:
        #   A = sqrt(N^4 * W_phi * df^2)  (no division by 2)
        #
        # This is equivalent to: A = N^2 * sqrt(W_phi) * df = N * sqrt(W_phi) / pixel_size / N
        #                         = sqrt(W_phi) / pixel_size * N
        self.df = 1.0 / (n_pix * pixel_size)
        
        # Precomputed frequency term
        self.f_term = (self.freq_grid.F ** (-11.0 / 3.0)).astype(xp.float32)
    
    def generate(self, n_screens: int, r0, pupil=None, seed: Optional[int] = None):
        """
        Generate high-frequency phase screens.
        
        Args:
            n_screens: Number of screens
            r0: Fried parameter (scalar or array of length n_screens)
            pupil: Optional pupil mask
            seed: Random seed
        
        Returns:
            Phase screens in radians, shape (n_screens, n_pix, n_pix)
        """
        backend = get_backend()
        xp = backend.xp
        n_pix = self.n_pix
        
        if seed is not None:
            xp.random.seed(seed)
        
        # Ensure r0 is array
        r0 = xp.asarray(r0, dtype=xp.float32)
        if r0.ndim == 0:
            r0 = xp.full(n_screens, float(r0), dtype=xp.float32)
        
        # Complex white noise
        noise_real = xp.random.randn(n_screens, n_pix, n_pix).astype(xp.float32)
        noise_imag = xp.random.randn(n_screens, n_pix, n_pix).astype(xp.float32)
        W = (noise_real + 1j * noise_imag).astype(xp.complex64)
        
        # PSD amplitude with FFT normalization
        # A = sqrt(N^4 * W_phi * df^2) where W_phi = C * r0^(-5/3) * f^(-11/3)
        # This simplifies to: A = N^2 * df * sqrt(W_phi) = N * sqrt(W_phi) / pixel_size
        r0_factor = (r0 ** (-5.0 / 3.0))[:, None, None]
        amplitude = xp.sqrt(
            self.C * r0_factor * self.f_term[None, :, :] * (n_pix ** 4) * (self.df ** 2)
        ).astype(xp.float32)
        
        # Apply PSD and high-pass filter
        Phi_f = W * amplitude * self.HP[None, :, :]
        Phi_f[:, 0, 0] = 0.0 + 0.0j  # Zero DC
        
        # Transform to spatial domain (take real part only)
        # The amplitude formula already accounts for the factor of 2 lost when
        # taking the real part of a non-Hermitian spectrum
        phi = xp.real(xp.fft.ifft2(Phi_f)).astype(xp.float32)
        
        # Remove piston over pupil
        if pupil is not None:
            pupil = backend.ensure_local(pupil).astype(xp.float32)
            pupil_sum = xp.sum(pupil)
            mean_phi = xp.sum(phi * pupil[None, :, :], axis=(1, 2)) / pupil_sum
            phi = (phi - mean_phi[:, None, None]) * pupil[None, :, :]
        
        return phi


# =============================================================================
# Combined Atmosphere Model
# =============================================================================

class AtmosphereModel:
    """
    Combined atmospheric phase model.
    
    Combines low-frequency (Zernike-based) and high-frequency (Kolmogorov)
    components with optional frequency-domain separation.
    """
    
    def __init__(
        self,
        n_pix: int,
        wavelength: float,
        pixel_scale: float,
        pupil_radius: float,
        actuator_pitch: Optional[float] = None,
        hf_transition: float = 0.15,
        zernike_n_range: Tuple[int, int] = (2, 5),
    ):
        """
        Initialize atmosphere model.
        
        Args:
            n_pix: Grid size
            wavelength: Wavelength (meters)
            pixel_scale: Pixel scale (rad/pixel)
            pupil_radius: Pupil radius (meters)
            actuator_pitch: DM pitch for HF cutoff. None disables HF.
            hf_transition: Smooth transition width
            zernike_n_range: (n_min, n_max) for Zernike modes
        """
        self.n_pix = n_pix
        self.wavelength = wavelength
        self.pixel_scale = pixel_scale
        self.pupil_radius = pupil_radius
        self.zernike_n_range = zernike_n_range
        
        # Pupil plane pixel size
        self.pupil_pixel_size = wavelength / n_pix / pixel_scale
        
        # High-frequency model
        self.hf_enabled = actuator_pitch is not None
        if self.hf_enabled:
            self.hf_generator = KolmogorovGenerator(
                n_pix, self.pupil_pixel_size, actuator_pitch, hf_transition
            )
            # Low-pass filter for LF
            self.LP = raised_cosine_filter(
                self.hf_generator.freq_grid.F,
                self.hf_generator.fc,
                hf_transition,
                highpass=False
            )
        else:
            self.hf_generator = None
            self.LP = None
    
    def generate_lf_phase(
        self,
        n_screens: int,
        power_law: float = 2.0,
        seed: Optional[int] = None,
    ):
        """
        Generate low-frequency Zernike phase screens.
        
        Returns unnormalized phase (call normalize_and_scale separately).
        """
        from .zernike import generate_zernike_phase
        
        return generate_zernike_phase(
            n_screens,
            self.n_pix,
            self.pupil_radius,
            self.zernike_n_range,
            power_law,
            seed,
        )
    
    def generate_hf_phase(self, n_screens: int, r0, pupil=None, seed: Optional[int] = None):
        """
        Generate high-frequency Kolmogorov phase screens.
        
        Returns None if HF model is disabled.
        """
        if not self.hf_enabled:
            return None
        return self.hf_generator.generate(n_screens, r0, pupil, seed)
    
    def normalize_and_scale(self, phase, pupil, rms_meters):
        """
        Normalize phase to unit RMS, then scale to target OPD.
        
        Also applies low-pass filter if HF is enabled.
        
        Args:
            phase: Raw phase screens
            pupil: Pupil mask
            rms_meters: Target RMS OPD (scalar or per-screen)
        
        Returns:
            Phase in radians
        """
        from .zernike import normalize_phase_rms
        
        backend = get_backend()
        xp = backend.xp
        
        # Normalize to target RMS OPD
        phase_opd = normalize_phase_rms(phase, pupil, rms_meters)
        
        # Convert OPD to phase (radians)
        phase_rad = phase_opd * (2 * xp.pi / self.wavelength)
        
        # Apply low-pass filter if HF enabled
        if self.hf_enabled and self.LP is not None:
            LP = backend.ensure_local(self.LP)
            pupil_local = backend.ensure_local(pupil).astype(xp.float32)
            pupil_sum = xp.sum(pupil_local)
            
            # Handle single/batch
            single = phase_rad.ndim == 2
            if single:
                phase_rad = phase_rad[None, :, :]
            
            # Filter in Fourier domain
            Phi = xp.fft.fft2(phase_rad)
            phase_rad = xp.real(xp.fft.ifft2(Phi * LP[None, :, :])).astype(xp.float32)
            
            # Remove filtering-induced piston
            mean = xp.sum(phase_rad * pupil_local[None, :, :], axis=(1, 2)) / pupil_sum
            phase_rad = (phase_rad - mean[:, None, None]) * pupil_local[None, :, :]
            
            if single:
                phase_rad = phase_rad[0]
        
        return phase_rad

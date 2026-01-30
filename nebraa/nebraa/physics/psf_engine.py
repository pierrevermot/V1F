"""
PSF Engine - Unified PSF Computation Module

Provides consistent PSF computation from phase screens and pupil masks.
Extracted from individual models to ensure uniform behavior across all
phase generation methods.

Features:
- Short-exposure PSF (single phase realization)
- Long-exposure PSF (average over realizations)
- Diffraction-limited reference PSF
- Strehl ratio computation
- Consistent normalization (sum=1 by default)

Author: NEBRAA
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Union, Any
from dataclasses import dataclass

from ..utils.compute import get_backend

# Type alias for array-like objects (numpy or cupy arrays)
ArrayLike = Any


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PSFEngineConfig:
    """
    Configuration for PSF computation.
    
    Attributes:
        n_pix: Grid size in pixels
        wavelength: Observation wavelength (meters)
        pixel_scale: PSF pixel scale (radians/pixel)
        normalize_to: Normalization mode ('sum' or 'peak')
        zero_pad_factor: Zero-padding factor for FFT (1 = no padding)
    """
    n_pix: int
    wavelength: float
    pixel_scale: float  # radians/pixel
    normalize_to: str = "sum"  # 'sum', 'peak', or 'none'
    zero_pad_factor: int = 1
    
    @property
    def pupil_pixel_size(self) -> float:
        """Physical pixel size in pupil plane (meters)."""
        return self.wavelength / (self.n_pix * self.pixel_scale)
    
    @property
    def pupil_extent(self) -> float:
        """Total physical extent of pupil grid (meters)."""
        return self.n_pix * self.pupil_pixel_size
    
    @property
    def airy_null_pix(self) -> float:
        """
        Airy disk first null in pixels.
        
        For a circular aperture of diameter D:
        θ_null = 1.22 × λ/D (radians)
        
        This requires knowing telescope diameter, so returns lambda/D scaling.
        """
        # Return lambda/pixel_scale in pixels (user divides by D to get actual)
        return self.wavelength / self.pixel_scale


# =============================================================================
# PSF Engine
# =============================================================================

class PSFEngine:
    """
    Unified PSF computation engine.
    
    Computes PSFs from phase screens using the direct FFT method:
        E_focal = FFT(pupil × exp(i × phase))
        PSF = |E_focal|²
    
    Supports:
    - Single phase screen → short-exposure PSF
    - Multiple phase screens → individual PSFs or averaged long-exposure
    - Diffraction-limited reference PSF
    - Strehl ratio computation
    
    All PSFs are normalized to sum=1 (energy conservation) by default.
    """
    
    def __init__(
        self,
        n_pix: int,
        wavelength: float,
        pixel_scale: float,
        normalize_to: str = "sum",
        zero_pad_factor: int = 1,
    ):
        """
        Initialize PSF engine.
        
        Args:
            n_pix: Grid size in pixels
            wavelength: Observation wavelength (meters)
            pixel_scale: PSF pixel scale (radians/pixel)
            normalize_to: 'sum' (default, energy conservation), 'peak' (peak=1), or 'none' (raw)
            zero_pad_factor: Zero-padding factor for FFT (1 = no padding, 2 = pad to 2x size, etc.)
        """
        self.n_pix = n_pix
        self.wavelength = wavelength
        self.pixel_scale = pixel_scale
        self.normalize_to = normalize_to
        self.zero_pad_factor = zero_pad_factor
        
        backend = get_backend()
        self._xp = backend.xp
        
        # Derived quantities
        self.pupil_pixel_size = wavelength / (n_pix * pixel_scale)
        self.pupil_extent = n_pix * self.pupil_pixel_size
        
        # Padded grid size
        self.n_pix_padded = n_pix * zero_pad_factor
        self.pixel_scale_effective = pixel_scale / zero_pad_factor
    
    @classmethod
    def from_config(cls, config: PSFEngineConfig) -> "PSFEngine":
        """Create PSFEngine from configuration."""
        return cls(
            n_pix=config.n_pix,
            wavelength=config.wavelength,
            pixel_scale=config.pixel_scale,
            normalize_to=config.normalize_to,
            zero_pad_factor=config.zero_pad_factor,
        )
    
    def compute_psf(
        self,
        pupil: ArrayLike,
        phase: Optional[ArrayLike] = None,
        normalize: bool = True,
    ) -> ArrayLike:
        """
        Compute PSF from pupil and phase.
        
        Args:
            pupil: 2D pupil amplitude mask (n_pix, n_pix)
            phase: 2D phase screen in radians (n_pix, n_pix). If None, returns DL PSF.
            normalize: Whether to normalize the PSF
            
        Returns:
            2D PSF array (n_pix_padded, n_pix_padded) where n_pix_padded = n_pix * zero_pad_factor
        """
        xp = self._xp
        
        # Ensure inputs are on correct backend
        pupil = xp.asarray(pupil)
        
        # Build complex field
        if phase is None:
            E_pupil = pupil.astype(xp.complex128)
        else:
            phase = xp.asarray(phase)
            E_pupil = pupil * xp.exp(1j * phase.astype(xp.float64))
        
        # Apply zero-padding if requested
        if self.zero_pad_factor > 1:
            E_pupil = self._zero_pad(E_pupil)
        
        # FFT to focal plane (standard convention: ifftshift before fft2)
        # This ensures proper centering regardless of input pupil indexing
        E_focal = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(E_pupil)))
        
        # Intensity
        psf = xp.abs(E_focal) ** 2
        
        # Normalize
        if normalize:
            psf = self._normalize(psf)
        
        return psf.astype(xp.float32)
    
    def compute_psf_batch(
        self,
        pupil: ArrayLike,
        phases: ArrayLike,
        normalize: bool = True,
        return_individual: bool = False,
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        """
        Compute PSFs for batch of phase screens.
        
        Args:
            pupil: 2D pupil amplitude mask (n_pix, n_pix)
            phases: 3D phase screens (n_screens, n_pix, n_pix) in radians
            normalize: Whether to normalize PSFs
            return_individual: If True, return individual PSFs as well as average
            
        Returns:
            If return_individual:
                (psf_avg, psfs_individual) tuple
            Else:
                psf_avg: Averaged long-exposure PSF (n_pix_padded, n_pix_padded)
        """
        xp = self._xp
        
        # Ensure inputs are on correct backend
        pupil = xp.asarray(pupil)
        phases = xp.asarray(phases)
        
        n_screens = phases.shape[0]
        
        # Build complex field for entire batch: (B, N, N)
        # Broadcast pupil to batch dimension
        E_pupil = pupil[None, :, :] * xp.exp(1j * phases.astype(xp.float64))
        
        # Apply zero-padding if requested
        if self.zero_pad_factor > 1:
            E_pupil = self._zero_pad_batch(E_pupil)
        
        # Batch FFT to focal plane (standard convention: ifftshift before fft2)
        # This ensures proper centering regardless of input pupil indexing
        E_focal = xp.fft.fftshift(
            xp.fft.fft2(xp.fft.ifftshift(E_pupil, axes=(-2, -1)), axes=(-2, -1)),
            axes=(-2, -1)
        )
        
        # Batch intensity
        psfs = xp.abs(E_focal) ** 2
        
        # Normalize each PSF in batch
        if normalize:
            psfs = self._normalize_batch(psfs)
        else:
            psfs = psfs.astype(xp.float32)
        
        # Compute average
        psf_avg = xp.mean(psfs, axis=0)
        
        # Re-normalize average (should already sum to ~1, but ensure it)
        if normalize:
            psf_avg = self._normalize(psf_avg)
        
        if return_individual:
            return psf_avg, psfs
        return psf_avg
    
    def compute_long_exposure_psf(
        self,
        pupil: ArrayLike,
        phases: ArrayLike,
        phases_lwe: Optional[ArrayLike] = None,
        normalize: bool = True,
    ) -> ArrayLike:
        """
        Compute long-exposure PSF by averaging over realizations.
        
        Args:
            pupil: 2D pupil amplitude mask
            phases: Phase screens (n_screens, n_pix, n_pix) in radians
            phases_lwe: Optional LWE phase screens to add
            normalize: Whether to normalize the PSF (default True)
            
        Returns:
            Long-exposure PSF (n_pix, n_pix), normalized according to normalize parameter
        """
        xp = self._xp
        
        # Ensure inputs on correct backend
        phases = xp.asarray(phases)
        
        # Combine with LWE if provided
        if phases_lwe is not None:
            phases_lwe = xp.asarray(phases_lwe)
            phases_total = phases + phases_lwe
        else:
            phases_total = phases
        
        return self.compute_psf_batch(pupil, phases_total, normalize=normalize)
    
    def compute_diffraction_limited_psf(self, pupil: ArrayLike, normalize: bool = True) -> ArrayLike:
        """
        Compute diffraction-limited (no aberrations) PSF.
        
        Args:
            pupil: 2D pupil amplitude mask
            normalize: Whether to normalize the PSF (default True)
            
        Returns:
            Diffraction-limited PSF (n_pix, n_pix), normalized according to normalize parameter
        """
        return self.compute_psf(pupil, phase=None, normalize=normalize)
    
    def compute_strehl_ratio(
        self,
        pupil: ArrayLike,
        phase: Optional[ArrayLike] = None,
        psf: Optional[ArrayLike] = None,
    ) -> float:
        """
        Compute Strehl ratio.
        
        Strehl = peak(aberrated PSF) / peak(diffraction-limited PSF)
        
        Args:
            pupil: 2D pupil amplitude mask
            phase: Phase screen (if psf not provided)
            psf: Pre-computed PSF (if phase not provided)
            
        Returns:
            Strehl ratio (0 to 1)
        """
        xp = self._xp
        
        # Get diffraction-limited PSF peak (always normalized for fair comparison)
        psf_dl = self.compute_diffraction_limited_psf(pupil, normalize=True)
        peak_dl = float(xp.max(psf_dl))
        
        # Get aberrated PSF peak (always normalized for fair comparison)
        if psf is not None:
            # Ensure consistent normalization
            psf_ab = self._normalize(xp.asarray(psf))
        elif phase is not None:
            psf_ab = self.compute_psf(pupil, phase, normalize=True)
        else:
            raise ValueError("Must provide either phase or psf")
        
        peak_ab = float(xp.max(psf_ab))
        
        # Strehl ratio
        strehl = peak_ab / peak_dl if peak_dl > 0 else 0.0
        
        return strehl
    
    def compute_strehl_from_rms(self, rms_rad: float) -> float:
        """
        Compute Maréchal approximation Strehl from phase RMS.
        
        S ≈ exp(-σ²) for small aberrations
        
        Args:
            rms_rad: Phase RMS in radians
            
        Returns:
            Approximate Strehl ratio
        """
        return math.exp(-rms_rad**2)
    
    def compute_otf(
        self,
        pupil: ArrayLike,
        phase: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """
        Compute Optical Transfer Function.
        
        OTF = autocorrelation(pupil × exp(i × phase)), normalized
        
        Args:
            pupil: 2D pupil amplitude mask
            phase: Optional phase screen
            
        Returns:
            Complex OTF array (n_pix, n_pix)
        """
        xp = self._xp
        
        # Ensure inputs on correct backend
        pupil = xp.asarray(pupil)
        
        # Build complex field
        if phase is None:
            E_pupil = pupil.astype(xp.complex128)
        else:
            phase = xp.asarray(phase)
            E_pupil = pupil * xp.exp(1j * phase.astype(xp.float64))
        
        # OTF via autocorrelation theorem
        # OTF = FFT^-1(|FFT(E)|²) = autocorr(E)
        # Use standard convention with ifftshift
        E_ft = xp.fft.fft2(xp.fft.ifftshift(E_pupil))
        autocorr = xp.fft.ifft2(xp.abs(E_ft) ** 2)
        
        # Normalize by DC value (= integral of |E|²)
        autocorr_shifted = xp.fft.fftshift(autocorr)
        dc = autocorr_shifted[self.n_pix // 2, self.n_pix // 2]
        
        otf = autocorr_shifted / dc if xp.abs(dc) > 0 else autocorr_shifted
        
        return otf
    
    def compute_telescope_otf(self, pupil: ArrayLike) -> ArrayLike:
        """
        Compute telescope (diffraction-limited) OTF.
        
        This is the autocorrelation of the pupil function.
        
        Args:
            pupil: 2D pupil amplitude mask
            
        Returns:
            Real-valued OTF array (n_pix, n_pix)
        """
        xp = self._xp
        return xp.real(self.compute_otf(pupil, phase=None))
    
    def _zero_pad(self, field: ArrayLike) -> ArrayLike:
        """
        Zero-pad a 2D field to n_pix_padded.
        
        Args:
            field: 2D complex field (n_pix, n_pix)
            
        Returns:
            Zero-padded field (n_pix_padded, n_pix_padded)
        """
        xp = self._xp
        
        # Calculate padding amounts
        pad_total = self.n_pix_padded - self.n_pix
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        
        # Pad with zeros
        padded = xp.pad(
            field,
            ((pad_before, pad_after), (pad_before, pad_after)),
            mode='constant',
            constant_values=0
        )
        
        return padded
    
    def _zero_pad_batch(self, fields: ArrayLike) -> ArrayLike:
        """
        Zero-pad a batch of 2D fields to n_pix_padded.
        
        Args:
            fields: 3D complex fields (n_batch, n_pix, n_pix)
            
        Returns:
            Zero-padded fields (n_batch, n_pix_padded, n_pix_padded)
        """
        xp = self._xp
        
        # Calculate padding amounts
        pad_total = self.n_pix_padded - self.n_pix
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        
        # Pad with zeros (no padding on batch dimension)
        padded = xp.pad(
            fields,
            ((0, 0), (pad_before, pad_after), (pad_before, pad_after)),
            mode='constant',
            constant_values=0
        )
        
        return padded
    
    def _normalize(self, psf: ArrayLike) -> ArrayLike:
        """Normalize PSF according to mode."""
        xp = self._xp
        
        if self.normalize_to == "sum":
            total = xp.sum(psf)
            if total > 0:
                return psf / total
        elif self.normalize_to == "peak":
            peak = xp.max(psf)
            if peak > 0:
                return psf / peak
        elif self.normalize_to == "none":
            return psf
        
        return psf
    
    def _normalize_batch(self, psfs: ArrayLike) -> ArrayLike:
        """Normalize batch of PSFs according to mode."""
        xp = self._xp
        
        if self.normalize_to == "sum":
            # Sum over spatial dimensions (last 2 axes)
            totals = xp.sum(psfs, axis=(-2, -1), keepdims=True)
            # Avoid division by zero
            totals = xp.maximum(totals, xp.float64(1e-30))
            return (psfs / totals).astype(xp.float32)
        elif self.normalize_to == "peak":
            # Max over spatial dimensions
            peaks = xp.max(psfs, axis=(-2, -1), keepdims=True)
            # Avoid division by zero
            peaks = xp.maximum(peaks, xp.float64(1e-30))
            return (psfs / peaks).astype(xp.float32)
        elif self.normalize_to == "none":
            return psfs.astype(xp.float32)
        
        return psfs.astype(xp.float32)
    
    def get_sampling_info(self, telescope_diameter: float) -> Dict[str, float]:
        """
        Get PSF sampling information.
        
        Args:
            telescope_diameter: Primary mirror diameter (meters)
            
        Returns:
            Dictionary with sampling metrics
        """
        # Airy disk first null: θ = 1.22 × λ/D
        airy_null_rad = 1.22 * self.wavelength / telescope_diameter
        airy_null_pix = airy_null_rad / self.pixel_scale
        
        # FWHM: θ ≈ 1.028 × λ/D
        fwhm_rad = 1.028 * self.wavelength / telescope_diameter
        fwhm_pix = fwhm_rad / self.pixel_scale
        
        # Nyquist criterion: >= 2 pixels per resolution element
        nyquist_ok = airy_null_pix >= 2.0
        
        return {
            "airy_null_rad": airy_null_rad,
            "airy_null_pix": airy_null_pix,
            "fwhm_rad": fwhm_rad,
            "fwhm_pix": fwhm_pix,
            "pixel_scale_rad": self.pixel_scale,
            "pixel_scale_mas": self.pixel_scale * 180 / math.pi * 3600 * 1000,
            "nyquist_satisfied": nyquist_ok,
            "pupil_pixel_size_m": self.pupil_pixel_size,
            "pupil_extent_m": self.pupil_extent,
        }
    
    def info(self) -> Dict[str, Any]:
        """Return engine configuration info."""
        return {
            "n_pix": self.n_pix,
            "n_pix_padded": self.n_pix_padded,
            "zero_pad_factor": self.zero_pad_factor,
            "wavelength_m": self.wavelength,
            "pixel_scale_rad": self.pixel_scale,
            "pixel_scale_mas": self.pixel_scale * 180 / math.pi * 3600 * 1000,
            "pixel_scale_effective_rad": self.pixel_scale_effective,
            "pixel_scale_effective_mas": self.pixel_scale_effective * 180 / math.pi * 3600 * 1000,
            "pupil_pixel_size_m": self.pupil_pixel_size,
            "pupil_extent_m": self.pupil_extent,
            "normalize_to": self.normalize_to,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_psf(
    pupil: ArrayLike,
    phase: Optional[ArrayLike] = None,
    wavelength: float = 2.2e-6,
    pixel_scale: float = 1e-5,
) -> ArrayLike:
    """
    Convenience function to compute a single PSF.
    
    Args:
        pupil: 2D pupil amplitude mask
        phase: Optional phase screen in radians
        wavelength: Observation wavelength (meters)
        pixel_scale: PSF pixel scale (radians/pixel)
        
    Returns:
        PSF normalized to sum=1
    """
    n_pix = pupil.shape[0]
    engine = PSFEngine(n_pix, wavelength, pixel_scale)
    return engine.compute_psf(pupil, phase)


def compute_long_exposure_psf(
    pupil: ArrayLike,
    phases: ArrayLike,
    wavelength: float = 2.2e-6,
    pixel_scale: float = 1e-5,
) -> ArrayLike:
    """
    Convenience function to compute long-exposure PSF.
    
    Args:
        pupil: 2D pupil amplitude mask
        phases: Phase screens (n_screens, n_pix, n_pix) in radians
        wavelength: Observation wavelength (meters)
        pixel_scale: PSF pixel scale (radians/pixel)
        
    Returns:
        Long-exposure PSF normalized to sum=1
    """
    n_pix = pupil.shape[0]
    engine = PSFEngine(n_pix, wavelength, pixel_scale)
    return engine.compute_long_exposure_psf(pupil, phases)


def compute_strehl_ratio(
    pupil: ArrayLike,
    phase: ArrayLike,
    wavelength: float = 2.2e-6,
    pixel_scale: float = 1e-5,
) -> float:
    """
    Convenience function to compute Strehl ratio.
    
    Args:
        pupil: 2D pupil amplitude mask
        phase: Phase screen in radians
        wavelength: Observation wavelength (meters)
        pixel_scale: PSF pixel scale (radians/pixel)
        
    Returns:
        Strehl ratio (0 to 1)
    """
    n_pix = pupil.shape[0]
    engine = PSFEngine(n_pix, wavelength, pixel_scale)
    return engine.compute_strehl_ratio(pupil, phase=phase)

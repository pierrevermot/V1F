"""
Basic optical propagation and PSF computation.

**LEGACY MODULE**: These functions are maintained for backward compatibility
but are deprecated in favor of PSFEngine. New code should use PSFEngine directly.

Provides functions for computing point spread functions from pupil
fields and related optical quantities.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

from ..utils.compute import get_backend


# =============================================================================
# PSF Computation (Legacy - Use PSFEngine Instead)
# =============================================================================

def compute_psf(pupil, phase, normalize: bool = True, normalize_to: str = "peak"):
    """
    Compute PSF from pupil amplitude and phase.
    
    **DEPRECATED**: This function is a legacy wrapper around PSFEngine.
    New code should use PSFEngine directly for better control and consistency.
    
    Args:
        pupil: 2D pupil transmission array
        phase: 2D phase array in radians
        normalize: If True, normalize PSF (legacy parameter, use normalize_to instead)
        normalize_to: Normalization mode - 'sum' (energy conservation), 'peak' (peak=1), or 'none'
    
    Returns:
        2D PSF (intensity) array
    
    Note:
        For backward compatibility, normalize=False sets normalize_to='none'.
        When both are specified, normalize_to takes precedence.
        
    Example (migrating to PSFEngine):
        # Old code:
        psf = compute_psf(pupil, phase, normalize=True)
        
        # New code:
        from nebraa.physics import PSFEngine
        engine = PSFEngine(n_pix=pupil.shape[0], wavelength=2.2e-6, pixel_scale=1e-5)
        psf = engine.compute_psf(pupil, phase, normalize=True)
    """
    warnings.warn(
        "compute_psf() from optics module is deprecated. "
        "Use PSFEngine.compute_psf() instead for better control and consistency. "
        "This legacy function will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Use PSFEngine as backend
    from .psf_engine import PSFEngine
    
    n_pix = pupil.shape[0]
    # Use default wavelength and pixel_scale (arbitrary but consistent)
    engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
    
    # Handle legacy normalize parameter
    if not normalize:
        normalize_to = "none"
    
    # Map normalize_to to PSFEngine's normalize parameter
    if normalize_to == "none":
        do_normalize = False
    else:
        do_normalize = True
    
    # Compute PSF using PSFEngine
    psf = engine.compute_psf(pupil, phase, normalize=do_normalize)
    
    # PSFEngine always normalizes to sum=1, handle peak normalization
    if normalize_to == "peak" and do_normalize:
        backend = get_backend()
        xp = backend.xp
        peak = xp.max(psf)
        if peak > 0:
            psf = psf / peak
    
    return psf


def compute_psf_batch(pupil, phase_screens, normalize: bool = True, normalize_to: str = "peak", batch_size: int = None):
    """
    Compute PSFs for multiple phase screens with vectorized batch processing.
    
    **DEPRECATED**: This function is a legacy wrapper around PSFEngine.
    New code should use PSFEngine.compute_long_exposure_psf() instead.
    
    Args:
        pupil: 2D pupil transmission array
        phase_screens: 3D array (n, H, W) of phase screens in radians
        normalize: If True, normalize PSFs (legacy parameter, use normalize_to instead)
        normalize_to: Normalization mode - 'sum' (energy conservation), 'peak' (peak=1), or 'none'
        batch_size: Deprecated, kept for API compatibility. Processing is fully vectorized.
    
    Returns:
        3D array (n, H, W) of PSFs (NOT averaged - returns individual PSFs)
    
    Note:
        For backward compatibility, normalize=False sets normalize_to='none'.
        When both are specified, normalize_to takes precedence.
        
        This function returns INDIVIDUAL PSFs, not the averaged long-exposure PSF.
        For long-exposure PSF, use PSFEngine.compute_long_exposure_psf() directly.
        
    Example (migrating to PSFEngine):
        # Old code (individual PSFs):
        psfs = compute_psf_batch(pupil, phases, normalize=True)
        
        # New code (individual PSFs):
        from nebraa.physics import PSFEngine
        engine = PSFEngine(n_pix=pupil.shape[0], wavelength=2.2e-6, pixel_scale=1e-5)
        _, psfs = engine.compute_psf_batch(pupil, phases, normalize=True, return_individual=True)
        
        # Or for long-exposure PSF (averaged):
        psf_le = engine.compute_long_exposure_psf(pupil, phases, normalize=True)
    """
    warnings.warn(
        "compute_psf_batch() from optics module is deprecated. "
        "Use PSFEngine.compute_psf_batch() or PSFEngine.compute_long_exposure_psf() instead. "
        "This legacy function will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Use PSFEngine as backend
    from .psf_engine import PSFEngine
    
    n_pix = pupil.shape[0]
    engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
    
    # Handle legacy normalize parameter
    if not normalize:
        normalize_to = "none"
    
    # Map normalize_to to PSFEngine's normalize parameter
    if normalize_to == "none":
        do_normalize = False
    else:
        do_normalize = True
    
    # Compute batch PSFs using PSFEngine (get individual PSFs)
    _, psfs = engine.compute_psf_batch(
        pupil, phase_screens, 
        normalize=do_normalize, 
        return_individual=True
    )
    
    # PSFEngine normalizes to sum=1, handle peak normalization
    if normalize_to == "peak" and do_normalize:
        backend = get_backend()
        xp = backend.xp
        peaks = xp.max(psfs, axis=(-2, -1), keepdims=True)
        peaks = xp.maximum(peaks, xp.float32(1e-30))
        psfs = psfs / peaks
    
    return psfs



def compute_reference_psf(pupil):
    """
    Compute diffraction-limited (zero phase) reference PSF.
    
    **DEPRECATED**: This function is a legacy wrapper around PSFEngine.
    Use PSFEngine.compute_diffraction_limited_psf() instead.
    
    Args:
        pupil: 2D pupil transmission array
    
    Returns:
        2D reference PSF, normalized to unit sum (not peak!)
        
    Example (migrating to PSFEngine):
        # Old code:
        ref_psf = compute_reference_psf(pupil)
        
        # New code:
        from nebraa.physics import PSFEngine
        engine = PSFEngine(n_pix=pupil.shape[0], wavelength=2.2e-6, pixel_scale=1e-5)
        ref_psf = engine.compute_diffraction_limited_psf(pupil)
    """
    warnings.warn(
        "compute_reference_psf() from optics module is deprecated. "
        "Use PSFEngine.compute_diffraction_limited_psf() instead. "
        "This legacy function will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    
    from .psf_engine import PSFEngine
    
    n_pix = pupil.shape[0]
    engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
    
    # PSFEngine.compute_psf with phase=None returns DL PSF
    return engine.compute_psf(pupil, phase=None, normalize=True)


# =============================================================================
# Strehl and Phase Statistics
# =============================================================================

def compute_strehl(phase, pupil):
    """
    Compute Strehl ratio from phase screen.
    
    Strehl = |<exp(i*phi)>|^2 where <> denotes average over pupil.
    
    Args:
        phase: Phase array (n, H, W) or (H, W) in radians
        pupil: 2D pupil mask
    
    Returns:
        Strehl ratio(s)
    """
    backend = get_backend()
    xp = backend.xp
    
    pupil = backend.ensure_local(pupil).astype(xp.float32)
    pupil_sum = xp.sum(pupil)
    
    # Handle single/batch
    single = phase.ndim == 2
    if single:
        phase = phase[None, :, :]
    
    # Complex phasor average
    phasor = xp.exp(1j * phase.astype(xp.float32))
    avg_phasor = xp.sum(phasor * pupil[None, :, :], axis=(1, 2)) / pupil_sum
    
    strehl = xp.abs(avg_phasor) ** 2
    
    return float(strehl[0]) if single else strehl.astype(xp.float32)


def compute_rms_phase(phase, pupil):
    """
    Compute RMS phase over pupil.
    
    Args:
        phase: Phase array (n, H, W) or (H, W) in radians
        pupil: 2D pupil mask
    
    Returns:
        RMS phase value(s) in radians
    """
    backend = get_backend()
    xp = backend.xp
    
    pupil = backend.ensure_local(pupil).astype(xp.float32)
    pupil_sum = xp.sum(pupil)
    
    single = phase.ndim == 2
    if single:
        phase = phase[None, :, :]
    
    phase = phase * pupil[None, :, :]
    
    # Remove piston
    mean = xp.sum(phase, axis=(1, 2)) / pupil_sum
    phase_centered = phase - mean[:, None, None]
    
    # RMS
    var = xp.sum(phase_centered**2 * pupil[None, :, :], axis=(1, 2)) / pupil_sum
    rms = xp.sqrt(var)
    
    return float(rms[0]) if single else rms.astype(xp.float32)


def compute_rms_opd(phase, pupil, wavelength: float):
    """
    Compute RMS OPD (optical path difference) in meters.
    
    Args:
        phase: Phase in radians
        pupil: Pupil mask
        wavelength: Wavelength in meters
    
    Returns:
        RMS OPD in meters
    """
    import math
    rms_rad = compute_rms_phase(phase, pupil)
    return rms_rad * wavelength / (2 * math.pi)


# =============================================================================
# Utility Functions
# =============================================================================

def center_crop(images, target_size: int):
    """
    Center-crop images to target size.
    
    Args:
        images: Array (n, H, W) or (H, W)
        target_size: Output size
    
    Returns:
        Cropped array
    """
    backend = get_backend()
    
    if images.ndim == 2:
        h, w = images.shape
        y0 = (h - target_size) // 2
        x0 = (w - target_size) // 2
        return images[y0:y0+target_size, x0:x0+target_size]
    else:
        n, h, w = images.shape
        y0 = (h - target_size) // 2
        x0 = (w - target_size) // 2
        return images[:, y0:y0+target_size, x0:x0+target_size]


def pad_to_size(image, target_size: int, value: float = 0.0):
    """
    Pad image to target size (centered).
    
    Args:
        image: 2D array
        target_size: Output size
        value: Padding value
    
    Returns:
        Padded array
    """
    backend = get_backend()
    xp = backend.xp
    
    h, w = image.shape
    result = xp.full((target_size, target_size), value, dtype=image.dtype)
    
    y0 = (target_size - h) // 2
    x0 = (target_size - w) // 2
    
    result[y0:y0+h, x0:x0+w] = image
    return result

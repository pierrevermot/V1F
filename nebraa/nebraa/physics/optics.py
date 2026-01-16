"""
Basic optical propagation and PSF computation.

Provides functions for computing point spread functions from pupil
fields and related optical quantities.
"""

from __future__ import annotations

from typing import Optional, Tuple

from ..utils.compute import get_backend


# =============================================================================
# PSF Computation
# =============================================================================

def compute_psf(pupil, phase, normalize: bool = True):
    """
    Compute PSF from pupil amplitude and phase.
    
    Args:
        pupil: 2D pupil transmission array
        phase: 2D phase array in radians
        normalize: If True, normalize to unit peak
    
    Returns:
        2D PSF (intensity) array
    """
    backend = get_backend()
    xp = backend.xp
    
    pupil = backend.ensure_local(pupil).astype(xp.float32)
    phase = backend.ensure_local(phase).astype(xp.float32)
    
    # Build complex field
    E_pupil = pupil * xp.exp(1j * phase)
    
    # FFT to focal plane
    E_focal = xp.fft.fftshift(xp.fft.fft2(E_pupil))
    
    # Intensity
    psf = xp.abs(E_focal) ** 2
    
    if normalize:
        psf = psf / xp.max(psf)
    
    return psf.astype(xp.float32)


def compute_psf_batch(pupil, phase_screens, normalize: bool = True, batch_size: int = 64):
    """
    Compute PSFs for multiple phase screens with batching.
    
    Args:
        pupil: 2D pupil transmission array
        phase_screens: 3D array (n, H, W) of phase screens in radians
        normalize: If True, normalize each PSF to unit peak
        batch_size: Number of PSFs per batch
    
    Returns:
        3D array (n, H, W) of PSFs
    """
    backend = get_backend()
    xp = backend.xp
    
    pupil = backend.ensure_local(pupil).astype(xp.float32)
    n_psfs = phase_screens.shape[0]
    n_pix = phase_screens.shape[1]
    
    psfs = xp.zeros((n_psfs, n_pix, n_pix), dtype=xp.float32)
    
    for i0 in range(0, n_psfs, batch_size):
        i1 = min(i0 + batch_size, n_psfs)
        batch_phase = phase_screens[i0:i1]
        
        # Build complex field
        E_pupil = pupil[None, :, :] * xp.exp(1j * batch_phase)
        
        # FFT to focal plane
        E_focal = xp.fft.fftshift(
            xp.fft.fft2(E_pupil, axes=(1, 2)),
            axes=(1, 2)
        )
        
        # Intensity
        batch_psf = xp.abs(E_focal) ** 2
        
        # Normalize
        if normalize:
            peak = xp.max(batch_psf, axis=(1, 2), keepdims=True)
            peak = xp.maximum(peak, xp.float32(1e-30))
            batch_psf = batch_psf / peak
        
        psfs[i0:i1] = batch_psf.astype(xp.float32)
    
    return psfs


def compute_reference_psf(pupil):
    """
    Compute diffraction-limited (zero phase) reference PSF.
    
    Args:
        pupil: 2D pupil transmission array
    
    Returns:
        2D reference PSF, normalized to unit peak
    """
    backend = get_backend()
    xp = backend.xp
    
    pupil = backend.ensure_local(pupil).astype(xp.float32)
    n_pix = pupil.shape[0]
    
    zero_phase = xp.zeros((n_pix, n_pix), dtype=xp.float32)
    return compute_psf(pupil, zero_phase, normalize=True)


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

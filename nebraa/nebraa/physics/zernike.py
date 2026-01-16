"""
Zernike polynomial wavefront representation.

Provides functions for computing Zernike polynomials and generating
phase screens from Zernike expansions. Supports per-device caching
for efficient multi-GPU execution.
"""

from __future__ import annotations

import math
from typing import Tuple, Dict, Optional, List
from functools import lru_cache

from ..utils.compute import get_backend


# =============================================================================
# Mathematical Functions
# =============================================================================

@lru_cache(maxsize=256)
def _factorial(n: int) -> int:
    """Cached factorial computation."""
    return math.factorial(n)


def noll_to_nm(j: int) -> Tuple[int, int]:
    """
    Convert Noll index to (n, m) Zernike indices.
    
    Args:
        j: Noll index (1-based)
    
    Returns:
        (n, m) tuple: radial order n, azimuthal frequency m
    """
    n = int(math.ceil((-3 + math.sqrt(9 + 8*(j-1))) / 2))
    m = 2*j - n*(n+2) - 1
    
    # Adjust sign based on parity
    if j % 2 == 0:
        m = abs(m)
    else:
        m = -abs(m) if m != 0 else 0
    
    return n, m


def nm_to_noll(n: int, m: int) -> int:
    """
    Convert (n, m) Zernike indices to Noll index.
    
    Args:
        n: Radial order
        m: Azimuthal frequency
    
    Returns:
        Noll index (1-based)
    """
    j = n * (n + 1) // 2 + 1
    
    if m > 0:
        j += 2 * ((n % 2 == 0) ^ (m % 2 == 0))
    elif m < 0:
        j += 2 * ((n % 2 == 0) ^ (abs(m) % 2 == 1)) + 1
    
    return j


def count_modes(n_min: int, n_max: int) -> int:
    """Count number of Zernike modes in radial order range [n_min, n_max)."""
    total = 0
    for n in range(n_min, n_max):
        total += n + 1  # Each order n has (n+1) modes
    return total


def get_nm_list(n_min: int, n_max: int) -> List[Tuple[int, int]]:
    """Get list of (n, m) pairs for radial orders [n_min, n_max)."""
    modes = []
    for n in range(n_min, n_max):
        for m in range(-n, n + 1, 2):
            modes.append((n, m))
    return modes


# =============================================================================
# Zernike Polynomial Computation
# =============================================================================

def zernike_radial(n: int, m: int, rho):
    """
    Compute radial Zernike polynomial R_n^m(rho).
    
    Args:
        n: Radial order
        m: Azimuthal frequency (absolute value used)
        rho: Radial coordinate array (normalized to [0, 1])
    
    Returns:
        Radial polynomial values
    """
    backend = get_backend()
    xp = backend.xp
    
    m = abs(m)
    R = xp.zeros_like(rho, dtype=xp.float32)
    
    for s in range((n - m) // 2 + 1):
        num = ((-1) ** s) * _factorial(n - s)
        den = _factorial(s) * _factorial((n + m) // 2 - s) * _factorial((n - m) // 2 - s)
        R = R + (num / den) * (rho ** (n - 2*s))
    
    return R


def zernike_nm(n: int, m: int, rho, theta, normalize: bool = True):
    """
    Compute Zernike polynomial Z_n^m(rho, theta).
    
    Uses the standard normalization where:
    - Z_n^m = R_n^m(rho) * cos(m*theta)  for m >= 0
    - Z_n^m = R_n^|m|(rho) * sin(|m|*theta)  for m < 0
    
    Args:
        n: Radial order
        m: Azimuthal frequency
        rho: Radial coordinate (normalized to [0, 1])
        theta: Angular coordinate (radians)
        normalize: If True, apply normalization factor
    
    Returns:
        Zernike polynomial values
    """
    backend = get_backend()
    xp = backend.xp
    
    R = zernike_radial(n, m, rho)
    
    if m >= 0:
        Z = R * xp.cos(m * theta)
    else:
        Z = R * xp.sin(abs(m) * theta)
    
    if normalize:
        # Normalization factor for unit RMS over unit circle
        if m == 0:
            norm = math.sqrt(n + 1)
        else:
            norm = math.sqrt(2 * (n + 1))
        Z = Z * norm
    
    return Z.astype(xp.float32)


# =============================================================================
# Caching for Multi-GPU Efficiency
# =============================================================================

class ZernikeModeCache:
    """
    Per-device cache for Zernike modes and coordinate grids.
    
    This prevents redundant computation and avoids cross-GPU memory
    access errors by maintaining separate caches per device.
    """
    
    def __init__(self):
        self._coord_cache: Dict[Tuple, Tuple] = {}
        self._mode_cache: Dict[Tuple, object] = {}
    
    def get_coordinates(self, n_pix: int, radius: float) -> Tuple:
        """
        Get or compute normalized polar coordinates.
        
        Args:
            n_pix: Grid size
            radius: Pupil radius for normalization
        
        Returns:
            (rho, theta) coordinate arrays
        """
        backend = get_backend()
        device_id = backend.device_id or 0
        key = (device_id, n_pix, radius)
        
        if key not in self._coord_cache:
            xp = backend.xp
            
            c = (n_pix - 1) / 2.0
            idx = xp.arange(n_pix, dtype=xp.float32)
            X, Y = xp.meshgrid(idx - c, idx - c)
            
            rho = xp.sqrt(X**2 + Y**2) / (n_pix / 2.0)  # Normalized to [0, 1] at edge
            theta = xp.arctan2(Y, X)
            
            self._coord_cache[key] = (rho, theta)
        
        return self._coord_cache[key]
    
    def get_modes(self, n_pix: int, radius: float, n_range: Tuple[int, int]) -> object:
        """
        Get or compute Zernike mode cube.
        
        Args:
            n_pix: Grid size
            radius: Pupil radius
            n_range: (n_min, n_max) radial order range
        
        Returns:
            Array of shape (n_modes, n_pix, n_pix)
        """
        backend = get_backend()
        device_id = backend.device_id or 0
        key = (device_id, n_pix, radius, n_range)
        
        if key not in self._mode_cache:
            xp = backend.xp
            
            rho, theta = self.get_coordinates(n_pix, radius)
            nm_list = get_nm_list(n_range[0], n_range[1])
            n_modes = len(nm_list)
            
            modes = xp.zeros((n_modes, n_pix, n_pix), dtype=xp.float32)
            
            for i, (n, m) in enumerate(nm_list):
                modes[i] = zernike_nm(n, m, rho, theta)
            
            self._mode_cache[key] = modes
        
        return self._mode_cache[key]
    
    def clear(self):
        """Clear all caches."""
        self._coord_cache.clear()
        self._mode_cache.clear()


# Global cache instance
_cache = ZernikeModeCache()


def build_zernike_modes(n_pix: int, radius: float, n_range: Tuple[int, int]):
    """
    Build Zernike mode cube (using global cache).
    
    Args:
        n_pix: Grid size
        radius: Pupil radius for normalization
        n_range: (n_min, n_max) radial order range
    
    Returns:
        Array of shape (n_modes, n_pix, n_pix)
    """
    return _cache.get_modes(n_pix, radius, n_range)


# =============================================================================
# Phase Screen Generation
# =============================================================================

def generate_zernike_phase(
    n_screens: int,
    n_pix: int,
    radius: float,
    n_range: Tuple[int, int] = (2, 5),
    power_law: float = 2.0,
    seed: Optional[int] = None,
):
    """
    Generate random phase screens from Zernike polynomials.
    
    Coefficients are drawn randomly with power-law scaling by radial order.
    
    Args:
        n_screens: Number of phase screens to generate
        n_pix: Grid size
        radius: Pupil radius
        n_range: (n_min, n_max) radial order range
        power_law: Exponent for amplitude decay with radial order
        seed: Random seed for reproducibility
    
    Returns:
        Phase screens array of shape (n_screens, n_pix, n_pix)
    """
    backend = get_backend()
    xp = backend.xp
    
    if seed is not None:
        xp.random.seed(seed)
    
    # Get or compute modes
    modes = build_zernike_modes(n_pix, radius, n_range)
    n_modes = modes.shape[0]
    nm_list = get_nm_list(n_range[0], n_range[1])
    
    # Generate random coefficients with power-law scaling
    coeffs = xp.random.randn(n_screens, n_modes).astype(xp.float32)
    
    # Apply power-law amplitude scaling
    for i, (n, m) in enumerate(nm_list):
        scale = (n + 1) ** (-power_law / 2)  # Variance scales as n^(-power_law)
        coeffs[:, i] *= scale
    
    # Compute phase screens: sum of coefficient * mode
    # Shape: (n_screens, n_modes) @ (n_modes, n_pix*n_pix)
    modes_flat = modes.reshape(n_modes, -1)
    phase_flat = xp.dot(coeffs, modes_flat)
    phase = phase_flat.reshape(n_screens, n_pix, n_pix)
    
    return phase


def normalize_phase_rms(phase, pupil, target_rms):
    """
    Normalize phase to target RMS over pupil.
    
    Args:
        phase: Phase array (n, H, W) or (H, W)
        pupil: Pupil mask
        target_rms: Target RMS (scalar or per-screen array)
    
    Returns:
        Normalized phase
    """
    backend = get_backend()
    xp = backend.xp
    
    pupil = backend.ensure_local(pupil).astype(xp.float32)
    pupil_sum = xp.sum(pupil)
    
    # Handle single screen
    single = phase.ndim == 2
    if single:
        phase = phase[None, :, :]
        target_rms = xp.array([target_rms])
    else:
        target_rms = xp.asarray(target_rms, dtype=xp.float32).reshape(-1)
    
    # Apply pupil
    phase = phase * pupil[None, :, :]
    
    # Remove mean (piston)
    mean = xp.sum(phase, axis=(1, 2)) / pupil_sum
    phase = phase - mean[:, None, None]
    
    # Compute RMS
    var = xp.sum(phase**2 * pupil[None, :, :], axis=(1, 2)) / pupil_sum
    rms = xp.sqrt(xp.maximum(var, xp.float32(1e-30)))
    
    # Normalize
    phase = phase / rms[:, None, None] * target_rms[:, None, None]
    
    return phase[0] if single else phase

"""
Zernike polynomial wavefront representation.

Provides functions for computing Zernike polynomials and generating
phase screens from Zernike expansions. Supports per-device caching
for efficient multi-GPU execution.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Tuple, Dict, Optional, List
from functools import lru_cache

from ..utils.compute import get_backend
from ..utils.rng import create_rng, get_rng_or_create, RNGType


# =============================================================================
# Mathematical Functions
# =============================================================================

@lru_cache(maxsize=256)
def _factorial(n: int) -> int:
    """Cached factorial computation."""
    return math.factorial(n)


@lru_cache(maxsize=1024)
def _log_factorial(n: int) -> float:
    """
    Cached log-factorial computation for numerical stability.
    
    Uses math.lgamma(n+1) = log(n!) which is numerically stable
    even for large n.
    """
    return math.lgamma(n + 1)


def noll_to_nm(j: int) -> Tuple[int, int]:
    """
    Convert Noll index to (n, m) Zernike indices.
    
    Uses the standard Noll (1976) indexing convention where within each
    radial order n, modes are ordered by increasing |m|, with the sine 
    term (m<0) before the cosine term (m>0):
    
    j=1:  (0,0)  piston
    j=2:  (1,1)  x-tilt
    j=3:  (1,-1) y-tilt
    j=4:  (2,0)  defocus
    j=5:  (2,-2) oblique astigmatism
    j=6:  (2,2)  vertical astigmatism
    j=7:  (3,-1) vertical coma
    j=8:  (3,1)  horizontal coma
    j=9:  (3,-3) oblique trefoil
    j=10: (3,3)  vertical trefoil
    ...
    
    Args:
        j: Noll index (1-based, j >= 1)
    
    Returns:
        (n, m) tuple: radial order n, azimuthal frequency m
    """
    if j < 1:
        raise ValueError(f"Noll index must be >= 1, got {j}")
    
    # Find radial order n such that n*(n+1)/2 < j <= (n+1)*(n+2)/2
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    
    # Position within this radial order (0-based)
    k = j - n * (n + 1) // 2 - 1
    
    # Noll ordering within radial order n:
    # - First: m=0 (if n is even) or lowest |m| pair
    # - Then pairs of increasing |m|: (m<0, m>0)
    # 
    # For even n: modes are 0, -2, 2, -4, 4, ... or similar
    # For odd n: modes are -1, 1, -3, 3, ...
    #
    # The pattern is: within order n, for position k (0-based):
    # |m| = 2*((k+1)//2) if n is even, else |m| = 2*((k)//2) + 1
    
    if n % 2 == 0:  # Even n: has m=0 term
        if k == 0:
            m = 0
        else:
            m_abs = 2 * ((k + 1) // 2)
            # Odd k -> negative m, even k -> positive m (for k > 0)
            m = -m_abs if k % 2 == 1 else m_abs
    else:  # Odd n: no m=0 term
        m_abs = 2 * (k // 2) + 1
        # Even k -> negative m, odd k -> positive m
        m = -m_abs if k % 2 == 0 else m_abs
    
    return n, m


def nm_to_noll(n: int, m: int) -> int:
    """
    Convert (n, m) Zernike indices to Noll index.
    
    Args:
        n: Radial order (n >= 0)
        m: Azimuthal frequency (|m| <= n, n-|m| must be even)
    
    Returns:
        Noll index (1-based)
    """
    if n < 0:
        raise ValueError(f"Radial order n must be >= 0, got {n}")
    if abs(m) > n:
        raise ValueError(f"|m|={abs(m)} must be <= n={n}")
    if (n - abs(m)) % 2 != 0:
        raise ValueError(f"n-|m| must be even: n={n}, m={m}")
    
    # Base index: first mode of radial order n
    j_base = n * (n + 1) // 2 + 1
    
    # Find position k within radial order
    if n % 2 == 0:  # Even n
        if m == 0:
            k = 0
        elif m < 0:
            k = abs(m) - 1  # -2 -> k=1, -4 -> k=3
        else:
            k = m  # 2 -> k=2, 4 -> k=4
    else:  # Odd n
        if m < 0:
            k = abs(m) - 1  # -1 -> k=0, -3 -> k=2
        else:
            k = m  # 1 -> k=1, 3 -> k=3
    
    return j_base + k


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
    Compute radial Zernike polynomial R_n^m(rho) using Jacobi polynomials.
    
    GPU-accelerated version that uses cupyx.scipy when available, otherwise
    falls back to scipy (with CPU transfer) or direct recurrence relation.
    
    Uses the relationship between Zernike and Jacobi polynomials for
    numerical stability with high radial orders:
    
        R_n^m(rho) = (-1)^k * rho^m * P_k^{(0,m)}(1 - 2*rho^2)
    
    where k = (n-m)/2 and P_k^{(α,β)} is the Jacobi polynomial.
    
    Args:
        n: Radial order (n >= 0)
        m: Azimuthal frequency (|m| <= n, n-|m| even)
        rho: Radial coordinate array (normalized to [0, 1])
    
    Returns:
        Radial polynomial values (zero outside unit circle)
    """
    backend = get_backend()
    xp = backend.xp
    
    m = abs(m)
    k = (n - m) // 2
    
    # Clip rho to [0, 1] - Zernike polynomials are only defined on unit disk
    rho_clipped = xp.clip(rho, 0.0, 1.0)
    
    # Determine if we're on GPU
    is_gpu = hasattr(xp, 'cuda')
    
    # Try GPU-accelerated path first (CuPy with cupyx.scipy)
    if is_gpu:
        try:
            from cupyx.scipy.special import eval_jacobi as eval_jacobi_gpu
            
            # Compute on GPU directly
            x = 1 - 2 * rho_clipped**2
            jacobi_vals = eval_jacobi_gpu(k, 0, m, x)
            
            if m > 0:
                R = ((-1)**k) * (rho_clipped**m) * jacobi_vals
            else:
                R = ((-1)**k) * jacobi_vals
            
            # Zero out values outside unit circle
            R = xp.where(rho <= 1.0, R, 0.0)
            
            return R.astype(xp.float64)
            
        except (ImportError, AttributeError):
            # cupyx.scipy not available, will try CPU fallback below
            pass
    
    # CPU path: use scipy (requires conversion)
    try:
        from scipy.special import eval_jacobi
        
        # Convert to numpy for scipy
        if hasattr(rho_clipped, 'get'):
            rho_np = rho_clipped.get().astype(np.float64)
        else:
            rho_np = np.asarray(rho_clipped, dtype=np.float64)
        
        # Compute using Jacobi polynomial relation
        x = 1 - 2 * rho_np**2
        jacobi_vals = eval_jacobi(k, 0, m, x)
        
        if m > 0:
            R_np = ((-1)**k) * (rho_np**m) * jacobi_vals
        else:
            R_np = ((-1)**k) * jacobi_vals
        
        # Zero out values outside unit circle
        if hasattr(rho, 'get'):
            rho_original_np = rho.get()
        else:
            rho_original_np = np.asarray(rho)
        R_np = np.where(rho_original_np <= 1.0, R_np, 0.0)
        
        # Convert back to backend array
        R = xp.asarray(R_np, dtype=xp.float64)
        
        return R
        
    except ImportError:
        # SciPy not available - use direct recurrence relation
        return _zernike_radial_recurrence(n, m, rho_clipped, rho, xp)


def _zernike_radial_recurrence(n: int, m: int, rho_clipped, rho_original, xp):
    """
    Compute radial Zernike polynomial using direct recurrence relation.
    
    This is a fallback when neither scipy nor cupyx.scipy are available.
    Uses the three-term recurrence relation for Zernike radials.
    
    Args:
        n: Radial order
        m: Azimuthal frequency (absolute value)
        rho_clipped: Clipped radial coordinates [0, 1]
        rho_original: Original radial coordinates (for masking)
        xp: Backend array module
        
    Returns:
        Radial polynomial values
    """
    m = abs(m)
    
    if n == m:
        # Base case: R_m^m(rho) = rho^m
        R = rho_clipped ** m
    elif n == m + 2:
        # R_{m+2}^m(rho) = (m+2)*rho^{m+2} - (m+1)*rho^m
        R = (m + 2) * rho_clipped**(m + 2) - (m + 1) * rho_clipped**m
    else:
        # Use three-term recurrence relation
        # Start with base cases
        R_prev2 = rho_clipped ** m  # R_m^m
        R_prev1 = (m + 2) * rho_clipped**(m + 2) - (m + 1) * rho_clipped**m  # R_{m+2}^m
        
        # Build up to desired n
        rho2 = rho_clipped ** 2
        for n_curr in range(m + 4, n + 1, 2):
            # Coefficients for recurrence
            c1 = (2 * n_curr * (n_curr + m - 1) * (n_curr - m - 1)) / ((n_curr + m) * (n_curr - m))
            c2 = ((n_curr - 1) * (n_curr + m - 2) * (n_curr - m)) / ((n_curr + m) * (n_curr - m))
            
            R_curr = c1 * rho2 * R_prev1 - c2 * R_prev2
            
            R_prev2 = R_prev1
            R_prev1 = R_curr
        
        R = R_prev1 if n > m + 2 else R_prev2
    
    # Zero out values outside unit circle
    R = xp.where(rho_original <= 1.0, R, 0.0)
    
    return R.astype(xp.float64)


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
        Zernike polynomial values (float64 for numerical stability)
    """
    backend = get_backend()
    xp = backend.xp
    
    # Get radial polynomial (already float64)
    R = zernike_radial(n, m, rho)
    
    # Ensure theta is float64
    theta = xp.asarray(theta, dtype=xp.float64)
    
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
    
    return Z  # Keep as float64


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
            radius: Pupil radius in pixels for normalization (rho=1 at this radius)
        
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
            
            # Normalize rho so that rho=1 at the specified radius (in pixels)
            rho = xp.sqrt(X**2 + Y**2) / radius
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
            
            # Use float64 for modes to avoid overflow with high radial orders
            modes = xp.zeros((n_modes, n_pix, n_pix), dtype=xp.float64)
            
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
    rng: Optional[RNGType] = None,
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
        seed: Random seed for reproducibility (ignored if rng provided)
        rng: Optional RNG object for reproducibility (preferred over seed)
    
    Returns:
        Phase screens array of shape (n_screens, n_pix, n_pix)
    """
    backend = get_backend()
    xp = backend.xp
    
    # Get RNG object (never mutates global state)
    rng = get_rng_or_create(rng=rng, seed=seed, backend=backend)
    
    # Get or compute modes
    modes = build_zernike_modes(n_pix, radius, n_range)
    n_modes = modes.shape[0]
    nm_list = get_nm_list(n_range[0], n_range[1])
    
    # Generate random coefficients with power-law scaling
    # Use float64 to avoid overflow when summing many modes
    coeffs = rng.standard_normal((n_screens, n_modes)).astype(xp.float64)
    
    # Apply power-law amplitude scaling
    for i, (n, m) in enumerate(nm_list):
        scale = (n + 1) ** (-power_law / 2)  # Variance scales as n^(-power_law)
        coeffs[:, i] *= scale
    
    # Compute phase screens: sum of coefficient * mode
    # Shape: (n_screens, n_modes) @ (n_modes, n_pix*n_pix)
    # Use float64 for the accumulation to avoid overflow with many modes
    modes_flat = modes.reshape(n_modes, -1).astype(xp.float64)
    phase_flat = xp.dot(coeffs, modes_flat)
    phase = phase_flat.reshape(n_screens, n_pix, n_pix).astype(xp.float32)
    
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
    
    # Use float64 for numerical stability with large phase values
    pupil = backend.ensure_local(pupil).astype(xp.float64)
    pupil_sum = xp.sum(pupil)
    
    # Handle single screen
    single = phase.ndim == 2
    if single:
        phase = phase[None, :, :].astype(xp.float64)
        target_rms = xp.array([target_rms], dtype=xp.float64)
    else:
        phase = phase.astype(xp.float64)
        target_rms = xp.asarray(target_rms, dtype=xp.float64).reshape(-1)
    
    # Apply pupil
    phase = phase * pupil[None, :, :]
    
    # Remove mean (piston)
    mean = xp.sum(phase, axis=(1, 2)) / pupil_sum
    phase = phase - mean[:, None, None]
    
    # Compute RMS
    var = xp.sum(phase**2 * pupil[None, :, :], axis=(1, 2)) / pupil_sum
    rms = xp.sqrt(xp.maximum(var, 1e-30))
    
    # Normalize
    phase = phase / rms[:, None, None] * target_rms[:, None, None]
    
    # Convert back to float32
    phase = phase.astype(xp.float32)
    
    return phase[0] if single else phase


# =============================================================================
# Zernike Phase Generator (Unified Interface)
# =============================================================================

from dataclasses import dataclass
from .phase_generator import PhaseGeneratorBase
from .low_wind_effect import LowWindEffectConfig


@dataclass
class ZernikeConfig:
    """
    Configuration for Zernike phase generator.
    
    Attributes:
        n_range: (n_min, n_max) radial order range. Reasonable values are
            up to ~50 for most applications.
        power_law: Exponent for amplitude decay with radial order.
            Must be non-negative (variance ~ n^(-power_law)). Higher values
            suppress high-order modes more strongly. Typical: 2-4.
            
            Note: With power_law near 0 and high n_range, the resulting
            phase will have most of its variance concentrated near the pupil
            edge (rho ~ 1) because high-order Zernike modes scale as rho^m.
            To get more uniform variance across the pupil, use higher power_law
            or smaller n_range.
        target_rms: DEPRECATED - use lf_rms instead for clearer semantics.
            Target RMS for generated phases (radians). When hf_rms is also set,
            this normalizes the TOTAL phase (LF+HF), which can cause unexpected
            HF power reduction. For new code, use lf_rms + hf_rms to control
            each component independently.
        lf_rms: Target RMS for low-frequency (Zernike) component only (radians).
            If None, uses natural scaling from power_law. This parameter allows
            independent control of LF and HF RMS, similar to DualPowerLawConfig.
            Note: If both target_rms and lf_rms are set, lf_rms takes precedence.
        seed: Random seed for reproducibility
        
        # High-frequency turbulence parameters (above cutoff)
        f_cutoff: Cutoff frequency separating Zernike (LF) and PSD (HF) regimes 
            (cycles/meter). If None, only Zernike modes are used.
        hf_alpha: Power-law exponent for high-frequency PSD component.
            Default is 11/3 for Kolmogorov turbulence.
        hf_rms: Target RMS of high-frequency phase component (radians).
            If None, no HF component is added.
        transition_width: Width of smooth transition as fraction of f_cutoff (0-1)
    """
    n_range: Tuple[int, int] = (2, 5)
    power_law: float = 2.0
    target_rms: Optional[float] = None
    lf_rms: Optional[float] = None  # New: independent LF RMS control
    seed: Optional[int] = None
    
    # High-frequency parameters
    f_cutoff: Optional[float] = None
    hf_alpha: float = 11.0 / 3.0  # Kolmogorov default
    hf_rms: Optional[float] = None
    transition_width: float = 0.2
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.power_law < 0:
            raise ValueError(
                f"power_law must be non-negative (got {self.power_law}). "
                f"Negative values amplify high-order modes exponentially."
            )
        
        # Validate HF parameters
        if self.hf_rms is not None and self.f_cutoff is None:
            raise ValueError(
                "f_cutoff must be specified when hf_rms is provided"
            )
        
        if self.transition_width < 0 or self.transition_width > 1:
            raise ValueError(
                f"transition_width must be in [0, 1], got {self.transition_width}"
            )


class ZernikePhaseGenerator(PhaseGeneratorBase):
    """
    Phase generator using Zernike polynomial expansion.
    
    Generates random phase screens as weighted sums of Zernike modes,
    with coefficients scaled by a power-law in radial order.
    
    This implements the unified PhaseGeneratorBase interface.
    
    Example:
        ```python
        # Create generator
        gen = ZernikePhaseGenerator(
            n_pix=256,
            pixel_size=0.032,  # 32mm per pixel → 8.2m extent
            zernike_config=ZernikeConfig(
                n_range=(2, 10),
                power_law=2.5,
                target_rms=0.5,  # radians
            ),
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
        zernike_config: Optional[ZernikeConfig] = None,
        seed: Optional[int] = None,
        lwe_config: Optional[LowWindEffectConfig] = None,
    ):
        """
        Initialize Zernike phase generator.
        
        Args:
            n_pix: Grid size in pixels
            pixel_size: Physical pixel size (meters)
            zernike_config: Zernike configuration (or uses defaults)
            seed: Random seed (overrides zernike_config.seed)
            lwe_config: Optional LWE configuration
        """
        # Use config seed unless explicitly overridden
        if zernike_config is not None and seed is None:
            seed = zernike_config.seed
        
        super().__init__(
            n_pix=n_pix,
            pixel_size=pixel_size,
            seed=seed,
            lwe_config=lwe_config,
        )
        
        self.zernike_config = zernike_config or ZernikeConfig()
        
        # Normalized pupil radius for Zernike computation
        # Zernike polynomials are defined on unit circle, so rho is normalized
        # Use pixel-centered convention: radius = (n_pix - 1) / 2
        self._radius = (n_pix - 1) / 2.0
        
        # Setup frequency grid for HF component if needed
        self._freq_grid = None
        self._hf_amplitude = None
        if self.zernike_config.hf_rms is not None:
            self._setup_hf_turbulence()
    
    @classmethod
    def from_telescope(
        cls,
        n_pix: int,
        telescope_diameter: float,
        zernike_config: Optional[ZernikeConfig] = None,
        seed: Optional[int] = None,
        lwe_config: Optional[LowWindEffectConfig] = None,
    ) -> "ZernikePhaseGenerator":
        """
        Create generator from telescope parameters.
        
        Args:
            n_pix: Grid size in pixels
            telescope_diameter: Primary mirror diameter (meters)
            zernike_config: Zernike configuration
            seed: Random seed
            lwe_config: Optional LWE configuration
            
        Returns:
            ZernikePhaseGenerator
        """
        pixel_size = telescope_diameter / n_pix
        return cls(
            n_pix=n_pix,
            pixel_size=pixel_size,
            zernike_config=zernike_config,
            seed=seed,
            lwe_config=lwe_config,
        )
    
    @classmethod
    def from_fourier_sampling(
        cls,
        n_pix: int,
        wavelength: float,
        pixel_scale_rad: float,
        zernike_config: Optional[ZernikeConfig] = None,
        seed: Optional[int] = None,
        lwe_config: Optional[LowWindEffectConfig] = None,
    ) -> "ZernikePhaseGenerator":
        """
        Create generator from Fourier sampling requirements.
        
        Args:
            n_pix: Grid size in pixels
            wavelength: Observation wavelength (meters)
            pixel_scale_rad: PSF pixel scale (radians/pixel)
            zernike_config: Zernike configuration
            seed: Random seed
            lwe_config: Optional LWE configuration
            
        Returns:
            ZernikePhaseGenerator
        """
        pixel_size = wavelength / (n_pix * pixel_scale_rad)
        return cls(
            n_pix=n_pix,
            pixel_size=pixel_size,
            zernike_config=zernike_config,
            seed=seed,
            lwe_config=lwe_config,
        )
    
    def generate(
        self,
        n_screens: int,
        pupil: Optional = None,
        seed: Optional[int] = None,
    ):
        """
        Generate Zernike phase screens with optional HF turbulence.
        
        If HF parameters are configured, generates combined phase screens:
        - Low frequencies: Zernike polynomial expansion
        - High frequencies: Power-law PSD above cutoff
        
        RMS Control Modes:
        - lf_rms + hf_rms: Independent control of each component (recommended)
        - target_rms alone: Normalizes total phase (LF or LF+HF)
        - target_rms + hf_rms (deprecated): Normalizes LF+HF together
        
        Args:
            n_screens: Number of phase screens
            pupil: Optional pupil mask (used for RMS normalization and to 
                   determine the Zernike normalization radius)
            seed: Random seed (overrides instance seed)
            
        Returns:
            Phase screens (n_screens, n_pix, n_pix) in radians
        """
        actual_seed = seed if seed is not None else self.seed
        
        # Determine the radius for Zernike normalization
        # If a pupil is provided, compute the radius from the pupil extent
        if pupil is not None:
            radius = self._compute_pupil_radius(pupil)
        else:
            radius = self._radius
        
        # Generate Zernike (low-frequency) component
        phases_lf = generate_zernike_phase(
            n_screens=n_screens,
            n_pix=self.n_pix,
            radius=radius,
            n_range=self.zernike_config.n_range,
            power_law=self.zernike_config.power_law,
            seed=actual_seed,
        )
        
        # Normalize LF component if lf_rms is specified (new behavior)
        # This allows independent control of LF and HF RMS
        if self.zernike_config.lf_rms is not None and pupil is not None:
            phases_lf = normalize_phase_rms(
                phases_lf, pupil, self.zernike_config.lf_rms
            )
        
        # Add high-frequency component if configured
        if self.zernike_config.hf_rms is not None:
            # Use different seed for HF to avoid correlation
            hf_seed = actual_seed + 10000 if actual_seed is not None else None
            phases_hf = self._generate_hf_component(n_screens, seed=hf_seed)
            
            # Remove piston from HF component over pupil if provided
            if pupil is not None and phases_hf is not None:
                backend = get_backend()
                xp = backend.xp
                pupil_local = backend.ensure_local(pupil).astype(xp.float64)
                pupil_sum = xp.sum(pupil_local)
                mean_hf = xp.sum(
                    phases_hf * pupil_local[None, :, :], axis=(1, 2)
                ) / pupil_sum
                phases_hf = phases_hf - mean_hf[:, None, None]
            
            # Combine LF and HF components
            phases = phases_lf + phases_hf
        else:
            phases = phases_lf
        
        # Normalize to target RMS if specified (legacy behavior)
        # Note: target_rms applies to total phase (LF + HF)
        # If lf_rms is also specified, skip this to preserve independent control
        if (self.zernike_config.target_rms is not None 
            and self.zernike_config.lf_rms is None 
            and pupil is not None):
            phases = normalize_phase_rms(
                phases, pupil, self.zernike_config.target_rms
            )
        
        return phases
    
    def _setup_hf_turbulence(self):
        """
        Setup frequency grid and amplitude array for HF turbulence generation.
        
        This precomputes the HF PSD component following the approach in
        powerlaw_psd.py, with a smooth transition from the Zernike domain.
        """
        xp = self._xp
        n_pix = self.n_pix
        pixel_size = self.pixel_size
        
        # Build frequency grid
        df = 1.0 / (n_pix * pixel_size)
        fx = xp.fft.fftfreq(n_pix, d=pixel_size).astype(xp.float64)
        FX, FY = xp.meshgrid(fx, fx, indexing='ij')
        F = xp.sqrt(FX**2 + FY**2)
        F_safe = xp.maximum(F, 1e-12)
        
        # Store frequency grid info
        self._freq_grid = {
            'FX': FX, 'FY': FY, 'F': F, 'F_safe': F_safe,
            'df': df, 'dA': df**2
        }
        
        # Build transition window (high-pass filter)
        fc = self.zernike_config.f_cutoff
        w = fc * self.zernike_config.transition_width
        f1 = max(fc - w, 1e-12)
        f2 = fc + w
        
        # High-pass window (0 below f1, 1 above f2, smooth transition)
        W_HF = xp.zeros_like(F, dtype=xp.float64)
        
        # Transition region (smooth cosine)
        mid_mask = (F >= f1) & (F <= f2)
        if xp.any(mid_mask):
            x = (F[mid_mask] - f1) / (f2 - f1)
            W_HF[mid_mask] = 0.5 * (1.0 - xp.cos(xp.pi * x))
        
        # Above transition
        W_HF[F > f2] = 1.0
        
        # Build HF PSD: PSD(f) = f^(-alpha) * W_HF(f)
        psd_hf_raw = (F_safe ** (-self.zernike_config.hf_alpha)) * W_HF
        
        # Zero DC component
        psd_hf_raw.flat[0] = 0.0
        
        # Scale to achieve target RMS
        var_hf_raw = float(xp.sum(psd_hf_raw) * self._freq_grid['dA'])
        var_hf_target = self.zernike_config.hf_rms ** 2
        
        if var_hf_raw > 0:
            scale_hf = math.sqrt(var_hf_target / var_hf_raw)
            psd_hf = psd_hf_raw * (scale_hf ** 2)
        else:
            psd_hf = psd_hf_raw
        
        # Precompute amplitude array for phase generation
        # Following kolmogorov.py convention: A = sqrt(N^4 × PSD × df²)
        self._hf_amplitude = xp.sqrt(
            (n_pix ** 4) * psd_hf * self._freq_grid['dA']
        ).astype(xp.float64)
        self._hf_amplitude.flat[0] = 0.0
    
    def _generate_hf_component(
        self, n_screens: int, seed: Optional[int] = None
    ):
        """
        Generate high-frequency turbulence component using PSD.
        
        Args:
            n_screens: Number of phase screens
            seed: Random seed (deprecated, use rng parameter)
            
        Returns:
            HF phase screens (n_screens, n_pix, n_pix)
        """
        if self._hf_amplitude is None:
            return None
        
        xp = self._xp
        n_pix = self.n_pix
        
        # Create RNG object (never mutates global state)
        rng = create_rng(seed=seed, backend=self._backend)
        
        # Generate complex white noise
        noise_real = rng.standard_normal((n_screens, n_pix, n_pix)).astype(xp.float64)
        noise_imag = rng.standard_normal((n_screens, n_pix, n_pix)).astype(xp.float64)
        W = noise_real + 1j * noise_imag
        
        # Apply PSD coloring
        Phi_f = W * self._hf_amplitude[None, :, :]
        
        # Transform to spatial domain
        phi_hf = xp.real(xp.fft.ifft2(Phi_f)).astype(xp.float64)
        
        return phi_hf
    
    def _compute_pupil_radius(self, pupil) -> float:
        """
        Compute the pupil radius in pixels from a pupil mask.
        
        Uses the maximum extent of the pupil to determine the radius.
        """
        backend = get_backend()
        xp = backend.xp
        
        # Convert to numpy for processing (needed for scipy)
        if hasattr(pupil, 'get'):
            pupil_np = pupil.get()
        else:
            pupil_np = np.asarray(pupil)
        
        # Find the center of mass
        y_indices, x_indices = np.where(pupil_np > 0)
        if len(y_indices) == 0:
            return self._radius
        
        cy = np.mean(y_indices)
        cx = np.mean(x_indices)
        
        # Compute distances from center to all pupil pixels
        distances = np.sqrt((y_indices - cy)**2 + (x_indices - cx)**2)
        
        # Use the maximum distance as the radius (outer edge of pupil)
        # Add a small margin to ensure rho <= 1 for all pupil pixels
        radius = distances.max() * 1.01
        
        return float(radius)
    
    @property
    def rms_expected(self) -> float:
        """
        Expected RMS of generated phase screens.
        
        If target_rms is specified, returns that. Otherwise returns an 
        approximate value based on power-law scaling.
        """
        if self.zernike_config.target_rms is not None:
            return self.zernike_config.target_rms
        
        # Approximate RMS from power-law: sum of mode variances
        # Each mode variance ~ (n+1)^(-power_law)
        xp = self._xp
        nm_list = get_nm_list(*self.zernike_config.n_range)
        
        var_lf = sum((n + 1) ** (-self.zernike_config.power_law) for n, m in nm_list)
        
        # Add HF component variance if present
        if self.zernike_config.hf_rms is not None:
            var_hf = self.zernike_config.hf_rms ** 2
            var_total = var_lf + var_hf
        else:
            var_total = var_lf
        
        return float(xp.sqrt(var_total))
    
    def info(self) -> Dict:
        """Return generator information."""
        base_info = super().info()
        base_info.update({
            "n_range": self.zernike_config.n_range,
            "power_law": self.zernike_config.power_law,
            "target_rms": self.zernike_config.target_rms,
            "n_modes": count_modes(*self.zernike_config.n_range),
        })
        
        # Add HF turbulence info if configured
        if self.zernike_config.hf_rms is not None:
            base_info.update({
                'hf_enabled': True,
                'f_cutoff': self.zernike_config.f_cutoff,
                'hf_alpha': self.zernike_config.hf_alpha,
                'hf_rms': self.zernike_config.hf_rms,
                'transition_width': self.zernike_config.transition_width,
            })
        else:
            base_info['hf_enabled'] = False
        
        return base_info

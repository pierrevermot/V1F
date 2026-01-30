"""
Jolissaint Analytical AO Model for Long-Exposure PSF

Implements the phase spatial power spectrum approach for computing
long-exposure, AO-corrected PSFs as described in:

    Jolissaint, Véran & Conan (2006)
    "Analytical modeling of adaptive optics: foundations of the 
     phase spatial power spectrum approach"
    J. Opt. Soc. Am. A, Vol. 23, No. 2, pp. 382-394

The method computes the residual phase power spectrum (PS) after AO correction
as a sum of five error terms:
1. Fitting error (high-frequency turbulence beyond AO correction)
2. Anisoplanatism error (off-axis science target)
3. Servo-lag error (temporal delay in AO loop)
4. WFS aliasing error (high frequencies aliased into low frequencies)
5. WFS noise error (detector and photon noise)

From the total residual PS, the structure function is computed, which
gives the AO-corrected OTF via OTF_ao(f) = exp(-D_phi(λf)/2).
The long-exposure PSF is then obtained via inverse Fourier transform.

Author: NEBRAA
"""

from __future__ import annotations

import math
from typing import Tuple, Optional, List, Dict, Union, Any
from dataclasses import dataclass, field
import numpy.typing as npt

from ..utils.compute import get_backend
from .low_wind_effect import LowWindEffect, LowWindEffectConfig

# Type alias for array-like objects (numpy or cupy arrays)
ArrayLike = Any


# =============================================================================
# Cache Dataclasses for Performance
# =============================================================================

@dataclass(frozen=True)
class FFTGrid:
    """
    Precomputed FFT frequency grid.
    
    Frozen dataclass ensures immutability and enables safe caching.
    All arrays are computed once in __post_init__.
    """
    n_pix: int
    pupil_pixel_size: float  # meters
    xp: Any  # numpy or cupy module
    
    # These are set in __post_init__
    FX: ArrayLike = field(init=False, repr=False)
    FY: ArrayLike = field(init=False, repr=False)
    F: ArrayLike = field(init=False, repr=False)
    F2: ArrayLike = field(init=False, repr=False)
    df: float = field(init=False)
    dA: float = field(init=False)
    
    def __post_init__(self):
        xp = self.xp
        n = self.n_pix
        
        # Frequency grid (cycles/meter)
        fx = xp.fft.fftfreq(n, d=self.pupil_pixel_size).astype(xp.float64)
        fy = xp.fft.fftfreq(n, d=self.pupil_pixel_size).astype(xp.float64)
        FX, FY = xp.meshgrid(fx, fy, indexing='xy')
        F2 = FX**2 + FY**2
        F = xp.sqrt(F2)
        
        # Use object.__setattr__ for frozen dataclass
        object.__setattr__(self, 'FX', FX)
        object.__setattr__(self, 'FY', FY)
        object.__setattr__(self, 'F', F)
        object.__setattr__(self, 'F2', F2)
        
        df_val = float(fx[1] - fx[0])
        object.__setattr__(self, 'df', df_val)
        object.__setattr__(self, 'dA', df_val**2)


@dataclass(frozen=True)
class AOMasks:
    """
    Precomputed AO correction masks and filters.
    
    These depend only on system geometry and are computed once.
    """
    xp: Any
    mu_LF: ArrayLike  # Low-frequency (corrected) domain mask
    mu_HF: ArrayLike  # High-frequency (uncorrected) domain mask  
    mu_WFS: ArrayLike  # WFS measurement domain mask
    Fp: ArrayLike  # Piston filter
    
    @staticmethod
    def build(
        grid: FFTGrid,
        D: float,
        f_ao: float,
        f_wfs: float,
        mask_geometry: str = 'circular',
        mask_rolloff: float = 0.0,
    ) -> 'AOMasks':
        """
        Build AO masks from grid and system parameters.
        
        Args:
            grid: FFT frequency grid
            D: Telescope diameter (meters)
            f_ao: AO correction cutoff frequency (cycles/meter)
            f_wfs: WFS measurement cutoff frequency (cycles/meter)
            mask_geometry: 'circular' or 'square'
            mask_rolloff: Smooth transition width (0 = hard cutoff)
        
        Returns:
            AOMasks instance
        """
        xp = grid.xp
        
        # Build LF/WFS masks based on geometry
        if mask_geometry == 'circular':
            if mask_rolloff > 0:
                # Smooth sigmoid roll-off
                mu_LF = 1.0 / (1.0 + xp.exp((grid.F - f_ao) / mask_rolloff))
                mu_WFS = 1.0 / (1.0 + xp.exp((grid.F - f_wfs) / mask_rolloff))
            else:
                # Hard circular cutoff
                mu_LF = (grid.F < f_ao).astype(xp.float64)
                mu_WFS = (grid.F < f_wfs).astype(xp.float64)
        else:  # square
            if mask_rolloff > 0:
                # Smooth square mask
                mask_x_ao = 1.0 / (1.0 + xp.exp((xp.abs(grid.FX) - f_ao) / mask_rolloff))
                mask_y_ao = 1.0 / (1.0 + xp.exp((xp.abs(grid.FY) - f_ao) / mask_rolloff))
                mu_LF = mask_x_ao * mask_y_ao
                
                mask_x_wfs = 1.0 / (1.0 + xp.exp((xp.abs(grid.FX) - f_wfs) / mask_rolloff))
                mask_y_wfs = 1.0 / (1.0 + xp.exp((xp.abs(grid.FY) - f_wfs) / mask_rolloff))
                mu_WFS = mask_x_wfs * mask_y_wfs
            else:
                # Hard square cutoff
                mu_LF = ((xp.abs(grid.FX) < f_ao) & (xp.abs(grid.FY) < f_ao)).astype(xp.float64)
                mu_WFS = ((xp.abs(grid.FX) < f_wfs) & (xp.abs(grid.FY) < f_wfs)).astype(xp.float64)
        
        # High-frequency mask
        mu_HF = 1.0 - mu_LF
        
        # Piston filter
        Fp = piston_filter(xp, grid.F, D)
        
        return AOMasks(xp=xp, mu_LF=mu_LF, mu_HF=mu_HF, mu_WFS=mu_WFS, Fp=Fp)


@dataclass(frozen=True)
class LayerCache:
    """
    Precomputed per-layer turbulence PSD and layer properties.
    
    This eliminates redundant Von Kármán PSD evaluations.
    """
    psd_phi: ArrayLike  # Φ_i(fx, fy) on full grid
    r0_sci: float  # r0 at science wavelength
    vx: float  # Wind velocity x-component
    vy: float  # Wind velocity y-component
    h: float  # Layer altitude


# =============================================================================
# Atmospheric Layer Model
# =============================================================================

@dataclass
class TurbulentLayer:
    """
    Single turbulent atmospheric layer.
    
    Attributes:
        altitude: Height above telescope (meters)
        r0: Fried parameter for this layer at reference wavelength (meters)
        wind_speed: Wind velocity magnitude (m/s)
        wind_direction: Wind direction (radians, 0 = +x axis)
        Cn2_fraction: Fraction of total Cn2 (optional, for weighting)
    """
    altitude: float
    r0: float
    wind_speed: float = 10.0
    wind_direction: float = 0.0
    Cn2_fraction: float = 1.0
    
    @property
    def wind_velocity(self) -> Tuple[float, float]:
        """Return (vx, vy) wind velocity components."""
        return (
            self.wind_speed * math.cos(self.wind_direction),
            self.wind_speed * math.sin(self.wind_direction)
        )


@dataclass
class AtmosphereProfile:
    """
    Multi-layer atmospheric turbulence profile.
    
    Supports two modes:
    1. Explicit per-layer r0: Each layer has its own r0 value
    2. Total r0 + Cn2 fractions: Layers have Cn2_fraction, r0 computed from total
    
    Attributes:
        layers: List of TurbulentLayer objects
        wavelength_ref: Reference wavelength for r0 values (meters)
        L0: Outer scale of turbulence (meters), None for Kolmogorov
        l0: Inner scale of turbulence (meters), default 0
        total_r0: Optional total r0 at reference wavelength (meters).
                  If provided, per-layer r0 values are computed from Cn2_fraction.
    """
    layers: List[TurbulentLayer]
    wavelength_ref: float = 0.5e-6
    L0: Optional[float] = 25.0
    l0: float = 0.0
    total_r0: Optional[float] = None
    
    def __post_init__(self):
        """
        If total_r0 is provided, compute per-layer r0 from Cn2_fraction.
        
        Uses: r0_i = r0_total * fraction_i^(-3/5)
        because r0^(-5/3) is linear in turbulence strength.
        """
        if self.total_r0 is not None:
            # Validate that Cn2_fractions sum to 1
            total_frac = sum(layer.Cn2_fraction for layer in self.layers)
            if not (0.99 < total_frac < 1.01):
                raise ValueError(
                    f"Cn2_fractions must sum to 1.0, got {total_frac:.4f}. "
                    "Either normalize fractions or provide explicit r0 per layer."
                )
            
            # Compute per-layer r0 from total_r0 and fractions
            # r0_i^(-5/3) = fraction_i * r0_total^(-5/3)
            # => r0_i = r0_total * fraction_i^(-3/5)
            r0_total_inv_53 = self.total_r0 ** (-5/3)
            for layer in self.layers:
                # Update layer r0 based on fraction (TurbulentLayer is not frozen)
                layer.r0 = (layer.Cn2_fraction * r0_total_inv_53) ** (-3/5)
    
    @property
    def r0_total_computed(self) -> float:
        """Compute total Fried parameter from per-layer r0 values."""
        # r0_total^(-5/3) = sum(r0_i^(-5/3))
        r0_inv_53 = sum(layer.r0 ** (-5/3) for layer in self.layers)
        return r0_inv_53 ** (-3/5)
    
    @property
    def r0_effective(self) -> float:
        """Return total r0 (provided or computed from layers)."""
        return self.total_r0 if self.total_r0 is not None else self.r0_total_computed
    
    @property
    def r0_total(self) -> float:
        """Backward compatibility alias for r0_effective."""
        return self.r0_effective
    
    @property
    def mean_altitude(self) -> float:
        """Cn2-weighted mean altitude."""
        total_weight = sum(layer.r0 ** (-5/3) for layer in self.layers)
        weighted_sum = sum(
            layer.altitude * layer.r0 ** (-5/3) 
            for layer in self.layers
        )
        return weighted_sum / total_weight
    
    @property
    def mean_wind_speed(self) -> float:
        """Cn2-weighted mean wind speed."""
        total_weight = sum(layer.r0 ** (-5/3) for layer in self.layers)
        weighted_sum = sum(
            layer.wind_speed * layer.r0 ** (-5/3) 
            for layer in self.layers
        )
        return weighted_sum / total_weight
    
    def r0_at_wavelength(self, wavelength: float) -> float:
        """
        Scale total r0 to different wavelength.
        
        r0(λ) = r0(λ_ref) * (λ/λ_ref)^(6/5)
        """
        return self.r0_total * (wavelength / self.wavelength_ref) ** (6/5)
    
    def isoplanatic_angle(self, wavelength: float) -> float:
        """
        Compute isoplanatic angle θ0.
        
        θ0 = 0.314 * r0 / h_bar
        where h_bar is the Cn2-weighted mean altitude.
        """
        r0 = self.r0_at_wavelength(wavelength)
        h_bar = self.mean_altitude
        return 0.314 * r0 / max(h_bar, 1.0)  # Avoid division by zero
    
    def coherence_time(self, wavelength: float) -> float:
        """
        Compute atmospheric coherence time τ0.
        
        τ0 = 0.314 * r0 / v_bar
        where v_bar is the Cn2-weighted mean wind speed.
        """
        r0 = self.r0_at_wavelength(wavelength)
        v_bar = self.mean_wind_speed
        return 0.314 * r0 / max(v_bar, 0.1)  # Avoid division by zero


# =============================================================================
# AO System Configuration
# =============================================================================

@dataclass
class AOSystemConfig:
    """
    Adaptive optics system configuration.
    
    Attributes:
        actuator_pitch: DM actuator spacing in pupil plane (meters)
        wfs_subaperture_size: WFS subaperture size (meters), typically = actuator_pitch
        integration_time: WFS integration time (seconds)
        loop_delay: Additional delay after integration (seconds)
        noise_variance: WFS slope noise variance (rad^2)
        science_field_offset: Angular separation between science target and NGS (radians).
                              This is the angle θ in the paper's Eq. 25. The science target
                              is at angle θ from the guide star, so correction degrades for
                              the science target due to anisoplanatism.
        science_field_direction: Direction of science target relative to NGS (radians, 0 = +x axis)
        include_aliasing: Whether to include WFS aliasing error
        include_fitting: Whether to include fitting error
        include_servo_lag: Whether to include servo-lag error
        include_anisoplanatism: Whether to include anisoplanatism error
        include_noise: Whether to include WFS noise error
        mask_geometry: Shape of LF/HF correction masks:
                       'circular' (default, realistic) or 'square' (Jolissaint paper convention)
        mask_rolloff: Smoothness of mask transition. 0.0 = hard cutoff, >0 = smooth transition.
                      Recommended: 0.1 * f_ao for realistic modal reconstructor behavior.
                      The mask uses a sigmoid: 1/(1 + exp((F - f0)/delta)) with delta = rolloff.
    """
    actuator_pitch: float
    wfs_subaperture_size: Optional[float] = None  # Default to actuator_pitch
    integration_time: float = 0.001  # 1 ms
    loop_delay: float = 0.0005  # 0.5 ms
    noise_variance: float = 0.0  # rad^2
    science_field_offset: float = 0.0  # On-axis science = same as NGS
    science_field_direction: float = 0.0
    include_aliasing: bool = True
    include_fitting: bool = True
    include_servo_lag: bool = True
    include_anisoplanatism: bool = True
    include_noise: bool = True
    mask_geometry: str = 'circular'  # 'circular' or 'square'
    mask_rolloff: float = 0.0  # Smooth transition width (0 = hard cutoff)
    
    def __post_init__(self):
        if self.wfs_subaperture_size is None:
            self.wfs_subaperture_size = self.actuator_pitch
        if self.mask_geometry not in ('circular', 'square'):
            raise ValueError(f"mask_geometry must be 'circular' or 'square', got '{self.mask_geometry}'")
    
    @property
    def f_ao(self) -> float:
        """AO cutoff frequency from DM (cycles/meter)."""
        return 1.0 / (2.0 * self.actuator_pitch)
    
    @property
    def f_wfs(self) -> float:
        """WFS Nyquist frequency (cycles/meter)."""
        return 1.0 / (2.0 * self.wfs_subaperture_size)
    
    @property
    def total_delay(self) -> float:
        """Total servo lag: dt/2 + delay. [Eq. after Eq. 10 in paper]"""
        return self.integration_time / 2.0 + self.loop_delay
    
    @property
    def field_offset(self) -> Tuple[float, float]:
        """Science field offset vector (θx, θy) relative to NGS, in radians."""
        return (
            self.science_field_offset * math.cos(self.science_field_direction),
            self.science_field_offset * math.sin(self.science_field_direction)
        )


# =============================================================================
# Turbulent Phase Power Spectrum Models
# =============================================================================

def kolmogorov_psd(xp, f, r0: float):
    """
    Kolmogorov phase power spectral density. [Eq. 19]
    
    Φ(f) = 0.023 * r0^(-5/3) * f^(-11/3)
    
    where f is spatial frequency in cycles/meter.
    
    Args:
        xp: Backend module (numpy or cupy)
        f: Spatial frequency magnitude (cycles/meter, from fftfreq)
        r0: Fried parameter (meters)
    
    Returns:
        Power spectral density (rad^2 per (cycles/m)^2)
        
    Note:
        The coefficient 0.023 is for f in cycles/m, integrated as ∫∫ Φ(f) d²f.
        This is the standard convention for FFT-based phase screen generation.
        The structure function D_φ(r0) = 6.88 is recovered with this normalization.
    """
    # Avoid division by zero
    f = xp.maximum(xp.asarray(f), xp.float64(1e-12))
    
    return 0.023 * (r0 ** (-5/3)) * (f ** (-11/3))


def von_karman_psd(xp, f, r0: float, L0: float):
    """
    Von Karman phase power spectral density (with outer scale).
    
    Φ(f) = 0.023 * r0^(-5/3) * (f^2 + f0^2)^(-11/6)
    
    where f is spatial frequency in cycles/meter and f0 = 1/L0.
    
    Args:
        xp: Backend module (numpy or cupy)
        f: Spatial frequency magnitude (cycles/meter, from fftfreq)
        r0: Fried parameter (meters)
        L0: Outer scale (meters)
    
    Returns:
        Power spectral density (rad^2 per (cycles/m)^2)
        
    Note:
        The coefficient 0.023 is for f in cycles/m, integrated as ∫∫ Φ(f) d²f.
        The outer scale cutoff frequency is f0 = 1/L0 in cycles/m.
    """
    f = xp.asarray(f)
    
    # Outer scale cutoff frequency in cycles/m
    f0 = 1.0 / L0
    
    return 0.023 * (r0 ** (-5/3)) * ((f**2 + f0**2) ** (-11/6))


def piston_filter(xp, f, D: float):
    """
    Piston filter for removing piston mode from phase PS. [Eq. 21]
    
    F_p(f) = 1 - (2*J1(π*D*f)/(π*D*f))^2
    
    Args:
        xp: Backend module (numpy or cupy)
        f: Spatial frequency magnitude (cycles/meter)
        D: Telescope diameter (meters)
    
    Returns:
        Piston filter values
    
    Raises:
        RuntimeError: If required Bessel function library is not available
    """
    
    f = xp.asarray(f, dtype=xp.float64)
    
    # Argument for Bessel function
    x = xp.pi * D * f
    
    # Handle x=0 case (filter should be 0 at f=0)
    x = xp.maximum(x, xp.float64(1e-12))
    
    # Compute J1 accurately for the active backend
    j1_vals = None
    
    # CuPy path - require cupyx.scipy.special.j1
    try:
        import cupy  # type: ignore
        if xp is cupy:
            try:
                from cupyx.scipy.special import j1 as cupy_j1  # type: ignore
                j1_vals = cupy_j1(x)
            except ImportError as e:
                raise RuntimeError(
                    "cupyx.scipy.special.j1 required for piston_filter on CuPy backend. "
                    "Install with: pip install cupy-cuda11x (or appropriate CUDA version)"
                ) from e
    except ImportError:
        pass
    
    # NumPy path (only reached if xp is not cupy)
    if j1_vals is None:
        try:
            from scipy.special import j1 as scipy_j1
        except ImportError as e:
            raise RuntimeError(
                "scipy.special.j1 required for piston_filter. "
                "Install with: pip install scipy"
            ) from e
        # x is guaranteed to be a numpy array here since xp is not cupy
        j1_vals = xp.asarray(scipy_j1(x), dtype=xp.float64)
    
    # Piston filter: Fp = 1 - (2*J1(x)/x)^2
    term = 2.0 * j1_vals / x
    Fp = 1.0 - term**2
    
    # Ensure non-negative (numerical issues near f=0)
    Fp = xp.maximum(Fp, xp.float64(0.0))
    
    return Fp


class PistonFilterLUT:
    """
    Lookup-table based piston filter for fast evaluation.
    
    Precomputes J1(x)/x on a fine grid and uses linear interpolation.
    This is ~2-3x faster than calling scipy.special.j1 repeatedly.
    
    Usage:
        lut = PistonFilterLUT(D=8.2, f_max=10.0, n_points=10000)
        Fp = lut(f_array)
    """
    
    def __init__(self, D: float, f_max: float = 10.0, n_points: int = 10000):
        """
        Initialize the LUT.
        
        Args:
            D: Telescope diameter (meters)
            f_max: Maximum spatial frequency to tabulate (cycles/meter)
            n_points: Number of points in the lookup table
        """
        import numpy as np
        from scipy.special import j1 as scipy_j1
        
        self.D = D
        self.x_max = np.pi * D * f_max
        self.n_points = n_points
        
        # Build LUT for J1(x)/x
        self.x_lut = np.linspace(0, self.x_max, n_points)
        self.x_lut[0] = 1e-12  # Avoid division by zero
        self.dx = self.x_lut[1] - self.x_lut[0]
        
        j1_vals = scipy_j1(self.x_lut)
        self.j1_over_x_lut = j1_vals / self.x_lut
        self.j1_over_x_lut[0] = 0.5  # Limit as x->0: J1(x)/x -> 1/2
        
        # Cache for GPU-side LUT to avoid repeated CPU->GPU transfers
        self._gpu_lut_cache = None
    
    def __call__(self, f, xp=None) -> ArrayLike:
        """
        Evaluate piston filter at frequencies f using LUT interpolation.
        
        Args:
            f: Spatial frequency magnitude (cycles/meter)
            xp: Array module (numpy or cupy). If None, inferred from f.
        
        Returns:
            Piston filter values F_p(f)
        """
        import numpy as np
        
        # Infer xp from input array if not provided
        if xp is None:
            try:
                import cupy
                if hasattr(f, '__cuda_array_interface__'):
                    xp = cupy
                else:
                    xp = np
            except ImportError:
                xp = np
        
        f = xp.asarray(f, dtype=xp.float64)
        x = xp.pi * self.D * xp.abs(f)
        
        # Clamp to LUT range
        x = xp.clip(x, 0, self.x_max - self.dx)
        
        # Linear interpolation
        idx = (x / self.dx).astype(xp.int64)
        frac = (x / self.dx) - idx
        
        # Use cached GPU array to avoid repeated CPU->GPU transfers
        is_gpu = xp.__name__ == 'cupy'
        if is_gpu:
            if self._gpu_lut_cache is None:
                self._gpu_lut_cache = xp.asarray(self.j1_over_x_lut)
            j1_over_x_lut = self._gpu_lut_cache
        else:
            j1_over_x_lut = self.j1_over_x_lut
        
        # Interpolate
        idx_next = xp.minimum(idx + 1, self.n_points - 1)
        j1_over_x = j1_over_x_lut[idx] * (1 - frac) + j1_over_x_lut[idx_next] * frac
        
        # Piston filter: Fp = 1 - (2*J1(x)/x)^2
        term = 2.0 * j1_over_x
        Fp = 1.0 - term**2
        
        return xp.maximum(Fp, xp.float64(0.0))


# =============================================================================
# Jolissaint AO Model
# =============================================================================

class JolissaintAOModel:
    """
    Analytical AO model following Jolissaint et al. (2006).
    
    Computes long-exposure, AO-corrected PSF using the phase spatial
    power spectrum approach.
    
    The model computes:
    1. Residual phase PS as sum of fitting, anisoplanatism, servo-lag,
       aliasing, and noise errors
    2. Phase structure function from PS via numerical integration
    3. AO-corrected OTF from structure function
    4. Long-exposure PSF via inverse Fourier transform
    """
    
    def __init__(
        self,
        n_pix: int,
        telescope_diameter: float,
        obstruction_diameter: float,
        wavelength: float,
        pixel_scale: float,
        atmosphere: AtmosphereProfile,
        ao_config: AOSystemConfig,
        lwe_config: Optional[LowWindEffectConfig] = None,
    ):
        """
        Initialize the Jolissaint AO model.
        
        Args:
            n_pix: Grid size (pixels)
            telescope_diameter: Primary mirror diameter (meters)
            obstruction_diameter: Central obstruction diameter (meters)
            wavelength: Science wavelength (meters)
            pixel_scale: Focal plane pixel scale (radians/pixel)
            atmosphere: Atmospheric turbulence profile
            ao_config: AO system configuration
            lwe_config: Optional Low Wind Effect configuration. If provided,
                       LWE will be included in long-exposure PSF computation.
        """
        # Cache backend reference ONCE at init to avoid repeated get_backend() calls
        self._backend = get_backend()
        self._xp = self._backend.xp
        xp = self._xp
        
        self.n_pix = n_pix
        self.D = telescope_diameter
        self.D_obs = obstruction_diameter
        self.wavelength = wavelength
        self.pixel_scale = pixel_scale
        self.atmosphere = atmosphere
        self.ao_config = ao_config
        self.lwe_config = lwe_config
        
        # LWE model (lazily initialized on first use)
        self._lwe_model = None
        
        # Pupil plane pixel size
        self.pupil_pixel_size = wavelength / (n_pix * pixel_scale)
        
        # Physical pupil extent
        self.pupil_extent = n_pix * self.pupil_pixel_size
        
        # Build immutable caches for performance (NEW APPROACH)
        self.grid = FFTGrid(n_pix, self.pupil_pixel_size, xp)
        self.masks = AOMasks.build(
            self.grid, self.D,
            self.ao_config.f_ao,
            self.ao_config.f_wfs,
            self.ao_config.mask_geometry,
            self.ao_config.mask_rolloff,
        )
        self.layer_cache = self._build_layer_cache()
        
        # Keep old attributes for backward compatibility
        self.FX = self.grid.FX
        self.FY = self.grid.FY
        self.F = self.grid.F
        self.df = self.grid.df
        self.dA = self.grid.dA
        self.mu_LF = self.masks.mu_LF
        self.mu_HF = self.masks.mu_HF
        self.mu_WFS = self.masks.mu_WFS
        self.Fp = self.masks.Fp
        
        # Piston filter LUT for fast aliasing computation
        f_max_grid = float(self._backend.to_numpy(xp.max(self.grid.F)))
        f_max_lut = f_max_grid + 5.0 / self.ao_config.wfs_subaperture_size
        self._piston_lut = PistonFilterLUT(self.D, f_max=f_max_lut, n_points=20000)
        
        # Precompute aliasing index arrays for vectorized computation
        self._precompute_aliasing_arrays()
    
    def _build_layer_cache(self) -> Tuple[LayerCache, ...]:
        """
        Build per-layer PSD cache to avoid redundant computations.
        
        Returns:
            Tuple of LayerCache objects, one per atmospheric layer.
        """
        xp = self._xp
        caches = []
        
        for layer in self.atmosphere.layers:
            # Scale r0 to science wavelength
            r0_sci = layer.r0 * (self.wavelength / self.atmosphere.wavelength_ref) ** (6/5)
            
            # Compute PSD for this layer on full grid
            if self.atmosphere.L0 is not None:
                psd_phi = von_karman_psd(xp, self.grid.F, r0_sci, self.atmosphere.L0)
            else:
                psd_phi = kolmogorov_psd(xp, self.grid.F, r0_sci)
            
            # Extract wind components
            vx, vy = layer.wind_velocity
            
            caches.append(LayerCache(
                psd_phi=psd_phi,
                r0_sci=r0_sci,
                vx=vx,
                vy=vy,
                h=layer.altitude,
            ))
        
        return tuple(caches)
    
    # NOTE: _setup_frequency_grid() and _setup_masks() have been removed.
    # Use FFTGrid and AOMasks frozen dataclasses instead (self.grid, self.masks).
    
    def _precompute_aliasing_arrays(self):
        """Precompute index arrays used in vectorized aliasing computation."""
        xp = self._xp
        n_alias = 3
        
        # (k,l) pairs excluding (0,0) for main aliasing sum
        kl_pairs = [(k, l) for k in range(-n_alias, n_alias + 1)
                    for l in range(-n_alias, n_alias + 1) if not (k == 0 and l == 0)]
        self._alias_k_arr = xp.array([kl[0] for kl in kl_pairs], dtype=xp.float64)[:, None, None]
        self._alias_l_arr = xp.array([kl[1] for kl in kl_pairs], dtype=xp.float64)[:, None, None]
        self._alias_sign_arr = xp.array([(-1) ** (kl[0] + kl[1]) for kl in kl_pairs], dtype=xp.float64)[:, None, None]
        
        # l != 0 for fx=0 singularity case
        self._alias_l_vals = xp.array([l for l in range(-n_alias, n_alias + 1) if l != 0], dtype=xp.float64)
        
        # k != 0 for fy=0 singularity case
        self._alias_k_vals = xp.array([k for k in range(-n_alias, n_alias + 1) if k != 0], dtype=xp.float64)
        
        # (k,l) with k != 0 AND l != 0 for (0,0) singularity case
        kl_00 = [(k, l) for k in range(-n_alias, n_alias + 1)
                 for l in range(-n_alias, n_alias + 1) if k != 0 and l != 0]
        self._alias_k_00 = xp.array([kl[0] for kl in kl_00], dtype=xp.float64)
        self._alias_l_00 = xp.array([kl[1] for kl in kl_00], dtype=xp.float64)
    
    def compute_fitting_psd(self) -> ArrayLike:
        """
        Compute fitting error PSD. [Eq. 22]
        
        High-frequency turbulence that cannot be corrected by the AO system.
        
        Φ_fit(f) = μ_HF(f) * F_p(f) * Φ_turb(f)
        """
        xp = self._xp
        
        if not self.ao_config.include_fitting:
            return xp.zeros_like(self.grid.F)
        
        # Sum over all layers using precomputed PSDs
        psd_fit = xp.zeros_like(self.grid.F)
        for lc in self.layer_cache:
            psd_fit = psd_fit + self.masks.mu_HF * self.masks.Fp * lc.psd_phi
        
        return psd_fit
    
    def compute_anisoplanatism_psd(self) -> ArrayLike:
        """
        Compute anisoplanatism error PSD. [Eq. 25]
        
        Error due to science target at angle θ from the guide star,
        seeing different turbulence than what the AO corrects for.
        
        Φ_aniso(f,θ) = 2 * μ_LF(f) * F_p(f) * Σ_n Φ_n(f) * [1 - cos(2πh_n f·θ)]
        
        where θ is the angular separation between science target and NGS.
        """
        xp = self._xp
        
        if not self.ao_config.include_anisoplanatism or self.ao_config.science_field_offset == 0:
            return xp.zeros_like(self.grid.F)
        
        theta_x, theta_y = self.ao_config.field_offset
        
        psd_aniso = xp.zeros_like(self.grid.F)
        for lc in self.layer_cache:
            # Phase term: 2π * h * (fx*θx + fy*θy)
            phase = 2.0 * xp.pi * lc.h * (self.grid.FX * theta_x + self.grid.FY * theta_y)
            psd_aniso = psd_aniso + 2.0 * self.masks.mu_LF * self.masks.Fp * lc.psd_phi * (1.0 - xp.cos(phase))
        
        return psd_aniso
    
    def compute_servo_lag_psd(self) -> ArrayLike:
        """
        Compute servo-lag error PSD. [Eq. 31]
        
        Error due to time delay between WFS measurement and DM correction.
        
        Φ_servo(f) = μ_LF(f) * F_p(f) * Σ_n Φ_n(f) * 
                     {1 - 2*cos(2π*t_l*f·v_n)*sinc(Δt*f·v_n) + sinc²(Δt*f·v_n)}
        """
        xp = self._xp
        
        if not self.ao_config.include_servo_lag:
            return xp.zeros_like(self.grid.F)
        
        dt = self.ao_config.integration_time
        tl = self.ao_config.total_delay
        
        psd_servo = xp.zeros_like(self.grid.F)
        for lc in self.layer_cache:
            # f·v = fx*vx + fy*vy
            f_dot_v = self.grid.FX * lc.vx + self.grid.FY * lc.vy
            
            # sinc and cos arguments
            sinc_val = xp.sinc(dt * f_dot_v)
            cos_arg = 2.0 * xp.pi * tl * f_dot_v
            
            # Error factor: {1 - 2*cos(...)*sinc(...) + sinc²(...)}
            error_factor = 1.0 - 2.0 * xp.cos(cos_arg) * sinc_val + sinc_val**2
            
            psd_servo = psd_servo + self.masks.mu_LF * self.masks.Fp * lc.psd_phi * error_factor
        
        return psd_servo
    
    def compute_aniso_servo_psd(self) -> ArrayLike:
        """
        Compute combined anisoplanatism + servo-lag PSD. [Eq. 33]
        
        This is the proper way to combine these correlated errors,
        rather than adding them independently.
        
        Φ_θ,s(f) = μ_LF(f) * F_p(f) * Σ_n Φ_n(f) * 
                   {1 - 2*cos(2π*f·[h_n*θ - t_l*v_n])*sinc(Δt*f·v_n) + sinc²(Δt*f·v_n)}
        
        where θ is the science field angle relative to NGS.
        
        Note: This method should only be called when BOTH anisoplanatism and
        servo-lag are enabled. For individual terms, use compute_anisoplanatism_psd()
        or compute_servo_lag_psd().
        """
        xp = self._xp
        
        # If both are disabled, return zero
        if not self.ao_config.include_anisoplanatism and not self.ao_config.include_servo_lag:
            return xp.zeros_like(self.grid.F)
        
        # If only one is enabled, fall back to separate terms
        # (The combined equation requires both effects to be present)
        if not self.ao_config.include_anisoplanatism:
            return self.compute_servo_lag_psd()
        if not self.ao_config.include_servo_lag:
            return self.compute_anisoplanatism_psd()
        
        # Both are enabled: compute combined term
        dt = self.ao_config.integration_time
        tl = self.ao_config.total_delay
        theta_x, theta_y = self.ao_config.field_offset
        
        psd_as = xp.zeros_like(self.grid.F)
        for lc in self.layer_cache:
            # Combined offset: h*θ - t_l*v
            offset_x = lc.h * theta_x - tl * lc.vx
            offset_y = lc.h * theta_y - tl * lc.vy
            
            # f·offset = fx*offset_x + fy*offset_y
            f_dot_offset = self.grid.FX * offset_x + self.grid.FY * offset_y
            
            # f·v for sinc
            f_dot_v = self.grid.FX * lc.vx + self.grid.FY * lc.vy
            sinc_val = xp.sinc(dt * f_dot_v)
            
            # cos argument
            cos_arg = 2.0 * xp.pi * f_dot_offset
            
            # Error factor
            error_factor = 1.0 - 2.0 * xp.cos(cos_arg) * sinc_val + sinc_val**2
            
            psd_as = psd_as + self.masks.mu_LF * self.masks.Fp * lc.psd_phi * error_factor
        
        return psd_as
    
    def compute_aliasing_psd(self) -> ArrayLike:
        """
        Compute WFS aliasing error PSD. [Eqs. 44-48]
        
        High-frequency turbulence aliased into low frequencies by the WFS.
        
        Implementation of Eq. (45) using decorrelated replica assumption:
        
        Φ_alias(f) = μ_LF_eff(f) * (fx²*fy²/f⁴) * Σ_n sinc²(Δt*f·v_n) *
                     Σ_{k,l≠0} [fx/(fy-l/Λ) + fy/(fx-k/Λ)]² *
                     F_p(|f-(k/Λ,l/Λ)|) * Φ_n(|f-(k/Λ,l/Λ)|)
        
        where μ_LF_eff = μ_LF * μ_WFS (intersection of DM and WFS domains).
        
        With special handling for singularities at fx=0 [Eq. 47], fy=0 [Eq. 46],
        and (fx,fy)=(0,0) [Eq. 48].
        
        Note: Cross-terms between different (k,l) replicas vanish due to
        decorrelation assumption in deriving Eq. (45).
        """
        xp = self._xp
        
        if not self.ao_config.include_aliasing:
            return xp.zeros_like(self.grid.F)
        
        # Check if we're on GPU - use vectorized path for better performance
        is_gpu = xp.__name__ == 'cupy'
        
        if is_gpu:
            return self._compute_aliasing_psd_vectorized()
        else:
            return self._compute_aliasing_psd_loop()
    
    def _compute_aliasing_psd_vectorized(self) -> ArrayLike:
        """
        Vectorized GPU-optimized aliasing PSD computation with memory chunking.
        
        Processes (k,l) terms in batches to avoid GPU OOM on large grids.
        Uses layer_cache for r0_sci and wind velocities.
        """
        xp = self._xp
        
        Lambda = self.ao_config.wfs_subaperture_size
        f_wfs = self.ao_config.f_wfs
        dt = self.ao_config.integration_time
        is_square = self.ao_config.mask_geometry == 'square'
        n_alias = 3
        
        # Memory-efficient chunking: process (k,l) pairs in batches
        # Each batch creates (batch_size, n_pix, n_pix) arrays
        # Batch size tuned for typical GPU memory (adjust if needed)
        chunk_size = 12  # Process 12 (k,l) pairs at a time
        
        # Build list of (k,l) pairs excluding (0,0)
        kl_pairs = [(k, l) for k in range(-n_alias, n_alias + 1)
                    for l in range(-n_alias, n_alias + 1) if not (k == 0 and l == 0)]
        
        psd_alias = xp.zeros_like(self.grid.F, dtype=xp.float64)
        
        for lc in self.layer_cache:
            # Temporal averaging factor: sinc²(Δt * f·v)
            f_dot_v = self.grid.FX * lc.vx + self.grid.FY * lc.vy
            sinc2_temporal = xp.sinc(dt * f_dot_v) ** 2
            
            # Von Karman coefficient (computed once per layer)
            L0_inv2 = 1.0 / self.atmosphere.L0**2 if self.atmosphere.L0 else 0
            coeff = 0.023 * (lc.r0_sci ** (-5/3))
            
            # Accumulate alias_sum over chunked (k,l) pairs (decorrelated replicas)
            alias_sum = xp.zeros_like(self.grid.F, dtype=xp.float64)
            
            for i in range(0, len(kl_pairs), chunk_size):
                batch = kl_pairs[i:i + chunk_size]
                batch_size = len(batch)
                
                # Build batch arrays: (batch_size, 1, 1) for broadcasting
                k_batch = xp.array([kl[0] for kl in batch], dtype=xp.float64)[:, None, None]
                l_batch = xp.array([kl[1] for kl in batch], dtype=xp.float64)[:, None, None]
                
                # Broadcast FX, FY to (batch_size, n_pix, n_pix)
                FX_3d = self.grid.FX[None, :, :]
                FY_3d = self.grid.FY[None, :, :]
                
                # Aliased frequencies
                fx_alias = FX_3d - k_batch / Lambda
                fy_alias = FY_3d - l_batch / Lambda
                f_alias = xp.sqrt(fx_alias**2 + fy_alias**2)
                
                # HF mask
                if is_square:
                    is_hf = (xp.abs(fx_alias) >= f_wfs) | (xp.abs(fy_alias) >= f_wfs)
                else:
                    is_hf = f_alias >= f_wfs
                
                # Von Karman PSD at aliased frequencies
                if self.atmosphere.L0 is not None:
                    psd_alias_f = coeff * xp.power(f_alias**2 + L0_inv2, -11/6)
                else:
                    f_alias_safe = xp.maximum(f_alias, 1e-12)
                    psd_alias_f = coeff * xp.power(f_alias_safe, -11/3)
                
                # Piston filter using LUT
                Fp_alias = self._piston_lut(f_alias, xp=xp)
                
                # PSD term (no sqrt needed)
                psd_term = xp.maximum(Fp_alias * psd_alias_f, 0.0)
                
                # Geometry factor with safe division
                eps = 1e-12
                safe_denom_x = xp.where(xp.abs(fy_alias) < eps,
                                        eps * xp.sign(fy_alias + eps), fy_alias)
                safe_denom_y = xp.where(xp.abs(fx_alias) < eps,
                                        eps * xp.sign(fx_alias + eps), fx_alias)
                
                term_x = FX_3d / safe_denom_x
                term_y = FY_3d / safe_denom_y
                geom_factor = term_x + term_y
                geom2 = geom_factor * geom_factor  # squared geometry term
                
                # PSD contribution for this batch
                contrib = is_hf.astype(xp.float64) * geom2 * psd_term
                
                # Accumulate sum over batch
                alias_sum = alias_sum + xp.sum(contrib, axis=0)
                
                # Free batch memory explicitly
                del fx_alias, fy_alias, f_alias, is_hf, psd_alias_f, Fp_alias
                del psd_term, geom_factor, geom2, contrib
            
            # Prefactor: fx²*fy²/f⁴
            f4 = xp.maximum(self.grid.F**4, 1e-40)
            prefactor = (self.grid.FX**2 * self.grid.FY**2) / f4
            
            # Main PSD contribution
            psd_layer = prefactor * sinc2_temporal * alias_sum
            
            # Handle axis singularities (small arrays, no chunking needed)
            fx_zero = xp.abs(self.grid.FX) < 1e-10
            fy_zero = xp.abs(self.grid.FY) < 1e-10
            
            # fx=0 case using precomputed l_vals (l != 0): shape (6,)
            l_vals = self._alias_l_vals
            FY_3d_l = self.grid.FY[None, :, :]
            fy_al_3d = FY_3d_l - l_vals[:, None, None] / Lambda
            f_al_fx0 = xp.abs(fy_al_3d)
            # At fx=0, HF check is |fy_alias| >= f_wfs (same for both geometries)
            is_hf_fx0 = f_al_fx0 >= f_wfs
            if self.atmosphere.L0 is not None:
                psd_fx0_3d = coeff * xp.power(f_al_fx0**2 + L0_inv2, -11/6)
            else:
                psd_fx0_3d = coeff * xp.power(xp.maximum(f_al_fx0, 1e-12), -11/3)
            Fp_fx0_3d = self._piston_lut(f_al_fx0, xp=xp)
            psd_fx0 = xp.sum(is_hf_fx0.astype(xp.float64) * Fp_fx0_3d * psd_fx0_3d, axis=0) * sinc2_temporal
            
            # fy=0 case using precomputed k_vals (k != 0): shape (6,)
            k_vals = self._alias_k_vals
            FX_3d_k = self.grid.FX[None, :, :]
            fx_al_3d = FX_3d_k - k_vals[:, None, None] / Lambda
            f_al_fy0 = xp.abs(fx_al_3d)
            # At fy=0, HF check is |fx_alias| >= f_wfs (same for both geometries)
            is_hf_fy0 = f_al_fy0 >= f_wfs
            if self.atmosphere.L0 is not None:
                psd_fy0_3d = coeff * xp.power(f_al_fy0**2 + L0_inv2, -11/6)
            else:
                psd_fy0_3d = coeff * xp.power(xp.maximum(f_al_fy0, 1e-12), -11/3)
            Fp_fy0_3d = self._piston_lut(f_al_fy0, xp=xp)
            psd_fy0 = xp.sum(is_hf_fy0.astype(xp.float64) * Fp_fy0_3d * psd_fy0_3d, axis=0) * sinc2_temporal
            
            # (0,0) case: only axis replicas contribute
            # Use existing precomputed arrays for k != 0 and l != 0
            k_vals = self._alias_k_vals  # k != 0
            l_vals = self._alias_l_vals  # l != 0
            
            # k-axis replicas: (|k|/Λ, 0)
            f_k = xp.abs(k_vals) / Lambda
            if self.atmosphere.L0 is not None:
                psd_k = coeff * xp.power(f_k**2 + L0_inv2, -11/6)
            else:
                psd_k = coeff * xp.power(xp.maximum(f_k, 1e-12), -11/3)
            Fp_k = self._piston_lut(f_k, xp=xp)
            
            # l-axis replicas: (0, |l|/Λ)
            f_l = xp.abs(l_vals) / Lambda
            if self.atmosphere.L0 is not None:
                psd_l = coeff * xp.power(f_l**2 + L0_inv2, -11/6)
            else:
                psd_l = coeff * xp.power(xp.maximum(f_l, 1e-12), -11/3)
            Fp_l = self._piston_lut(f_l, xp=xp)
            
            psd_00 = xp.sum(Fp_k * psd_k) + xp.sum(Fp_l * psd_l)
            
            # Combine
            both_zero = fx_zero & fy_zero
            psd_layer = xp.where(both_zero, psd_00, psd_layer)
            psd_layer = xp.where(fx_zero & ~both_zero, psd_fx0, psd_layer)
            psd_layer = xp.where(fy_zero & ~both_zero, psd_fy0, psd_layer)
            
            # Aliasing is a LF residual term, but WFS-bounded.
            # Use effective LF mask (intersection of DM and WFS domains).
            mu_LF_eff = self.masks.mu_LF * self.masks.mu_WFS
            psd_alias = psd_alias + mu_LF_eff * psd_layer
        
        return psd_alias
    
    def _compute_aliasing_psd_loop(self) -> ArrayLike:
        """
        Loop-based aliasing PSD computation (CPU-optimized with LUT).
        
        Uses precomputed layer cache (self.layer_cache) to avoid redundant PSD evaluation.
        
        Note: This method still accumulates (k,l) terms in memory. For GPU usage with large
        arrays, use _compute_aliasing_psd_vectorized with chunking.
        """
        xp = self._xp
        
        Lambda = self.ao_config.wfs_subaperture_size
        f_wfs = self.ao_config.f_wfs
        dt = self.ao_config.integration_time
        
        psd_alias = xp.zeros_like(self.grid.F, dtype=xp.float64)
        n_alias = 3
        
        for lc in self.layer_cache:
            # Temporal averaging factor: sinc²(Δt * f·v)
            f_dot_v = self.grid.FX * lc.vx + self.grid.FY * lc.vy
            sinc2_temporal = xp.sinc(dt * f_dot_v) ** 2
            
            # Sum over (k,l) replica contributions (decorrelated assumption)
            # Each replica contributes: geom_factor² * Fp * Φ
            alias_sum = xp.zeros_like(self.grid.F, dtype=xp.float64)
            
            for k in range(-n_alias, n_alias + 1):
                for l in range(-n_alias, n_alias + 1):
                    if k == 0 and l == 0:
                        continue
                    
                    # Aliased frequency components
                    fx_alias = self.grid.FX - k / Lambda
                    fy_alias = self.grid.FY - l / Lambda
                    f_alias = xp.sqrt(fx_alias**2 + fy_alias**2)
                    
                    # Only spatial frequencies outside the WFS Nyquist alias into LF
                    # Geometry matches mask_geometry setting
                    if self.ao_config.mask_geometry == 'circular':
                        is_hf = f_alias >= f_wfs
                    else:
                        is_hf = (xp.abs(fx_alias) >= f_wfs) | (xp.abs(fy_alias) >= f_wfs)
                    
                    # Turbulent PSD at aliased frequency
                    if self.atmosphere.L0 is not None:
                        psd_alias_f = von_karman_psd(xp, f_alias, lc.r0_sci, self.atmosphere.L0)
                    else:
                        psd_alias_f = kolmogorov_psd(xp, f_alias, lc.r0_sci)
                    
                    # Piston filter at aliased frequency (use LUT for speed)
                    Fp_alias = self._piston_lut(f_alias, xp=xp)
                    
                    # PSD term (no sqrt needed)
                    psd_term = xp.maximum(Fp_alias * psd_alias_f, 0.0)
                    
                    # Geometry factor: fx/(fy - l/Λ) + fy/(fx - k/Λ)
                    # Robust division based on actual denominator magnitude
                    eps = 1e-12
                    
                    denom_x = fy_alias  # denominator for fx/(...) term
                    denom_y = fx_alias  # denominator for fy/(...) term
                    safe_denom_x = xp.where(xp.abs(denom_x) < eps, eps * xp.sign(denom_x + eps), denom_x)
                    safe_denom_y = xp.where(xp.abs(denom_y) < eps, eps * xp.sign(denom_y + eps), denom_y)
                    
                    term_x = self.grid.FX / safe_denom_x  # fx / (fy - l/Λ)
                    term_y = self.grid.FY / safe_denom_y  # fy / (fx - k/Λ)
                    
                    geom_factor = term_x + term_y
                    geom2 = geom_factor * geom_factor  # squared geometry term
                    
                    # Add PSD contribution (only where HF)
                    alias_sum = alias_sum + is_hf.astype(xp.float64) * geom2 * psd_term
            
            # Compute the PSD from the sum of replica contributions
            # Eq. (45): Φ_alias = μ_LF_eff * (fx²fy²/f⁴) * sinc² * sum
            
            # Prefactor: fx²*fy²/f⁴
            # Handle singularities at axes
            f4 = self.grid.F**4
            f4 = xp.maximum(f4, xp.float64(1e-40))
            prefactor = (self.grid.FX**2 * self.grid.FY**2) / f4
            
            # Main contribution from Eq. (45)
            psd_layer = prefactor * sinc2_temporal * alias_sum
            
            # Handle singularities using Eqs. (46-48)
            # These are limit cases that need special treatment
            is_square = self.ao_config.mask_geometry == 'square'
            
            # Eq. (46): fx = 0 case (only l ≠ 0 terms contribute)
            # Φ_alias(0, fy) = μ_WFS * Σ_l≠0 F_p(0, fy-l/Λ) * Φ(0, fy-l/Λ)
            fx_zero = xp.abs(self.grid.FX) < 1e-10
            psd_fx0 = xp.zeros_like(self.grid.F)
            for l in range(-n_alias, n_alias + 1):
                if l == 0:
                    continue
                fy_alias = self.grid.FY - l / Lambda
                f_alias = xp.abs(fy_alias)
                # HF check: at fx=0, aliased point is (0, fy-l/Λ)
                # For square: |fy-l/Λ| >= f_wfs; for circular: same (radial = |fy| when fx=0)
                is_hf_l = f_alias >= f_wfs
                if self.atmosphere.L0 is not None:
                    psd_l = von_karman_psd(xp, f_alias, lc.r0_sci, self.atmosphere.L0)
                else:
                    psd_l = kolmogorov_psd(xp, f_alias, lc.r0_sci)
                Fp_l = self._piston_lut(f_alias, xp=xp)
                psd_fx0 = psd_fx0 + is_hf_l.astype(xp.float64) * Fp_l * psd_l * sinc2_temporal
            
            # Eq. (47): fy = 0 case (only k ≠ 0 terms contribute)
            fy_zero = xp.abs(self.grid.FY) < 1e-10
            psd_fy0 = xp.zeros_like(self.grid.F)
            for k in range(-n_alias, n_alias + 1):
                if k == 0:
                    continue
                fx_alias = self.grid.FX - k / Lambda
                f_alias = xp.abs(fx_alias)
                # HF check: at fy=0, aliased point is (fx-k/Λ, 0)
                # For square: |fx-k/Λ| >= f_wfs; for circular: same (radial = |fx| when fy=0)
                is_hf_k = f_alias >= f_wfs
                if self.atmosphere.L0 is not None:
                    psd_k = von_karman_psd(xp, f_alias, lc.r0_sci, self.atmosphere.L0)
                else:
                    psd_k = kolmogorov_psd(xp, f_alias, lc.r0_sci)
                Fp_k = self._piston_lut(f_alias, xp=xp)
                psd_fy0 = psd_fy0 + is_hf_k.astype(xp.float64) * Fp_k * psd_k * sinc2_temporal
            
            # Eq. (48): (fx, fy) = (0, 0) case
            # At (0,0), only axis replicas contribute: (k/Λ, 0) and (0, l/Λ)
            psd_00 = xp.float64(0.0)
            
            # k-axis replicas: (k/Λ, 0) for k != 0
            for k in range(-n_alias, n_alias + 1):
                if k == 0:
                    continue
                f_alias = abs(k) / Lambda
                if self.atmosphere.L0 is not None:
                    psd_k = float(von_karman_psd(xp, xp.array([f_alias]), lc.r0_sci, self.atmosphere.L0)[0])
                else:
                    psd_k = float(kolmogorov_psd(xp, xp.array([f_alias]), lc.r0_sci)[0])
                Fp_k = float(self._piston_lut(xp.array([f_alias]), xp=xp)[0])
                psd_00 += Fp_k * psd_k
            
            # l-axis replicas: (0, l/Λ) for l != 0
            for l in range(-n_alias, n_alias + 1):
                if l == 0:
                    continue
                f_alias = abs(l) / Lambda
                if self.atmosphere.L0 is not None:
                    psd_l = float(von_karman_psd(xp, xp.array([f_alias]), lc.r0_sci, self.atmosphere.L0)[0])
                else:
                    psd_l = float(kolmogorov_psd(xp, xp.array([f_alias]), lc.r0_sci)[0])
                Fp_l = float(self._piston_lut(xp.array([f_alias]), xp=xp)[0])
                psd_00 += Fp_l * psd_l
            
            # Combine: use special cases where applicable
            both_zero = fx_zero & fy_zero
            psd_layer = xp.where(both_zero, psd_00, psd_layer)
            psd_layer = xp.where(fx_zero & ~both_zero, psd_fx0, psd_layer)
            psd_layer = xp.where(fy_zero & ~both_zero, psd_fy0, psd_layer)
            
            # Aliasing is a LF residual term, but WFS-bounded.
            # Use effective LF mask (intersection of DM and WFS domains).
            mu_LF_eff = self.masks.mu_LF * self.masks.mu_WFS
            psd_alias = psd_alias + mu_LF_eff * psd_layer
        
        return psd_alias
    
    def compute_noise_psd(self) -> ArrayLike:
        """
        Compute WFS noise error PSD. [Eq. 50]
        
        Noise propagated through the reconstructor into the corrected phase.
        
        Φ_noise(f) = μ_LF(f) * N(f) / (4π² * f² * sinc²(Λfx) * sinc²(Λfy))
        
        where N(f) is the slope noise power spectrum, constant within the
        WFS domain |fx|,|fy| < f_WFS = 1/(2Λ) [per paper text after Eq. 50].
        
        Note: The paper specifies that N(f) is bounded to the WFS domain,
        not the DM domain. When wfs_subaperture_size != actuator_pitch,
        these domains differ.
        """
        xp = self._xp
        
        if not self.ao_config.include_noise or self.ao_config.noise_variance == 0:
            return xp.zeros_like(self.grid.F)
        
        Lambda = self.ao_config.wfs_subaperture_size
        sigma_n2 = self.ao_config.noise_variance
        
        # Noise PSD is flat within WFS domain (|fx|,|fy| < f_WFS)
        # N * Λ² = σ_n² [Eq. 51]
        N = sigma_n2 / Lambda**2
        
        # sinc² terms with subaperture size
        sinc_x = xp.sinc(Lambda * self.grid.FX)
        sinc_y = xp.sinc(Lambda * self.grid.FY)
        sinc2 = sinc_x**2 * sinc_y**2
        
        # Avoid division by zero
        f2 = self.grid.F**2
        f2 = xp.maximum(f2, xp.float64(1e-20))
        sinc2 = xp.maximum(sinc2, xp.float64(1e-20))
        
        # Use effective LF mask: intersection of DM-correctable and WFS-measurable domains
        # Noise is a LF residual term, but WFS quantities are bounded to WFS domain.
        # When f_wfs != f_ao, the correct support is the intersection.
        mu_LF_eff = self.masks.mu_LF * self.masks.mu_WFS
        psd_noise = mu_LF_eff * N / (4.0 * xp.pi**2 * f2 * sinc2)
        
        # Set DC to zero (no noise at f=0)
        psd_noise = xp.where(self.grid.F < 1e-10, xp.float64(0.0), psd_noise)
        
        return psd_noise
    
    def compute_total_residual_psd(self, use_combined_aniso_servo: bool = True) -> ArrayLike:
        """
        Compute total residual phase PSD. [Eq. 17]
        
        Sum of all error components.
        
        Args:
            use_combined_aniso_servo: If True, use combined aniso+servo term [Eq. 33]
                                     which properly accounts for their correlation.
                                     If False, add them independently [Eqs. 25 + 31].
        
        Returns:
            Total residual phase PSD
        """
        # Fitting error (HF)
        psd_fit = self.compute_fitting_psd()
        
        # LF errors: either combined or separate aniso + servo
        if use_combined_aniso_servo:
            psd_aniso_servo = self.compute_aniso_servo_psd()
            psd_total = psd_fit + psd_aniso_servo
        else:
            psd_aniso = self.compute_anisoplanatism_psd()
            psd_servo = self.compute_servo_lag_psd()
            psd_total = psd_fit + psd_aniso + psd_servo
        
        # Aliasing and noise
        psd_alias = self.compute_aliasing_psd()
        psd_noise = self.compute_noise_psd()
        
        psd_total = psd_total + psd_alias + psd_noise
        
        return psd_total
    
    def compute_structure_function(self, psd: ArrayLike) -> ArrayLike:
        """
        Compute phase structure function from power spectrum. [Eq. 8]
        
        D_φ(ρ) = 2 ∫∫ [1 - cos(2π f·ρ)] Φ(f) d²f
        
        This is computed efficiently via FFT:
        D_φ(ρ) = 2 * [B_φ(0) - B_φ(ρ)]
        where B_φ = IFT(Φ) is the autocorrelation of the phase.
        
        Args:
            psd: Phase power spectral density
        
        Returns:
            Structure function D_φ(ρ) sampled on pupil grid
        """
        xp = self._xp
        
        # Phase variance = integral of PSD
        # B_φ(0) = ∫∫ Φ(f) d²f
        var_phi = xp.sum(psd) * self.dA
        
        # Autocorrelation B_φ(ρ) = IFT(Φ(f))
        # Need to account for FFT normalization
        B_phi = xp.real(xp.fft.ifft2(psd)) * self.n_pix**2 * self.dA
        
        # Structure function
        D_phi = 2.0 * (var_phi - B_phi)
        
        # Shift to center
        D_phi = xp.fft.fftshift(D_phi)
        
        return D_phi
    
    def compute_ao_otf(self, D_phi: ArrayLike) -> ArrayLike:
        """
        Compute AO-corrected OTF from structure function. [Eq. 7]
        
        OTF_ao(ρ) = exp(-D_φ(ρ) / 2)
        
        Note on coordinates: The structure function D_φ(ρ) is computed on a grid
        of pupil-plane shift coordinates ρ. The OTF is naturally a function of
        focal-plane spatial frequency f, with the relationship ρ = λf (Eq. 3).
        
        In this implementation, D_phi is already computed on the pupil-plane
        grid with spacing pupil_pixel_size = λ/(n_pix * pixel_scale), which
        corresponds to shifts in meters. The resulting OTF is therefore sampled
        at focal-plane frequencies consistent with our pixel_scale.
        
        Args:
            D_phi: Phase structure function D_φ(ρ), centered on the grid
        
        Returns:
            AO-corrected OTF, centered
        """
        xp = self._xp
        
        # OTF from structure function
        OTF_ao = xp.exp(-D_phi / 2.0)
        
        return OTF_ao
    
    def compute_telescope_otf(self, pupil: ArrayLike) -> ArrayLike:
        """
        Compute telescope OTF (diffraction limit + pupil geometry).
        
        OTF_tsc(f) = autocorrelation of pupil function
        
        For aberration-free telescope, this is just the pupil autocorrelation.
        
        Args:
            pupil: 2D pupil amplitude array
        
        Returns:
            Telescope OTF (centered)
        """
        xp = self._xp
        
        pupil = xp.asarray(pupil, dtype=xp.float64)
        
        # OTF = |FT(pupil)|² normalized = autocorrelation(pupil) / pupil_area
        # Equivalently: OTF = IFT(|FT(pupil)|²) / |FT(pupil)|²_max
        Pupil_f = xp.fft.fft2(pupil)
        OTF_tsc = xp.real(xp.fft.ifft2(xp.abs(Pupil_f)**2))
        
        # Normalize
        OTF_tsc = OTF_tsc / xp.max(OTF_tsc)
        
        # Shift to center
        OTF_tsc = xp.fft.fftshift(OTF_tsc)
        
        return OTF_tsc
    
    def compute_long_exposure_psf(
        self,
        pupil: ArrayLike,
        return_components: bool = False,
        include_lwe: bool = True,
    ) -> Union[ArrayLike, Dict[str, ArrayLike]]:
        """
        Compute long-exposure AO-corrected PSF.
        
        This is the main output of the Jolissaint model.
        
        PSF = IFT(OTF_ao × OTF_tsc)
        
        If LWE is configured and include_lwe=True, the PSF is averaged over
        multiple realizations with different LWE phase screens to simulate
        the effect of quasi-static aberrations.
        
        Args:
            pupil: 2D pupil amplitude array
            return_components: If True, return dict with intermediate results
            include_lwe: If True and lwe_config is set, include Low Wind Effect
        
        Returns:
            If return_components=False: PSF array (normalized to sum=1)
            If return_components=True: Dict with 'psf', 'otf_total', 'otf_ao',
                                       'otf_tsc', 'structure_function', 'psd_total'
        """
        xp = self._xp
        
        # Convert pupil to backend array (CPU or GPU)
        pupil = xp.asarray(pupil, dtype=xp.float64)
        
        # Compute residual phase PSD
        psd_total = self.compute_total_residual_psd()
        
        # Structure function
        D_phi = self.compute_structure_function(psd_total)
        
        # AO OTF
        OTF_ao = self.compute_ao_otf(D_phi)
        
        # Telescope OTF
        OTF_tsc = self.compute_telescope_otf(pupil)
        
        # Total system OTF (before LWE)
        OTF_total = OTF_ao * OTF_tsc
        
        # Apply LWE if configured
        if include_lwe and self.lwe_config is not None:
            PSF = self._compute_psf_with_lwe(pupil, OTF_total)
        else:
            # PSF via inverse FFT
            # Need to ifftshift before ifft2 since OTF is centered
            # Use real part (PSF should be real for symmetric OTF) and clamp negatives
            PSF = xp.real(xp.fft.ifft2(xp.fft.ifftshift(OTF_total)))
            
            # Shift to center
            PSF = xp.fft.fftshift(PSF)
            
            # Clamp any numerical negatives (should be negligible if OTF is physical)
            PSF = xp.maximum(PSF, 0.0)
            
            # Normalize to sum = 1 (energy conservation)
            PSF = PSF / xp.sum(PSF)
        
        if return_components:
            return {
                'psf': PSF,
                'otf_total': OTF_total,
                'otf_ao': OTF_ao,
                'otf_tsc': OTF_tsc,
                'structure_function': D_phi,
                'psd_total': psd_total,
            }
        
        return PSF
    
    def _compute_psf_with_lwe(self, pupil: ArrayLike, OTF_base: ArrayLike) -> ArrayLike:
        """
        Compute long-exposure PSF including Low Wind Effect.
        
        The LWE is applied by averaging over multiple realizations:
        1. Generate n_realizations LWE phase screens
        2. For each screen, compute OTF_lwe and multiply with base OTF
        3. Average the resulting PSFs
        
        This simulates the long-exposure effect of quasi-static aberrations
        that vary slowly compared to the AO correction loop but faster than
        the exposure time.
        
        Args:
            pupil: 2D pupil amplitude array
            OTF_base: Base OTF (OTF_ao × OTF_tsc) before LWE
        
        Returns:
            PSF averaged over LWE realizations
        """
        xp = self._xp
        
        # Initialize LWE model if needed
        if self._lwe_model is None:
            self._lwe_model = LowWindEffect(
                pupil=pupil,
                piston_rms_rad=self.lwe_config.piston_rms_rad,
                tilt_rms_rad=self.lwe_config.tilt_rms_rad,
                ar_coeff=self.lwe_config.ar_coeff,
            )
        
        # Generate LWE phase screens
        n_realizations = self.lwe_config.n_realizations
        seed = self.lwe_config.seed
        lwe_phases = self._lwe_model.generate(n_realizations, seed=seed)
        
        # Ensure LWE phases are on correct backend
        lwe_phases = xp.asarray(lwe_phases)
        
        # Accumulate PSFs over realizations
        PSF_avg = xp.zeros((self.n_pix, self.n_pix), dtype=xp.float64)
        
        for i in range(n_realizations):
            lwe_phase = lwe_phases[i]
            
            # Compute LWE OTF degradation
            # The OTF degradation from a phase error is given by:
            #   OTF_lwe(f) = <exp(i*[phi(r) - phi(r+f)])>
            # For uncorrelated phase errors, this is exp(-D_phi/2)
            # where D_phi is the structure function.
            #
            # We compute this as the autocorrelation of exp(i*phi) over the pupil,
            # then divide by the pupil autocorrelation (telescope OTF) to get
            # just the phase degradation factor.
            #
            # OTF_lwe = autocorr(pupil * exp(i*phi)) / autocorr(pupil)
            #         = FT⁻¹(|FT(pupil * exp(i*phi))|²) / FT⁻¹(|FT(pupil)|²)
            
            pupil_with_lwe = pupil * xp.exp(1j * lwe_phase)
            
            # Autocorrelation of aberrated pupil
            Pupil_f_lwe = xp.fft.fft2(xp.fft.ifftshift(pupil_with_lwe))
            autocorr_lwe = xp.fft.fftshift(xp.fft.ifft2(xp.abs(Pupil_f_lwe)**2))
            
            # Autocorrelation of reference pupil (telescope OTF, not normalized)
            Pupil_f_ref = xp.fft.fft2(xp.fft.ifftshift(pupil))
            autocorr_ref = xp.fft.fftshift(xp.fft.ifft2(xp.abs(Pupil_f_ref)**2))
            
            # LWE OTF is the ratio (avoid division by zero)
            # This extracts just the phase degradation, independent of telescope OTF
            epsilon = 1e-10 * xp.max(xp.abs(autocorr_ref))
            OTF_lwe = xp.real(autocorr_lwe) / (xp.real(autocorr_ref) + epsilon)
            
            # Clip to valid range (should be between 0 and 1 for small aberrations)
            OTF_lwe = xp.clip(OTF_lwe, 0.0, 1.0)
            
            # Total OTF for this realization (OTF_base already includes telescope OTF)
            OTF_total = OTF_base * OTF_lwe
            
            # Compute PSF
            PSF_i = xp.real(xp.fft.ifft2(xp.fft.ifftshift(OTF_total)))
            PSF_i = xp.fft.fftshift(PSF_i)
            PSF_i = xp.maximum(PSF_i, 0.0)
            
            # Normalize this realization
            PSF_i = PSF_i / xp.sum(PSF_i)
            
            # Accumulate
            PSF_avg += PSF_i
        
        # Average over realizations
        PSF_avg = PSF_avg / n_realizations
        
        # Final normalization (should already be ~1, but ensure it)
        PSF_avg = PSF_avg / xp.sum(PSF_avg)
        
        return PSF_avg
    
    def compute_strehl_ratio(self, pupil: ArrayLike) -> float:
        """
        Compute Strehl ratio from the model.
        
        Strehl = OTF(0) / OTF_tsc(0) = exp(-σ²_φ)
        
        where σ²_φ is the residual phase variance.
        
        Args:
            pupil: 2D pupil amplitude array
        
        Returns:
            Strehl ratio (0 to 1)
        """
        xp = self._xp
        
        # Residual phase variance = integral of PSD
        psd_total = self.compute_total_residual_psd()
        var_phi = float(xp.sum(psd_total) * self.dA)
        
        # Strehl from Maréchal approximation
        strehl = math.exp(-var_phi)
        
        return min(1.0, max(0.0, strehl))
    
    def compute_psd_terms(self) -> Dict[str, ArrayLike]:
        """
        Compute all PSD error terms without mutating config.
        
        Returns:
            Dict mapping error source names to PSD arrays.
        """
        terms = {}
        terms['fitting'] = self.compute_fitting_psd()
        terms['anisoplanatism'] = self.compute_anisoplanatism_psd()
        terms['servo_lag'] = self.compute_servo_lag_psd()
        terms['aliasing'] = self.compute_aliasing_psd()
        terms['noise'] = self.compute_noise_psd()
        return terms
    
    def get_error_breakdown(self) -> Dict[str, float]:
        """
        Get breakdown of error contributions as variances.
        
        Returns:
            Dict with variance contributions from each error source (rad²).
        """
        xp = self._xp
        
        # Compute all terms (respects include_* flags in each compute method)
        psd_terms = self.compute_psd_terms()
        
        # Integrate each PSD to get variance
        errors = {}
        for name, psd in psd_terms.items():
            errors[name] = float(xp.sum(psd) * self.dA)
        
        # Total variance
        errors['total'] = sum(errors.values())
        
        # Convert to RMS (radians)
        errors['rms_rad'] = math.sqrt(errors['total'])
        
        return errors
    
    # =========================================================================
    # Monte Carlo Phase Sampling (Unified Interface)
    # =========================================================================
    
    def generate_phase_screens(
        self,
        n_screens: int,
        pupil: Optional[ArrayLike] = None,
        seed: Optional[int] = None,
        include_lwe: bool = False,
    ) -> ArrayLike:
        """
        Generate phase screens by Monte Carlo sampling from the analytical PSD.
        
        This provides compatibility with the unified PhaseGeneratorBase interface.
        The method samples from the total residual PSD to produce phase screens
        with the correct spatial correlation structure.
        
        Args:
            n_screens: Number of phase screens to generate
            pupil: Optional pupil mask for piston removal and masking
            seed: Random seed for reproducibility
            include_lwe: If True, add LWE phase to each screen
            
        Returns:
            Phase screens array (n_screens, n_pix, n_pix) in radians
        """
        xp = self._xp
        n_pix = self.n_pix
        
        # Set random seed
        if seed is not None:
            if hasattr(xp.random, 'seed'):
                xp.random.seed(seed)
        
        # Get total residual PSD
        psd_total = self.compute_total_residual_psd()
        
        # Compute amplitude for phase generation
        # A = sqrt(N^4 × PSD × df²) for real output from complex noise
        amplitude = xp.sqrt((n_pix ** 4) * psd_total * self.dA).astype(xp.float64)
        
        # Zero DC to ensure zero mean phase
        amplitude.flat[0] = 0.0
        
        # Generate complex white noise
        noise_real = xp.random.randn(n_screens, n_pix, n_pix).astype(xp.float64)
        noise_imag = xp.random.randn(n_screens, n_pix, n_pix).astype(xp.float64)
        W = noise_real + 1j * noise_imag
        
        # Apply PSD coloring
        Phi_f = W * amplitude[None, :, :]
        
        # Transform to spatial domain (take real part)
        phi = xp.real(xp.fft.ifft2(Phi_f)).astype(xp.float64)
        
        # Remove piston over pupil if provided
        if pupil is not None:
            pupil = xp.asarray(pupil, dtype=xp.float64)
            pupil_sum = xp.sum(pupil)
            if pupil_sum > 0:
                mean_phi = xp.sum(phi * pupil[None, :, :], axis=(1, 2)) / pupil_sum
                phi = (phi - mean_phi[:, None, None]) * pupil[None, :, :]
        
        # Add LWE if requested
        if include_lwe and self.lwe_config is not None:
            # Initialize LWE model if needed
            if self._lwe_model is None and pupil is not None:
                self._lwe_model = LowWindEffect(
                    pupil=pupil,
                    piston_rms_rad=self.lwe_config.piston_rms_rad,
                    tilt_rms_rad=self.lwe_config.tilt_rms_rad,
                    ar_coeff=self.lwe_config.ar_coeff,
                )
            
            if self._lwe_model is not None:
                lwe_seed = self.lwe_config.seed
                if seed is not None:
                    lwe_seed = seed + 1000  # Offset to avoid correlation
                lwe_phases = self._lwe_model.generate(n_screens, seed=lwe_seed)
                phi = phi + lwe_phases
        
        return phi
    
    def generate_phase_screens_with_lwe(
        self,
        n_screens: int,
        pupil: ArrayLike,
        seed: Optional[int] = None,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Generate phase screens with separate AO residual and LWE components.
        
        Args:
            n_screens: Number of phase screens
            pupil: Pupil mask (required for LWE)
            seed: Random seed
            
        Returns:
            Tuple of:
            - phase_ao: (n_screens, n_pix, n_pix) AO residual phase
            - phase_lwe: (n_screens, n_pix, n_pix) LWE phase
            - phase_total: (n_screens, n_pix, n_pix) combined phase
        """
        xp = self._xp
        
        # Generate AO residual phases
        phase_ao = self.generate_phase_screens(
            n_screens, pupil, seed, include_lwe=False
        )
        
        # Generate LWE phases
        if self.lwe_config is None:
            phase_lwe = xp.zeros_like(phase_ao)
        else:
            if self._lwe_model is None:
                self._lwe_model = LowWindEffect(
                    pupil=pupil,
                    piston_rms_rad=self.lwe_config.piston_rms_rad,
                    tilt_rms_rad=self.lwe_config.tilt_rms_rad,
                    ar_coeff=self.lwe_config.ar_coeff,
                )
            lwe_seed = self.lwe_config.seed
            if seed is not None:
                lwe_seed = seed + 1000
            phase_lwe = self._lwe_model.generate(n_screens, seed=lwe_seed)
            # Ensure LWE phases are on same backend
            phase_lwe = xp.asarray(phase_lwe)
        
        # Combine
        phase_total = phase_ao + phase_lwe
        
        return phase_ao, phase_lwe, phase_total
    
    @property
    def rms_expected(self) -> float:
        """Expected RMS of residual phase (radians)."""
        return self.get_error_breakdown()['rms_rad']
    
    @property
    def pixel_size(self) -> float:
        """Pixel size in pupil plane (meters) - for interface compatibility."""
        return self.pupil_pixel_size
    
    # =========================================================================
    # Alternative constructors
    # =========================================================================
    
    @classmethod
    def from_pupil(
        cls,
        pupil: 'Pupil',
        atmosphere: AtmosphereProfile,
        ao_config: AOSystemConfig,
    ) -> 'JolissaintAOModel':
        """
        Create model from a Pupil object.
        
        This is the preferred way to initialize the model as it ensures
        consistent physical parameters between the pupil and the model.
        
        Args:
            pupil: Pupil object with amplitude map and physical parameters
            atmosphere: Atmospheric turbulence profile
            ao_config: AO system configuration
        
        Returns:
            JolissaintAOModel instance
        
        Example:
            from nebraa.physics.pupil import Pupil
            
            # Create a VLT pupil
            pupil = Pupil.from_vlt(
                n_pix=256, wavelength=2.2e-6, pixel_scale=13e-3/206265
            )
            
            # Create model
            model = JolissaintAOModel.from_pupil(pupil, atmosphere, ao_config)
            
            # Compute PSF (pass pupil.amplitude)
            psf = model.compute_long_exposure_psf(pupil.amplitude)
        """
        return cls(
            n_pix=pupil.n_pix,
            telescope_diameter=pupil.diameter,
            obstruction_diameter=pupil.obstruction_diameter,
            wavelength=pupil.wavelength,
            pixel_scale=pupil.pixel_scale,
            atmosphere=atmosphere,
            ao_config=ao_config,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_simple_atmosphere(
    r0: float = 0.15,
    wavelength_ref: float = 0.5e-6,
    L0: float = 25.0,
    wind_speed: float = 10.0,
    wind_direction: float = 0.0,
    altitude: float = 0.0,
) -> AtmosphereProfile:
    """
    Create a simple single-layer atmosphere.
    
    Args:
        r0: Fried parameter at reference wavelength (meters)
        wavelength_ref: Reference wavelength for r0 (meters)
        L0: Outer scale (meters), None for Kolmogorov
        wind_speed: Wind speed (m/s)
        wind_direction: Wind direction (radians)
        altitude: Layer altitude (meters)
    
    Returns:
        AtmosphereProfile with single layer
    """
    layer = TurbulentLayer(
        altitude=altitude,
        r0=r0,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
    )
    return AtmosphereProfile(layers=[layer], wavelength_ref=wavelength_ref, L0=L0)


def create_mauna_kea_atmosphere(
    r0_total: float = 0.15,
    wavelength_ref: float = 0.5e-6,
    L0: float = 25.0,
) -> AtmosphereProfile:
    """
    Create a typical Mauna Kea 7-layer atmosphere profile.
    
    Based on Gemini site characterization data.
    
    Args:
        r0_total: Total Fried parameter at reference wavelength (meters)
        wavelength_ref: Reference wavelength (meters)
        L0: Outer scale (meters)
    
    Returns:
        AtmosphereProfile with 7 layers
    """
    # Typical Mauna Kea profile
    # Heights (m), Cn2 fractions, wind speeds (m/s), directions (deg)
    profile_data = [
        (0, 0.646, 5.6, 0),
        (1800, 0.080, 11.2, 30),
        (3300, 0.119, 5.3, 45),
        (5800, 0.035, 4.9, 60),
        (7400, 0.025, 4.6, 90),
        (13100, 0.080, 17.5, 180),
        (15800, 0.015, 26.2, 270),
    ]
    
    # Convert Cn2 fractions to r0 for each layer
    # r0_i^(-5/3) = fraction_i * r0_total^(-5/3)
    r0_total_inv53 = r0_total ** (-5/3)
    
    layers = []
    for h, frac, v, theta in profile_data:
        r0_layer = (frac * r0_total_inv53) ** (-3/5)
        layers.append(TurbulentLayer(
            altitude=h,
            r0=r0_layer,
            wind_speed=v,
            wind_direction=math.radians(theta),
            Cn2_fraction=frac,
        ))
    
    return AtmosphereProfile(layers=layers, wavelength_ref=wavelength_ref, L0=L0)


def create_ao_config(
    n_actuators: int,
    telescope_diameter: float,
    sampling_frequency: float = 1000.0,
    loop_gain: float = 0.5,
    noise_variance: float = 0.0,
    science_field_offset_arcsec: float = 0.0,
    science_field_direction_deg: float = 0.0,
    wfs_subaperture_size: float = None,
) -> AOSystemConfig:
    """
    Create AO system configuration from common parameters.
    
    Args:
        n_actuators: Number of actuators across pupil diameter
        telescope_diameter: Primary mirror diameter (meters)
        sampling_frequency: WFS frame rate (Hz)
        loop_gain: AO loop gain (affects effective delay)
        noise_variance: WFS slope noise variance (rad²)
        science_field_offset_arcsec: Angular offset of science target from 
            guide star (arcseconds). This is the separation between where 
            the NGS is and where you want to image.
        science_field_direction_deg: Direction of offset (degrees)
        wfs_subaperture_size: WFS subaperture size (meters). If None, 
            defaults to actuator pitch.
    
    Returns:
        AOSystemConfig
    """
    # Actuator pitch
    pitch = telescope_diameter / n_actuators
    
    # Integration time from sampling frequency
    dt = 1.0 / sampling_frequency
    
    # Loop delay (approximately one frame)
    delay = dt
    
    # WFS subaperture defaults to actuator pitch if not specified
    wfs_size = wfs_subaperture_size if wfs_subaperture_size is not None else pitch
    
    return AOSystemConfig(
        actuator_pitch=pitch,
        integration_time=dt,
        loop_delay=delay,
        noise_variance=noise_variance,
        science_field_offset=science_field_offset_arcsec * math.pi / (180 * 3600),
        science_field_direction=math.radians(science_field_direction_deg),
        wfs_subaperture_size=wfs_size,
    )

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

# Type alias for array-like objects (numpy or cupy arrays)
ArrayLike = Any


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
    
    Attributes:
        layers: List of TurbulentLayer objects
        wavelength_ref: Reference wavelength for r0 values (meters)
        L0: Outer scale of turbulence (meters), None for Kolmogorov
        l0: Inner scale of turbulence (meters), default 0
    """
    layers: List[TurbulentLayer]
    wavelength_ref: float = 0.5e-6
    L0: Optional[float] = 25.0
    l0: float = 0.0
    
    @property
    def r0_total(self) -> float:
        """Compute total Fried parameter from all layers."""
        # r0_total^(-5/3) = sum(r0_i^(-5/3))
        r0_inv_53 = sum(layer.r0 ** (-5/3) for layer in self.layers)
        return r0_inv_53 ** (-3/5)
    
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
    
    def __post_init__(self):
        if self.wfs_subaperture_size is None:
            self.wfs_subaperture_size = self.actuator_pitch
    
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

def kolmogorov_psd(f, r0: float):
    """
    Kolmogorov phase power spectral density. [Eq. 19]
    
    Φ(f) = 0.023 * r0^(-5/3) * f^(-11/3)
    
    Args:
        f: Spatial frequency magnitude (cycles/meter)
        r0: Fried parameter (meters)
    
    Returns:
        Power spectral density (rad^2 per (cycles/m)^2)
    """
    backend = get_backend()
    xp = backend.xp
    
    # Avoid division by zero
    f = xp.maximum(xp.asarray(f), xp.float64(1e-12))
    
    return 0.023 * (r0 ** (-5/3)) * (f ** (-11/3))


def von_karman_psd(f, r0: float, L0: float):
    """
    Von Karman phase power spectral density (with outer scale).
    
    Φ(f) = 0.023 * r0^(-5/3) * (f^2 + 1/L0^2)^(-11/6)
    
    Args:
        f: Spatial frequency magnitude (cycles/meter)
        r0: Fried parameter (meters)
        L0: Outer scale (meters)
    
    Returns:
        Power spectral density
    """
    backend = get_backend()
    xp = backend.xp
    
    f = xp.asarray(f)
    return 0.023 * (r0 ** (-5/3)) * ((f**2 + 1.0/L0**2) ** (-11/6))


def piston_filter(f, D: float):
    """
    Piston filter for removing piston mode from phase PS. [Eq. 21]
    
    F_p(f) = 1 - (2*J1(π*D*f)/(π*D*f))^2
    
    Args:
        f: Spatial frequency magnitude (cycles/meter)
        D: Telescope diameter (meters)
    
    Returns:
        Piston filter values
    
    Raises:
        RuntimeError: If required Bessel function library is not available
    """
    backend = get_backend()
    xp = backend.xp
    
    f = xp.asarray(f, dtype=xp.float64)
    
    # Argument for Bessel function
    x = xp.pi * D * f
    
    # Handle x=0 case (filter should be 0 at f=0)
    x = xp.maximum(x, xp.float64(1e-12))
    
    # Compute J1 accurately for the active backend
    j1_vals = None
    
    # CuPy path
    try:
        import cupy  # type: ignore
        if xp is cupy:
            try:
                from cupyx.scipy.special import j1 as cupy_j1  # type: ignore
                j1_vals = cupy_j1(x)
            except Exception as e:
                raise RuntimeError(
                    "cupyx.scipy.special.j1 required for piston_filter on CuPy backend. "
                    "Install with: pip install cupy-cuda11x (or appropriate CUDA version)"
                ) from e
    except ImportError:
        pass
    
    # NumPy path
    if j1_vals is None:
        try:
            from scipy.special import j1 as scipy_j1
        except ImportError as e:
            raise RuntimeError(
                "scipy.special.j1 required for piston_filter. "
                "Install with: pip install scipy"
            ) from e
        x_np = backend.to_numpy(x)
        j1_vals = xp.asarray(scipy_j1(x_np), dtype=xp.float64)
    
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
    
    def __call__(self, f) -> ArrayLike:
        """
        Evaluate piston filter at frequencies f using LUT interpolation.
        
        Args:
            f: Spatial frequency magnitude (cycles/meter)
        
        Returns:
            Piston filter values F_p(f)
        """
        backend = get_backend()
        xp = backend.xp
        
        f = xp.asarray(f, dtype=xp.float64)
        x = xp.pi * self.D * xp.abs(f)
        
        # Clamp to LUT range
        x = xp.clip(x, 0, self.x_max - self.dx)
        
        # Linear interpolation
        idx = (x / self.dx).astype(xp.int64)
        frac = (x / self.dx) - idx
        
        # Convert LUT to backend array if needed
        j1_over_x_lut = xp.asarray(self.j1_over_x_lut)
        
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
        """
        backend = get_backend()
        xp = backend.xp
        
        self.n_pix = n_pix
        self.D = telescope_diameter
        self.D_obs = obstruction_diameter
        self.wavelength = wavelength
        self.pixel_scale = pixel_scale
        self.atmosphere = atmosphere
        self.ao_config = ao_config
        
        # Pupil plane pixel size
        self.pupil_pixel_size = wavelength / (n_pix * pixel_scale)
        
        # Physical pupil extent
        self.pupil_extent = n_pix * self.pupil_pixel_size
        
        # Setup frequency grids
        self._setup_frequency_grid()
        
        # Precompute masks and filters
        self._setup_masks()
    
    def _setup_frequency_grid(self):
        """Setup spatial frequency grid in pupil plane."""
        backend = get_backend()
        xp = backend.xp
        
        n = self.n_pix
        
        # Frequency grid (cycles/meter)
        # Using fftfreq convention for consistency with FFT operations
        df = 1.0 / self.pupil_extent
        fx = xp.fft.fftfreq(n, d=self.pupil_pixel_size).astype(xp.float64)
        fy = xp.fft.fftfreq(n, d=self.pupil_pixel_size).astype(xp.float64)
        self.FX, self.FY = xp.meshgrid(fx, fy)
        
        # Frequency magnitude
        self.F = xp.sqrt(self.FX**2 + self.FY**2)
        
        # Area element for integration
        self.df = df
        self.dA = df**2
    
    def _setup_masks(self):
        """Setup LF/HF masks and filters."""
        backend = get_backend()
        xp = backend.xp
        
        f_ao = self.ao_config.f_ao
        f_wfs = self.ao_config.f_wfs
        
        # Low-frequency mask for DM: |fx|, |fy| < f_ao (square geometry for SH-WFS)
        # [Section 4.C: "the LF domain is defined by |fx|,|fy| < f_ao (a square)"]
        self.mu_LF = ((xp.abs(self.FX) < f_ao) & (xp.abs(self.FY) < f_ao)).astype(xp.float64)
        
        # High-frequency mask (DM)
        self.mu_HF = 1.0 - self.mu_LF
        
        # WFS domain mask: |fx|, |fy| < f_wfs (for noise PSD, per paper after Eq. 50)
        self.mu_WFS = ((xp.abs(self.FX) < f_wfs) & (xp.abs(self.FY) < f_wfs)).astype(xp.float64)
        
        # Piston filter (for fitting PSD etc.)
        self.Fp = piston_filter(self.F, self.D)
        
        # Piston filter LUT for fast aliasing computation
        # Estimate max frequency needed: f_max ~ n_alias / Lambda + max(|F|)
        f_max_grid = float(backend.to_numpy(xp.max(self.F)))
        f_max_lut = f_max_grid + 5.0 / self.ao_config.wfs_subaperture_size
        self._piston_lut = PistonFilterLUT(self.D, f_max=f_max_lut, n_points=20000)
    
    def _turbulent_psd_layer(self, layer: TurbulentLayer) -> ArrayLike:
        """
        Compute turbulent phase PSD for a single layer at science wavelength.
        
        Uses Kolmogorov or Von Karman model depending on outer scale.
        """
        backend = get_backend()
        xp = backend.xp
        
        # Scale r0 to science wavelength
        r0_sci = layer.r0 * (self.wavelength / self.atmosphere.wavelength_ref) ** (6/5)
        
        if self.atmosphere.L0 is not None:
            psd = von_karman_psd(self.F, r0_sci, self.atmosphere.L0)
        else:
            psd = kolmogorov_psd(self.F, r0_sci)
        
        return psd
    
    def compute_fitting_psd(self) -> ArrayLike:
        """
        Compute fitting error PSD. [Eq. 22]
        
        High-frequency turbulence that cannot be corrected by the AO system.
        
        Φ_fit(f) = μ_HF(f) * F_p(f) * Φ_turb(f)
        """
        backend = get_backend()
        xp = backend.xp
        
        if not self.ao_config.include_fitting:
            return xp.zeros_like(self.F)
        
        # Sum over all layers
        psd_fit = xp.zeros_like(self.F)
        for layer in self.atmosphere.layers:
            psd_layer = self._turbulent_psd_layer(layer)
            psd_fit = psd_fit + self.mu_HF * self.Fp * psd_layer
        
        return psd_fit
    
    def compute_anisoplanatism_psd(self) -> ArrayLike:
        """
        Compute anisoplanatism error PSD. [Eq. 25]
        
        Error due to science target at angle θ from the guide star,
        seeing different turbulence than what the AO corrects for.
        
        Φ_aniso(f,θ) = 2 * μ_LF(f) * F_p(f) * Σ_n Φ_n(f) * [1 - cos(2πh_n f·θ)]
        
        where θ is the angular separation between science target and NGS.
        """
        backend = get_backend()
        xp = backend.xp
        
        if not self.ao_config.include_anisoplanatism or self.ao_config.science_field_offset == 0:
            return xp.zeros_like(self.F)
        
        theta_x, theta_y = self.ao_config.field_offset
        
        psd_aniso = xp.zeros_like(self.F)
        for layer in self.atmosphere.layers:
            psd_layer = self._turbulent_psd_layer(layer)
            h = layer.altitude
            
            # Phase term: 2πh(f·θ) = 2π*h*(fx*θx + fy*θy)
            phase_term = 2.0 * xp.pi * h * (self.FX * theta_x + self.FY * theta_y)
            
            # 1 - cos(phase) factor gives the decorrelation
            decorr = 1.0 - xp.cos(phase_term)
            
            psd_aniso = psd_aniso + 2.0 * self.mu_LF * self.Fp * psd_layer * decorr
        
        return psd_aniso
    
    def compute_servo_lag_psd(self) -> ArrayLike:
        """
        Compute servo-lag error PSD. [Eq. 31]
        
        Error due to time delay between WFS measurement and DM correction.
        
        Φ_servo(f) = μ_LF(f) * F_p(f) * Σ_n Φ_n(f) * 
                     {1 - 2*cos(2π*t_l*f·v_n)*sinc(Δt*f·v_n) + sinc²(Δt*f·v_n)}
        """
        backend = get_backend()
        xp = backend.xp
        
        if not self.ao_config.include_servo_lag:
            return xp.zeros_like(self.F)
        
        dt = self.ao_config.integration_time
        tl = self.ao_config.total_delay
        
        psd_servo = xp.zeros_like(self.F)
        for layer in self.atmosphere.layers:
            psd_layer = self._turbulent_psd_layer(layer)
            vx, vy = layer.wind_velocity
            
            # f·v = fx*vx + fy*vy
            f_dot_v = self.FX * vx + self.FY * vy
            
            # Arguments for sinc and cos
            sinc_arg = dt * f_dot_v  # sinc(Δt * f·v)
            cos_arg = 2.0 * xp.pi * tl * f_dot_v  # 2π * t_l * f·v
            
            # sinc function: sinc(x) = sin(πx)/(πx)
            # Handle x=0 case
            sinc_val = xp.sinc(sinc_arg)  # numpy's sinc includes the π factor
            
            # Error factor: {1 - 2*cos(...)*sinc(...) + sinc²(...)}
            error_factor = 1.0 - 2.0 * xp.cos(cos_arg) * sinc_val + sinc_val**2
            
            psd_servo = psd_servo + self.mu_LF * self.Fp * psd_layer * error_factor
        
        return psd_servo
    
    def compute_aniso_servo_psd(self) -> ArrayLike:
        """
        Compute combined anisoplanatism + servo-lag PSD. [Eq. 33]
        
        This is the proper way to combine these correlated errors,
        rather than adding them independently.
        
        Φ_θ,s(f) = μ_LF(f) * F_p(f) * Σ_n Φ_n(f) * 
                   {1 - 2*cos(2π*f·[h_n*θ - t_l*v_n])*sinc(Δt*f·v_n) + sinc²(Δt*f·v_n)}
        
        where θ is the science field angle relative to NGS.
        """
        backend = get_backend()
        xp = backend.xp
        
        # If both are disabled, return zero
        if not self.ao_config.include_anisoplanatism and not self.ao_config.include_servo_lag:
            return xp.zeros_like(self.F)
        
        dt = self.ao_config.integration_time
        tl = self.ao_config.total_delay
        theta_x, theta_y = self.ao_config.field_offset
        
        psd_as = xp.zeros_like(self.F)
        for layer in self.atmosphere.layers:
            psd_layer = self._turbulent_psd_layer(layer)
            h = layer.altitude
            vx, vy = layer.wind_velocity
            
            # Combined offset: h*θ - t_l*v
            offset_x = h * theta_x - tl * vx
            offset_y = h * theta_y - tl * vy
            
            # f·offset = fx*offset_x + fy*offset_y
            f_dot_offset = self.FX * offset_x + self.FY * offset_y
            
            # f·v for sinc
            f_dot_v = self.FX * vx + self.FY * vy
            sinc_arg = dt * f_dot_v
            sinc_val = xp.sinc(sinc_arg)
            
            # cos argument
            cos_arg = 2.0 * xp.pi * f_dot_offset
            
            # Error factor
            error_factor = 1.0 - 2.0 * xp.cos(cos_arg) * sinc_val + sinc_val**2
            
            psd_as = psd_as + self.mu_LF * self.Fp * psd_layer * error_factor
        
        return psd_as
    
    def compute_aliasing_psd(self) -> ArrayLike:
        """
        Compute WFS aliasing error PSD. [Eqs. 44-48]
        
        High-frequency turbulence aliased into low frequencies by the WFS.
        
        Full implementation of Eq. (45):
        
        Φ_alias(f) = μ_WFS(f) * (fx²*fy²/f⁴) * Σ_n sinc²(Δt*f·v_n) *
                     |Σ_{k,l≠0} [fx/(fy-l/Λ) + fy/(fx-k/Λ)] * (-1)^(k+l) *
                      √[F_p(f-k/Λ,f-l/Λ) * Φ_n(f-k/Λ,f-l/Λ)]|²
        
        With special handling for singularities at fx=0 [Eq. 47], fy=0 [Eq. 46],
        and (fx,fy)=(0,0) [Eq. 48].
        
        Note: Aliasing is a WFS phenomenon, so domains are defined by the WFS
        Nyquist frequency f_WFS = 1/(2Λ), not the DM cutoff f_AO.
        """
        backend = get_backend()
        xp = backend.xp
        
        if not self.ao_config.include_aliasing:
            return xp.zeros_like(self.F)
        
        # Check if we're on GPU - use vectorized path for better performance
        is_gpu = backend.gpu
        
        if is_gpu:
            return self._compute_aliasing_psd_vectorized()
        else:
            return self._compute_aliasing_psd_loop()
    
    def _compute_aliasing_psd_vectorized(self) -> ArrayLike:
        """
        Vectorized GPU-optimized aliasing PSD computation.
        
        Stacks all (k,l) terms into a 3D tensor for parallel processing.
        """
        backend = get_backend()
        xp = backend.xp
        
        Lambda = self.ao_config.wfs_subaperture_size
        f_wfs = self.ao_config.f_wfs
        dt = self.ao_config.integration_time
        n_alias = 3
        
        # Build list of (k, l) pairs excluding (0, 0)
        kl_pairs = [(k, l) for k in range(-n_alias, n_alias + 1)
                    for l in range(-n_alias, n_alias + 1) if not (k == 0 and l == 0)]
        n_terms = len(kl_pairs)
        
        # Create arrays for k and l values: shape (n_terms, 1, 1)
        k_arr = xp.array([kl[0] for kl in kl_pairs], dtype=xp.float64)[:, None, None]
        l_arr = xp.array([kl[1] for kl in kl_pairs], dtype=xp.float64)[:, None, None]
        
        # Sign factors: (-1)^(k+l), shape (n_terms, 1, 1)
        sign_arr = xp.array([(-1) ** (kl[0] + kl[1]) for kl in kl_pairs], dtype=xp.float64)[:, None, None]
        
        psd_alias = xp.zeros_like(self.F, dtype=xp.float64)
        
        for layer in self.atmosphere.layers:
            r0_sci = layer.r0 * (self.wavelength / self.atmosphere.wavelength_ref) ** (6/5)
            vx, vy = layer.wind_velocity
            
            # Temporal averaging factor: sinc²(Δt * f·v)
            f_dot_v = self.FX * vx + self.FY * vy
            sinc2_temporal = xp.sinc(dt * f_dot_v) ** 2
            
            # Vectorized computation: broadcast FX, FY to (n_terms, n_pix, n_pix)
            FX_3d = self.FX[None, :, :]  # (1, n_pix, n_pix)
            FY_3d = self.FY[None, :, :]
            
            # Aliased frequencies: (n_terms, n_pix, n_pix)
            fx_alias = FX_3d - k_arr / Lambda
            fy_alias = FY_3d - l_arr / Lambda
            f_alias = xp.sqrt(fx_alias**2 + fy_alias**2)
            
            # HF mask: outside WFS Nyquist
            is_hf = (xp.abs(fx_alias) >= f_wfs) | (xp.abs(fy_alias) >= f_wfs)
            
            # Von Karman PSD at aliased frequencies
            L0_inv2 = 1.0 / self.atmosphere.L0**2 if self.atmosphere.L0 else 0
            coeff = 0.023 * (r0_sci ** (-5/3))
            if self.atmosphere.L0 is not None:
                psd_alias_f = coeff * xp.power(f_alias**2 + L0_inv2, -11/6)
            else:
                f_alias_safe = xp.maximum(f_alias, 1e-12)
                psd_alias_f = coeff * xp.power(f_alias_safe, -11/3)
            
            # Piston filter using LUT (GPU-compatible)
            Fp_alias = self._piston_lut(f_alias)
            
            # Amplitude: sqrt(F_p * Φ)
            amplitude = xp.sqrt(xp.maximum(Fp_alias * psd_alias_f, 0.0))
            
            # Geometry factor with safe division
            eps = 1e-12
            safe_denom_x = xp.where(xp.abs(fy_alias) < eps, 
                                     eps * xp.sign(fy_alias + eps), fy_alias)
            safe_denom_y = xp.where(xp.abs(fx_alias) < eps,
                                     eps * xp.sign(fx_alias + eps), fx_alias)
            
            term_x = FX_3d / safe_denom_x  # fx / (fy - l/Λ)
            term_y = FY_3d / safe_denom_y  # fy / (fx - k/Λ)
            geom_factor = term_x + term_y
            
            # Weighted contribution per (k,l): shape (n_terms, n_pix, n_pix)
            contrib = is_hf.astype(xp.float64) * sign_arr * geom_factor * amplitude
            
            # Sum over all (k,l) terms
            complex_sum = xp.sum(contrib, axis=0)  # (n_pix, n_pix)
            
            # Prefactor: fx²*fy²/f⁴
            f4 = xp.maximum(self.F**4, 1e-40)
            prefactor = (self.FX**2 * self.FY**2) / f4
            
            # Main PSD contribution
            psd_layer = prefactor * sinc2_temporal * complex_sum**2
            
            # Handle axis singularities (simplified for GPU)
            fx_zero = xp.abs(self.FX) < 1e-10
            fy_zero = xp.abs(self.FY) < 1e-10
            
            # For fx=0: sum over l≠0
            psd_fx0 = xp.zeros_like(self.F)
            for l in range(-n_alias, n_alias + 1):
                if l == 0:
                    continue
                fy_al = self.FY - l / Lambda
                f_al = xp.abs(fy_al)
                is_hf_l = xp.abs(fy_al) >= f_wfs
                if self.atmosphere.L0 is not None:
                    psd_l = coeff * xp.power(f_al**2 + L0_inv2, -11/6)
                else:
                    psd_l = coeff * xp.power(xp.maximum(f_al, 1e-12), -11/3)
                Fp_l = self._piston_lut(f_al)
                psd_fx0 = psd_fx0 + is_hf_l.astype(xp.float64) * Fp_l * psd_l * sinc2_temporal
            
            # For fy=0: sum over k≠0
            psd_fy0 = xp.zeros_like(self.F)
            for k in range(-n_alias, n_alias + 1):
                if k == 0:
                    continue
                fx_al = self.FX - k / Lambda
                f_al = xp.abs(fx_al)
                is_hf_k = xp.abs(fx_al) >= f_wfs
                if self.atmosphere.L0 is not None:
                    psd_k = coeff * xp.power(f_al**2 + L0_inv2, -11/6)
                else:
                    psd_k = coeff * xp.power(xp.maximum(f_al, 1e-12), -11/3)
                Fp_k = self._piston_lut(f_al)
                psd_fy0 = psd_fy0 + is_hf_k.astype(xp.float64) * Fp_k * psd_k * sinc2_temporal
            
            # (0,0) case
            psd_00 = 0.0
            for k in range(-n_alias, n_alias + 1):
                for l in range(-n_alias, n_alias + 1):
                    if k == 0 or l == 0:
                        continue
                    f_00 = math.sqrt((k/Lambda)**2 + (l/Lambda)**2)
                    if self.atmosphere.L0 is not None:
                        psd_kl = coeff * (f_00**2 + L0_inv2) ** (-11/6)
                    else:
                        psd_kl = coeff * f_00 ** (-11/3)
                    Fp_kl = float(self._piston_lut(xp.array([f_00]))[0])
                    psd_00 += Fp_kl * psd_kl
            
            # Combine
            both_zero = fx_zero & fy_zero
            psd_layer = xp.where(both_zero, psd_00, psd_layer)
            psd_layer = xp.where(fx_zero & ~both_zero, psd_fx0, psd_layer)
            psd_layer = xp.where(fy_zero & ~both_zero, psd_fy0, psd_layer)
            
            psd_alias = psd_alias + self.mu_WFS * psd_layer
        
        return psd_alias
    
    def _compute_aliasing_psd_loop(self) -> ArrayLike:
        """
        Loop-based aliasing PSD computation (CPU-optimized with LUT).
        """
        backend = get_backend()
        xp = backend.xp
        
        Lambda = self.ao_config.wfs_subaperture_size
        f_wfs = self.ao_config.f_wfs
        dt = self.ao_config.integration_time
        
        psd_alias = xp.zeros_like(self.F, dtype=xp.float64)
        n_alias = 3
        
        for layer in self.atmosphere.layers:
            r0_sci = layer.r0 * (self.wavelength / self.atmosphere.wavelength_ref) ** (6/5)
            vx, vy = layer.wind_velocity
            
            # Temporal averaging factor: sinc²(Δt * f·v)
            f_dot_v = self.FX * vx + self.FY * vy
            sinc2_temporal = xp.sinc(dt * f_dot_v) ** 2
            
            # Build the complex sum over (k,l) for Eq. (44)
            # The term inside the absolute value squared
            complex_sum = xp.zeros_like(self.F, dtype=xp.complex128)
            
            for k in range(-n_alias, n_alias + 1):
                for l in range(-n_alias, n_alias + 1):
                    if k == 0 and l == 0:
                        continue
                    
                    # Aliased frequency components
                    fx_alias = self.FX - k / Lambda
                    fy_alias = self.FY - l / Lambda
                    f_alias = xp.sqrt(fx_alias**2 + fy_alias**2)
                    
                    # Only spatial frequencies outside the WFS Nyquist square alias into LF
                    is_hf = (xp.abs(fx_alias) >= f_wfs) | (xp.abs(fy_alias) >= f_wfs)
                    
                    # Turbulent PSD at aliased frequency
                    if self.atmosphere.L0 is not None:
                        psd_alias_f = von_karman_psd(f_alias, r0_sci, self.atmosphere.L0)
                    else:
                        psd_alias_f = kolmogorov_psd(f_alias, r0_sci)
                    
                    # Piston filter at aliased frequency (use LUT for speed)
                    Fp_alias = self._piston_lut(f_alias)
                    
                    # Amplitude: sqrt(F_p * Φ)
                    amplitude = xp.sqrt(xp.maximum(Fp_alias * psd_alias_f, 0.0))
                    
                    # Sign factor: (-1)^(k+l)
                    sign = (-1) ** (k + l)
                    
                    # Geometry factor: fx/(fy - l/Λ) + fy/(fx - k/Λ)
                    # Robust division based on actual denominator magnitude, not k/l values
                    eps = 1e-12
                    
                    denom_x = fy_alias  # denominator for fx/(...) term
                    denom_y = fx_alias  # denominator for fy/(...) term
                    safe_denom_x = xp.where(xp.abs(denom_x) < eps, eps * xp.sign(denom_x + eps), denom_x)
                    safe_denom_y = xp.where(xp.abs(denom_y) < eps, eps * xp.sign(denom_y + eps), denom_y)
                    
                    term_x = self.FX / safe_denom_x  # fx / (fy - l/Λ)
                    term_y = self.FY / safe_denom_y  # fy / (fx - k/Λ)
                    
                    geom_factor = term_x + term_y
                    
                    # Add to complex sum (only where HF)
                    complex_sum = complex_sum + is_hf.astype(xp.float64) * sign * geom_factor * amplitude
            
            # Now compute the PSD from the squared magnitude of the sum
            # Eq. (45): Φ_alias = μ_LF * (fx²fy²/f⁴) * sinc² * |sum|²
            
            # Prefactor: fx²*fy²/f⁴
            # Handle singularities at axes
            f4 = self.F**4
            f4 = xp.maximum(f4, xp.float64(1e-40))
            prefactor = (self.FX**2 * self.FY**2) / f4
            
            # Main contribution from Eq. (45)
            psd_layer = prefactor * sinc2_temporal * xp.abs(complex_sum)**2
            
            # Handle singularities using Eqs. (46-48)
            # These are limit cases that need special treatment
            
            # Eq. (46): fx = 0 case (only l ≠ 0 terms contribute)
            # Φ_alias(0, fy) = μ_WFS * Σ_l≠0 F_p(0, fy-l/Λ) * Φ(0, fy-l/Λ)
            fx_zero = xp.abs(self.FX) < 1e-10
            psd_fx0 = xp.zeros_like(self.F)
            for l in range(-n_alias, n_alias + 1):
                if l == 0:
                    continue
                fy_alias = self.FY - l / Lambda
                f_alias = xp.abs(fy_alias)
                is_hf_l = xp.abs(fy_alias) >= f_wfs
                if self.atmosphere.L0 is not None:
                    psd_l = von_karman_psd(f_alias, r0_sci, self.atmosphere.L0)
                else:
                    psd_l = kolmogorov_psd(f_alias, r0_sci)
                Fp_l = self._piston_lut(f_alias)
                psd_fx0 = psd_fx0 + is_hf_l.astype(xp.float64) * Fp_l * psd_l * sinc2_temporal
            
            # Eq. (47): fy = 0 case (only k ≠ 0 terms contribute)
            fy_zero = xp.abs(self.FY) < 1e-10
            psd_fy0 = xp.zeros_like(self.F)
            for k in range(-n_alias, n_alias + 1):
                if k == 0:
                    continue
                fx_alias = self.FX - k / Lambda
                f_alias = xp.abs(fx_alias)
                is_hf_k = xp.abs(fx_alias) >= f_wfs
                if self.atmosphere.L0 is not None:
                    psd_k = von_karman_psd(f_alias, r0_sci, self.atmosphere.L0)
                else:
                    psd_k = kolmogorov_psd(f_alias, r0_sci)
                Fp_k = self._piston_lut(f_alias)
                psd_fy0 = psd_fy0 + is_hf_k.astype(xp.float64) * Fp_k * psd_k * sinc2_temporal
            
            # Eq. (48): (fx, fy) = (0, 0) case
            # Φ_alias(0,0) = μ_LF(0,0) * (fxfy/f⁴) * Σ_{k≠0} Σ_{l≠0} F_p(-k/Λ,-l/Λ) * Φ(-k/Λ,-l/Λ)
            # This simplifies since fx=fy=0
            psd_00 = xp.float64(0.0)
            for k in range(-n_alias, n_alias + 1):
                for l in range(-n_alias, n_alias + 1):
                    if k == 0 or l == 0:
                        continue
                    f_alias_00 = math.sqrt((k/Lambda)**2 + (l/Lambda)**2)
                    if self.atmosphere.L0 is not None:
                        psd_kl = float(von_karman_psd(xp.array([f_alias_00]), r0_sci, self.atmosphere.L0)[0])
                    else:
                        psd_kl = float(kolmogorov_psd(xp.array([f_alias_00]), r0_sci)[0])
                    Fp_kl = float(self._piston_lut(xp.array([f_alias_00]))[0])
                    psd_00 = psd_00 + Fp_kl * psd_kl
            
            # Combine: use special cases where applicable
            both_zero = fx_zero & fy_zero
            psd_layer = xp.where(both_zero, psd_00, psd_layer)
            psd_layer = xp.where(fx_zero & ~both_zero, psd_fx0, psd_layer)
            psd_layer = xp.where(fy_zero & ~both_zero, psd_fy0, psd_layer)
            
            # Aliasing PSD is defined over the WFS LF square, not the DM LF square
            psd_alias = psd_alias + self.mu_WFS * psd_layer
        
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
        backend = get_backend()
        xp = backend.xp
        
        if not self.ao_config.include_noise or self.ao_config.noise_variance == 0:
            return xp.zeros_like(self.F)
        
        Lambda = self.ao_config.wfs_subaperture_size
        sigma_n2 = self.ao_config.noise_variance
        
        # Noise PSD is flat within WFS domain (|fx|,|fy| < f_WFS)
        # N * Λ² = σ_n² [Eq. 51]
        N = sigma_n2 / Lambda**2
        
        # sinc² terms with subaperture size
        sinc_x = xp.sinc(Lambda * self.FX)
        sinc_y = xp.sinc(Lambda * self.FY)
        sinc2 = sinc_x**2 * sinc_y**2
        
        # Avoid division by zero
        f2 = self.F**2
        f2 = xp.maximum(f2, xp.float64(1e-20))
        sinc2 = xp.maximum(sinc2, xp.float64(1e-20))
        
        # Use WFS domain mask (mu_WFS), not DM domain (mu_LF)
        # Per paper: "The WFS noise [...] is a random quantity, with a white
        # spectrum bounded to the LF domain |fx|,|fy| < f_WFS"
        psd_noise = self.mu_WFS * N / (4.0 * xp.pi**2 * f2 * sinc2)
        
        # Set DC to zero (no noise at f=0)
        psd_noise = xp.where(self.F < 1e-10, xp.float64(0.0), psd_noise)
        
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
        backend = get_backend()
        xp = backend.xp
        
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
        backend = get_backend()
        xp = backend.xp
        
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
        backend = get_backend()
        xp = backend.xp
        
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
        backend = get_backend()
        xp = backend.xp
        
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
        return_components: bool = False
    ) -> Union[ArrayLike, Dict[str, ArrayLike]]:
        """
        Compute long-exposure AO-corrected PSF.
        
        This is the main output of the Jolissaint model.
        
        PSF = IFT(OTF_ao × OTF_tsc)
        
        Args:
            pupil: 2D pupil amplitude array
            return_components: If True, return dict with intermediate results
        
        Returns:
            If return_components=False: PSF array (normalized to sum=1)
            If return_components=True: Dict with 'psf', 'otf_total', 'otf_ao',
                                       'otf_tsc', 'structure_function', 'psd_total'
        """
        backend = get_backend()
        xp = backend.xp
        
        # Compute residual phase PSD
        psd_total = self.compute_total_residual_psd()
        
        # Structure function
        D_phi = self.compute_structure_function(psd_total)
        
        # AO OTF
        OTF_ao = self.compute_ao_otf(D_phi)
        
        # Telescope OTF
        OTF_tsc = self.compute_telescope_otf(pupil)
        
        # Total system OTF
        OTF_total = OTF_ao * OTF_tsc
        
        # PSF via inverse FFT
        # Need to ifftshift before ifft2 since OTF is centered
        PSF = xp.abs(xp.fft.ifft2(xp.fft.ifftshift(OTF_total)))
        
        # Shift to center
        PSF = xp.fft.fftshift(PSF)
        
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
        backend = get_backend()
        xp = backend.xp
        
        # Residual phase variance = integral of PSD
        psd_total = self.compute_total_residual_psd()
        var_phi = float(xp.sum(psd_total) * self.dA)
        
        # Strehl from Maréchal approximation
        strehl = math.exp(-var_phi)
        
        return min(1.0, max(0.0, strehl))
    
    def get_error_breakdown(self) -> Dict[str, float]:
        """
        Get breakdown of error contributions.
        
        Returns:
            Dict with variance contributions from each error source.
        """
        backend = get_backend()
        xp = backend.xp
        
        # Store current settings
        orig_settings = {
            'fitting': self.ao_config.include_fitting,
            'aniso': self.ao_config.include_anisoplanatism,
            'servo': self.ao_config.include_servo_lag,
            'alias': self.ao_config.include_aliasing,
            'noise': self.ao_config.include_noise,
        }
        
        errors = {}
        
        # Fitting error
        self.ao_config.include_fitting = True
        self.ao_config.include_anisoplanatism = False
        self.ao_config.include_servo_lag = False
        self.ao_config.include_aliasing = False
        self.ao_config.include_noise = False
        psd = self.compute_fitting_psd()
        errors['fitting'] = float(xp.sum(psd) * self.dA)
        
        # Anisoplanatism
        self.ao_config.include_fitting = False
        self.ao_config.include_anisoplanatism = True
        psd = self.compute_anisoplanatism_psd()
        errors['anisoplanatism'] = float(xp.sum(psd) * self.dA)
        
        # Servo-lag
        self.ao_config.include_anisoplanatism = False
        self.ao_config.include_servo_lag = True
        psd = self.compute_servo_lag_psd()
        errors['servo_lag'] = float(xp.sum(psd) * self.dA)
        
        # Aliasing
        self.ao_config.include_servo_lag = False
        self.ao_config.include_aliasing = True
        psd = self.compute_aliasing_psd()
        errors['aliasing'] = float(xp.sum(psd) * self.dA)
        
        # Noise
        self.ao_config.include_aliasing = False
        self.ao_config.include_noise = True
        psd = self.compute_noise_psd()
        errors['noise'] = float(xp.sum(psd) * self.dA)
        
        # Restore settings
        self.ao_config.include_fitting = orig_settings['fitting']
        self.ao_config.include_anisoplanatism = orig_settings['aniso']
        self.ao_config.include_servo_lag = orig_settings['servo']
        self.ao_config.include_aliasing = orig_settings['alias']
        self.ao_config.include_noise = orig_settings['noise']
        
        # Total variance
        errors['total'] = sum(errors.values())
        
        # Convert to RMS (radians)
        errors['rms_rad'] = math.sqrt(errors['total'])
        
        return errors


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

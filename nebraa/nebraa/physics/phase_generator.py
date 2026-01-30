"""
Phase Generator Base Classes

Provides a unified interface for generating residual phase screens from different
physical models (Zernike, Power-Law PSD, Jolissaint AO).

All generators share:
- Common grid parameters (n_pix, pixel_size in meters)
- Optional LWE integration
- Standardized output: (n_screens, n_pix, n_pix) phase arrays in radians

Author: NEBRAA
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass

from ..utils.compute import get_backend
from .low_wind_effect import LowWindEffect, LowWindEffectConfig

# Type alias for array-like objects (numpy or cupy arrays)
ArrayLike = Any


# =============================================================================
# Common Configuration
# =============================================================================

@dataclass
class PhaseGeneratorConfig:
    """
    Base configuration shared by all phase generators.
    
    Attributes:
        n_pix: Grid size in pixels
        pixel_size: Physical size of each pixel in pupil plane (meters)
        seed: Random seed for reproducibility. If None, random each time.
        lwe_config: Optional Low Wind Effect configuration
    """
    n_pix: int
    pixel_size: float  # meters
    seed: Optional[int] = None
    lwe_config: Optional[LowWindEffectConfig] = None
    
    @classmethod
    def from_telescope(
        cls,
        n_pix: int,
        telescope_diameter: float,
        obstruction_ratio: float = 0.0,
        seed: Optional[int] = None,
        lwe_config: Optional[LowWindEffectConfig] = None,
    ) -> "PhaseGeneratorConfig":
        """
        Create config from telescope parameters.
        
        The pixel_size is chosen so the telescope diameter fits in n_pix pixels.
        
        Args:
            n_pix: Grid size in pixels
            telescope_diameter: Primary mirror diameter (meters)
            obstruction_ratio: Central obstruction as fraction of diameter
            seed: Random seed
            lwe_config: Optional LWE configuration
            
        Returns:
            PhaseGeneratorConfig
        """
        # Pixel size such that telescope fits in grid with some margin
        pixel_size = telescope_diameter / n_pix
        
        return cls(
            n_pix=n_pix,
            pixel_size=pixel_size,
            seed=seed,
            lwe_config=lwe_config,
        )
    
    @classmethod
    def from_fourier_sampling(
        cls,
        n_pix: int,
        wavelength: float,
        pixel_scale_rad: float,
        seed: Optional[int] = None,
        lwe_config: Optional[LowWindEffectConfig] = None,
    ) -> "PhaseGeneratorConfig":
        """
        Create config from Fourier sampling requirements.
        
        This ensures proper Nyquist sampling for PSF computation:
        pupil_pixel_size = wavelength / (n_pix × pixel_scale)
        
        Args:
            n_pix: Grid size in pixels
            wavelength: Observation wavelength (meters)
            pixel_scale_rad: Desired PSF pixel scale (radians/pixel)
            seed: Random seed
            lwe_config: Optional LWE configuration
            
        Returns:
            PhaseGeneratorConfig
        """
        pixel_size = wavelength / (n_pix * pixel_scale_rad)
        
        return cls(
            n_pix=n_pix,
            pixel_size=pixel_size,
            seed=seed,
            lwe_config=lwe_config,
        )
    
    @property
    def extent(self) -> float:
        """Total physical extent of the grid (meters)."""
        return self.n_pix * self.pixel_size
    
    def telescope_fits(self, telescope_diameter: float, margin: float = 1.1) -> bool:
        """Check if telescope diameter fits in grid with given margin."""
        return telescope_diameter * margin <= self.extent


# =============================================================================
# Abstract Base Class
# =============================================================================

class PhaseGeneratorBase(ABC):
    """
    Abstract base class for phase screen generators.
    
    All implementations must provide:
    - generate(n_screens, pupil, seed) → phase screens
    - rms_expected property (expected RMS from model parameters)
    
    Optional:
    - LWE integration via add_lwe() or automatic combination
    """
    
    def __init__(
        self,
        n_pix: int,
        pixel_size: float,
        seed: Optional[int] = None,
        lwe_config: Optional[LowWindEffectConfig] = None,
    ):
        """
        Initialize base phase generator.
        
        Args:
            n_pix: Grid size in pixels
            pixel_size: Physical pixel size in pupil plane (meters)
            seed: Random seed for reproducibility
            lwe_config: Optional LWE configuration for combined generation
        """
        self.n_pix = n_pix
        self.pixel_size = pixel_size
        self.seed = seed
        self.lwe_config = lwe_config
        
        # Backend for GPU/CPU operations
        self._backend = get_backend()
        self._xp = self._backend.xp
        
        # LWE model (lazy initialization)
        self._lwe_model: Optional[LowWindEffect] = None
        self._lwe_pupil_hash: Optional[int] = None
    
    @classmethod
    def from_config(cls, config: PhaseGeneratorConfig, **kwargs) -> "PhaseGeneratorBase":
        """
        Create generator from PhaseGeneratorConfig.
        
        Subclasses should override this to accept their specific config.
        """
        return cls(
            n_pix=config.n_pix,
            pixel_size=config.pixel_size,
            seed=config.seed,
            lwe_config=config.lwe_config,
            **kwargs,
        )
    
    @abstractmethod
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
            pupil: Optional pupil mask for piston removal / masking
            seed: Random seed (overrides instance seed if provided)
            
        Returns:
            Phase screens array of shape (n_screens, n_pix, n_pix) in radians
        """
        pass
    
    @property
    @abstractmethod
    def rms_expected(self) -> float:
        """Expected RMS of generated phase screens (radians)."""
        pass
    
    @property
    def extent(self) -> float:
        """Total physical extent of grid (meters)."""
        return self.n_pix * self.pixel_size
    
    @property
    def df(self) -> float:
        """Spatial frequency resolution (cycles/meter)."""
        return 1.0 / self.extent
    
    def _get_lwe_model(self, pupil: ArrayLike) -> LowWindEffect:
        """
        Get or create LWE model for the given pupil.
        
        Caches the model to avoid re-detecting islands.
        """
        if self.lwe_config is None:
            raise ValueError("No LWE config provided")
        
        # Check if pupil changed (simple hash based on shape and sum)
        xp = self._xp
        pupil_hash = hash((pupil.shape, float(xp.sum(pupil))))
        
        if self._lwe_model is None or self._lwe_pupil_hash != pupil_hash:
            self._lwe_model = LowWindEffect(
                pupil=pupil,
                piston_rms_rad=self.lwe_config.piston_rms_rad,
                tilt_rms_rad=self.lwe_config.tilt_rms_rad,
                ar_coeff=self.lwe_config.ar_coeff,
            )
            self._lwe_pupil_hash = pupil_hash
        
        return self._lwe_model
    
    def generate_with_lwe(
        self,
        n_screens: int,
        pupil: ArrayLike,
        seed: Optional[int] = None,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Generate phase screens with separate LWE component.
        
        Args:
            n_screens: Number of phase screens
            pupil: Pupil mask (required for LWE island detection)
            seed: Random seed
            
        Returns:
            Tuple of:
            - phase_residual: (n_screens, n_pix, n_pix) base residual phase
            - phase_lwe: (n_screens, n_pix, n_pix) LWE phase
            - phase_total: (n_screens, n_pix, n_pix) combined phase
        """
        if self.lwe_config is None:
            raise ValueError("No LWE config provided")
        
        xp = self._xp
        
        # Generate base phase screens
        phase_residual = self.generate(n_screens, pupil, seed)
        
        # Generate LWE screens
        lwe_model = self._get_lwe_model(pupil)
        lwe_seed = self.lwe_config.seed if seed is None else seed + 1000
        phase_lwe = lwe_model.generate(n_screens, seed=lwe_seed)
        
        # Ensure LWE phases are on same backend as residual phases
        phase_lwe = xp.asarray(phase_lwe)
        
        # Combine
        phase_total = phase_residual + phase_lwe
        
        return phase_residual, phase_lwe, phase_total
    
    def compute_rms(self, phase: ArrayLike, pupil: Optional[ArrayLike] = None) -> float:
        """
        Compute RMS of phase over pupil.
        
        Args:
            phase: Phase array (n, H, W) or (H, W)
            pupil: Optional pupil mask (H, W)
            
        Returns:
            RMS in radians
        """
        xp = self._xp
        
        if pupil is None:
            pupil = xp.ones((self.n_pix, self.n_pix), dtype=xp.float32)
        else:
            # Ensure pupil is on same backend as phase
            pupil = xp.asarray(pupil, dtype=xp.float32)
        
        pupil_sum = xp.sum(pupil)
        
        # Handle batch dimension
        if phase.ndim == 2:
            phase = phase[None, :, :]
        
        # Masked variance
        var = xp.sum(phase**2 * pupil[None, :, :], axis=(1, 2)) / pupil_sum
        rms = xp.sqrt(xp.mean(var))
        
        return float(rms)
    
    def info(self) -> Dict[str, Any]:
        """Return generator information."""
        return {
            "type": self.__class__.__name__,
            "n_pix": self.n_pix,
            "pixel_size_m": self.pixel_size,
            "extent_m": self.extent,
            "rms_expected_rad": self.rms_expected,
            "has_lwe": self.lwe_config is not None,
            "seed": self.seed,
        }


# =============================================================================
# Output Container
# =============================================================================

@dataclass
class PhaseScreenResult:
    """
    Container for phase screen generation results.
    
    Provides a standardized output format with optional metadata.
    """
    phase: ArrayLike  # (n_screens, n_pix, n_pix) in radians
    n_pix: int
    pixel_size: float  # meters
    rms_actual: float  # measured RMS
    rms_expected: float  # model-predicted RMS
    
    # Optional components
    phase_lwe: Optional[ArrayLike] = None
    seed: Optional[int] = None
    generator_type: str = "unknown"
    
    @property
    def n_screens(self) -> int:
        return self.phase.shape[0]
    
    @property
    def extent(self) -> float:
        return self.n_pix * self.pixel_size
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.phase.shape
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without large arrays)."""
        return {
            "n_screens": self.n_screens,
            "n_pix": self.n_pix,
            "pixel_size_m": self.pixel_size,
            "extent_m": self.extent,
            "rms_actual_rad": self.rms_actual,
            "rms_expected_rad": self.rms_expected,
            "has_lwe": self.phase_lwe is not None,
            "seed": self.seed,
            "generator_type": self.generator_type,
        }

"""
LBTI (Large Binocular Telescope Interferometer) instrument implementation.

Two 8.4m primary mirrors with 14.4m center-to-center separation.
Supports Fizeau interferometry mode.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from . import Instrument, register_instrument
from ..config import InstrumentConfig
from ..utils.compute import get_backend
from ..physics.zernike import build_zernike_modes, generate_zernike_phase
from ..physics.kolmogorov import KolmogorovGenerator
from ..physics.optics import compute_psf
from ..physics.noise import simulate_observation


# =============================================================================
# LBTI Parameters
# =============================================================================

# Primary mirror parameters
D_PRIMARY = 8.4  # Primary diameter [m]
D_SECONDARY = 0.911  # Central obscuration diameter [m]
BASELINE = 14.4  # Center-to-center baseline [m]

# Default wavelength (N-band)
LAM_DEFAULT = 11.0e-6  # 11 microns

# Spider geometry (4 vanes per aperture)
SPIDER_WIDTH = 0.01  # meters
SPIDER_ANGLES = [45, 135, 225, 315]  # degrees


@dataclass
class LBTIConfig:
    """LBTI-specific configuration."""
    D: float = D_PRIMARY
    D2: float = D_SECONDARY
    baseline: float = BASELINE
    lam: float = LAM_DEFAULT
    n_pix: int = 256
    enable_spiders: bool = True
    enable_piston: bool = True  # Inter-aperture piston
    enable_tip_tilt: bool = True  # Inter-aperture tip-tilt
    piston_rms: float = 100e-9  # meters RMS
    tip_tilt_rms: float = 0.05  # arcsec RMS


# =============================================================================
# Pupil Generation
# =============================================================================

def generate_lbti_pupil(
    n_pix: int = 256,
    D: float = D_PRIMARY,
    D2: float = D_SECONDARY,
    baseline: float = BASELINE,
    enable_spiders: bool = True,
) -> np.ndarray:
    """
    Generate LBTI dual-aperture pupil mask.
    
    Args:
        n_pix: Pupil array size
        D: Primary diameter [m]
        D2: Central obscuration diameter [m]
        baseline: Center-to-center separation [m]
        enable_spiders: Include spider vanes
    
    Returns:
        Binary pupil mask (n_pix, n_pix)
    """
    # Total extent: baseline + D
    total_extent = baseline + D
    scale = total_extent / 2  # half-extent
    
    x = np.linspace(-scale, scale, n_pix)
    xx, yy = np.meshgrid(x, x)
    
    # Left aperture center
    x_left = -baseline / 2
    # Right aperture center
    x_right = baseline / 2
    
    # Left aperture
    r_left = np.sqrt((xx - x_left)**2 + yy**2)
    left_primary = r_left <= D / 2
    left_secondary = r_left <= D2 / 2
    left_aperture = left_primary & ~left_secondary
    
    # Right aperture
    r_right = np.sqrt((xx - x_right)**2 + yy**2)
    right_primary = r_right <= D / 2
    right_secondary = r_right <= D2 / 2
    right_aperture = right_primary & ~right_secondary
    
    pupil = left_aperture | right_aperture
    
    if enable_spiders:
        pupil = _add_spiders(pupil, xx, yy, x_left, x_right, D)
    
    return pupil.astype(np.float32)


def _add_spiders(
    pupil: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    x_left: float,
    x_right: float,
    D: float,
) -> np.ndarray:
    """Add spider vanes to both apertures."""
    width = SPIDER_WIDTH
    
    for x_center in [x_left, x_right]:
        for angle_deg in SPIDER_ANGLES:
            angle = np.radians(angle_deg)
            
            # Rotate coordinates to spider frame
            dx = xx - x_center
            u = dx * np.cos(angle) + yy * np.sin(angle)
            v = -dx * np.sin(angle) + yy * np.cos(angle)
            
            # Spider extends from center outward
            spider = (np.abs(v) < width / 2) & (u >= 0) & (u <= D / 2)
            pupil = pupil & ~spider
    
    return pupil


# =============================================================================
# LBTI Instrument Class
# =============================================================================

@register_instrument('lbti')
class LBTIInstrument(Instrument):
    """
    LBTI dual-aperture interferometric instrument.
    
    Supports:
    - Dual 8.4m apertures with 14.4m baseline
    - Inter-aperture piston and tip-tilt
    - Individual aperture AO residuals
    - Spider vanes
    - Fizeau imaging mode
    """
    
    def __init__(self, config: InstrumentConfig = None):
        """
        Initialize LBTI instrument.
        
        Args:
            config: Instrument configuration (uses defaults if None)
        """
        if config is None:
            config = InstrumentConfig(name='lbti')
        super().__init__(config)
    
    def _setup(self):
        """Setup LBTI-specific components."""
        self.backend = get_backend()
        self.xp = self.backend.xp
        
        # LBTI-specific parameters
        self.D = D_PRIMARY
        self.D2 = D_SECONDARY
        self.baseline = BASELINE
        self.lam = getattr(self.config, 'wavelength', None) or LAM_DEFAULT
        self.n_pix = getattr(self.config, 'n_pix', None) or 256
        
        # Feature flags
        self.enable_spiders = getattr(self.config, 'enable_spiders', True)
        self.enable_piston = getattr(self.config, 'enable_differential_piston', True)
        self.enable_tip_tilt = getattr(self.config, 'enable_differential_tip_tilt', True)
        
        # Zernike for each aperture
        self.n_zernike = getattr(self.config, 'n_zernike', 50)
        self._zernike_gen = None
        
        # Piston/tip-tilt state
        self.piston_rms = 100e-9  # meters
        self.tip_tilt_rms = 0.05  # arcsec
        
        # Cache
        self._pupil_mask = None
        self._aperture_masks = None
    
    @property
    def pupil(self):
        """Get pupil amplitude array."""
        return self.generate_pupil()
    
    def _get_zernike_modes(self):
        """Get Zernike modes for individual aperture."""
        if self._zernike_gen is None:
            # Aperture size in the full array
            aperture_pix = int(self.n_pix * self.D / (self.baseline + self.D))
            self._zernike_gen = build_zernike_modes(
                n_pix=aperture_pix, 
                radius=1.0, 
                n_range=(2, 5)
            )
        return self._zernike_gen
    
    def generate_pupil(self) -> np.ndarray:
        """Generate LBTI pupil mask."""
        if self._pupil_mask is None:
            pupil_np = generate_lbti_pupil(
                n_pix=self.n_pix,
                D=self.D,
                D2=self.D2,
                baseline=self.baseline,
                enable_spiders=self.enable_spiders,
            )
            self._pupil_mask = self.xp.asarray(pupil_np)
        return self._pupil_mask
    
    def _get_aperture_masks(self):
        """Get individual aperture masks."""
        if self._aperture_masks is None:
            total_extent = self.baseline + self.D
            scale = total_extent / 2
            
            x = self.xp.linspace(-scale, scale, self.n_pix)
            xx, yy = self.xp.meshgrid(x, x)
            
            x_left = -self.baseline / 2
            x_right = self.baseline / 2
            
            r_left = self.xp.sqrt((xx - x_left)**2 + yy**2)
            r_right = self.xp.sqrt((xx - x_right)**2 + yy**2)
            
            left_mask = r_left <= self.D / 2
            right_mask = r_right <= self.D / 2
            
            self._aperture_masks = (left_mask, right_mask)
        
        return self._aperture_masks
    
    def generate_phase(self) -> np.ndarray:
        """
        Generate wavefront phase for both apertures.
        
        Includes:
        - Differential piston
        - Differential tip-tilt
        - Individual AO residuals
        """
        xp = self.xp
        phase = xp.zeros((self.n_pix, self.n_pix), dtype=xp.float32)
        
        left_mask, right_mask = self._get_aperture_masks()
        
        # Differential piston
        if self.enable_piston:
            piston = xp.random.randn() * self.piston_rms
            piston_rad = 2 * xp.pi * piston / self.lam
            
            # Apply to right aperture relative to left
            phase = xp.where(right_mask, phase + piston_rad, phase)
        
        # Differential tip-tilt
        if self.enable_tip_tilt:
            # Convert arcsec to radians
            tip_tilt_rad = self.tip_tilt_rms * (xp.pi / 180 / 3600)
            
            tip = xp.random.randn() * tip_tilt_rad
            tilt = xp.random.randn() * tip_tilt_rad
            
            # Apply gradient across right aperture
            total_extent = self.baseline + self.D
            scale = total_extent / 2
            x = xp.linspace(-scale, scale, self.n_pix)
            xx, yy = xp.meshgrid(x, x)
            
            gradient = tip * xx + tilt * yy
            gradient_rad = 2 * xp.pi * gradient / self.lam
            
            phase = xp.where(right_mask, phase + gradient_rad, phase)
        
        # TODO: Add individual AO residuals for each aperture
        
        return phase
    
    def generate_psf(self, phase: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate LBTI PSF (Fizeau fringe pattern).
        
        Args:
            phase: Optional wavefront phase (generates random if None)
        
        Returns:
            PSF array
        """
        xp = self.xp
        
        pupil = self.generate_pupil()
        
        if phase is None:
            phase = self.generate_phase()
        
        # Compute PSF using optics module
        psf = compute_psf(pupil, phase, normalize=True)
        
        return psf
    
    def simulate_observation(
        self,
        source_image: np.ndarray,
        n_photons: float = 1e6,
        read_noise: float = 10.0,
    ) -> np.ndarray:
        """
        Simulate LBTI observation of a source.
        
        Args:
            source_image: Ground truth source image
            n_photons: Total photon count
            read_noise: Read noise in electrons
        
        Returns:
            Simulated observation
        """
        xp = self.xp
        
        # Ensure source is on GPU
        if hasattr(source_image, '__cuda_array_interface__'):
            source = source_image
        else:
            source = xp.asarray(source_image)
        
        # Generate PSF
        psf = self.generate_psf()
        
        # Convolve source with PSF
        from scipy.signal import fftconvolve
        
        source_np = self.backend.to_numpy(source)
        psf_np = self.backend.to_numpy(psf)
        
        # Resize PSF if needed
        if psf_np.shape != source_np.shape:
            from scipy.ndimage import zoom
            scale = source_np.shape[0] / psf_np.shape[0]
            psf_np = zoom(psf_np, scale, order=1)
        
        convolved = fftconvolve(source_np, psf_np, mode='same')
        convolved = xp.asarray(convolved.astype(np.float32))
        
        # Add noise
        observation = simulate_observation(convolved, n_photons, read_noise)
        
        return observation
    
    def generate_psfs(self, n: int, **kwargs) -> np.ndarray:
        """Generate multiple PSFs."""
        xp = self.xp
        psfs = []
        for _ in range(n):
            psfs.append(self.generate_psf(**kwargs))
        return xp.stack(psfs, axis=0)
    
    def generate_observation(self, psf, **kwargs):
        """Generate noisy observation from PSF."""
        n_photons = kwargs.get('n_photons', 1e6)
        read_noise = kwargs.get('read_noise', 10.0)
        return simulate_observation(psf, n_photons, read_noise)
    
    def generate_observations(self, psfs, **kwargs):
        """Generate noisy observations from PSFs."""
        xp = self.xp
        observations = []
        for psf in psfs:
            observations.append(self.generate_observation(psf, **kwargs))
        return xp.stack(observations, axis=0)


# Register instrument
def get_lbti_instrument(config: Optional[InstrumentConfig] = None) -> LBTIInstrument:
    """Factory function for LBTI instrument."""
    return LBTIInstrument(config)


__all__ = [
    'LBTIInstrument',
    'generate_lbti_pupil',
    'get_lbti_instrument',
    'LBTIConfig',
]

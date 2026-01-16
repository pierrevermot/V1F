"""
VLT (Very Large Telescope) instrument simulation.

Implements the VLT 8.2m telescope with:
- Central obscuration (secondary mirror)
- Spider vanes (4 vanes from 2 attachment points)
- Atmospheric turbulence (Zernike + Kolmogorov)
- Low Wind Effect (LWE)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

from ..config import InstrumentConfig
from ..utils.compute import get_backend
from ..physics import zernike, kolmogorov, optics, noise
from . import Instrument, register_instrument


# =============================================================================
# Pupil Generation
# =============================================================================

class VLTPupil:
    """
    VLT telescope pupil with spider vanes.
    """
    
    def __init__(
        self,
        n_pix: int,
        wavelength: float,
        pixel_scale: float,
        primary_diameter: float = 8.2,
        obstruction_diameter: float = 1.116,
    ):
        """
        Initialize VLT pupil.
        
        Args:
            n_pix: Grid size
            wavelength: Wavelength (meters)
            pixel_scale: Pixel scale (rad/pixel)
            primary_diameter: Primary mirror diameter (meters)
            obstruction_diameter: Central obstruction diameter (meters)
        """
        backend = get_backend()
        xp = backend.xp
        
        self.n_pix = n_pix
        self.wavelength = wavelength
        self.pixel_scale = pixel_scale
        self.D = primary_diameter
        self.D2 = obstruction_diameter
        
        # Pupil plane pixel size
        self.pupil_pixel_size = wavelength / n_pix / pixel_scale
        self.radius = self.D / 2
        
        # Build annular pupil
        self._build_base_pupil()
        
        # Spider mask (will be applied)
        self._spider_mask = None
    
    def _build_base_pupil(self):
        """Build annular pupil (primary - obstruction)."""
        backend = get_backend()
        xp = backend.xp
        
        # Coordinate grid (high resolution for smooth edges)
        upscaling = 32
        n_large = upscaling * self.n_pix
        
        x = xp.arange(n_large, dtype=xp.float32) * self.pupil_pixel_size / upscaling
        x = x - xp.mean(x)
        X, Y = xp.meshgrid(x, x)
        R = xp.sqrt(X**2 + Y**2)
        
        # Annular pupil
        pupil_large = ((R > self.D2/2) & (R < self.D/2)).astype(xp.float32)
        
        # Rebin to target size
        self._amplitude = self._rebin(pupil_large, (self.n_pix, self.n_pix))
    
    def _rebin(self, array, shape):
        """Rebin array to new shape by averaging."""
        backend = get_backend()
        xp = backend.xp
        
        sh = shape[0], array.shape[0]//shape[0], shape[1], array.shape[1]//shape[1]
        return array.reshape(sh).mean(-1).mean(1)
    
    def add_vlt_spiders(
        self,
        width: float = 0.04,
        half_opening_deg: float = 51.3,
        attach_angles_deg: List[float] = [0.0, 180.0],
    ):
        """
        Add VLT-like spider vanes.
        
        Args:
            width: Spider vane width (meters)
            half_opening_deg: Half-angle between vanes at each attachment
            attach_angles_deg: Angles of spider attachment points
        """
        # Generate spider segments
        segments = vlt_spider_segments(
            self.D, self.D2, half_opening_deg, attach_angles_deg
        )
        
        # Build spider mask
        self._spider_mask = self._build_spider_mask(segments, width)
        
        # Apply to amplitude
        self._amplitude = self._amplitude * self._spider_mask
    
    def _build_spider_mask(self, segments, width: float):
        """Build spider mask from line segments."""
        backend = get_backend()
        xp = backend.xp
        
        # High resolution grid
        upscaling = 32
        n_large = upscaling * self.n_pix
        
        x = xp.arange(n_large, dtype=xp.float32) * self.pupil_pixel_size / upscaling
        x = x - xp.mean(x)
        X, Y = xp.meshgrid(x, x)
        
        mask = xp.ones((n_large, n_large), dtype=xp.float32)
        half_w = width / 2
        
        for (x0, y0, x1, y1) in segments:
            # Vector along segment
            vx, vy = x1 - x0, y1 - y0
            length = math.sqrt(vx**2 + vy**2)
            if length < 1e-10:
                continue
            
            # Normalize
            ux, uy = vx / length, vy / length
            
            # Vector from start to each point
            dx = X - x0
            dy = Y - y0
            
            # Parallel distance (along segment)
            t = dx * ux + dy * uy
            
            # Perpendicular distance
            perp = xp.abs(-dx * uy + dy * ux)
            
            # Points within segment bounds and within width
            in_segment = (t >= 0) & (t <= length) & (perp <= half_w)
            mask = mask * (~in_segment).astype(xp.float32)
        
        # Rebin to target size
        return self._rebin(mask, (self.n_pix, self.n_pix))
    
    @property
    def amplitude(self):
        """Get pupil amplitude array."""
        return self._amplitude


def vlt_spider_segments(
    D: float,
    D2: float,
    half_opening_deg: float = 51.3,
    attach_angles_deg: List[float] = [0.0, 180.0],
) -> List[Tuple[float, float, float, float]]:
    """
    Generate VLT-like spider segment coordinates.
    
    VLT has 4 spider vanes from 2 attachment points. Each attachment
    point has 2 vanes separated by half_opening_deg from the radial
    direction.
    
    Args:
        D: Primary diameter
        D2: Obstruction diameter
        half_opening_deg: Half-angle between vane pairs
        attach_angles_deg: Attachment point angles
    
    Returns:
        List of (x0, y0, x1, y1) segment coordinates in meters
    """
    segments = []
    half_open_rad = math.radians(half_opening_deg)
    
    r_inner = D2 / 2  # Start from obstruction edge
    r_outer = D / 2   # End at primary edge
    
    for attach_deg in attach_angles_deg:
        attach_rad = math.radians(attach_deg)
        
        # Two vanes per attachment point
        for sign in [-1, +1]:
            vane_angle = attach_rad + sign * half_open_rad
            
            # Start point (on obstruction rim)
            x0 = r_inner * math.cos(attach_rad)
            y0 = r_inner * math.sin(attach_rad)
            
            # End point (on primary edge, along vane direction)
            x1 = r_outer * math.cos(vane_angle)
            y1 = r_outer * math.sin(vane_angle)
            
            segments.append((x0, y0, x1, y1))
    
    return segments


# =============================================================================
# Low Wind Effect Model
# =============================================================================

class VLTLowWindEffect:
    """
    VLT Low Wind Effect (LWE) model.
    
    Models differential wavefront errors between spider-separated
    pupil sectors with AR(1) temporal correlation.
    """
    
    def __init__(
        self,
        n_pix: int,
        wavelength: float,
        pixel_scale: float,
        D: float,
        piston_rms_rad: float = 0.5,
        tilt_rms_rad: float = 0.3,
        ar_coeff: float = 0.95,
        half_opening_deg: float = 51.3,
    ):
        backend = get_backend()
        xp = backend.xp
        
        self.n_pix = n_pix
        self.piston_rms = piston_rms_rad
        self.tilt_rms = tilt_rms_rad
        self.ar_coeff = ar_coeff
        
        # Build sector masks
        self.masks = self._build_sector_masks(
            n_pix, wavelength, pixel_scale, D, half_opening_deg
        )
        self.n_sectors = self.masks.shape[0]
        
        # Normalized coordinates for tilt
        c = (n_pix - 1) / 2.0
        idx = xp.arange(n_pix, dtype=xp.float32)
        X, Y = xp.meshgrid(idx, idx)
        self.X_norm = (X - c) / c
        self.Y_norm = (Y - c) / c
    
    def _build_sector_masks(
        self, n_pix, wavelength, pixel_scale, D, half_opening_deg
    ):
        """Build 4 sector masks for VLT spider geometry."""
        backend = get_backend()
        xp = backend.xp
        
        pupil_pixel_size = wavelength / n_pix / pixel_scale
        c = (n_pix - 1) / 2.0
        
        idx = xp.arange(n_pix, dtype=xp.float32)
        X, Y = xp.meshgrid(idx, idx)
        X = (X - c) * pupil_pixel_size
        Y = (Y - c) * pupil_pixel_size
        
        theta = xp.arctan2(Y, X)
        
        half_open = math.radians(half_opening_deg)
        
        # Boundary angles
        b1 = half_open
        b2 = math.pi - half_open
        b3 = math.pi + half_open - 2*math.pi  # Wrap to [-pi, pi]
        b4 = -half_open
        
        masks = xp.zeros((4, n_pix, n_pix), dtype=xp.float32)
        
        masks[0] = ((theta >= b4) & (theta < b1)).astype(xp.float32)
        masks[1] = ((theta >= b1) & (theta < b2)).astype(xp.float32)
        masks[2] = ((theta >= b2) | (theta < b3)).astype(xp.float32)
        masks[3] = ((theta >= b3) & (theta < b4)).astype(xp.float32)
        
        return masks
    
    def generate(self, n_screens: int, pupil=None, seed: Optional[int] = None):
        """Generate LWE phase screens."""
        backend = get_backend()
        xp = backend.xp
        
        if seed is not None:
            xp.random.seed(seed)
        
        # Random coefficients
        pistons = xp.random.randn(self.n_sectors, n_screens).astype(xp.float32) * self.piston_rms
        tips_x = xp.random.randn(self.n_sectors, n_screens).astype(xp.float32) * self.tilt_rms
        tips_y = xp.random.randn(self.n_sectors, n_screens).astype(xp.float32) * self.tilt_rms
        
        # Build phase screens
        phase = xp.zeros((n_screens, self.n_pix, self.n_pix), dtype=xp.float32)
        
        for s in range(self.n_sectors):
            mask = self.masks[s]
            phase += pistons[s, :, None, None] * mask[None, :, :]
            phase += tips_x[s, :, None, None] * self.X_norm[None, :, :] * mask[None, :, :]
            phase += tips_y[s, :, None, None] * self.Y_norm[None, :, :] * mask[None, :, :]
        
        if pupil is not None:
            pupil = backend.ensure_local(pupil).astype(xp.float32)
            phase = phase * pupil[None, :, :]
        
        return phase


# =============================================================================
# VLT Instrument
# =============================================================================

@register_instrument('vlt')
class VLTInstrument(Instrument):
    """
    VLT instrument simulation.
    
    Includes:
    - 8.2m primary with 1.116m central obscuration
    - 4 spider vanes (2 attachment points)
    - Zernike-based low-frequency AO residuals
    - Kolmogorov high-frequency turbulence
    - Low Wind Effect
    """
    
    def _setup(self):
        """Initialize VLT components."""
        from ..utils.compute import init_backend
        
        cfg = self.config
        init_backend(cfg.compute_mode)
        
        # Build pupil
        self._pupil = VLTPupil(
            cfg.n_pix,
            cfg.wavelength,
            cfg.pixel_scale,
            cfg.telescope.primary_diameter,
            cfg.telescope.obstruction_diameter,
        )
        
        # Add spiders if configured
        if cfg.telescope.spider_model == 'VLT_LIKE':
            self._pupil.add_vlt_spiders(
                width=cfg.telescope.spider_width,
                half_opening_deg=cfg.telescope.spider_half_opening_deg,
                attach_angles_deg=cfg.telescope.spider_attach_angles_deg,
            )
        
        # Atmosphere model
        atm_cfg = cfg.atmosphere
        self._atmosphere = kolmogorov.AtmosphereModel(
            cfg.n_pix,
            cfg.wavelength,
            cfg.pixel_scale,
            self._pupil.radius,
            actuator_pitch=atm_cfg.actuator_pitch if atm_cfg.hf_enable else None,
            hf_transition=atm_cfg.hf_transition,
            zernike_n_range=(atm_cfg.zernike_n_min, atm_cfg.zernike_n_max),
        )
        
        # LWE model
        if atm_cfg.lwe_enable:
            self._lwe = VLTLowWindEffect(
                cfg.n_pix,
                cfg.wavelength,
                cfg.pixel_scale,
                cfg.telescope.primary_diameter,
                piston_rms_rad=atm_cfg.lwe_piston_rms_rad,
                tilt_rms_rad=atm_cfg.lwe_tilt_rms_rad,
                ar_coeff=atm_cfg.lwe_ar_coeff,
                half_opening_deg=cfg.telescope.spider_half_opening_deg,
            )
        else:
            self._lwe = None
    
    @property
    def pupil(self):
        """Get pupil amplitude."""
        return self._pupil.amplitude
    
    def generate_psf(
        self,
        rms_opd: Optional[float] = None,
        r0: Optional[float] = None,
        include_lwe: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Generate a single PSF.
        
        Args:
            rms_opd: RMS OPD for low-freq component (meters). Random if None.
            r0: Fried parameter for high-freq (meters). Random if None.
            include_lwe: Include Low Wind Effect
            seed: Random seed
        
        Returns:
            2D PSF array
        """
        backend = get_backend()
        xp = backend.xp
        
        cfg = self.config
        atm_cfg = cfg.atmosphere
        
        if seed is not None:
            xp.random.seed(seed)
        
        # Random parameters if not specified
        if rms_opd is None:
            rms_opd = float(xp.random.uniform(atm_cfg.rms_min, atm_cfg.rms_max))
        if r0 is None and atm_cfg.hf_enable:
            r0 = float(xp.random.uniform(atm_cfg.r0_min, atm_cfg.r0_max))
        
        # Generate phase components
        lf_raw = self._atmosphere.generate_lf_phase(1, atm_cfg.power_law_exponent)
        lf_phase = self._atmosphere.normalize_and_scale(lf_raw, self.pupil, xp.array([rms_opd]))
        total_phase = lf_phase[0]
        
        # High-frequency
        if atm_cfg.hf_enable and r0 is not None:
            hf_phase = self._atmosphere.generate_hf_phase(1, xp.array([r0]), self.pupil)
            total_phase = total_phase + hf_phase[0]
        
        # LWE
        if include_lwe and self._lwe is not None:
            lwe_phase = self._lwe.generate(1, self.pupil)
            total_phase = total_phase + lwe_phase[0]
        
        # Compute PSF
        psf = optics.compute_psf(self.pupil, total_phase)
        
        # Crop to output size
        if cfg.n_pix_output != cfg.n_pix:
            psf = optics.center_crop(psf, cfg.n_pix_output)
        
        return psf
    
    def generate_psfs(
        self,
        n: int,
        rms_opd=None,
        r0=None,
        include_lwe: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Generate multiple PSFs.
        
        Args:
            n: Number of PSFs
            rms_opd: RMS OPD array or (min, max) tuple for random
            r0: r0 array or (min, max) tuple for random
            include_lwe: Include LWE
            seed: Random seed
        
        Returns:
            3D PSF array (n, H, W)
        """
        backend = get_backend()
        xp = backend.xp
        
        cfg = self.config
        atm_cfg = cfg.atmosphere
        
        if seed is not None:
            xp.random.seed(seed)
        
        # Handle random parameters
        if rms_opd is None:
            rms_opd = xp.random.uniform(atm_cfg.rms_min, atm_cfg.rms_max, n).astype(xp.float32)
        elif isinstance(rms_opd, (tuple, list)):
            rms_opd = xp.random.uniform(rms_opd[0], rms_opd[1], n).astype(xp.float32)
        else:
            rms_opd = xp.asarray(rms_opd, dtype=xp.float32)
        
        if atm_cfg.hf_enable:
            if r0 is None:
                r0 = xp.random.uniform(atm_cfg.r0_min, atm_cfg.r0_max, n).astype(xp.float32)
            elif isinstance(r0, (tuple, list)):
                r0 = xp.random.uniform(r0[0], r0[1], n).astype(xp.float32)
            else:
                r0 = xp.asarray(r0, dtype=xp.float32)
        
        # Generate phase components
        lf_raw = self._atmosphere.generate_lf_phase(n, atm_cfg.power_law_exponent)
        total_phase = self._atmosphere.normalize_and_scale(lf_raw, self.pupil, rms_opd)
        
        # High-frequency
        if atm_cfg.hf_enable and r0 is not None:
            hf_phase = self._atmosphere.generate_hf_phase(n, r0, self.pupil)
            total_phase = total_phase + hf_phase
        
        # LWE
        if include_lwe and self._lwe is not None:
            lwe_phase = self._lwe.generate(n, self.pupil)
            total_phase = total_phase + lwe_phase
        
        # Compute PSFs
        psfs = optics.compute_psf_batch(self.pupil, total_phase, batch_size=cfg.batch_size)
        
        # Crop to output size
        if cfg.n_pix_output != cfg.n_pix:
            psfs = optics.center_crop(psfs, cfg.n_pix_output)
        
        return psfs
    
    def generate_observation(self, psf, **kwargs):
        """Generate noisy observation from PSF."""
        cfg = self.config.noise
        return noise.simulate_observation(
            psf,
            peak_flux=kwargs.get('peak_flux', cfg.peak_flux),
            background=kwargs.get('background', cfg.background_level),
            read_noise=kwargs.get('read_noise', cfg.read_noise),
        )
    
    def generate_observations(self, psfs, **kwargs):
        """Generate noisy observations from PSFs."""
        cfg = self.config.noise
        return noise.simulate_observations_batch(
            psfs,
            peak_flux=kwargs.get('peak_flux', cfg.peak_flux),
            background=kwargs.get('background', cfg.background_level),
            read_noise=kwargs.get('read_noise', cfg.read_noise),
        )

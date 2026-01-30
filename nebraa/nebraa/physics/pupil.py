"""
Generic Pupil class for telescope simulations.

Provides a unified interface for pupil definition that can be used
across different models (Jolissaint AO, VLT instrument, etc.).

The pupil encapsulates:
- 2D amplitude/transmission map
- Physical properties (diameter, pixel scale, wavelength)
- Optional segment masks (for segmented mirrors, LWE effects)
- Coordinate grids for Zernike polynomials, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Union
import numpy as np

from ..utils.compute import get_backend

# Type alias for array-like objects
ArrayLike = Any


@dataclass
class Pupil:
    """
    Generic telescope pupil representation.
    
    The pupil is defined by its amplitude map and physical parameters.
    The FFT relationship between pupil and focal plane is:
    
        pupil_pixel_size = wavelength / (n_pix * pixel_scale)
    
    where pixel_scale is the angular size of PSF pixels (rad/pixel).
    
    Attributes:
        amplitude: 2D pupil amplitude/transmission map (0 to 1)
        wavelength: Reference wavelength (meters)
        pixel_scale: Angular pixel scale in focal plane (rad/pixel)
        diameter: Primary mirror diameter (meters)
        obstruction_diameter: Central obstruction diameter (meters), 0 if none
        segment_masks: Optional dict mapping segment names to boolean masks
        metadata: Optional dict for additional properties
    
    Properties (computed):
        n_pix: Grid size (from amplitude shape)
        pupil_pixel_size: Physical pixel size in pupil plane (meters)
        area: Total pupil area in pixels
        filling_factor: Fraction of array filled by pupil
    
    Example:
        # Create from explicit parameters
        pupil = Pupil.from_circular(
            n_pix=256, wavelength=2.2e-6, pixel_scale=13e-3/206265,
            diameter=8.2, obstruction_diameter=1.116
        )
        
        # Create from existing amplitude map
        pupil = Pupil.from_amplitude(
            amplitude=my_pupil_array, wavelength=2.2e-6,
            pixel_scale=13e-3/206265, diameter=8.2
        )
        
        # Use with Jolissaint AO model
        model = JolissaintAOModel.from_pupil(pupil, atmosphere, ao_config)
    """
    
    amplitude: ArrayLike
    wavelength: float
    pixel_scale: float
    diameter: float
    obstruction_diameter: float = 0.0
    segment_masks: Optional[Dict[str, ArrayLike]] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        """Validate inputs and compute derived quantities."""
        if self.amplitude.ndim != 2:
            raise ValueError(f"Amplitude must be 2D, got shape {self.amplitude.shape}")
        if self.amplitude.shape[0] != self.amplitude.shape[1]:
            raise ValueError(f"Amplitude must be square, got shape {self.amplitude.shape}")
        if self.wavelength <= 0:
            raise ValueError(f"Wavelength must be positive, got {self.wavelength}")
        if self.pixel_scale <= 0:
            raise ValueError(f"Pixel scale must be positive, got {self.pixel_scale}")
        if self.diameter <= 0:
            raise ValueError(f"Diameter must be positive, got {self.diameter}")
        if self.obstruction_diameter < 0:
            raise ValueError(f"Obstruction diameter must be non-negative, got {self.obstruction_diameter}")
    
    @property
    def n_pix(self) -> int:
        """Grid size in pixels."""
        return self.amplitude.shape[0]
    
    @property
    def pupil_pixel_size(self) -> float:
        """Physical pixel size in pupil plane (meters)."""
        return self.wavelength / (self.n_pix * self.pixel_scale)
    
    @property
    def pupil_extent(self) -> float:
        """Total physical extent of pupil array (meters)."""
        return self.n_pix * self.pupil_pixel_size
    
    @property
    def diameter_pixels(self) -> float:
        """Diameter in pixels."""
        return self.diameter / self.pupil_pixel_size
    
    @property
    def obstruction_pixels(self) -> float:
        """Obstruction diameter in pixels."""
        return self.obstruction_diameter / self.pupil_pixel_size
    
    @property
    def area(self) -> float:
        """Total pupil area in pixels (sum of amplitude squared)."""
        backend = get_backend()
        return float(backend.to_numpy(backend.xp.sum(self.amplitude ** 2)))
    
    @property
    def filling_factor(self) -> float:
        """Fraction of array filled by pupil."""
        return self.area / (self.n_pix ** 2)
    
    @property
    def n_segments(self) -> int:
        """Number of segments (0 if no segment masks defined)."""
        return len(self.segment_masks) if self.segment_masks else 0
    
    def get_coordinate_grid(self, units: str = 'meters') -> Tuple[ArrayLike, ArrayLike]:
        """
        Get 2D coordinate grids.
        
        Args:
            units: 'meters', 'pixels', or 'normalized' (to diameter)
        
        Returns:
            (X, Y) coordinate arrays
        
        Note:
            Uses pixel-centered convention: center at (n_pix-1)/2
        """
        backend = get_backend()
        xp = backend.xp
        
        n = self.n_pix
        center = (n - 1) / 2.0  # Pixel-centered grid
        
        if units == 'pixels':
            x = xp.arange(n, dtype=xp.float64) - center
        elif units == 'meters':
            x = (xp.arange(n, dtype=xp.float64) - center) * self.pupil_pixel_size
        elif units == 'normalized':
            x = (xp.arange(n, dtype=xp.float64) - center) * self.pupil_pixel_size / (self.diameter / 2)
        else:
            raise ValueError(f"Unknown units: {units}")
        
        X, Y = xp.meshgrid(x, x)
        return X, Y
    
    def get_polar_coordinates(self, normalized: bool = True) -> Tuple[ArrayLike, ArrayLike]:
        """
        Get polar coordinate grids.
        
        Args:
            normalized: If True, normalize radius to [0, 1] at pupil edge
        
        Returns:
            (rho, theta) arrays
        """
        backend = get_backend()
        xp = backend.xp
        
        X, Y = self.get_coordinate_grid('meters')
        rho = xp.sqrt(X**2 + Y**2)
        theta = xp.arctan2(Y, X)
        
        if normalized:
            rho = rho / (self.diameter / 2)
        
        return rho, theta
    
    def to_backend(self, backend_name: str = None) -> 'Pupil':
        """
        Convert amplitude array to specified backend (CPU/GPU).
        
        Args:
            backend_name: 'CPU' or 'GPU', or None to use current backend
        
        Returns:
            New Pupil with converted arrays
        """
        from ..utils.compute import init_backend
        
        if backend_name:
            init_backend(backend_name)
        
        backend = get_backend()
        xp = backend.xp
        
        # Convert amplitude
        amplitude = backend.ensure_local(self.amplitude)
        amplitude = xp.asarray(amplitude)
        
        # Convert segment masks if present
        segment_masks = None
        if self.segment_masks:
            segment_masks = {
                name: xp.asarray(backend.ensure_local(mask))
                for name, mask in self.segment_masks.items()
            }
        
        return Pupil(
            amplitude=amplitude,
            wavelength=self.wavelength,
            pixel_scale=self.pixel_scale,
            diameter=self.diameter,
            obstruction_diameter=self.obstruction_diameter,
            segment_masks=segment_masks,
            metadata=self.metadata.copy(),
        )
    
    def resample(self, new_n_pix: int) -> 'Pupil':
        """
        Resample pupil to new grid size.
        
        Note: This changes the PSF field of view while keeping the same
        pixel scale. The pupil diameter in pixels will change.
        
        Args:
            new_n_pix: New grid size
        
        Returns:
            New Pupil with resampled amplitude
        """
        backend = get_backend()
        xp = backend.xp
        
        # Use scipy for resampling
        from scipy.ndimage import zoom as scipy_zoom
        
        amp_np = backend.to_numpy(self.amplitude)
        factor = new_n_pix / self.n_pix
        new_amp = scipy_zoom(amp_np, factor, order=1)
        new_amp = xp.asarray(new_amp)
        
        # Resample segment masks if present
        segment_masks = None
        if self.segment_masks:
            segment_masks = {}
            for name, mask in self.segment_masks.items():
                mask_np = backend.to_numpy(mask)
                new_mask = scipy_zoom(mask_np, factor, order=0)  # Nearest neighbor for masks
                segment_masks[name] = xp.asarray(new_mask)
        
        return Pupil(
            amplitude=new_amp,
            wavelength=self.wavelength,
            pixel_scale=self.pixel_scale,
            diameter=self.diameter,
            obstruction_diameter=self.obstruction_diameter,
            segment_masks=segment_masks,
            metadata=self.metadata.copy(),
        )
    
    def with_wavelength(self, new_wavelength: float) -> 'Pupil':
        """
        Create new pupil with different wavelength.
        
        This keeps the same amplitude map but changes the wavelength,
        which affects the pupil_pixel_size and PSF scaling.
        
        Args:
            new_wavelength: New wavelength (meters)
        
        Returns:
            New Pupil with updated wavelength
        """
        return Pupil(
            amplitude=self.amplitude,
            wavelength=new_wavelength,
            pixel_scale=self.pixel_scale,
            diameter=self.diameter,
            obstruction_diameter=self.obstruction_diameter,
            segment_masks=self.segment_masks,
            metadata=self.metadata.copy(),
        )
    
    def with_pixel_scale(self, new_pixel_scale: float) -> 'Pupil':
        """
        Create new pupil with different pixel scale.
        
        Warning: This changes the pupil_pixel_size, so the amplitude map
        will no longer correctly represent the physical pupil. Use this
        only if you understand the implications.
        
        Args:
            new_pixel_scale: New pixel scale (rad/pixel)
        
        Returns:
            New Pupil with updated pixel scale
        """
        return Pupil(
            amplitude=self.amplitude,
            wavelength=self.wavelength,
            pixel_scale=new_pixel_scale,
            diameter=self.diameter,
            obstruction_diameter=self.obstruction_diameter,
            segment_masks=self.segment_masks,
            metadata=self.metadata.copy(),
        )
    
    def copy(self) -> 'Pupil':
        """Create a deep copy of the pupil."""
        backend = get_backend()
        xp = backend.xp
        
        segment_masks = None
        if self.segment_masks:
            segment_masks = {name: xp.copy(mask) for name, mask in self.segment_masks.items()}
        
        return Pupil(
            amplitude=xp.copy(self.amplitude),
            wavelength=self.wavelength,
            pixel_scale=self.pixel_scale,
            diameter=self.diameter,
            obstruction_diameter=self.obstruction_diameter,
            segment_masks=segment_masks,
            metadata=self.metadata.copy(),
        )
    
    # =========================================================================
    # Factory methods
    # =========================================================================
    
    @classmethod
    def from_amplitude(
        cls,
        amplitude: ArrayLike,
        wavelength: float,
        pixel_scale: float,
        diameter: float,
        obstruction_diameter: float = 0.0,
        segment_masks: Optional[Dict[str, ArrayLike]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'Pupil':
        """
        Create Pupil from existing amplitude map.
        
        Args:
            amplitude: 2D amplitude/transmission array
            wavelength: Wavelength (meters)
            pixel_scale: Angular pixel scale (rad/pixel)
            diameter: Primary diameter (meters)
            obstruction_diameter: Central obstruction diameter (meters)
            segment_masks: Optional segment masks
            metadata: Optional metadata dict
        
        Returns:
            Pupil instance
        """
        return cls(
            amplitude=amplitude,
            wavelength=wavelength,
            pixel_scale=pixel_scale,
            diameter=diameter,
            obstruction_diameter=obstruction_diameter,
            segment_masks=segment_masks,
            metadata=metadata or {},
        )
    
    @classmethod
    def from_circular(
        cls,
        n_pix: int,
        wavelength: float,
        pixel_scale: float,
        diameter: float,
        obstruction_diameter: float = 0.0,
        soft_edge: bool = False,
        edge_width: float = 0.01,
    ) -> 'Pupil':
        """
        Create circular pupil with optional central obstruction.
        
        The pupil is created with correct physical scaling for the given
        wavelength and pixel scale.
        
        Args:
            n_pix: Grid size
            wavelength: Wavelength (meters)
            pixel_scale: Angular pixel scale (rad/pixel)
            diameter: Primary diameter (meters)
            obstruction_diameter: Central obstruction diameter (meters)
            soft_edge: If True, apply smooth edges
            edge_width: Width of soft edge (fraction of diameter)
        
        Returns:
            Pupil instance
        """
        backend = get_backend()
        xp = backend.xp
        
        # Compute pupil pixel size
        pupil_pixel_size = wavelength / (n_pix * pixel_scale)
        
        # Create coordinate grid in physical units (pixel-centered)
        center = (n_pix - 1) / 2.0
        x_m = (xp.arange(n_pix, dtype=xp.float64) - center) * pupil_pixel_size
        X, Y = xp.meshgrid(x_m, x_m)
        R = xp.sqrt(X**2 + Y**2)
        
        # Create annular pupil
        outer_radius = diameter / 2
        inner_radius = obstruction_diameter / 2
        
        if soft_edge:
            # Smooth sigmoid edges
            edge_w = edge_width * diameter
            outer_edge = 0.5 * (1.0 - xp.tanh((R - outer_radius) / edge_w))
            if obstruction_diameter > 0:
                inner_edge = 0.5 * (1.0 + xp.tanh((R - inner_radius) / edge_w))
                amplitude = outer_edge * inner_edge
            else:
                amplitude = outer_edge
        else:
            # Hard edges
            if obstruction_diameter > 0:
                amplitude = ((R <= outer_radius) & (R >= inner_radius)).astype(xp.float64)
            else:
                amplitude = (R <= outer_radius).astype(xp.float64)
        
        return cls(
            amplitude=amplitude,
            wavelength=wavelength,
            pixel_scale=pixel_scale,
            diameter=diameter,
            obstruction_diameter=obstruction_diameter,
            metadata={'type': 'circular'},
        )
    
    @classmethod
    def from_vlt(
        cls,
        n_pix: int,
        wavelength: float,
        pixel_scale: float,
        include_spiders: bool = True,
        spider_width: float = 0.04,
    ) -> 'Pupil':
        """
        Create VLT pupil with optional spider vanes.
        
        Args:
            n_pix: Grid size
            wavelength: Wavelength (meters)
            pixel_scale: Angular pixel scale (rad/pixel)
            include_spiders: Include spider vane obstruction
            spider_width: Spider vane width (meters)
        
        Returns:
            Pupil instance with VLT geometry
        """
        # VLT parameters
        diameter = 8.2  # meters
        obstruction_diameter = 1.116  # meters
        
        # Create base circular pupil
        pupil = cls.from_circular(
            n_pix=n_pix,
            wavelength=wavelength,
            pixel_scale=pixel_scale,
            diameter=diameter,
            obstruction_diameter=obstruction_diameter,
        )
        
        if include_spiders:
            # Add spider vanes
            pupil = pupil._add_vlt_spiders(spider_width)
        
        # Detect segments for LWE
        pupil = pupil._detect_segments()
        
        pupil.metadata['type'] = 'vlt'
        pupil.metadata['include_spiders'] = include_spiders
        
        return pupil
    
    def _add_vlt_spiders(self, width: float = 0.04) -> 'Pupil':
        """Add VLT-like spider vanes to the pupil."""
        backend = get_backend()
        xp = backend.xp
        
        # VLT spider geometry
        half_opening_deg = 51.3
        attach_angles_deg = [0.0, 180.0]
        
        import math
        half_open_rad = math.radians(half_opening_deg)
        
        X, Y = self.get_coordinate_grid('meters')
        spider_mask = xp.ones_like(self.amplitude)
        
        r_inner = self.obstruction_diameter / 2
        r_outer = self.diameter / 2
        half_w = width / 2
        
        for attach_deg in attach_angles_deg:
            attach_rad = math.radians(attach_deg)
            
            for sign in [-1, +1]:
                vane_angle = attach_rad + sign * half_open_rad
                
                # Line from inner to outer radius
                x0 = r_inner * math.cos(vane_angle)
                y0 = r_inner * math.sin(vane_angle)
                x1 = r_outer * math.cos(vane_angle)
                y1 = r_outer * math.sin(vane_angle)
                
                # Direction vector
                dx = x1 - x0
                dy = y1 - y0
                length = math.sqrt(dx**2 + dy**2)
                dx, dy = dx/length, dy/length
                
                # Normal vector
                nx, ny = -dy, dx
                
                # Distance from line
                dist = xp.abs((X - x0) * nx + (Y - y0) * ny)
                
                # Along-line parameter
                t = (X - x0) * dx + (Y - y0) * dy
                
                # Mask where within vane
                in_vane = (dist < half_w) & (t > -half_w) & (t < length + half_w)
                spider_mask = spider_mask * (~in_vane).astype(xp.float64)
        
        new_amplitude = self.amplitude * spider_mask
        
        return Pupil(
            amplitude=new_amplitude,
            wavelength=self.wavelength,
            pixel_scale=self.pixel_scale,
            diameter=self.diameter,
            obstruction_diameter=self.obstruction_diameter,
            segment_masks=self.segment_masks,
            metadata=self.metadata.copy(),
        )
    
    def _detect_segments(self) -> 'Pupil':
        """Detect disconnected segments in the pupil using connected components."""
        backend = get_backend()
        xp = backend.xp
        
        try:
            from scipy import ndimage
        except ImportError:
            # No scipy, skip segment detection
            return self
        
        amp_np = backend.to_numpy(self.amplitude)
        
        # Binarize with high threshold
        binary = (amp_np > 0.5).astype(np.int32)
        
        # Label connected components
        labeled, n_segments = ndimage.label(binary)
        
        if n_segments <= 1:
            # No segments to store
            return self
        
        # Create segment masks
        segment_masks = {}
        for i in range(1, n_segments + 1):
            mask = (labeled == i).astype(np.float32)
            segment_masks[f'segment_{i}'] = xp.asarray(mask)
        
        return Pupil(
            amplitude=self.amplitude,
            wavelength=self.wavelength,
            pixel_scale=self.pixel_scale,
            diameter=self.diameter,
            obstruction_diameter=self.obstruction_diameter,
            segment_masks=segment_masks,
            metadata=self.metadata.copy(),
        )
    
    # =========================================================================
    # Display and diagnostics
    # =========================================================================
    
    def __repr__(self) -> str:
        seg_str = f", {self.n_segments} segments" if self.n_segments > 0 else ""
        return (
            f"Pupil(n_pix={self.n_pix}, D={self.diameter}m, "
            f"D_obs={self.obstruction_diameter}m, λ={self.wavelength*1e6:.2f}μm, "
            f"scale={self.pixel_scale*206265*1000:.1f}mas/pix{seg_str})"
        )
    
    def summary(self) -> str:
        """Return detailed summary string."""
        lines = [
            "Pupil Summary",
            "=" * 40,
            f"Grid size:         {self.n_pix} × {self.n_pix} pixels",
            f"Wavelength:        {self.wavelength * 1e6:.3f} μm",
            f"Pixel scale:       {self.pixel_scale * 206265 * 1000:.2f} mas/pixel",
            f"Pupil pixel size:  {self.pupil_pixel_size * 1000:.3f} mm",
            f"Pupil extent:      {self.pupil_extent:.2f} m",
            f"",
            f"Diameter:          {self.diameter:.3f} m ({self.diameter_pixels:.1f} pixels)",
            f"Obstruction:       {self.obstruction_diameter:.3f} m ({self.obstruction_pixels:.1f} pixels)",
            f"Filling factor:    {self.filling_factor * 100:.1f}%",
            f"",
            f"Segments:          {self.n_segments}",
        ]
        if self.metadata:
            lines.append(f"Metadata:          {self.metadata}")
        return "\n".join(lines)
    
    def plot(self, ax=None, show_segments: bool = False, **kwargs):
        """
        Plot the pupil amplitude.
        
        Args:
            ax: Matplotlib axes (created if None)
            show_segments: Overlay segment boundaries
            **kwargs: Additional arguments for imshow
        
        Returns:
            matplotlib axes
        """
        import matplotlib.pyplot as plt
        
        backend = get_backend()
        amp = backend.to_numpy(self.amplitude)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        extent_m = self.pupil_extent / 2
        extent = [-extent_m, extent_m, -extent_m, extent_m]
        
        im = ax.imshow(amp, extent=extent, origin='lower', cmap='gray', **kwargs)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'Pupil (D={self.diameter}m)')
        plt.colorbar(im, ax=ax, label='Amplitude')
        
        if show_segments and self.segment_masks:
            # Overlay segment boundaries
            for name, mask in self.segment_masks.items():
                mask_np = backend.to_numpy(mask)
                ax.contour(mask_np, levels=[0.5], extent=extent, colors='r', linewidths=0.5)
        
        return ax


# =============================================================================
# Convenience function for creating pupils
# =============================================================================

def create_pupil(
    n_pix: int,
    wavelength: float,
    pixel_scale: float,
    diameter: float,
    obstruction_diameter: float = 0.0,
    telescope: Optional[str] = None,
    **kwargs,
) -> Pupil:
    """
    Convenience function to create a Pupil.
    
    Args:
        n_pix: Grid size
        wavelength: Wavelength (meters)
        pixel_scale: Angular pixel scale (rad/pixel)
        diameter: Primary diameter (meters)
        obstruction_diameter: Central obstruction diameter (meters)
        telescope: Optional telescope name ('vlt', 'elt', etc.)
        **kwargs: Additional arguments for specific telescope types
    
    Returns:
        Pupil instance
    
    Examples:
        # Simple circular pupil
        p = create_pupil(256, 2.2e-6, 13e-3/206265, 8.2, 1.116)
        
        # VLT with spiders
        p = create_pupil(256, 2.2e-6, 13e-3/206265, 8.2, telescope='vlt')
    """
    if telescope:
        telescope = telescope.lower()
        if telescope == 'vlt':
            return Pupil.from_vlt(n_pix, wavelength, pixel_scale, **kwargs)
        else:
            raise ValueError(f"Unknown telescope: {telescope}")
    
    return Pupil.from_circular(
        n_pix=n_pix,
        wavelength=wavelength,
        pixel_scale=pixel_scale,
        diameter=diameter,
        obstruction_diameter=obstruction_diameter,
        **kwargs,
    )

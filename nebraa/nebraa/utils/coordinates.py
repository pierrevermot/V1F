"""
Coordinate grid utilities for consistent centering conventions.

This module provides standardized coordinate grid generation to eliminate
half-pixel centering inconsistencies across the codebase.

Convention: Pixel-centered grids
--------------------------------
For an n_pix x n_pix array, pixel centers are indexed [0, 1, ..., n_pix-1].
The center of the array is at index (n_pix-1)/2, giving coordinates:

    coordinates = [-(n_pix-1)/2, ..., -1, 0, 1, ..., (n_pix-1)/2]

For even n_pix, the center falls between pixels (e.g., n=4 → center at 1.5).
For odd n_pix, the center is exactly on a pixel (e.g., n=5 → center at 2).

This convention:
- Aligns with pixel indexing (center is the mean of first/last indices)
- Produces symmetric grids for symmetric inputs
- Matches standard practice in astronomical image processing
- Is consistent with FFT centering (fftshift centers at (n-1)/2)

Example:
    n_pix = 5
    center = (5-1)/2 = 2.0
    indices = [0, 1, 2, 3, 4]
    coordinates = [0-2, 1-2, 2-2, 3-2, 4-2] = [-2, -1, 0, 1, 2]  ✓ symmetric
    
    n_pix = 4  
    center = (4-1)/2 = 1.5
    indices = [0, 1, 2, 3]
    coordinates = [0-1.5, 1-1.5, 2-1.5, 3-1.5] = [-1.5, -0.5, 0.5, 1.5]  ✓ symmetric

Compare with incorrect n_pix/2:
    n_pix = 4
    center = 4/2 = 2.0
    coordinates = [0-2, 1-2, 2-2, 3-2] = [-2, -1, 0, 1]  ✗ asymmetric!
"""

from typing import Tuple, Optional, Literal
import numpy as np
from .compute import get_backend

ArrayLike = object  # Type alias for backend arrays


def get_grid_center(n_pix: int) -> float:
    """
    Get the center coordinate for a pixel-centered grid.
    
    Args:
        n_pix: Grid size
    
    Returns:
        Center coordinate in pixel units
    
    Example:
        >>> get_grid_center(5)  # Odd: center on pixel
        2.0
        >>> get_grid_center(4)  # Even: center between pixels
        1.5
    """
    return (n_pix - 1) / 2.0


def get_pixel_radius(n_pix: int) -> float:
    """
    Get the nominal radius for a pixel-centered grid.
    
    This is the radius from center to edge, useful for normalizing
    radial coordinates to [0, 1] at the pupil edge.
    
    Args:
        n_pix: Grid size
    
    Returns:
        Radius in pixel units (distance from center to edge)
    
    Example:
        >>> get_pixel_radius(5)  # Odd: center to edge
        2.0
        >>> get_pixel_radius(4)  # Even
        1.5
    """
    return (n_pix - 1) / 2.0


def make_1d_grid(
    n_pix: int,
    pixel_size: float = 1.0,
    center: Literal['pixel', 'corner'] = 'pixel',
    backend: Optional[str] = None,
) -> ArrayLike:
    """
    Create 1D coordinate grid.
    
    Args:
        n_pix: Number of pixels
        pixel_size: Physical size per pixel
        center: 'pixel' for pixel-centered [-(n-1)/2, ..., (n-1)/2], 
                'corner' for corner-centered [0, ..., n-1]
        backend: Backend to use (None for current)
    
    Returns:
        1D coordinate array
    
    Example:
        >>> make_1d_grid(5, pixel_size=0.1, center='pixel')
        array([-0.2, -0.1,  0.0,  0.1,  0.2])
        >>> make_1d_grid(4, pixel_size=0.1, center='corner')
        array([0.0, 0.1, 0.2, 0.3])
    """
    from .compute import init_backend
    if backend:
        init_backend(backend)
    
    xp = get_backend().xp
    
    idx = xp.arange(n_pix, dtype=xp.float64)
    
    if center == 'pixel':
        coords = (idx - (n_pix - 1) / 2.0) * pixel_size
    elif center == 'corner':
        coords = idx * pixel_size
    else:
        raise ValueError(f"Unknown center type: {center}. Use 'pixel' or 'corner'")
    
    return coords


def make_xy_grid(
    n_pix: int,
    pixel_size: float = 1.0,
    center: Literal['pixel', 'corner'] = 'pixel',
    backend: Optional[str] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Create 2D Cartesian coordinate grids (X, Y).
    
    This is the standard way to create coordinate grids for pupil functions,
    Zernike polynomials, and other 2D operations.
    
    Args:
        n_pix: Grid size (square array)
        pixel_size: Physical size per pixel
        center: 'pixel' for pixel-centered grids (recommended),
                'corner' for corner-centered [0, ..., n-1]
        backend: Backend to use (None for current)
    
    Returns:
        (X, Y) coordinate grids using default 'xy' indexing where:
            - X[i,j] varies along j (columns), constant along i (rows)
            - Y[i,j] varies along i (rows), constant along j (columns)
        This matches standard numpy meshgrid and existing code.
    
    Example:
        >>> X, Y = make_xy_grid(3, pixel_size=1.0)
        >>> X[0, :]  # First row - varies in x
        array([-1.,  0.,  1.])
        >>> Y[:, 0]  # First column - varies in y
        array([-1.,  0.,  1.])
    """
    from .compute import init_backend
    if backend:
        init_backend(backend)
    
    xp = get_backend().xp
    
    coords = make_1d_grid(n_pix, pixel_size, center)
    # Use 'xy' indexing (default) to match existing behavior
    X, Y = xp.meshgrid(coords, coords)
    
    return X, Y


def make_polar_grid(
    n_pix: int,
    pixel_size: float = 1.0,
    center: Literal['pixel', 'corner'] = 'pixel',
    normalize_radius: Optional[float] = None,
    backend: Optional[str] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Create 2D polar coordinate grids (rho, theta).
    
    Args:
        n_pix: Grid size
        pixel_size: Physical size per pixel  
        center: Centering convention
        normalize_radius: If provided, normalize rho so rho=1 at this radius
        backend: Backend to use (None for current)
    
    Returns:
        (rho, theta) coordinate grids
            rho: Radius from center
            theta: Angle in radians, [-π, π], zero at +x axis
    
    Example:
        >>> rho, theta = make_polar_grid(5, normalize_radius=2.0)
        >>> rho[2, 2]  # Center
        0.0
        >>> rho[2, 4]  # Edge
        1.0
    """
    from .compute import init_backend
    if backend:
        init_backend(backend)
    
    xp = get_backend().xp
    
    X, Y = make_xy_grid(n_pix, pixel_size, center)
    
    rho = xp.sqrt(X**2 + Y**2)
    theta = xp.arctan2(Y, X)
    
    if normalize_radius is not None:
        rho = rho / normalize_radius
    
    return rho, theta


def make_frequency_grid(
    n_pix: int,
    pixel_size: float,
    backend: Optional[str] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Create 2D frequency grids for FFT operations.
    
    Uses FFT-shifted frequency convention: DC at center, frequencies
    ordered from -Nyquist to +Nyquist.
    
    Args:
        n_pix: Grid size
        pixel_size: Physical pixel size (spatial domain)
        backend: Backend to use (None for current)
    
    Returns:
        (FX, FY) frequency grids in cycles per unit length
    
    Note:
        Frequency spacing is df = 1 / (n_pix * pixel_size)
        Nyquist frequency is ±1 / (2 * pixel_size)
    
    Example:
        >>> FX, FY = make_frequency_grid(4, pixel_size=0.1)
        >>> # Frequencies: [-5, -2.5, 0, 2.5] cycles/unit
    """
    from .compute import init_backend
    if backend:
        init_backend(backend)
    
    xp = get_backend().xp
    
    # Frequency grid using FFT convention
    freq = xp.fft.fftshift(xp.fft.fftfreq(n_pix, pixel_size))
    FX, FY = xp.meshgrid(freq, freq, indexing='xy')
    
    return FX, FY


def check_grid_symmetry(
    grid: ArrayLike,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> bool:
    """
    Check if a 2D grid is symmetric (for testing centering).
    
    Args:
        grid: 2D array to check
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        True if grid is symmetric about center
    
    Example:
        >>> X, Y = make_xy_grid(5)
        >>> check_grid_symmetry(X)
        True
        >>> check_grid_symmetry(X**2 + Y**2)  # Radial function
        True
    """
    xp = get_backend().xp
    
    # Check square
    if grid.shape[0] != grid.shape[1]:
        return False
    
    # Convert to numpy if needed
    if hasattr(grid, 'get'):  # CuPy array
        grid_np = grid.get()
    elif hasattr(grid, '__array__'):  # Generic array-like
        grid_np = np.asarray(grid)
    else:
        grid_np = grid
    
    # Flip both axes and compare
    flipped = np.flip(np.flip(grid_np, axis=0), axis=1)
    
    return np.allclose(grid_np, flipped, rtol=rtol, atol=atol)


# Backward compatibility: export old names
def get_coordinate_grid(*args, **kwargs):
    """Deprecated: Use make_xy_grid instead."""
    import warnings
    warnings.warn(
        "get_coordinate_grid is deprecated, use make_xy_grid",
        DeprecationWarning,
        stacklevel=2,
    )
    return make_xy_grid(*args, **kwargs)

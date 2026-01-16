"""
Detector noise models.

Provides functions for simulating realistic detector noise including
photon noise (Poisson), read noise (Gaussian), and background.
"""

from __future__ import annotations

from ..utils.compute import get_backend


# =============================================================================
# Noise Sources
# =============================================================================

def add_photon_noise(image, peak_flux: float):
    """
    Add Poisson (photon) noise to image.
    
    Args:
        image: Input image, normalized to [0, 1]
        peak_flux: Peak photon count
    
    Returns:
        Noisy image (in photon counts)
    """
    backend = get_backend()
    xp = backend.xp
    
    # Scale to expected counts
    expected = image.astype(xp.float64) * peak_flux
    
    # Poisson sampling
    noisy = xp.random.poisson(expected).astype(xp.float32)
    
    return noisy


def add_read_noise(image, sigma: float):
    """
    Add Gaussian read noise.
    
    Args:
        image: Input image
        sigma: Read noise standard deviation
    
    Returns:
        Image with read noise added
    """
    backend = get_backend()
    xp = backend.xp
    
    noise = xp.random.randn(*image.shape).astype(xp.float32) * sigma
    return image + noise


def add_background(image, level: float):
    """
    Add uniform background level.
    
    Args:
        image: Input image
        level: Background level per pixel
    
    Returns:
        Image with background added
    """
    return image + level


# =============================================================================
# SNR Computation
# =============================================================================

def compute_snr(signal, background: float, read_noise: float):
    """
    Compute signal-to-noise ratio.
    
    SNR = signal / sqrt(signal + background + read_noise^2)
    
    Args:
        signal: Signal counts
        background: Background counts
        read_noise: Read noise standard deviation
    
    Returns:
        SNR value(s)
    """
    backend = get_backend()
    xp = backend.xp
    
    variance = signal + background + read_noise**2
    variance = xp.maximum(variance, xp.float32(1e-10))
    
    return signal / xp.sqrt(variance)


def compute_peak_snr(psf, peak_flux: float, background: float, read_noise: float):
    """
    Compute peak SNR for a PSF.
    
    Args:
        psf: PSF array (normalized to unit peak)
        peak_flux: Peak signal level
        background: Background level
        read_noise: Read noise sigma
    
    Returns:
        Peak SNR value
    """
    backend = get_backend()
    xp = backend.xp
    
    signal = psf * peak_flux
    snr_map = compute_snr(signal, background, read_noise)
    
    return float(xp.max(snr_map))


# =============================================================================
# Complete Noise Simulation
# =============================================================================

def simulate_observation(
    psf,
    peak_flux: float = 1e4,
    background: float = 100.0,
    read_noise: float = 5.0,
    normalize_output: bool = True,
):
    """
    Simulate realistic observation from ideal PSF.
    
    Applies:
    1. Scale PSF to peak flux
    2. Add background
    3. Poisson (shot) noise
    4. Read noise
    5. Optional normalization
    
    Args:
        psf: Input PSF (normalized to unit peak)
        peak_flux: Peak signal in photons
        background: Background level per pixel
        read_noise: Read noise sigma
        normalize_output: If True, normalize output to [0, 1]
    
    Returns:
        Simulated observation
    """
    backend = get_backend()
    xp = backend.xp
    
    # Scale to photons
    signal = psf.astype(xp.float32) * peak_flux
    
    # Add background
    with_bg = signal + background
    
    # Poisson noise
    noisy = xp.random.poisson(with_bg.astype(xp.float64)).astype(xp.float32)
    
    # Read noise
    noisy = noisy + xp.random.randn(*noisy.shape).astype(xp.float32) * read_noise
    
    # Ensure non-negative
    observation = xp.maximum(noisy, xp.float32(0))
    
    # Normalize
    if normalize_output:
        observation = normalize_to_range(observation)
    
    return observation


def simulate_observations_batch(
    psfs,
    peak_flux: float = 1e4,
    background: float = 100.0,
    read_noise: float = 5.0,
    normalize_output: bool = True,
):
    """
    Simulate observations for multiple PSFs.
    
    Args:
        psfs: Array of PSFs (n, H, W)
        peak_flux, background, read_noise: Noise parameters
        normalize_output: If True, normalize each to [0, 1]
    
    Returns:
        Array of observations (n, H, W)
    """
    backend = get_backend()
    xp = backend.xp
    
    n = psfs.shape[0]
    
    # Scale to photons
    signal = psfs.astype(xp.float32) * peak_flux
    
    # Add background
    with_bg = signal + background
    
    # Poisson noise
    noisy = xp.random.poisson(with_bg.astype(xp.float64)).astype(xp.float32)
    
    # Read noise
    noisy = noisy + xp.random.randn(*noisy.shape).astype(xp.float32) * read_noise
    
    # Ensure non-negative
    observations = xp.maximum(noisy, xp.float32(0))
    
    # Normalize each image
    if normalize_output:
        for i in range(n):
            observations[i] = normalize_to_range(observations[i])
    
    return observations


# =============================================================================
# Normalization Utilities
# =============================================================================

def normalize_to_range(image, vmin: float = 0.0, vmax: float = 1.0):
    """
    Normalize image to specified range.
    
    Args:
        image: Input image
        vmin: Output minimum
        vmax: Output maximum
    
    Returns:
        Normalized image
    """
    backend = get_backend()
    xp = backend.xp
    
    img_min = xp.min(image)
    img_max = xp.max(image)
    
    scale = img_max - img_min
    if scale < 1e-10:
        return xp.full_like(image, (vmin + vmax) / 2)
    
    normalized = (image - img_min) / scale
    return normalized * (vmax - vmin) + vmin


def standardize(image):
    """
    Standardize to zero mean and unit variance.
    
    Args:
        image: Input image
    
    Returns:
        Standardized image
    """
    backend = get_backend()
    xp = backend.xp
    
    mean = xp.mean(image)
    std = xp.std(image)
    
    if std < 1e-10:
        return image - mean
    
    return (image - mean) / std


# =============================================================================
# Visualization Stretches
# =============================================================================

def log_stretch(image, a: float = 1000.0):
    """
    Apply logarithmic stretch for visualization.
    
    f(x) = log(1 + a*x) / log(1 + a)
    
    Args:
        image: Image array (assumed [0, 1])
        a: Stretch parameter
    
    Returns:
        Stretched image
    """
    backend = get_backend()
    xp = backend.xp
    
    image = xp.maximum(image, xp.float32(0))
    return xp.log1p(a * image) / xp.log1p(xp.float32(a))


def asinh_stretch(image, a: float = 1.0):
    """
    Apply asinh stretch for visualization.
    
    Args:
        image: Image array
        a: Stretch parameter
    
    Returns:
        Stretched image
    """
    backend = get_backend()
    xp = backend.xp
    
    return xp.arcsinh(a * image) / xp.arcsinh(xp.float32(a))

"""
Shared physics modules for optical simulation.

This package contains physical models that are shared across different
instruments and simulation modes:

- zernike: Zernike polynomial wavefront representation
- kolmogorov: Atmospheric turbulence models
- optics: Basic optical propagation (PSF computation, Fourier optics)
- noise: Detector noise models
"""

from .zernike import (
    zernike_nm,
    build_zernike_modes,
    generate_zernike_phase,
    ZernikeModeCache,
)

from .kolmogorov import (
    kolmogorov_psd,
    FrequencyGrid,
    KolmogorovGenerator,
    raised_cosine_filter,
)

from .optics import (
    compute_psf,
    compute_psf_batch,
    compute_strehl,
    compute_rms_phase,
)

from .noise import (
    add_photon_noise,
    add_read_noise,
    add_background,
    compute_snr,
)

__all__ = [
    # Zernike
    'zernike_nm',
    'build_zernike_modes',
    'generate_zernike_phase',
    'ZernikeModeCache',
    # Kolmogorov
    'kolmogorov_psd',
    'FrequencyGrid', 
    'KolmogorovGenerator',
    'raised_cosine_filter',
    # Optics
    'compute_psf',
    'compute_psf_batch',
    'compute_strehl',
    'compute_rms_phase',
    # Noise
    'add_photon_noise',
    'add_read_noise',
    'add_background',
    'compute_snr',
]

"""
Shared physics modules for optical simulation.

This package contains physical models that are shared across different
instruments and simulation modes:

- phase_generator: Unified interface for phase screen generators
- psf_engine: Unified PSF computation from phase screens
- zernike: Zernike polynomial wavefront representation
- kolmogorov: Atmospheric turbulence models
- optics: Basic optical propagation (PSF computation, Fourier optics)
- noise: Detector noise models
- jolissaint_ao: Analytical AO model for long-exposure PSF (Jolissaint et al. 2006)
- powerlaw_psd: Dual power-law PSD phase generator
- low_wind_effect: Low Wind Effect model
"""

# Unified interface components
from .phase_generator import (
    PhaseGeneratorBase,
    PhaseGeneratorConfig,
    PhaseScreenResult,
)

from .psf_engine import (
    PSFEngine,
    PSFEngineConfig,
    compute_psf,
    compute_long_exposure_psf,
    compute_strehl_ratio,
)

from .zernike import (
    zernike_nm,
    build_zernike_modes,
    generate_zernike_phase,
    ZernikeModeCache,
    # Unified interface
    ZernikeConfig,
    ZernikePhaseGenerator,
)

from .kolmogorov import (
    kolmogorov_psd,
    FrequencyGrid,
    KolmogorovGenerator,
    raised_cosine_filter,
)

from .optics import (
    # Keep legacy optics functions for backward compatibility
    compute_psf as _compute_psf_legacy,
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

from .jolissaint_ao import (
    TurbulentLayer,
    AtmosphereProfile,
    AOSystemConfig,
    JolissaintAOModel,
    create_simple_atmosphere,
    create_mauna_kea_atmosphere,
    create_ao_config,
)

from .low_wind_effect import (
    LowWindEffectConfig,
    LowWindEffect,
)

from .powerlaw_psd import (
    DualPowerLawConfig,
    DualPowerLawPSD,
    DualPowerLawPhaseGenerator,
    DualPowerLawPSFModel,
    create_simple_psd_config,
)

from .pupil import (
    Pupil,
    create_pupil,
)

__all__ = [
    # Pupil
    'Pupil',
    'create_pupil',
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
    # Jolissaint AO model
    'TurbulentLayer',
    'AtmosphereProfile',
    'AOSystemConfig',
    'JolissaintAOModel',
    'create_simple_atmosphere',
    'create_mauna_kea_atmosphere',
    'create_ao_config',
    # Low Wind Effect
    'LowWindEffectConfig',
    'LowWindEffect',
    # Dual Power-Law PSD
    'DualPowerLawConfig',
    'DualPowerLawPSD',
    'DualPowerLawPhaseGenerator',
    'DualPowerLawPSFModel',
    'create_simple_psd_config',
    # Unified Interface - Phase Generators
    'PhaseGeneratorBase',
    'PhaseGeneratorConfig',
    'PhaseScreenResult',
    'ZernikeConfig',
    'ZernikePhaseGenerator',
    # Unified Interface - PSF Engine
    'PSFEngine',
    'PSFEngineConfig',
    'compute_psf',
    'compute_long_exposure_psf',
    'compute_strehl_ratio',
]

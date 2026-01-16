"""
Centralized configuration system.

This module provides a hierarchical, type-validated configuration system
using dataclasses. Supports YAML/JSON loading and experiment reproducibility.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

import yaml


# =============================================================================
# Source Generation Configuration
# =============================================================================

@dataclass
class SourceConfig:
    """Configuration for astronomical source image generation."""
    
    # Generation mode
    mode: Literal['fourier', 'fourier_ps', 'fourier_ifu', '3d'] = 'fourier_ps'
    
    # Output parameters
    n_files: int = 4096
    n_images_per_file: int = 256
    
    # Image content
    max_objects_per_image: int = 40
    max_std_threshold: float = 5.0
    max_point_sources: int = 10  # For fourier_ps mode
    
    # Compute settings
    compute_mode: Literal['CPU', 'GPU'] = 'GPU'
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.n_files > 0, "n_files must be positive"
        assert self.n_images_per_file > 0, "n_images_per_file must be positive"


# =============================================================================
# Instrument Configuration
# =============================================================================

@dataclass
class TelescopeConfig:
    """Telescope physical parameters."""
    
    primary_diameter: float = 8.2      # meters
    obstruction_diameter: float = 1.116  # meters (secondary/central)
    
    # Spider vanes
    spider_model: Optional[Literal['VLT_LIKE']] = 'VLT_LIKE'
    spider_width: float = 0.04  # meters
    spider_half_opening_deg: float = 51.3
    spider_attach_angles_deg: List[float] = field(default_factory=lambda: [0.0, 180.0])


@dataclass
class AtmosphereConfig:
    """Atmospheric turbulence configuration."""
    
    # Low-frequency (Zernike-based AO residuals)
    zernike_n_min: int = 2
    zernike_n_max: int = 5
    power_law_exponent: float = 2.0
    rms_min: float = 50e-9   # meters
    rms_max: float = 1e-6    # meters (default ~lam/5)
    
    # High-frequency (Kolmogorov above AO cutoff)
    hf_enable: bool = True
    actuator_pitch: float = 0.20  # meters
    r0_min: float = 0.05  # meters
    r0_max: float = 0.30  # meters
    hf_transition: float = 0.15  # smooth cutoff width
    
    # Low Wind Effect
    lwe_enable: bool = True
    lwe_piston_rms_rad: float = 0.5
    lwe_tilt_rms_rad: float = 0.3
    lwe_ar_coeff: float = 0.95


@dataclass 
class NoiseConfig:
    """Detector noise configuration."""
    
    peak_flux: float = 1e4       # photons
    background_level: float = 100.0  # photons/pixel
    read_noise: float = 5.0      # electrons
    snr_min: float = 0.0
    snr_max: float = 50.0


@dataclass
class InstrumentConfig:
    """Complete instrument simulation configuration."""
    
    # Instrument identification
    name: str = 'vlt'
    
    # Grid parameters
    n_pix: int = 512         # Computation grid
    n_pix_output: int = 128  # Output PSF size
    n_frames: int = 32       # Temporal frames
    
    # Optical parameters
    wavelength: float = 4.78e-6  # meters
    pixel_scale: float = 0.02719 / 206265  # rad/pixel
    
    # Sub-configurations
    telescope: TelescopeConfig = field(default_factory=TelescopeConfig)
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    
    # Compute settings
    compute_mode: Literal['CPU', 'GPU'] = 'GPU'
    batch_size: int = 100
    
    def __post_init__(self):
        """Set derived parameters."""
        if self.atmosphere.rms_max <= 0:
            self.atmosphere.rms_max = self.wavelength / 5


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Neural network training configuration."""
    
    # Model selection
    model_name: str = 'unet'
    
    # Training parameters
    batch_size: int = 16
    epochs: int = 2500
    learning_rate: float = 5e-4
    clipnorm: float = 5.0
    
    # Loss and activation
    loss: str = 'mean_absolute_error'
    inner_activation: str = 'softplus'
    output_activation: str = 'linear'
    
    # Precision
    float_precision: Literal['float16', 'float32', 'float64', 'mixed'] = 'float32'
    
    # Validation
    test_dataset_length: int = 1000
    
    # Visualization
    plot_predictions: bool = True
    observation_paths: List[str] = field(default_factory=list)
    observation_names: List[str] = field(default_factory=list)
    
    # Callbacks
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    reduce_lr_min: float = 5e-6


# =============================================================================
# Data Pipeline Configuration
# =============================================================================

@dataclass
class DataConfig:
    """Data generation and storage configuration."""
    
    # Regeneration policy: 'always', 'if_missing', 'never'
    regen_images: Literal['always', 'if_missing', 'never'] = 'if_missing'
    regen_observations: Literal['always', 'if_missing', 'never'] = 'if_missing'
    regen_tfrecords: Literal['always', 'if_missing', 'never'] = 'if_missing'
    
    # Parallelization
    n_workers: Optional[int] = None  # None = auto-detect


# =============================================================================
# Path Configuration
# =============================================================================

@dataclass
class PathConfig:
    """Directory structure configuration."""
    
    # Base directories
    scratch_dir: str = field(default_factory=lambda: os.environ.get('SCRATCH', '/tmp'))
    output_dir: str = './outputs'
    
    # Experiment naming
    name_images: str = 'fpm_128'
    name_observations: str = 'NACO_M'
    
    # Derived paths (set in __post_init__)
    images_dir: str = ''
    observations_dir: str = ''
    tfrecords_dir: str = ''
    run_dir: str = ''
    log_dir: str = ''
    model_dir: str = ''
    plot_dir: str = ''
    
    def __post_init__(self):
        """Compute derived paths."""
        tag = f'{self.name_images}_{self.name_observations}'
        self.images_dir = os.path.join(self.scratch_dir, self.name_images)
        self.observations_dir = os.path.join(self.scratch_dir, tag)
        self.tfrecords_dir = os.path.join(self.observations_dir, 'tfrecords')
        
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_dir = os.path.join(self.output_dir, timestamp)
        self.log_dir = os.path.join(self.run_dir, 'logs')
        self.model_dir = os.path.join(self.run_dir, 'model')
        self.plot_dir = os.path.join(self.run_dir, 'plots')
    
    def ensure_dirs(self):
        """Create all necessary directories."""
        for d in [self.images_dir, self.observations_dir, self.tfrecords_dir,
                  self.output_dir, self.run_dir, self.log_dir, 
                  self.model_dir, self.plot_dir]:
            os.makedirs(d, exist_ok=True)


# =============================================================================
# Master Configuration
# =============================================================================

@dataclass
class Config:
    """Master configuration combining all sub-configurations."""
    
    # Run mode: what stages to execute
    run_mode: Literal['full', 'generate_only', 'train'] = 'full'
    
    # Sub-configurations
    source: SourceConfig = field(default_factory=SourceConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Grid dimensions (shared across modules)
    n_pix: int = 128
    n_frames_in: int = 32
    n_frames_out: int = 1
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary, handling nested dataclasses."""
        # Extract sub-configs
        source_data = data.pop('source', {})
        instrument_data = data.pop('instrument', {})
        training_data = data.pop('training', {})
        data_config = data.pop('data', {})
        paths_data = data.pop('paths', {})
        
        # Handle nested instrument config
        telescope_data = instrument_data.pop('telescope', {})
        atmosphere_data = instrument_data.pop('atmosphere', {})
        noise_data = instrument_data.pop('noise', {})
        
        return cls(
            source=SourceConfig(**source_data),
            instrument=InstrumentConfig(
                telescope=TelescopeConfig(**telescope_data),
                atmosphere=AtmosphereConfig(**atmosphere_data),
                noise=NoiseConfig(**noise_data),
                **instrument_data
            ),
            training=TrainingConfig(**training_data),
            data=DataConfig(**data_config),
            paths=PathConfig(**paths_data),
            **data
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_json(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self):
        """Validate entire configuration."""
        assert self.n_pix > 0, "n_pix must be positive"
        assert self.n_frames_in > 0, "n_frames_in must be positive"
        # Add more validation as needed


# =============================================================================
# Convenience Functions
# =============================================================================

def load_config(path: str) -> Config:
    """Load configuration from file (auto-detects format)."""
    path = str(path)
    if path.endswith('.yaml') or path.endswith('.yml'):
        return Config.from_yaml(path)
    elif path.endswith('.json'):
        return Config.from_json(path)
    else:
        raise ValueError(f"Unknown config format: {path}")


def save_config(config: Config, path: str):
    """Save configuration to file (auto-detects format)."""
    path = str(path)
    if path.endswith('.yaml') or path.endswith('.yml'):
        config.to_yaml(path)
    elif path.endswith('.json'):
        config.to_json(path)
    else:
        raise ValueError(f"Unknown config format: {path}")


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


# =============================================================================
# Instrument Presets (migrated from legacy overrides)
# =============================================================================

INSTRUMENT_PRESETS: Dict[str, Dict[str, Any]] = {
    'vlt_l': {
        'wavelength': 4.78e-6,
        'pixel_scale': 0.02719 / 206265,
        'telescope': {'primary_diameter': 8.2, 'obstruction_diameter': 1.116},
    },
    'vlt_k': {
        'wavelength': 2.18e-6,
        'pixel_scale': 0.013 / 206265,
        'telescope': {'primary_diameter': 8.2, 'obstruction_diameter': 1.116},
    },
    'lbti': {
        'wavelength': 8.699e-6,
        'pixel_scale': 0.018 / 206265,
        'n_pix_output': 100,
        'telescope': {'primary_diameter': 8.4, 'obstruction_diameter': 0.9},
    },
}


def apply_preset(config: Config, preset_name: str) -> Config:
    """Apply an instrument preset to configuration."""
    if preset_name not in INSTRUMENT_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(INSTRUMENT_PRESETS.keys())}")
    
    preset = INSTRUMENT_PRESETS[preset_name]
    
    # Apply instrument-level settings
    for key, value in preset.items():
        if key == 'telescope':
            for tk, tv in value.items():
                setattr(config.instrument.telescope, tk, tv)
        elif key == 'atmosphere':
            for ak, av in value.items():
                setattr(config.instrument.atmosphere, ak, av)
        elif key == 'noise':
            for nk, nv in value.items():
                setattr(config.instrument.noise, nk, nv)
        else:
            setattr(config.instrument, key, value)
    
    return config

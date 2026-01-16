"""
NEBRAA: Neural Enhanced Bayesian Reconstruction for Astronomical Algorithms
============================================================================

A modular framework for astronomical image simulation and neural network-based
deconvolution, particularly focused on adaptive optics systems.

Package Structure
-----------------
- config: Centralized configuration system with type validation
- physics: Shared physical models (Zernike, Kolmogorov, optics)
- instruments: Telescope-specific PSF simulation
- sources: Astronomical source image generation
- models: Neural network architectures
- data: Data I/O and TFRecord handling
- utils: Compute backends, logging, distributed execution

Quick Start
-----------
    from nebraa import Config, Pipeline
    
    cfg = Config.from_yaml('experiment.yaml')
    pipeline = Pipeline(cfg)
    pipeline.run()

Or run from command line:
    python -m nebraa --config experiment.yaml
"""

__version__ = '2.0.0'
__author__ = 'Pierre Vermot'

# Re-export main interfaces
from .config import (
    Config,
    SourceConfig,
    InstrumentConfig,
    TrainingConfig,
    load_config,
    save_config,
)
from .pipeline import Pipeline
from .utils.logging import get_logger

# Lazy imports for optional heavy dependencies
def get_simulator(instrument_name: str):
    """Factory function to get instrument simulator."""
    from .instruments import get_instrument
    return get_instrument(instrument_name)

def get_model(model_name: str):
    """Factory function to get neural network model."""
    from .models import get_model as _get_model
    return _get_model(model_name)

__all__ = [
    'Config',
    'SourceConfig', 
    'InstrumentConfig',
    'TrainingConfig',
    'Pipeline',
    'get_simulator',
    'get_model',
    'get_logger',
    'load_config',
    'save_config',
]

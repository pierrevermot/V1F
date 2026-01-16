"""
Instrument simulation package.

Provides telescope-specific PSF simulation with standardized interface.
Each instrument (VLT, VLTI, LBTI, etc.) implements the base Instrument
class with its specific optical configuration.
"""

from __future__ import annotations

from typing import Type, Dict
from abc import ABC, abstractmethod

from ..config import InstrumentConfig


class Instrument(ABC):
    """
    Abstract base class for telescope instruments.
    
    All instrument implementations must provide:
    - Pupil generation (with optional spiders)
    - PSF computation
    - Observation simulation with noise
    """
    
    def __init__(self, config: InstrumentConfig):
        """
        Initialize instrument.
        
        Args:
            config: Instrument configuration
        """
        self.config = config
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """Setup instrument-specific components (pupil, atmosphere, etc.)."""
        pass
    
    @abstractmethod
    def generate_psf(self, **kwargs):
        """Generate a single PSF."""
        pass
    
    @abstractmethod
    def generate_psfs(self, n: int, **kwargs):
        """Generate multiple PSFs."""
        pass
    
    @abstractmethod
    def generate_observation(self, psf, **kwargs):
        """Generate noisy observation from PSF."""
        pass
    
    @abstractmethod
    def generate_observations(self, psfs, **kwargs):
        """Generate noisy observations from PSFs."""
        pass
    
    @property
    @abstractmethod
    def pupil(self):
        """Get pupil amplitude array."""
        pass


# Instrument registry
_INSTRUMENTS: Dict[str, Type[Instrument]] = {}


def register_instrument(name: str):
    """
    Decorator to register an instrument class.
    
    Usage:
        @register_instrument('vlt')
        class VLTInstrument(Instrument):
            ...
    """
    def decorator(cls: Type[Instrument]):
        _INSTRUMENTS[name.lower()] = cls
        return cls
    return decorator


def get_instrument(name: str, config: InstrumentConfig = None) -> Instrument:
    """
    Get instrument instance by name.
    
    Args:
        name: Instrument name ('vlt', 'vlti', etc.)
        config: Optional configuration (uses defaults if None)
    
    Returns:
        Instrument instance
    """
    name = name.lower()
    
    if name not in _INSTRUMENTS:
        available = list(_INSTRUMENTS.keys())
        raise ValueError(f"Unknown instrument '{name}'. Available: {available}")
    
    if config is None:
        config = InstrumentConfig(name=name)
    
    return _INSTRUMENTS[name](config)


def list_instruments():
    """List available instrument names."""
    return list(_INSTRUMENTS.keys())


# Import instrument implementations to register them
from . import vlt
from . import lbti

__all__ = [
    'Instrument',
    'register_instrument',
    'get_instrument',
    'list_instruments',
]

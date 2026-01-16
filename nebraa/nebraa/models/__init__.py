"""
Neural network models for image reconstruction.

Provides model architectures for PSF deconvolution and image
reconstruction from atmospheric-degraded observations.
"""

from __future__ import annotations

from typing import Type, Dict

# Model registry
_MODELS: Dict[str, Type] = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls):
        _MODELS[name.lower()] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs):
    """
    Get model instance by name.
    
    Args:
        name: Model name ('unet', etc.)
        **kwargs: Model-specific arguments
    
    Returns:
        Model instance
    """
    name = name.lower()
    if name not in _MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(_MODELS.keys())}")
    return _MODELS[name](**kwargs)


def list_models():
    """List available model names."""
    return list(_MODELS.keys())


# Import implementations to register them
from . import unet

__all__ = [
    'register_model',
    'get_model',
    'list_models',
]

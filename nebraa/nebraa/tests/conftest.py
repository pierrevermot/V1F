"""
Test configuration for NEBRAA.
"""

import pytest
import sys
import os

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.fixture
def backend():
    """Fixture providing compute backend."""
    from nebraa.utils.compute import init_backend
    return init_backend(compute_mode="CPU")


@pytest.fixture
def xp(backend):
    """NumPy/CuPy array module."""
    return backend.xp


@pytest.fixture
def default_config():
    """Default configuration for testing."""
    from nebraa.config import Config
    return Config()


@pytest.fixture
def small_image(xp):
    """Small test image."""
    import numpy as np
    np.random.seed(42)
    return np.random.rand(64, 64).astype(np.float32)


@pytest.fixture
def source_config():
    """Source configuration for testing."""
    from nebraa.config import SourceConfig
    return SourceConfig(image_size=64, n_sources_max=5)


@pytest.fixture
def instrument_config():
    """Instrument configuration for testing."""
    from nebraa.config import InstrumentConfig
    return InstrumentConfig(name='vlt')

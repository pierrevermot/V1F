"""
Random Number Generator (RNG) utilities.

Provides thread-safe, reproducible random number generation for both
CPU (NumPy) and GPU (CuPy) backends.

Key Features:
- No global RNG state mutation (thread-safe)
- Reproducible with explicit seeds
- Backend-agnostic API
- Parallel-execution safe

Author: NEBRAA
"""

from __future__ import annotations

from typing import Optional, Union, Any
import numpy as np


# Type alias for RNG objects
RNGType = Union[np.random.Generator, Any]  # Any for CuPy RandomState


def create_rng(seed: Optional[int] = None, backend: Optional[Any] = None) -> RNGType:
    """
    Create a thread-safe RNG object for the specified backend.
    
    This function never mutates global RNG state, making it safe for
    parallel execution and reproducible testing.
    
    Args:
        seed: Random seed for reproducibility. If None, RNG is non-deterministic.
        backend: Backend object (from get_backend()). If None, uses NumPy.
        
    Returns:
        RNG object (np.random.Generator for NumPy, RandomState for CuPy)
        
    Examples:
        # CPU (NumPy)
        rng = create_rng(seed=42)
        data = rng.normal(size=(100, 100))
        
        # GPU (CuPy)
        backend = get_backend()
        rng = create_rng(seed=42, backend=backend)
        data = rng.normal(size=(100, 100))
    """
    if backend is None:
        # NumPy backend
        return np.random.default_rng(seed)
    
    xp = backend.xp
    
    if xp.__name__ == 'numpy':
        # NumPy backend
        return np.random.default_rng(seed)
    else:
        # CuPy backend
        # Use RandomState for broader CuPy version compatibility
        # (default_rng was added in CuPy 10.0, but RandomState works everywhere)
        import cupy as cp
        if seed is None:
            return cp.random.RandomState()
        else:
            return cp.random.RandomState(seed=seed)


def get_rng_or_create(
    rng: Optional[RNGType] = None,
    seed: Optional[int] = None,
    backend: Optional[Any] = None,
) -> RNGType:
    """
    Get existing RNG or create a new one.
    
    Utility function for APIs that accept optional RNG objects.
    
    Args:
        rng: Existing RNG object. If provided, returned as-is.
        seed: Random seed if creating new RNG. Ignored if rng is provided.
        backend: Backend object for creating new RNG. Ignored if rng is provided.
        
    Returns:
        RNG object (existing or newly created)
        
    Examples:
        # User provides RNG
        user_rng = create_rng(seed=42)
        rng = get_rng_or_create(rng=user_rng)  # Returns user_rng
        
        # User provides seed
        rng = get_rng_or_create(seed=42)  # Creates new RNG with seed
        
        # User provides neither (non-deterministic)
        rng = get_rng_or_create()  # Creates new RNG without seed
    """
    if rng is not None:
        return rng
    return create_rng(seed=seed, backend=backend)


def check_rng_determinism(
    func,
    rng_seed: int,
    backend: Optional[Any] = None,
    n_trials: int = 2,
    **kwargs
) -> bool:
    """
    Check if a function produces deterministic output with same RNG seed.
    
    Useful for testing reproducibility.
    
    Args:
        func: Function to test (should accept rng= parameter)
        rng_seed: Seed to use for RNG
        backend: Backend object
        n_trials: Number of trials to compare
        **kwargs: Additional arguments to pass to func
        
    Returns:
        True if all trials produce identical output
        
    Examples:
        def my_func(rng=None):
            rng = get_rng_or_create(rng)
            return rng.normal(size=(10, 10))
        
        # Should be True
        is_deterministic = check_rng_determinism(my_func, rng_seed=42)
    """
    import numpy as np
    
    # Get backend
    if backend is None:
        xp = np
    else:
        xp = backend.xp
    
    outputs = []
    for _ in range(n_trials):
        rng = create_rng(seed=rng_seed, backend=backend)
        output = func(rng=rng, **kwargs)
        
        # Convert to numpy for comparison
        if xp.__name__ == 'cupy':
            output = xp.asnumpy(output)
        
        outputs.append(output)
    
    # Check all outputs are identical
    for i in range(1, n_trials):
        if not np.allclose(outputs[0], outputs[i], rtol=1e-10, atol=1e-12):
            return False
    
    return True


def get_rng_info(rng: RNGType) -> dict:
    """
    Get information about an RNG object.
    
    Args:
        rng: RNG object
        
    Returns:
        Dictionary with RNG type and backend info
    """
    import numpy as np
    
    info = {}
    
    if isinstance(rng, np.random.Generator):
        info['type'] = 'numpy.random.Generator'
        info['backend'] = 'numpy'
    elif hasattr(rng, '__module__') and 'cupy' in rng.__module__:
        info['type'] = 'cupy.random.RandomState'
        info['backend'] = 'cupy'
    else:
        info['type'] = str(type(rng))
        info['backend'] = 'unknown'
    
    return info

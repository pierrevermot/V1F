"""
Compute backend utilities.

Provides unified CPU/GPU computing with automatic fallback and 
distributed execution support for SLURM/MPI environments.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple, Any


# =============================================================================
# Distributed Execution Identifiers
# =============================================================================

def _env_int(keys: list, default: int) -> int:
    """Parse first available environment variable as int."""
    for k in keys:
        v = os.environ.get(k)
        if v:
            try:
                return int(v)
            except ValueError:
                pass
    return default


# Global distributed execution state
WORLD_SIZE = _env_int(["WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"], 1)
RANK = _env_int(["RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK"], 0)
LOCAL_RANK = _env_int(["LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"], 0)
IS_CHIEF = (RANK == 0)


# =============================================================================
# Backend State
# =============================================================================

class Backend:
    """
    Unified compute backend for CPU/GPU operations.
    
    Provides a consistent interface regardless of whether NumPy or CuPy
    is being used underneath.
    """
    
    def __init__(self):
        self.xp = None           # NumPy or CuPy module
        self.fftconvolve = None  # Convolution function
        self.np = None           # Always NumPy (for CPU operations)
        self.gpu = False
        self._initialized = False
        self._device_id = None
    
    def init(self, compute_mode: str = "GPU", device_id: Optional[int] = None) -> 'Backend':
        """
        Initialize the computation backend.
        
        Args:
            compute_mode: "GPU" or "CPU"
            device_id: Specific GPU device ID. None uses LOCAL_RANK.
        
        Returns:
            Self for chaining
        """
        import numpy as np
        self.np = np
        
        if compute_mode == "CPU":
            from scipy.signal import fftconvolve
            self.xp = np
            self.fftconvolve = fftconvolve
            self.gpu = False
            if IS_CHIEF:
                print("[BACKEND] Using CPU (NumPy)")
        else:
            try:
                import cupy as cp
                from cupyx.scipy.signal import fftconvolve
                self.xp = cp
                self.fftconvolve = fftconvolve
                self.gpu = True
                
                # Pin to specific GPU
                n_devices = cp.cuda.runtime.getDeviceCount()
                if n_devices > 0:
                    self._device_id = device_id if device_id is not None else (LOCAL_RANK % n_devices)
                    cp.cuda.Device(self._device_id).use()
                    if IS_CHIEF:
                        print(f"[BACKEND] Using GPU {self._device_id}/{n_devices} (CuPy)")
                else:
                    raise RuntimeError("No CUDA devices available")
                    
            except Exception as e:
                if IS_CHIEF:
                    print(f"[BACKEND] CuPy unavailable ({e}), falling back to CPU")
                from scipy.signal import fftconvolve
                self.xp = np
                self.fftconvolve = fftconvolve
                self.gpu = False
        
        self._initialized = True
        return self
    
    def ensure_initialized(self):
        """Initialize with defaults if not already done."""
        if not self._initialized:
            self.init()
    
    @property
    def device_id(self) -> Optional[int]:
        """Current GPU device ID, or None for CPU."""
        return self._device_id if self.gpu else None
    
    def to_numpy(self, array) -> Any:
        """Convert array to NumPy (from GPU if necessary)."""
        if hasattr(array, 'get'):
            return array.get()
        return self.np.asarray(array)
    
    def to_device(self, array) -> Any:
        """Convert NumPy array to device array."""
        self.ensure_initialized()
        return self.xp.asarray(array)
    
    def ensure_local(self, array) -> Any:
        """
        Ensure array is local to the current device.
        
        This prevents cross-device memory access errors in multi-GPU setups.
        """
        self.ensure_initialized()
        
        if not self.gpu:
            return self.xp.asarray(array)
        
        # Check if already on current device
        if hasattr(array, 'device'):
            if array.device.id == self._device_id:
                return array
        
        # Move to current device
        return self.xp.asarray(self.to_numpy(array))
    
    def get_eps(self) -> float:
        """Get machine epsilon for current dtype."""
        return float(sys.float_info.epsilon)
    
    def zeros(self, shape, dtype=None):
        """Create zero array on current device."""
        self.ensure_initialized()
        dtype = dtype or self.xp.float32
        return self.xp.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=None):
        """Create ones array on current device."""
        self.ensure_initialized()
        dtype = dtype or self.xp.float32
        return self.xp.ones(shape, dtype=dtype)
    
    def arange(self, *args, **kwargs):
        """Create range array on current device."""
        self.ensure_initialized()
        return self.xp.arange(*args, **kwargs)


# Global singleton backend
_backend = Backend()


def init_backend(compute_mode: str = "GPU", device_id: Optional[int] = None) -> Backend:
    """Initialize the global backend."""
    return _backend.init(compute_mode, device_id)


def get_backend() -> Backend:
    """Get the global backend instance."""
    _backend.ensure_initialized()
    return _backend


def get_xp():
    """Get the array module (NumPy or CuPy)."""
    _backend.ensure_initialized()
    return _backend.xp


def is_gpu() -> bool:
    """Check if GPU is being used."""
    _backend.ensure_initialized()
    return _backend.gpu


# =============================================================================
# Parallel Execution Utilities
# =============================================================================

def split_work(items: list, n_workers: int) -> list:
    """
    Split work items across workers evenly.
    
    Args:
        items: List of work items
        n_workers: Number of workers
    
    Returns:
        List of lists, one per worker
    """
    k, m = divmod(len(items), n_workers)
    return [items[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n_workers)]


def get_worker_items(items: list, rank: int, world_size: int) -> list:
    """
    Get the items assigned to a specific worker.
    
    Args:
        items: Full list of items
        rank: Worker rank
        world_size: Total number of workers
    
    Returns:
        Subset of items for this worker
    """
    return [item for i, item in enumerate(items) if i % world_size == rank]


def get_device_count() -> int:
    """Get number of available GPU devices."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def get_cpu_count() -> int:
    """Get number of CPU cores."""
    return os.cpu_count() or 1

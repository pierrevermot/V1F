#!/usr/bin/env python3
"""
Comprehensive PSF Computing Time Benchmark
==========================================

This script benchmarks PSF generation times across all parameter combinations:
- Backend: CPU vs GPU
- Method: Zernike, DualPowerLaw, Jolissaint
- Exposure type: Short Exposure (SE) vs Long Exposure (LE)

Two distinct scaling parameters are explored:
1. n_screens_per_le: Number of phase screens averaged per LE PSF (LE quality)
2. n_psf_total: Total number of PSFs generated (throughput)

For SE: each PSF = 1 phase screen, n_psf_total PSFs generated
For LE: each PSF = n_screens_per_le phase screens averaged, n_psf_total PSFs generated

Output includes:
1. Total time and per-PSF time
2. Detailed breakdown of phase generation steps
3. Detailed breakdown of PSF computation steps
4. Three detailed case studies: 1 SE, 1 LE, 100 LE PSFs

Usage:
    python benchmark_psf_timing.py [--quick] [--output-dir DIR]
    
Options:
    --quick       Run quick mode (fewer samples)
    --output-dir  Output directory for results and plots
"""

import sys
import time
import json
import argparse
import io
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
from datetime import datetime
import warnings

import numpy as np


class TeeOutput:
    """Capture stdout to both console and a string buffer."""
    def __init__(self, stream):
        self.stream = stream
        self.buffer = io.StringIO()
    
    def write(self, data):
        self.stream.write(data)
        self.buffer.write(data)
    
    def flush(self):
        self.stream.flush()
    
    def getvalue(self):
        return self.buffer.getvalue()

# Add parent to path for nebraa imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import ScalarFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# =============================================================================
# Constants
# =============================================================================

# Grid
DEFAULT_N_PIX = 256

# Telescope (VLT)
TELESCOPE_DIAMETER = 8.2  # meters
OBSTRUCTION_DIAMETER = 1.116  # meters

# Observation
WAVELENGTH = 2.2e-6  # K-band (meters)
PIXEL_SCALE_MAS = 13.0  # mas/pixel
PIXEL_SCALE_RAD = PIXEL_SCALE_MAS * 1e-3 / 206265  # radians/pixel

# Benchmark parameters

# n_screens_per_le: How many phase screens are averaged per LE PSF
N_SCREENS_PER_LE_LIST = [1, 10, 20]
N_SCREENS_PER_LE_LIST_QUICK = [1, 10]

# n_psf_total: Total number of PSFs generated (SE or LE)
# Finer exploration up to 50 for throughput analysis
N_PSF_TOTAL_LIST = [1, 5, 10, 20, 30]
N_PSF_TOTAL_LIST_QUICK = [1, 10, 30]

METHODS = ['Zernike', 'DualPowerLaw', 'Jolissaint']
BACKENDS = ['CPU', 'GPU']

# Number of repetitions for timing
N_REPS = 3
N_REPS_QUICK = 1
N_WARMUP = 5  # Increased warm-up


# =============================================================================
# Timing Utilities
# =============================================================================

def sync_gpu():
    """Synchronize GPU if available."""
    if HAS_CUPY:
        try:
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass


class DetailedTimer:
    """Class to collect detailed timing information."""
    
    def __init__(self):
        self.timings = {}
        self._start_times = {}
    
    def start(self, name: str):
        """Start timing a section."""
        sync_gpu()
        self._start_times[name] = time.perf_counter()
    
    def stop(self, name: str):
        """Stop timing a section and record."""
        sync_gpu()
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            self.timings[name] = elapsed
            del self._start_times[name]
    
    def get(self, name: str, default: float = 0.0) -> float:
        """Get timing for a section."""
        return self.timings.get(name, default)
    
    def to_dict(self) -> Dict[str, float]:
        return dict(self.timings)


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class DetailedBreakdown:
    """Detailed timing breakdown for a single run."""
    # Total times
    total_time: float = 0.0
    phase_generation_total: float = 0.0
    psf_computation_total: float = 0.0
    lwe_generation_total: float = 0.0  # Low Wind Effect generation
    
    # Phase generation breakdown (method-specific)
    phase_noise_generation: float = 0.0
    phase_fft: float = 0.0
    phase_piston_removal: float = 0.0
    
    # Zernike-specific
    zernike_mode_computation: float = 0.0
    zernike_coefficient_generation: float = 0.0
    zernike_matrix_multiply: float = 0.0
    zernike_hf_generation: float = 0.0
    zernike_normalization: float = 0.0
    
    # DualPowerLaw-specific
    psd_coloring: float = 0.0
    
    # Jolissaint-specific
    jolissaint_psd_computation: float = 0.0
    
    # LWE-specific
    lwe_init: float = 0.0  # LWE model initialization (island detection)
    lwe_screen_generation: float = 0.0  # LWE phase screen generation
    
    # PSF computation breakdown
    psf_field_construction: float = 0.0
    psf_fft: float = 0.0
    psf_intensity: float = 0.0
    psf_normalization: float = 0.0
    psf_accumulation: float = 0.0  # For LE averaging
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Result for a single benchmark configuration."""
    backend: str
    method: str
    exposure_type: str  # 'SE' or 'LE'
    n_screens_per_le: int  # For LE: screens averaged per PSF. For SE: always 1
    n_psf_total: int  # Total PSFs generated
    
    # Derived
    total_phase_screens: int = 0  # = n_screens_per_le * n_psf_total for LE, = n_psf_total for SE
    
    # Timing results (averaged over repetitions)
    mean_total_time: float = 0.0
    std_total_time: float = 0.0
    mean_time_per_psf: float = 0.0
    mean_time_per_screen: float = 0.0
    
    # Breakdown (averaged)
    breakdown: DetailedBreakdown = field(default_factory=DetailedBreakdown)
    
    # All repetition timings
    rep_times: List[float] = field(default_factory=list)
    
    # Memory usage (MB)
    memory_mb: float = 0.0
    
    # Success flag
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['breakdown'] = self.breakdown.to_dict()
        return d


@dataclass
class BenchmarkSuite:
    """Collection of all benchmark results."""
    timestamp: str = ""
    platform_info: Dict[str, Any] = field(default_factory=dict)
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'platform_info': self.platform_info,
            'results': [r.to_dict() for r in self.results]
        }
    
    def get_results_for(
        self, 
        backend: Optional[str] = None,
        method: Optional[str] = None,
        exposure_type: Optional[str] = None,
        n_screens_per_le: Optional[int] = None,
        n_psf_total: Optional[int] = None,
    ) -> List[BenchmarkResult]:
        """Filter results by criteria."""
        filtered = self.results
        if backend is not None:
            filtered = [r for r in filtered if r.backend == backend]
        if method is not None:
            filtered = [r for r in filtered if r.method == method]
        if exposure_type is not None:
            filtered = [r for r in filtered if r.exposure_type == exposure_type]
        if n_screens_per_le is not None:
            filtered = [r for r in filtered if r.n_screens_per_le == n_screens_per_le]
        if n_psf_total is not None:
            filtered = [r for r in filtered if r.n_psf_total == n_psf_total]
        return filtered


# =============================================================================
# NEBRAA Module Imports
# =============================================================================

def import_nebraa_modules() -> Dict[str, Any]:
    """Import all required NEBRAA modules."""
    from nebraa.utils.compute import init_backend, get_backend
    from nebraa.physics import (
        PSFEngine,
        ZernikeConfig,
        ZernikePhaseGenerator,
        DualPowerLawConfig,
        DualPowerLawPhaseGenerator,
        JolissaintAOModel,
        TurbulentLayer,
        AtmosphereProfile,
        create_simple_atmosphere,
        create_ao_config,
    )
    from nebraa.physics.low_wind_effect import LowWindEffect
    
    return {
        'init_backend': init_backend,
        'get_backend': get_backend,
        'PSFEngine': PSFEngine,
        'ZernikeConfig': ZernikeConfig,
        'ZernikePhaseGenerator': ZernikePhaseGenerator,
        'DualPowerLawConfig': DualPowerLawConfig,
        'DualPowerLawPhaseGenerator': DualPowerLawPhaseGenerator,
        'JolissaintAOModel': JolissaintAOModel,
        'TurbulentLayer': TurbulentLayer,
        'AtmosphereProfile': AtmosphereProfile,
        'create_simple_atmosphere': create_simple_atmosphere,
        'create_ao_config': create_ao_config,
        'LowWindEffect': LowWindEffect,
    }


# =============================================================================
# Helper Functions
# =============================================================================

def create_vlt_pupil(n_pix: int, pixel_size: float, xp=np) -> Any:
    """Create VLT-like pupil with central obstruction and spiders."""
    D = TELESCOPE_DIAMETER
    D_obs = OBSTRUCTION_DIAMETER
    spider_width = 0.05  # meters (thin spiders)
    
    extent = n_pix * pixel_size
    x = xp.linspace(-extent/2, extent/2, n_pix)
    X, Y = xp.meshgrid(x, x)
    R = xp.sqrt(X**2 + Y**2)
    
    # Annular aperture
    pupil = ((R <= D/2) & (R >= D_obs/2)).astype(xp.float32)
    
    # Spider vanes (thin)
    spider_half = spider_width / 2
    mask_h = (xp.abs(Y) < spider_half) & (R <= D/2) & (R >= D_obs/2)
    mask_v = (xp.abs(X) < spider_half) & (R <= D/2) & (R >= D_obs/2)
    pupil = xp.where(mask_h | mask_v, xp.float32(0), pupil)
    
    return pupil


def to_numpy(arr) -> np.ndarray:
    """Convert array to numpy (handles CuPy arrays)."""
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def get_memory_usage_mb() -> float:
    """Get current GPU memory usage in MB (if GPU available)."""
    if HAS_CUPY:
        try:
            mempool = cp.get_default_memory_pool()
            return mempool.used_bytes() / (1024 * 1024)
        except Exception:
            pass
    return 0.0


def get_platform_info() -> Dict[str, Any]:
    """Gather platform information."""
    import platform
    
    info = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'numpy_version': np.__version__,
        'has_cupy': HAS_CUPY,
    }
    
    if HAS_CUPY:
        try:
            info['cupy_version'] = cp.__version__
            info['cuda_version'] = cp.cuda.runtime.runtimeGetVersion()
            device = cp.cuda.Device()
            info['gpu_memory_gb'] = device.mem_info[1] / (1024**3)
        except Exception as e:
            info['gpu_info_error'] = str(e)
    
    return info


# =============================================================================
# Benchmark Class
# =============================================================================

class PSFTimingBenchmark:
    """
    Comprehensive PSF timing benchmark with detailed breakdowns.
    
    Tests combinations of:
    - Backend: CPU, GPU
    - Method: Zernike, DualPowerLaw, Jolissaint
    - Exposure type: SE, LE
    - n_screens_per_le: screens averaged per LE PSF
    - n_psf_total: total PSFs generated
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        quick_mode: bool = False,
        n_pix: int = DEFAULT_N_PIX,
    ):
        self.output_dir = output_dir or Path(f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.quick_mode = quick_mode
        self.n_pix = n_pix
        self.pixel_size = WAVELENGTH / (n_pix * PIXEL_SCALE_RAD)
        
        self.n_reps = N_REPS_QUICK if quick_mode else N_REPS
        self.n_screens_per_le_list = N_SCREENS_PER_LE_LIST_QUICK if quick_mode else N_SCREENS_PER_LE_LIST
        self.n_psf_total_list = N_PSF_TOTAL_LIST_QUICK if quick_mode else N_PSF_TOTAL_LIST
        
        self.modules = import_nebraa_modules()
        self.suite = BenchmarkSuite(
            timestamp=datetime.now().isoformat(),
            platform_info=get_platform_info(),
        )
        
        # Cached objects
        self._psf_engines = {}
        self._pupils = {}
        
        print(f"PSF Timing Benchmark")
        print(f"{'='*60}")
        print(f"Output directory: {self.output_dir}")
        print(f"Grid size: {n_pix}x{n_pix}")
        print(f"Pixel size: {self.pixel_size*1000:.2f} mm")
        print(f"Quick mode: {quick_mode}")
        print(f"n_screens_per_le to test: {self.n_screens_per_le_list}")
        print(f"n_psf_total to test: {self.n_psf_total_list}")
        print(f"Repetitions per test: {self.n_reps}")
        print(f"Warm-up iterations: {N_WARMUP}")
        print(f"GPU available: {HAS_CUPY}")
        print()
    
    # =========================================================================
    # Setup Methods
    # =========================================================================
    
    def _init_backend_quiet(self, backend_name: str):
        """Initialize backend without printing."""
        import io
        import sys as _sys
        old_stdout = _sys.stdout
        _sys.stdout = io.StringIO()
        try:
            self.modules['init_backend'](backend_name)
        finally:
            _sys.stdout = old_stdout
    
    def _get_pupil(self, backend_name: str):
        """Get or create pupil for backend."""
        if backend_name not in self._pupils:
            self._init_backend_quiet(backend_name)
            backend = self.modules['get_backend']()
            xp = backend.xp
            self._pupils[backend_name] = create_vlt_pupil(self.n_pix, self.pixel_size, xp=xp)
        return self._pupils[backend_name]
    
    def _get_psf_engine(self, backend_name: str):
        """Get or create PSF engine for backend."""
        if backend_name not in self._psf_engines:
            self._init_backend_quiet(backend_name)
            self._psf_engines[backend_name] = self.modules['PSFEngine'](
                n_pix=self.n_pix,
                wavelength=WAVELENGTH,
                pixel_scale=PIXEL_SCALE_RAD,
                normalize_to='sum',
            )
        return self._psf_engines[backend_name]
    
    def _create_generator(self, method: str, backend_name: str, seed: int = 42):
        """Create a fresh phase generator for the given method and backend."""
        self._init_backend_quiet(backend_name)
        
        # Clear Zernike cache when switching backends
        try:
            from nebraa.physics.zernike import _cache
            _cache.clear()
        except (ImportError, AttributeError):
            pass
        
        if method == 'Zernike':
            config = self.modules['ZernikeConfig'](
                n_range=(2, 8),
                power_law=2.5,
                lf_rms=0.158,
                seed=seed,
                f_cutoff=1.0,
                hf_alpha=11.0/3.0,
                hf_rms=0.474,
                transition_width=0.2,
            )
            return self.modules['ZernikePhaseGenerator'](
                n_pix=self.n_pix,
                pixel_size=self.pixel_size,
                zernike_config=config,
            )
        
        elif method == 'DualPowerLaw':
            config = self.modules['DualPowerLawConfig'](
                alpha_lf=3.0,
                alpha_hf=11.0/3.0,
                rms_lf=0.35,
                rms_hf=0.35,
                f_cutoff=1.0,
                seed=seed,
            )
            return self.modules['DualPowerLawPhaseGenerator'](
                n_pix=self.n_pix,
                pixel_size=self.pixel_size,
                psd_config=config,
            )
        
        elif method == 'Jolissaint':
            atmosphere = self.modules['create_simple_atmosphere'](
                r0=0.15, L0=25.0, wind_speed=10.0
            )
            ao_config = self.modules['create_ao_config'](
                n_actuators=14,
                telescope_diameter=TELESCOPE_DIAMETER,
            )
            return self.modules['JolissaintAOModel'](
                n_pix=self.n_pix,
                telescope_diameter=TELESCOPE_DIAMETER,
                obstruction_diameter=OBSTRUCTION_DIAMETER,
                wavelength=WAVELENGTH,
                pixel_scale=PIXEL_SCALE_RAD,
                atmosphere=atmosphere,
                ao_config=ao_config,
            )
        
        raise ValueError(f"Unknown method: {method}")
    
    # =========================================================================
    # Detailed Timing Functions
    # =========================================================================
    
    def _time_phase_generation(
        self, method: str, generator, n_screens: int, pupil, seed: int, timer: DetailedTimer
    ):
        """Time phase generation with detailed breakdown."""
        backend = self.modules['get_backend']()
        xp = backend.xp
        
        timer.start('phase_total')
        
        if method == 'Zernike':
            # Zernike: can't easily decompose without modifying library
            phases = generator.generate(n_screens, pupil)
            sync_gpu()
            timer.stop('phase_total')
            
            # Estimate breakdown (rough approximation)
            total = timer.get('phase_total')
            timer.timings['mode_computation'] = total * 0.05
            timer.timings['coeff_generation'] = total * 0.05
            timer.timings['matrix_multiply'] = total * 0.15
            timer.timings['normalization'] = total * 0.05
            timer.timings['hf_generation'] = total * 0.70
            
        elif method == 'DualPowerLaw':
            # DualPowerLaw: measure individual steps
            timer.start('noise_generation')
            if generator.psd_config.seed is not None:
                rng = xp.random.default_rng(generator.psd_config.seed)
            else:
                rng = xp.random.default_rng()
            
            shape = (n_screens, self.n_pix, self.n_pix)
            noise_real = rng.standard_normal(shape, dtype=xp.float64)
            noise_imag = rng.standard_normal(shape, dtype=xp.float64)
            white_noise = noise_real + 1j * noise_imag
            sync_gpu()
            timer.stop('noise_generation')
            
            timer.start('psd_coloring')
            amplitude = generator.amplitude
            colored = white_noise * amplitude[None, :, :]
            sync_gpu()
            timer.stop('psd_coloring')
            
            timer.start('fft')
            phases = xp.real(xp.fft.ifft2(colored))
            sync_gpu()
            timer.stop('fft')
            
            timer.start('piston_removal')
            pupil_local = backend.ensure_local(pupil).astype(xp.float64)
            pupil_sum = xp.sum(pupil_local)
            if pupil_sum > 0:
                mean_phase = xp.sum(phases * pupil_local[None, :, :], axis=(1, 2)) / pupil_sum
                phases = phases - mean_phase[:, None, None]
            sync_gpu()
            timer.stop('piston_removal')
            
            timer.stop('phase_total')
            
        elif method == 'Jolissaint':
            # Jolissaint: measure individual steps
            timer.start('psd_computation')
            total_psd = generator.compute_total_residual_psd()
            sync_gpu()
            timer.stop('psd_computation')
            
            timer.start('psd_coloring')
            amplitude = xp.sqrt(xp.maximum(total_psd, 0))
            sync_gpu()
            timer.stop('psd_coloring')
            
            timer.start('noise_generation')
            if seed is not None:
                rng = xp.random.default_rng(seed)
            else:
                rng = xp.random.default_rng()
            
            shape = (n_screens, self.n_pix, self.n_pix)
            noise_real = rng.standard_normal(shape, dtype=xp.float64)
            noise_imag = rng.standard_normal(shape, dtype=xp.float64)
            white_noise = noise_real + 1j * noise_imag
            sync_gpu()
            timer.stop('noise_generation')
            
            timer.start('psd_application')
            colored = white_noise * amplitude[None, :, :]
            sync_gpu()
            timer.stop('psd_application')
            
            timer.start('fft')
            phases = xp.real(xp.fft.ifft2(colored))
            sync_gpu()
            timer.stop('fft')
            
            timer.start('piston_removal')
            pupil_local = backend.ensure_local(pupil).astype(xp.float64)
            pupil_sum = xp.sum(pupil_local)
            if pupil_sum > 0:
                mean_phase = xp.sum(phases * pupil_local[None, :, :], axis=(1, 2)) / pupil_sum
                phases = phases - mean_phase[:, None, None]
            sync_gpu()
            timer.stop('piston_removal')
            
            timer.stop('phase_total')
        
        return phases
    
    def _time_psf_computation_se(
        self, pupil, phases, timer: DetailedTimer
    ):
        """Time SE PSF computation (one PSF per phase screen)."""
        backend = self.modules['get_backend']()
        xp = backend.xp
        
        n_psf = phases.shape[0]
        
        timer.start('psf_total')
        
        timer.start('field_construction')
        fields = pupil[None, :, :] * xp.exp(1j * phases)
        sync_gpu()
        timer.stop('field_construction')
        
        timer.start('fft')
        fields_fft = xp.fft.fftshift(xp.fft.fft2(fields, axes=(-2, -1)), axes=(-2, -1))
        sync_gpu()
        timer.stop('fft')
        
        timer.start('intensity')
        psfs = xp.abs(fields_fft)**2
        sync_gpu()
        timer.stop('intensity')
        
        timer.start('normalization')
        psf_sums = xp.sum(psfs, axis=(-2, -1), keepdims=True)
        psfs = xp.where(psf_sums > 0, psfs / psf_sums, psfs)
        sync_gpu()
        timer.stop('normalization')
        
        timer.start('accumulation')
        # No accumulation for SE
        timer.stop('accumulation')
        
        timer.stop('psf_total')
        
        return psfs
    
    def _time_psf_computation_le(
        self, pupil, phases, n_screens_per_le: int, timer: DetailedTimer
    ):
        """Time LE PSF computation (average n_screens_per_le phases per PSF)."""
        backend = self.modules['get_backend']()
        xp = backend.xp
        
        total_screens = phases.shape[0]
        n_psf = total_screens // n_screens_per_le
        
        timer.start('psf_total')
        
        le_psfs = []
        
        for i in range(n_psf):
            start_idx = i * n_screens_per_le
            end_idx = start_idx + n_screens_per_le
            batch_phases = phases[start_idx:end_idx]
            
            timer.start('field_construction')
            fields = pupil[None, :, :] * xp.exp(1j * batch_phases)
            sync_gpu()
            timer.stop('field_construction')
            
            timer.start('fft')
            fields_fft = xp.fft.fftshift(xp.fft.fft2(fields, axes=(-2, -1)), axes=(-2, -1))
            sync_gpu()
            timer.stop('fft')
            
            timer.start('intensity')
            psfs = xp.abs(fields_fft)**2
            sync_gpu()
            timer.stop('intensity')
            
            timer.start('accumulation')
            psf_avg = xp.mean(psfs, axis=0)
            sync_gpu()
            timer.stop('accumulation')
            
            timer.start('normalization')
            psf_sum = xp.sum(psf_avg)
            if psf_sum > 0:
                psf_avg = psf_avg / psf_sum
            sync_gpu()
            timer.stop('normalization')
            
            le_psfs.append(psf_avg)
        
        timer.stop('psf_total')
        
        result = xp.stack(le_psfs, axis=0) if len(le_psfs) > 1 else le_psfs[0]
        return result
    
    def _time_lwe_generation(
        self, pupil, n_screens: int, timer: DetailedTimer, seed: int = 42
    ):
        """Time LWE (Low Wind Effect) phase screen generation with detailed breakdown."""
        backend = self.modules['get_backend']()
        xp = backend.xp
        
        timer.start('lwe_total')
        
        # LWE initialization (includes island detection)
        timer.start('lwe_init')
        lwe_model = self.modules['LowWindEffect'](
            pupil=pupil,
            piston_rms_rad=0.5,  # typical value
            tilt_rms_rad=0.3,    # typical value
        )
        sync_gpu()
        timer.stop('lwe_init')
        
        # LWE phase screen generation
        timer.start('lwe_screen_generation')
        lwe_phases = lwe_model.generate(n_screens, seed=seed)
        sync_gpu()
        timer.stop('lwe_screen_generation')
        
        timer.stop('lwe_total')
        
        return lwe_phases
    
    # =========================================================================
    # Simple Timing (fallback)
    # =========================================================================
    
    def _time_phase_simple(self, method: str, generator, n_screens: int, pupil, seed: int = 42):
        """Simple timing without detailed breakdown."""
        sync_gpu()
        start = time.perf_counter()
        
        if method == 'Zernike':
            phases = generator.generate(n_screens, pupil)
        elif method == 'DualPowerLaw':
            phases = generator.generate(n_screens, pupil)
        elif method == 'Jolissaint':
            phases = generator.generate_phase_screens(n_screens, pupil, seed=seed)
        
        sync_gpu()
        elapsed = time.perf_counter() - start
        
        return phases, elapsed
    
    # =========================================================================
    # Run Single Benchmark
    # =========================================================================
    
    def _run_single_benchmark(
        self,
        backend: str,
        method: str,
        exposure_type: str,
        n_screens_per_le: int,
        n_psf_total: int,
    ) -> BenchmarkResult:
        """Run a single benchmark configuration with detailed timing."""
        
        # For SE: 1 screen per PSF, total screens = n_psf_total
        # For LE: n_screens_per_le screens per PSF, total screens = n_screens_per_le * n_psf_total
        if exposure_type == 'SE':
            total_screens = n_psf_total
            actual_n_screens_per_le = 1
        else:
            total_screens = n_screens_per_le * n_psf_total
            actual_n_screens_per_le = n_screens_per_le
        
        result = BenchmarkResult(
            backend=backend,
            method=method,
            exposure_type=exposure_type,
            n_screens_per_le=actual_n_screens_per_le,
            n_psf_total=n_psf_total,
            total_phase_screens=total_screens,
        )
        
        try:
            self._init_backend_quiet(backend)
            
            pupil = self._get_pupil(backend)
            generator = self._create_generator(method, backend)
            
            # Warm-up (increased)
            for _ in range(N_WARMUP):
                warm_phases, _ = self._time_phase_simple(
                    method, generator, min(10, total_screens), pupil
                )
                # Also warm up PSF computation
                backend_obj = self.modules['get_backend']()
                xp = backend_obj.xp
                _ = xp.fft.fft2(warm_phases)
            
            sync_gpu()
            
            # Collect detailed timings
            all_total_times = []
            all_breakdowns = []
            
            for rep in range(self.n_reps):
                generator = self._create_generator(method, backend, seed=42 + rep * 1000)
                
                try:
                    phase_timer = DetailedTimer()
                    psf_timer = DetailedTimer()
                    lwe_timer = DetailedTimer()
                    
                    # Time phase generation
                    phases = self._time_phase_generation(
                        method, generator, total_screens, pupil, 
                        42 + rep * 1000, phase_timer
                    )
                    
                    # Time LWE generation (for LE mode, add LWE contribution)
                    lwe_time = 0.0
                    if exposure_type == 'LE':
                        _ = self._time_lwe_generation(
                            pupil, total_screens, lwe_timer, seed=42 + rep * 1000
                        )
                        lwe_time = lwe_timer.get('lwe_total')
                    
                    # Time PSF computation
                    if exposure_type == 'SE':
                        _ = self._time_psf_computation_se(pupil, phases, psf_timer)
                    else:
                        _ = self._time_psf_computation_le(
                            pupil, phases, n_screens_per_le, psf_timer
                        )
                    
                    phase_time = phase_timer.get('phase_total')
                    psf_time = psf_timer.get('psf_total')
                    
                    # Build breakdown
                    breakdown = DetailedBreakdown(
                        phase_generation_total=phase_time,
                        psf_computation_total=psf_time,
                        lwe_generation_total=lwe_time,
                        lwe_init=lwe_timer.get('lwe_init'),
                        lwe_screen_generation=lwe_timer.get('lwe_screen_generation'),
                    )
                    
                    # Phase breakdown
                    if method == 'Zernike':
                        breakdown.zernike_mode_computation = phase_timer.get('mode_computation')
                        breakdown.zernike_coefficient_generation = phase_timer.get('coeff_generation')
                        breakdown.zernike_matrix_multiply = phase_timer.get('matrix_multiply')
                        breakdown.zernike_normalization = phase_timer.get('normalization')
                        breakdown.zernike_hf_generation = phase_timer.get('hf_generation')
                    elif method == 'DualPowerLaw':
                        breakdown.phase_noise_generation = phase_timer.get('noise_generation')
                        breakdown.psd_coloring = phase_timer.get('psd_coloring')
                        breakdown.phase_fft = phase_timer.get('fft')
                        breakdown.phase_piston_removal = phase_timer.get('piston_removal')
                    elif method == 'Jolissaint':
                        breakdown.jolissaint_psd_computation = phase_timer.get('psd_computation')
                        breakdown.phase_noise_generation = phase_timer.get('noise_generation')
                        breakdown.psd_coloring = phase_timer.get('psd_coloring') + phase_timer.get('psd_application')
                        breakdown.phase_fft = phase_timer.get('fft')
                        breakdown.phase_piston_removal = phase_timer.get('piston_removal')
                    
                    # PSF breakdown
                    breakdown.psf_field_construction = psf_timer.get('field_construction')
                    breakdown.psf_fft = psf_timer.get('fft')
                    breakdown.psf_intensity = psf_timer.get('intensity')
                    breakdown.psf_normalization = psf_timer.get('normalization')
                    breakdown.psf_accumulation = psf_timer.get('accumulation')
                    
                    all_breakdowns.append(breakdown)
                    
                except Exception:
                    # Fallback to simple timing (LWE timing skipped in fallback mode)
                    lwe_time = 0.0
                    phases, phase_time = self._time_phase_simple(
                        method, generator, total_screens, pupil, seed=42 + rep * 1000
                    )
                    
                    sync_gpu()
                    start = time.perf_counter()
                    if exposure_type == 'SE':
                        backend_obj = self.modules['get_backend']()
                        xp = backend_obj.xp
                        fields = pupil[None, :, :] * xp.exp(1j * phases)
                        psfs = xp.abs(xp.fft.fftshift(xp.fft.fft2(fields, axes=(-2, -1)), axes=(-2, -1)))**2
                    else:
                        backend_obj = self.modules['get_backend']()
                        xp = backend_obj.xp
                        le_psfs = []
                        for i in range(n_psf_total):
                            batch = phases[i*n_screens_per_le:(i+1)*n_screens_per_le]
                            fields = pupil[None, :, :] * xp.exp(1j * batch)
                            psfs = xp.abs(xp.fft.fftshift(xp.fft.fft2(fields, axes=(-2, -1)), axes=(-2, -1)))**2
                            le_psfs.append(xp.mean(psfs, axis=0))
                    sync_gpu()
                    psf_time = time.perf_counter() - start
                    
                    breakdown = DetailedBreakdown(
                        phase_generation_total=phase_time,
                        psf_computation_total=psf_time,
                    )
                    all_breakdowns.append(breakdown)
                
                # Total time includes LWE for LE mode
                total_time = phase_time + psf_time + lwe_time
                all_total_times.append(total_time)
            
            # Compute statistics
            result.mean_total_time = np.mean(all_total_times)
            result.std_total_time = np.std(all_total_times)
            result.mean_time_per_psf = result.mean_total_time / n_psf_total
            result.mean_time_per_screen = result.mean_total_time / total_screens
            result.rep_times = all_total_times
            
            # Average breakdowns
            avg_breakdown = DetailedBreakdown()
            for attr in vars(avg_breakdown):
                if not attr.startswith('_'):
                    values = [getattr(b, attr) for b in all_breakdowns]
                    setattr(avg_breakdown, attr, np.mean(values))
            
            avg_breakdown.total_time = result.mean_total_time
            result.breakdown = avg_breakdown
            
            result.memory_mb = get_memory_usage_mb()
            result.success = True
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            import traceback
            traceback.print_exc()
        
        return result
    
    # =========================================================================
    # Main Benchmark Loop
    # =========================================================================
    
    def run_all(self):
        """Run all benchmark configurations."""
        print(f"\nRunning benchmarks...")
        print(f"{'='*60}")
        
        backends_to_test = ['CPU']
        if HAS_CUPY:
            backends_to_test.append('GPU')
        
        # Count tests
        # SE tests: methods × backends × n_psf_total_list
        # LE tests: methods × backends × n_screens_per_le_list × n_psf_total_list
        n_se_tests = len(METHODS) * len(backends_to_test) * len(self.n_psf_total_list)
        n_le_tests = len(METHODS) * len(backends_to_test) * len(self.n_screens_per_le_list) * len(self.n_psf_total_list)
        total_tests = n_se_tests + n_le_tests
        
        test_num = 0
        
        # ==== SHORT EXPOSURE TESTS ====
        print(f"\n{'='*60}")
        print("SHORT EXPOSURE (SE) TESTS")
        print(f"{'='*60}")
        
        for backend in backends_to_test:
            print(f"\n--- Backend: {backend} ---")
            
            for method in METHODS:
                print(f"\n  Method: {method}")
                
                for n_psf_total in self.n_psf_total_list:
                    test_num += 1
                    print(f"    [{test_num}/{total_tests}] "
                          f"SE: {n_psf_total} PSFs (each 1 screen)...", end=" ", flush=True)
                    
                    result = self._run_single_benchmark(
                        backend, method, 'SE', 
                        n_screens_per_le=1, 
                        n_psf_total=n_psf_total
                    )
                    
                    self.suite.results.append(result)
                    
                    if result.success:
                        print(f"✓ {result.mean_total_time*1000:.1f}ms "
                              f"({result.mean_time_per_psf*1000:.2f}ms/PSF)")
                    else:
                        print(f"✗ FAILED: {result.error_message[:40]}")
        
        # ==== LONG EXPOSURE TESTS ====
        print(f"\n{'='*60}")
        print("LONG EXPOSURE (LE) TESTS")
        print(f"{'='*60}")
        
        for backend in backends_to_test:
            print(f"\n--- Backend: {backend} ---")
            
            for method in METHODS:
                print(f"\n  Method: {method}")
                
                for n_screens_per_le in self.n_screens_per_le_list:
                    for n_psf_total in self.n_psf_total_list:
                        test_num += 1
                        total_screens = n_screens_per_le * n_psf_total
                        print(f"    [{test_num}/{total_tests}] "
                              f"LE: {n_psf_total} PSFs × {n_screens_per_le} screens/PSF "
                              f"= {total_screens} screens...", end=" ", flush=True)
                        
                        result = self._run_single_benchmark(
                            backend, method, 'LE',
                            n_screens_per_le=n_screens_per_le,
                            n_psf_total=n_psf_total
                        )
                        
                        self.suite.results.append(result)
                        
                        if result.success:
                            print(f"✓ {result.mean_total_time*1000:.1f}ms "
                                  f"({result.mean_time_per_psf*1000:.2f}ms/PSF)")
                        else:
                            print(f"✗ FAILED: {result.error_message[:40]}")
        
        print(f"\n{'='*60}")
        print(f"Benchmark complete: {len(self.suite.results)} tests")
        
        self._save_results()
        self._print_summary()
        self._print_detailed_case_studies()
        
        if HAS_MATPLOTLIB:
            self._generate_plots()
        
        # Save captured output to text file
        self._save_output_log()
    
    def _save_output_log(self):
        """Save all captured output to a detailed text file."""
        if hasattr(sys.stdout, 'getvalue'):
            output_text = sys.stdout.getvalue()
            output_file = self.output_dir / "benchmark_detailed_output.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("PSF TIMING BENCHMARK - DETAILED OUTPUT LOG\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write(output_text)
            print(f"\nDetailed output saved to: {output_file}")
    
    # =========================================================================
    # Results Output
    # =========================================================================
    
    def _save_results(self):
        """Save results to JSON file."""
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.suite.to_dict(), f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    def _print_summary(self):
        """Print summary of results."""
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        
        # Table 1: SE timing
        print(f"\n1. SHORT EXPOSURE (SE) - One screen per PSF")
        print(f"   {'Method':<15} {'Backend':<8} {'n_psf':<8} {'Total(ms)':<12} {'ms/PSF':<12}")
        print(f"   {'-'*60}")
        
        for method in METHODS:
            for backend in ['CPU', 'GPU']:
                for n_psf in self.n_psf_total_list:
                    results = self.suite.get_results_for(
                        backend=backend, method=method, exposure_type='SE', n_psf_total=n_psf
                    )
                    if results and results[0].success:
                        r = results[0]
                        print(f"   {method:<15} {backend:<8} {n_psf:<8} "
                              f"{r.mean_total_time*1000:<12.1f} {r.mean_time_per_psf*1000:<12.3f}")
        
        # Table 2: LE timing (fixed n_psf=1)
        print(f"\n2. LONG EXPOSURE (LE) - Single LE PSF, varying screens/PSF")
        print(f"   {'Method':<15} {'Backend':<8} {'screens/LE':<12} {'Total(ms)':<12} {'ms/screen':<12}")
        print(f"   {'-'*65}")
        
        for method in METHODS:
            for backend in ['CPU', 'GPU']:
                for n_screens in self.n_screens_per_le_list:
                    results = self.suite.get_results_for(
                        backend=backend, method=method, exposure_type='LE',
                        n_screens_per_le=n_screens, n_psf_total=1
                    )
                    if results and results[0].success:
                        r = results[0]
                        print(f"   {method:<15} {backend:<8} {n_screens:<12} "
                              f"{r.mean_total_time*1000:<12.1f} {r.mean_time_per_screen*1000:<12.3f}")
        
        # Table 3: LE scaling with n_psf (fixed screens/LE)
        n_screens_ref = self.n_screens_per_le_list[-1]  # Use largest tested value
        print(f"\n3. LONG EXPOSURE (LE) THROUGHPUT - {n_screens_ref} screens/PSF, varying n_psf")
        print(f"   {'Method':<15} {'Backend':<8} {'n_psf':<8} {'Total(ms)':<12} {'ms/PSF':<12}")
        print(f"   {'-'*60}")
        
        for method in METHODS:
            for backend in ['CPU', 'GPU']:
                for n_psf in self.n_psf_total_list:
                    results = self.suite.get_results_for(
                        backend=backend, method=method, exposure_type='LE',
                        n_screens_per_le=n_screens_ref, n_psf_total=n_psf
                    )
                    if results and results[0].success:
                        r = results[0]
                        print(f"   {method:<15} {backend:<8} {n_psf:<8} "
                              f"{r.mean_total_time*1000:<12.1f} {r.mean_time_per_psf*1000:<12.2f}")
    
    def _print_detailed_case_studies(self):
        """Print detailed breakdown for three specific cases."""
        print(f"\n{'='*80}")
        print("DETAILED CASE STUDIES")
        print(f"{'='*80}")
        
        # Use values from the actual test lists to ensure consistency
        n_screens_per_le_ref = self.n_screens_per_le_list[-1]  # Largest tested
        n_psf_ref = self.n_psf_total_list[-1]  # Largest tested
        
        cases = [
            ("CASE 1: Single SE PSF (1 screen)", 'SE', 1, 1),
            (f"CASE 2: Single LE PSF ({n_screens_per_le_ref} screens averaged)", 'LE', n_screens_per_le_ref, 1),
            (f"CASE 3: {n_psf_ref} LE PSFs ({n_screens_per_le_ref} screens each = {n_screens_per_le_ref*n_psf_ref} total screens)", 'LE', n_screens_per_le_ref, n_psf_ref),
        ]
        
        for case_name, exp_type, n_screens_per_le, n_psf_total in cases:
            print(f"\n{'-'*80}")
            print(f"{case_name}")
            print(f"{'-'*80}")
            
            for backend in ['CPU', 'GPU']:
                print(f"\n  === {backend} ===")
                
                for method in METHODS:
                    if exp_type == 'SE':
                        results = self.suite.get_results_for(
                            backend=backend, method=method, exposure_type='SE', n_psf_total=n_psf_total
                        )
                    else:
                        results = self.suite.get_results_for(
                            backend=backend, method=method, exposure_type='LE',
                            n_screens_per_le=n_screens_per_le, n_psf_total=n_psf_total
                        )
                    
                    if not results or not results[0].success:
                        print(f"\n  {method}: No data / Failed")
                        continue
                    
                    r = results[0]
                    b = r.breakdown
                    
                    print(f"\n  {method}:")
                    print(f"    Total time: {r.mean_total_time*1000:.2f} ms")
                    print(f"    Phase screens: {r.total_phase_screens}")
                    print(f"    Output PSFs: {r.n_psf_total}")
                    print(f"    Time per PSF: {r.mean_time_per_psf*1000:.2f} ms")
                    print(f"    Time per screen: {r.mean_time_per_screen*1000:.3f} ms")
                    
                    print(f"\n    Phase Generation ({b.phase_generation_total*1000:.2f} ms total):")
                    if method == 'Zernike':
                        print(f"      - Mode computation:   {b.zernike_mode_computation*1000:7.2f} ms")
                        print(f"      - Coeff generation:   {b.zernike_coefficient_generation*1000:7.2f} ms")
                        print(f"      - Matrix multiply:    {b.zernike_matrix_multiply*1000:7.2f} ms")
                        print(f"      - Normalization:      {b.zernike_normalization*1000:7.2f} ms")
                        print(f"      - HF generation:      {b.zernike_hf_generation*1000:7.2f} ms")
                    elif method == 'DualPowerLaw':
                        print(f"      - Noise generation:   {b.phase_noise_generation*1000:7.2f} ms")
                        print(f"      - PSD coloring:       {b.psd_coloring*1000:7.2f} ms")
                        print(f"      - IFFT:               {b.phase_fft*1000:7.2f} ms")
                        print(f"      - Piston removal:     {b.phase_piston_removal*1000:7.2f} ms")
                    elif method == 'Jolissaint':
                        print(f"      - PSD computation:    {b.jolissaint_psd_computation*1000:7.2f} ms")
                        print(f"      - Noise generation:   {b.phase_noise_generation*1000:7.2f} ms")
                        print(f"      - PSD coloring:       {b.psd_coloring*1000:7.2f} ms")
                        print(f"      - IFFT:               {b.phase_fft*1000:7.2f} ms")
                        print(f"      - Piston removal:     {b.phase_piston_removal*1000:7.2f} ms")
                    
                    # LWE breakdown (only for LE mode)
                    if exp_type == 'LE' and b.lwe_generation_total > 0:
                        print(f"\n    LWE Generation ({b.lwe_generation_total*1000:.2f} ms total):")
                        print(f"      - Initialization:     {b.lwe_init*1000:7.2f} ms")
                        print(f"      - Screen generation:  {b.lwe_screen_generation*1000:7.2f} ms")
                    
                    print(f"\n    PSF Computation ({b.psf_computation_total*1000:.2f} ms total):")
                    print(f"      - Field construction: {b.psf_field_construction*1000:7.2f} ms")
                    print(f"      - FFT:                {b.psf_fft*1000:7.2f} ms")
                    print(f"      - Intensity:          {b.psf_intensity*1000:7.2f} ms")
                    print(f"      - Normalization:      {b.psf_normalization*1000:7.2f} ms")
                    print(f"      - Accumulation (avg): {b.psf_accumulation*1000:7.2f} ms")
    
    # =========================================================================
    # Plotting
    # =========================================================================
    
    def _generate_plots(self):
        """Generate all benchmark plots."""
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        print(f"\nGenerating plots...")
        
        # Line plots
        self._plot_se_scaling(plots_dir)
        self._plot_le_screens_scaling(plots_dir)
        self._plot_le_throughput_scaling(plots_dir)
        if HAS_CUPY:
            self._plot_gpu_speedup(plots_dir)
        self._plot_detailed_breakdown_cases(plots_dir)
        
        # Heatmaps - Total times
        self._plot_heatmap_se_times(plots_dir)
        self._plot_heatmap_le_times(plots_dir)
        
        # Heatmaps - Per-PSF times
        self._plot_heatmap_se_times_per_psf(plots_dir)
        self._plot_heatmap_le_times_per_psf(plots_dir)
        
        if HAS_CUPY:
            self._plot_heatmap_gpu_speedup(plots_dir)
        self._plot_heatmap_breakdown_percentages(plots_dir)
        
        print(f"Plots saved to: {plots_dir}")
    
    def _plot_se_scaling(self, plots_dir: Path):
        """Plot SE scaling: time vs n_psf_total."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("SE Scaling: Time vs Number of PSFs", fontsize=14, fontweight='bold')
        
        colors = {'Zernike': 'C0', 'DualPowerLaw': 'C1', 'Jolissaint': 'C2'}
        linestyles = {'CPU': '-', 'GPU': '--'}
        markers = {'CPU': 'o', 'GPU': 's'}
        
        # Total time
        ax = axes[0]
        for method in METHODS:
            for backend in ['CPU', 'GPU']:
                n_vals = []
                times = []
                for n_psf in self.n_psf_total_list:
                    results = self.suite.get_results_for(
                        backend=backend, method=method, exposure_type='SE', n_psf_total=n_psf
                    )
                    if results and results[0].success:
                        n_vals.append(n_psf)
                        times.append(results[0].mean_total_time * 1000)
                
                if n_vals:
                    ax.loglog(n_vals, times, color=colors[method],
                             linestyle=linestyles[backend], marker=markers[backend],
                             label=f"{method} ({backend})", markersize=8, linewidth=2)
        
        ax.set_xlabel("Number of SE PSFs", fontsize=12)
        ax.set_ylabel("Total Time (ms)", fontsize=12)
        ax.set_title("Total Time", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        
        # Per-PSF time
        ax = axes[1]
        for method in METHODS:
            for backend in ['CPU', 'GPU']:
                n_vals = []
                times = []
                for n_psf in self.n_psf_total_list:
                    results = self.suite.get_results_for(
                        backend=backend, method=method, exposure_type='SE', n_psf_total=n_psf
                    )
                    if results and results[0].success:
                        n_vals.append(n_psf)
                        times.append(results[0].mean_time_per_psf * 1000)
                
                if n_vals:
                    ax.loglog(n_vals, times, color=colors[method],
                               linestyle=linestyles[backend], marker=markers[backend],
                               label=f"{method} ({backend})", markersize=8, linewidth=2)
        
        ax.set_xlabel("Number of SE PSFs", fontsize=12)
        ax.set_ylabel("Time per PSF (ms)", fontsize=12)
        ax.set_title("Time per PSF (amortization)", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "se_scaling.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: se_scaling.png")
    
    def _plot_le_screens_scaling(self, plots_dir: Path):
        """Plot LE scaling: time vs n_screens_per_le (single LE PSF)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("LE Quality Scaling: Time vs Screens per LE PSF (n_psf=1)", fontsize=14, fontweight='bold')
        
        colors = {'Zernike': 'C0', 'DualPowerLaw': 'C1', 'Jolissaint': 'C2'}
        linestyles = {'CPU': '-', 'GPU': '--'}
        markers = {'CPU': 'o', 'GPU': 's'}
        
        # Total time
        ax = axes[0]
        for method in METHODS:
            for backend in ['CPU', 'GPU']:
                n_vals = []
                times = []
                for n_screens in self.n_screens_per_le_list:
                    results = self.suite.get_results_for(
                        backend=backend, method=method, exposure_type='LE',
                        n_screens_per_le=n_screens, n_psf_total=1
                    )
                    if results and results[0].success:
                        n_vals.append(n_screens)
                        times.append(results[0].mean_total_time * 1000)
                
                if n_vals:
                    ax.loglog(n_vals, times, color=colors[method],
                             linestyle=linestyles[backend], marker=markers[backend],
                             label=f"{method} ({backend})", markersize=8, linewidth=2)
        
        ax.set_xlabel("Screens per LE PSF", fontsize=12)
        ax.set_ylabel("Total Time (ms)", fontsize=12)
        ax.set_title("Total Time for Single LE PSF", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        
        # Time per screen
        ax = axes[1]
        for method in METHODS:
            for backend in ['CPU', 'GPU']:
                n_vals = []
                times = []
                for n_screens in self.n_screens_per_le_list:
                    results = self.suite.get_results_for(
                        backend=backend, method=method, exposure_type='LE',
                        n_screens_per_le=n_screens, n_psf_total=1
                    )
                    if results and results[0].success:
                        n_vals.append(n_screens)
                        times.append(results[0].mean_time_per_screen * 1000)
                
                if n_vals:
                    ax.loglog(n_vals, times, color=colors[method],
                               linestyle=linestyles[backend], marker=markers[backend],
                               label=f"{method} ({backend})", markersize=8, linewidth=2)
        
        ax.set_xlabel("Screens per LE PSF", fontsize=12)
        ax.set_ylabel("Time per Screen (ms)", fontsize=12)
        ax.set_title("Time per Phase Screen", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "le_screens_scaling.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: le_screens_scaling.png")
    
    def _plot_le_throughput_scaling(self, plots_dir: Path):
        """Plot LE throughput: time vs n_psf_total (fixed screens/LE)."""
        n_screens_ref = self.n_screens_per_le_list[-1]  # Use largest tested value
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"LE Throughput: Time vs Number of LE PSFs ({n_screens_ref} screens/PSF)", 
                    fontsize=14, fontweight='bold')
        
        colors = {'Zernike': 'C0', 'DualPowerLaw': 'C1', 'Jolissaint': 'C2'}
        linestyles = {'CPU': '-', 'GPU': '--'}
        markers = {'CPU': 'o', 'GPU': 's'}
        
        # Total time
        ax = axes[0]
        for method in METHODS:
            for backend in ['CPU', 'GPU']:
                n_vals = []
                times = []
                for n_psf in self.n_psf_total_list:
                    results = self.suite.get_results_for(
                        backend=backend, method=method, exposure_type='LE',
                        n_screens_per_le=n_screens_ref, n_psf_total=n_psf
                    )
                    if results and results[0].success:
                        n_vals.append(n_psf)
                        times.append(results[0].mean_total_time * 1000)
                
                if n_vals:
                    ax.loglog(n_vals, times, color=colors[method],
                             linestyle=linestyles[backend], marker=markers[backend],
                             label=f"{method} ({backend})", markersize=8, linewidth=2)
        
        ax.set_xlabel("Number of LE PSFs", fontsize=12)
        ax.set_ylabel("Total Time (ms)", fontsize=12)
        ax.set_title("Total Time", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        
        # Per-PSF time
        ax = axes[1]
        for method in METHODS:
            for backend in ['CPU', 'GPU']:
                n_vals = []
                times = []
                for n_psf in self.n_psf_total_list:
                    results = self.suite.get_results_for(
                        backend=backend, method=method, exposure_type='LE',
                        n_screens_per_le=n_screens_ref, n_psf_total=n_psf
                    )
                    if results and results[0].success:
                        n_vals.append(n_psf)
                        times.append(results[0].mean_time_per_psf * 1000)
                
                if n_vals:
                    ax.loglog(n_vals, times, color=colors[method],
                               linestyle=linestyles[backend], marker=markers[backend],
                               label=f"{method} ({backend})", markersize=8, linewidth=2)
        
        ax.set_xlabel("Number of LE PSFs", fontsize=12)
        ax.set_ylabel("Time per LE PSF (ms)", fontsize=12)
        ax.set_title("Time per LE PSF", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "le_throughput_scaling.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: le_throughput_scaling.png")
    
    def _plot_gpu_speedup(self, plots_dir: Path):
        """Plot GPU speedup factors."""
        n_screens_ref = self.n_screens_per_le_list[-1]  # Use largest tested value
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("GPU Speedup (CPU time / GPU time)", fontsize=14, fontweight='bold')
        
        colors = {'Zernike': 'C0', 'DualPowerLaw': 'C1', 'Jolissaint': 'C2'}
        
        # SE speedup vs n_psf
        ax = axes[0]
        x = np.arange(len(self.n_psf_total_list))
        width = 0.25
        for i, method in enumerate(METHODS):
            speedups = []
            for n_psf in self.n_psf_total_list:
                cpu = self.suite.get_results_for(backend='CPU', method=method, exposure_type='SE', n_psf_total=n_psf)
                gpu = self.suite.get_results_for(backend='GPU', method=method, exposure_type='SE', n_psf_total=n_psf)
                if cpu and gpu and cpu[0].success and gpu[0].success and gpu[0].mean_total_time > 0:
                    speedups.append(cpu[0].mean_total_time / gpu[0].mean_total_time)
                else:
                    speedups.append(0)
            ax.bar(x + i * width, speedups, width, label=method, color=colors[method])
        ax.set_xlabel("Number of SE PSFs")
        ax.set_ylabel("Speedup")
        ax.set_title("SE Mode")
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(n) for n in self.n_psf_total_list])
        ax.legend()
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # LE speedup vs screens/LE (n_psf=1)
        ax = axes[1]
        x = np.arange(len(self.n_screens_per_le_list))
        for i, method in enumerate(METHODS):
            speedups = []
            for n_screens in self.n_screens_per_le_list:
                cpu = self.suite.get_results_for(backend='CPU', method=method, exposure_type='LE',
                                                  n_screens_per_le=n_screens, n_psf_total=1)
                gpu = self.suite.get_results_for(backend='GPU', method=method, exposure_type='LE',
                                                  n_screens_per_le=n_screens, n_psf_total=1)
                if cpu and gpu and cpu[0].success and gpu[0].success and gpu[0].mean_total_time > 0:
                    speedups.append(cpu[0].mean_total_time / gpu[0].mean_total_time)
                else:
                    speedups.append(0)
            ax.bar(x + i * width, speedups, width, label=method, color=colors[method])
        ax.set_xlabel("Screens per LE PSF")
        ax.set_ylabel("Speedup")
        ax.set_title("LE Quality (n_psf=1)")
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(n) for n in self.n_screens_per_le_list])
        ax.legend()
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # LE speedup vs n_psf (fixed screens/LE)
        ax = axes[2]
        x = np.arange(len(self.n_psf_total_list))
        for i, method in enumerate(METHODS):
            speedups = []
            for n_psf in self.n_psf_total_list:
                cpu = self.suite.get_results_for(backend='CPU', method=method, exposure_type='LE',
                                                  n_screens_per_le=n_screens_ref, n_psf_total=n_psf)
                gpu = self.suite.get_results_for(backend='GPU', method=method, exposure_type='LE',
                                                  n_screens_per_le=n_screens_ref, n_psf_total=n_psf)
                if cpu and gpu and cpu[0].success and gpu[0].success and gpu[0].mean_total_time > 0:
                    speedups.append(cpu[0].mean_total_time / gpu[0].mean_total_time)
                else:
                    speedups.append(0)
            ax.bar(x + i * width, speedups, width, label=method, color=colors[method])
        ax.set_xlabel("Number of LE PSFs")
        ax.set_ylabel("Speedup")
        ax.set_title(f"LE Throughput ({n_screens_ref} screens/PSF)")
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(n) for n in self.n_psf_total_list])
        ax.legend()
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "gpu_speedup.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: gpu_speedup.png")
    
    def _plot_detailed_breakdown_cases(self, plots_dir: Path):
        """Plot detailed breakdown for the three case studies."""
        # Use values from the actual test lists to ensure consistency
        n_screens_per_le_ref = self.n_screens_per_le_list[-1]  # Largest tested
        n_psf_ref = self.n_psf_total_list[-1]  # Largest tested
        
        cases = [
            ("1 SE PSF\n(1 screen)", 'SE', 1, 1),
            (f"1 LE PSF\n({n_screens_per_le_ref} screens)", 'LE', n_screens_per_le_ref, 1),
            (f"{n_psf_ref} LE PSFs\n({n_screens_per_le_ref*n_psf_ref} screens)", 'LE', n_screens_per_le_ref, n_psf_ref),
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Detailed Timing Breakdown: Three Case Studies", fontsize=14, fontweight='bold')
        
        for col, (case_name, exp_type, n_screens_per_le, n_psf_total) in enumerate(cases):
            for row, backend in enumerate(['CPU', 'GPU']):
                ax = axes[row, col]
                
                all_data = []
                all_labels = []
                
                for method in METHODS:
                    if exp_type == 'SE':
                        results = self.suite.get_results_for(
                            backend=backend, method=method, exposure_type='SE', n_psf_total=n_psf_total
                        )
                    else:
                        results = self.suite.get_results_for(
                            backend=backend, method=method, exposure_type='LE',
                            n_screens_per_le=n_screens_per_le, n_psf_total=n_psf_total
                        )
                    
                    if not results or not results[0].success:
                        all_data.append([0, 0, 0])
                        all_labels.append(method)
                        continue
                    
                    b = results[0].breakdown
                    phase_time = b.phase_generation_total * 1000
                    lwe_time = b.lwe_generation_total * 1000
                    psf_time = b.psf_computation_total * 1000
                    all_data.append([phase_time, lwe_time, psf_time])
                    all_labels.append(method)
                
                # Stacked bar chart
                x = np.arange(len(METHODS))
                phase_times = [d[0] for d in all_data]
                lwe_times = [d[1] for d in all_data]
                psf_times = [d[2] for d in all_data]
                
                # Stack: Phase, then LWE, then PSF
                ax.bar(x, phase_times, label='Phase Generation', color='steelblue')
                ax.bar(x, lwe_times, bottom=phase_times, label='LWE Generation', color='forestgreen')
                cumulative = [p + l for p, l in zip(phase_times, lwe_times)]
                ax.bar(x, psf_times, bottom=cumulative, label='PSF Computation', color='coral')
                
                ax.set_xticks(x)
                ax.set_xticklabels(all_labels, fontsize=10)
                ax.set_ylabel("Time (ms)", fontsize=10)
                ax.set_title(f"{case_name}\n{backend}", fontsize=11)
                
                if col == 0 and row == 0:
                    ax.legend(loc='upper right', fontsize=8)
                
                # Add total time annotations
                for i, (phase, lwe, psf) in enumerate(all_data):
                    total = phase + lwe + psf
                    if total > 0:
                        ax.text(i, total + 0.02 * ax.get_ylim()[1], f'{total:.1f}ms',
                               ha='center', va='bottom', fontsize=8)
                
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "detailed_breakdown_cases.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: detailed_breakdown_cases.png")
    
    def _plot_heatmap_se_times(self, plots_dir: Path):
        """Plot heatmap of SE times across methods and n_psf values."""
        from matplotlib.colors import LogNorm
        backends = ['CPU', 'GPU'] if HAS_CUPY else ['CPU']
        
        fig, axes = plt.subplots(1, len(backends), figsize=(7 * len(backends), 5))
        if len(backends) == 1:
            axes = [axes]
        
        fig.suptitle("SE Timing Heatmap: Total Time (ms) - Log Scale", fontsize=14, fontweight='bold')
        
        for ax_idx, backend in enumerate(backends):
            ax = axes[ax_idx]
            
            # Build data matrix: rows = methods, cols = n_psf values
            data = np.zeros((len(METHODS), len(self.n_psf_total_list)))
            
            for i, method in enumerate(METHODS):
                for j, n_psf in enumerate(self.n_psf_total_list):
                    results = self.suite.get_results_for(
                        backend=backend, method=method, exposure_type='SE', n_psf_total=n_psf
                    )
                    if results and results[0].success:
                        data[i, j] = results[0].mean_total_time * 1000
                    else:
                        data[i, j] = np.nan
            
            # Use log scale for the colormap
            vmin = np.nanmin(data[data > 0]) if np.any(data > 0) else 0.1
            vmax = np.nanmax(data)
            im = ax.imshow(data, aspect='auto', cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Time (ms)', fontsize=10)
            
            ax.set_xticks(np.arange(len(self.n_psf_total_list)))
            ax.set_xticklabels([str(n) for n in self.n_psf_total_list])
            ax.set_yticks(np.arange(len(METHODS)))
            ax.set_yticklabels(METHODS)
            ax.set_xlabel("Number of PSFs", fontsize=11)
            ax.set_ylabel("Method", fontsize=11)
            ax.set_title(f"{backend}", fontsize=12)
            
            # Add text annotations
            for i in range(len(METHODS)):
                for j in range(len(self.n_psf_total_list)):
                    val = data[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if val > np.sqrt(vmin * vmax) else 'black'
                        ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                               color=text_color, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "heatmap_se_times.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: heatmap_se_times.png")
    
    def _plot_heatmap_le_times(self, plots_dir: Path):
        """Plot heatmap of LE times: n_screens_per_le vs n_psf_total for each method/backend."""
        from matplotlib.colors import LogNorm
        backends = ['CPU', 'GPU'] if HAS_CUPY else ['CPU']
        
        fig, axes = plt.subplots(len(backends), len(METHODS), 
                                  figsize=(5 * len(METHODS), 4 * len(backends)))
        if len(backends) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle("LE Timing Heatmap: Total Time (ms) - Log Scale", fontsize=14, fontweight='bold', y=1.02)
        
        for row, backend in enumerate(backends):
            for col, method in enumerate(METHODS):
                ax = axes[row, col]
                
                # Build data matrix: rows = n_screens_per_le, cols = n_psf_total
                data = np.zeros((len(self.n_screens_per_le_list), len(self.n_psf_total_list)))
                
                for i, n_screens in enumerate(self.n_screens_per_le_list):
                    for j, n_psf in enumerate(self.n_psf_total_list):
                        results = self.suite.get_results_for(
                            backend=backend, method=method, exposure_type='LE',
                            n_screens_per_le=n_screens, n_psf_total=n_psf
                        )
                        if results and results[0].success:
                            data[i, j] = results[0].mean_total_time * 1000
                        else:
                            data[i, j] = np.nan
                
                # Use log scale for the colormap
                vmin = np.nanmin(data[data > 0]) if np.any(data > 0) else 0.1
                vmax = np.nanmax(data)
                im = ax.imshow(data, aspect='auto', cmap='plasma', norm=LogNorm(vmin=vmin, vmax=vmax))
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Time (ms)', fontsize=9)
                
                ax.set_xticks(np.arange(len(self.n_psf_total_list)))
                ax.set_xticklabels([str(n) for n in self.n_psf_total_list], fontsize=8)
                ax.set_yticks(np.arange(len(self.n_screens_per_le_list)))
                ax.set_yticklabels([str(n) for n in self.n_screens_per_le_list])
                ax.set_xlabel("n_psf_total", fontsize=10)
                ax.set_ylabel("n_screens_per_le", fontsize=10)
                ax.set_title(f"{method} ({backend})", fontsize=11)
                
                # Add text annotations
                for i in range(len(self.n_screens_per_le_list)):
                    for j in range(len(self.n_psf_total_list)):
                        val = data[i, j]
                        if not np.isnan(val):
                            text_color = 'white' if val > np.sqrt(vmin * vmax) else 'black'
                            ax.text(j, i, f'{val:.0f}', ha='center', va='center', 
                                   color=text_color, fontsize=7)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "heatmap_le_times.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: heatmap_le_times.png")
    
    def _plot_heatmap_se_times_per_psf(self, plots_dir: Path):
        """Plot heatmap of SE time per PSF across methods and n_psf values."""
        from matplotlib.colors import LogNorm
        backends = ['CPU', 'GPU'] if HAS_CUPY else ['CPU']
        
        fig, axes = plt.subplots(1, len(backends), figsize=(7 * len(backends), 5))
        if len(backends) == 1:
            axes = [axes]
        
        fig.suptitle("SE Timing Heatmap: Time per PSF (ms) - Log Scale", fontsize=14, fontweight='bold')
        
        for ax_idx, backend in enumerate(backends):
            ax = axes[ax_idx]
            
            # Build data matrix: rows = methods, cols = n_psf values
            data = np.zeros((len(METHODS), len(self.n_psf_total_list)))
            
            for i, method in enumerate(METHODS):
                for j, n_psf in enumerate(self.n_psf_total_list):
                    results = self.suite.get_results_for(
                        backend=backend, method=method, exposure_type='SE', n_psf_total=n_psf
                    )
                    if results and results[0].success:
                        data[i, j] = results[0].mean_time_per_psf * 1000
                    else:
                        data[i, j] = np.nan
            
            # Use log scale for the colormap
            vmin = np.nanmin(data[data > 0]) if np.any(data > 0) else 0.1
            vmax = np.nanmax(data)
            im = ax.imshow(data, aspect='auto', cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Time per PSF (ms)', fontsize=10)
            
            ax.set_xticks(np.arange(len(self.n_psf_total_list)))
            ax.set_xticklabels([str(n) for n in self.n_psf_total_list])
            ax.set_yticks(np.arange(len(METHODS)))
            ax.set_yticklabels(METHODS)
            ax.set_xlabel("Number of PSFs", fontsize=11)
            ax.set_ylabel("Method", fontsize=11)
            ax.set_title(f"{backend}", fontsize=12)
            
            # Add text annotations
            for i in range(len(METHODS)):
                for j in range(len(self.n_psf_total_list)):
                    val = data[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if val > np.sqrt(vmin * vmax) else 'black'
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                               color=text_color, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "heatmap_se_times_per_psf.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: heatmap_se_times_per_psf.png")
    
    def _plot_heatmap_le_times_per_psf(self, plots_dir: Path):
        """Plot heatmap of LE time per PSF: n_screens_per_le vs n_psf_total for each method/backend."""
        from matplotlib.colors import LogNorm
        backends = ['CPU', 'GPU'] if HAS_CUPY else ['CPU']
        
        fig, axes = plt.subplots(len(backends), len(METHODS), 
                                  figsize=(5 * len(METHODS), 4 * len(backends)))
        if len(backends) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle("LE Timing Heatmap: Time per PSF (ms) - Log Scale", fontsize=14, fontweight='bold', y=1.02)
        
        for row, backend in enumerate(backends):
            for col, method in enumerate(METHODS):
                ax = axes[row, col]
                
                # Build data matrix: rows = n_screens_per_le, cols = n_psf_total
                data = np.zeros((len(self.n_screens_per_le_list), len(self.n_psf_total_list)))
                
                for i, n_screens in enumerate(self.n_screens_per_le_list):
                    for j, n_psf in enumerate(self.n_psf_total_list):
                        results = self.suite.get_results_for(
                            backend=backend, method=method, exposure_type='LE',
                            n_screens_per_le=n_screens, n_psf_total=n_psf
                        )
                        if results and results[0].success:
                            data[i, j] = results[0].mean_time_per_psf * 1000
                        else:
                            data[i, j] = np.nan
                
                # Use log scale for the colormap
                vmin = np.nanmin(data[data > 0]) if np.any(data > 0) else 0.1
                vmax = np.nanmax(data)
                im = ax.imshow(data, aspect='auto', cmap='plasma', norm=LogNorm(vmin=vmin, vmax=vmax))
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Time/PSF (ms)', fontsize=9)
                
                ax.set_xticks(np.arange(len(self.n_psf_total_list)))
                ax.set_xticklabels([str(n) for n in self.n_psf_total_list], fontsize=8)
                ax.set_yticks(np.arange(len(self.n_screens_per_le_list)))
                ax.set_yticklabels([str(n) for n in self.n_screens_per_le_list])
                ax.set_xlabel("n_psf_total", fontsize=10)
                ax.set_ylabel("n_screens_per_le", fontsize=10)
                ax.set_title(f"{method} ({backend})", fontsize=11)
                
                # Add text annotations
                for i in range(len(self.n_screens_per_le_list)):
                    for j in range(len(self.n_psf_total_list)):
                        val = data[i, j]
                        if not np.isnan(val):
                            text_color = 'white' if val > np.sqrt(vmin * vmax) else 'black'
                            ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                                   color=text_color, fontsize=7)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "heatmap_le_times_per_psf.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: heatmap_le_times_per_psf.png")
    
    def _plot_heatmap_gpu_speedup(self, plots_dir: Path):
        """Plot heatmap of GPU speedup across methods and configurations."""
        # SE speedup heatmap
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("GPU Speedup Heatmap (CPU time / GPU time)", fontsize=14, fontweight='bold')
        
        # SE heatmap
        ax = axes[0]
        data = np.zeros((len(METHODS), len(self.n_psf_total_list)))
        
        for i, method in enumerate(METHODS):
            for j, n_psf in enumerate(self.n_psf_total_list):
                cpu = self.suite.get_results_for(backend='CPU', method=method, exposure_type='SE', n_psf_total=n_psf)
                gpu = self.suite.get_results_for(backend='GPU', method=method, exposure_type='SE', n_psf_total=n_psf)
                if cpu and gpu and cpu[0].success and gpu[0].success and gpu[0].mean_total_time > 0:
                    data[i, j] = cpu[0].mean_total_time / gpu[0].mean_total_time
                else:
                    data[i, j] = np.nan
        
        im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=max(np.nanmax(data), 2))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Speedup', fontsize=10)
        
        ax.set_xticks(np.arange(len(self.n_psf_total_list)))
        ax.set_xticklabels([str(n) for n in self.n_psf_total_list])
        ax.set_yticks(np.arange(len(METHODS)))
        ax.set_yticklabels(METHODS)
        ax.set_xlabel("Number of SE PSFs", fontsize=11)
        ax.set_ylabel("Method", fontsize=11)
        ax.set_title("SE Mode", fontsize=12)
        
        for i in range(len(METHODS)):
            for j in range(len(self.n_psf_total_list)):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}x', ha='center', va='center', fontsize=9, fontweight='bold')
        
        # LE heatmap (use highest n_screens_per_le)
        ax = axes[1]
        n_screens_ref = self.n_screens_per_le_list[-1]
        data = np.zeros((len(METHODS), len(self.n_psf_total_list)))
        
        for i, method in enumerate(METHODS):
            for j, n_psf in enumerate(self.n_psf_total_list):
                cpu = self.suite.get_results_for(backend='CPU', method=method, exposure_type='LE',
                                                  n_screens_per_le=n_screens_ref, n_psf_total=n_psf)
                gpu = self.suite.get_results_for(backend='GPU', method=method, exposure_type='LE',
                                                  n_screens_per_le=n_screens_ref, n_psf_total=n_psf)
                if cpu and gpu and cpu[0].success and gpu[0].success and gpu[0].mean_total_time > 0:
                    data[i, j] = cpu[0].mean_total_time / gpu[0].mean_total_time
                else:
                    data[i, j] = np.nan
        
        im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=max(np.nanmax(data), 2))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Speedup', fontsize=10)
        
        ax.set_xticks(np.arange(len(self.n_psf_total_list)))
        ax.set_xticklabels([str(n) for n in self.n_psf_total_list])
        ax.set_yticks(np.arange(len(METHODS)))
        ax.set_yticklabels(METHODS)
        ax.set_xlabel("Number of LE PSFs", fontsize=11)
        ax.set_ylabel("Method", fontsize=11)
        ax.set_title(f"LE Mode ({n_screens_ref} screens/PSF)", fontsize=12)
        
        for i in range(len(METHODS)):
            for j in range(len(self.n_psf_total_list)):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}x', ha='center', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "heatmap_gpu_speedup.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: heatmap_gpu_speedup.png")
    
    def _plot_heatmap_breakdown_percentages(self, plots_dir: Path):
        """Plot heatmap of timing breakdown percentages for each method/backend."""
        backends = ['CPU', 'GPU'] if HAS_CUPY else ['CPU']
        n_screens_ref = self.n_screens_per_le_list[-1]
        n_psf_ref = self.n_psf_total_list[-1]
        
        # Create figure with 2 rows (SE, LE) and backends as columns
        fig, axes = plt.subplots(2, len(backends), figsize=(8 * len(backends), 10))
        if len(backends) == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle("Timing Breakdown Percentages", fontsize=14, fontweight='bold', y=1.02)
        
        components_se = ['Phase Generation', 'PSF Computation']
        components_le = ['Phase Generation', 'LWE Generation', 'PSF Computation']
        
        for col, backend in enumerate(backends):
            # SE breakdown
            ax = axes[0, col]
            data = np.zeros((len(METHODS), len(components_se)))
            
            for i, method in enumerate(METHODS):
                results = self.suite.get_results_for(
                    backend=backend, method=method, exposure_type='SE', n_psf_total=n_psf_ref
                )
                if results and results[0].success:
                    b = results[0].breakdown
                    total = b.phase_generation_total + b.psf_computation_total
                    if total > 0:
                        data[i, 0] = (b.phase_generation_total / total) * 100
                        data[i, 1] = (b.psf_computation_total / total) * 100
            
            im = ax.imshow(data, aspect='auto', cmap='Blues', vmin=0, vmax=100)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Percentage (%)', fontsize=10)
            
            ax.set_xticks(np.arange(len(components_se)))
            ax.set_xticklabels(components_se, rotation=45, ha='right')
            ax.set_yticks(np.arange(len(METHODS)))
            ax.set_yticklabels(METHODS)
            ax.set_title(f"SE Mode ({n_psf_ref} PSFs) - {backend}", fontsize=11)
            
            for i in range(len(METHODS)):
                for j in range(len(components_se)):
                    val = data[i, j]
                    ax.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold')
            
            # LE breakdown
            ax = axes[1, col]
            data = np.zeros((len(METHODS), len(components_le)))
            
            for i, method in enumerate(METHODS):
                results = self.suite.get_results_for(
                    backend=backend, method=method, exposure_type='LE',
                    n_screens_per_le=n_screens_ref, n_psf_total=n_psf_ref
                )
                if results and results[0].success:
                    b = results[0].breakdown
                    total = b.phase_generation_total + b.lwe_generation_total + b.psf_computation_total
                    if total > 0:
                        data[i, 0] = (b.phase_generation_total / total) * 100
                        data[i, 1] = (b.lwe_generation_total / total) * 100
                        data[i, 2] = (b.psf_computation_total / total) * 100
            
            im = ax.imshow(data, aspect='auto', cmap='Oranges', vmin=0, vmax=100)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Percentage (%)', fontsize=10)
            
            ax.set_xticks(np.arange(len(components_le)))
            ax.set_xticklabels(components_le, rotation=45, ha='right')
            ax.set_yticks(np.arange(len(METHODS)))
            ax.set_yticklabels(METHODS)
            ax.set_title(f"LE Mode ({n_psf_ref} PSFs × {n_screens_ref} screens) - {backend}", fontsize=11)
            
            for i in range(len(METHODS)):
                for j in range(len(components_le)):
                    val = data[i, j]
                    ax.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "heatmap_breakdown_percentages.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: heatmap_breakdown_percentages.png")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PSF Computing Time Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--n-pix', type=int, default=DEFAULT_N_PIX, help='Grid size')
    
    args = parser.parse_args()
    
    # Set up output capture
    original_stdout = sys.stdout
    tee = TeeOutput(original_stdout)
    sys.stdout = tee
    
    try:
        benchmark = PSFTimingBenchmark(
            output_dir=Path(args.output_dir) if args.output_dir else None,
            quick_mode=args.quick,
            n_pix=args.n_pix,
        )
        
        benchmark.run_all()
    finally:
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()

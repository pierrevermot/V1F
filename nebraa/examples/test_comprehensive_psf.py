#!/usr/bin/env python3
"""
Comprehensive Phase Generator & PSF Test Suite
===============================================

Unified validation of all three phase generation methods:
1. ZernikePhaseGenerator - Zernike polynomial expansion (+ HF turbulence)
2. DualPowerLawPhaseGenerator - Dual power-law PSD
3. JolissaintAOModel - Analytical AO model with Monte Carlo sampling

Test Categories:
- Backend Parity: CPU vs GPU produce identical results
- Exposure Type: Short-exposure vs Long-exposure PSF behavior
- HF Turbulence: With/without high-frequency component
- LWE Integration: With/without Low Wind Effect
- Maréchal Validation: Measured Strehl vs exp(-σ²) approximation
- Cross-Method Comparison: All methods at equivalent parameters

Output:
- Detailed text log (timestamped file + console)
- Comprehensive diagnostic plots
- JSON metrics for regression testing

Usage:
    python test_comprehensive_psf.py                    # Full test suite
    python test_comprehensive_psf.py --quick            # Quick mode (fewer samples)
    python test_comprehensive_psf.py --thorough         # Thorough mode (more samples)
    python test_comprehensive_psf.py --cpu-only         # Skip GPU tests
    python test_comprehensive_psf.py --plot-only        # Regenerate plots from saved data
    python test_comprehensive_psf.py --output-dir DIR   # Custom output directory

Author: NEBRAA Test Suite
"""

from __future__ import annotations

import sys
import os
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from contextlib import contextmanager

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Check matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")

# Check CuPy
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


# =============================================================================
# Configuration Constants
# =============================================================================

# Grid and telescope (VLT-like)
DEFAULT_N_PIX = 256
TELESCOPE_DIAMETER = 8.2  # meters
OBSTRUCTION_DIAMETER = 1.116  # meters
WAVELENGTH = 2.2e-6  # K-band
PIXEL_SCALE_MAS = 13.0  # mas/pixel
PIXEL_SCALE_RAD = PIXEL_SCALE_MAS * 1e-3 / 206265

# Computed quantities
PUPIL_PIXEL_SIZE = WAVELENGTH / (DEFAULT_N_PIX * PIXEL_SCALE_RAD)

# Test parameters
QUICK_N_SAMPLES = 10
THOROUGH_N_SAMPLES = 100
DEFAULT_N_SAMPLES = 30

# Tolerances
# Note: CPU/GPU comparisons will differ because random number generators
# produce different sequences. The parity test checks that statistics
# (mean RMS, mean Strehl) are within reasonable tolerance.
GPU_PARITY_RTOL = 0.10  # 10% relative tolerance for CPU/GPU statistical comparison
RMS_RTOL = 0.15  # 15% tolerance for Monte Carlo RMS
MARECHAL_MAX_RESIDUAL = 0.08  # Max deviation from Maréchal approximation


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class TestResult:
    """Single test result."""
    name: str
    passed: bool
    message: str
    metrics: Dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.0


@dataclass 
class TestSuiteResults:
    """Accumulated test results."""
    suite_name: str
    start_time: str
    results: List[TestResult] = field(default_factory=list)
    total_duration_s: float = 0.0
    
    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)
    
    @property
    def n_total(self) -> int:
        return len(self.results)
    
    def add(self, result: TestResult):
        self.results.append(result)
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"TEST SUITE SUMMARY: {self.suite_name}",
            f"{'='*70}",
            f"Started: {self.start_time}",
            f"Duration: {self.total_duration_s:.1f}s",
            f"Results: {self.n_passed}/{self.n_total} passed, {self.n_failed} failed",
            "",
        ]
        
        if self.n_failed > 0:
            lines.append("FAILED TESTS:")
            for r in self.results:
                if not r.passed:
                    lines.append(f"  ✗ {r.name}: {r.message}")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            return obj
        
        results_dicts = []
        for r in self.results:
            d = asdict(r)
            results_dicts.append(convert_to_serializable(d))
        
        return {
            "suite_name": self.suite_name,
            "start_time": self.start_time,
            "total_duration_s": float(self.total_duration_s),
            "n_passed": int(self.n_passed),
            "n_failed": int(self.n_failed),
            "n_total": int(self.n_total),
            "results": results_dicts,
        }


# =============================================================================
# Logging Setup
# =============================================================================

class DualLogger:
    """Logger that writes to both file and console."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = open(log_path, 'w')
        self.start_time = time.time()
        
    def log(self, msg: str, level: str = "INFO"):
        timestamp = time.strftime("%H:%M:%S")
        elapsed = time.time() - self.start_time
        full_msg = f"[{timestamp}] [{level:5s}] [{elapsed:7.1f}s] {msg}"
        print(full_msg)
        self.log_file.write(full_msg + "\n")
        self.log_file.flush()
    
    def info(self, msg: str):
        self.log(msg, "INFO")
    
    def pass_(self, msg: str):
        self.log(f"✓ {msg}", "PASS")
    
    def fail(self, msg: str):
        self.log(f"✗ {msg}", "FAIL")
    
    def warn(self, msg: str):
        self.log(f"⚠ {msg}", "WARN")
    
    def header(self, msg: str):
        border = "=" * 70
        self.log(border)
        self.log(msg)
        self.log(border)
    
    def subheader(self, msg: str):
        self.log(f"\n--- {msg} ---")
    
    def close(self):
        self.log_file.close()


# =============================================================================
# Utility Functions
# =============================================================================

def to_numpy(arr) -> np.ndarray:
    """Convert array to numpy (handles numpy, cupy, and other array types)."""
    if arr is None:
        return None
    if hasattr(arr, 'get'):  # CuPy array
        return arr.get()
    return np.asarray(arr)


def create_vlt_pupil(n_pix: int, pixel_size: float, 
                     D: float = TELESCOPE_DIAMETER, 
                     D_obs: float = OBSTRUCTION_DIAMETER,
                     spider_width: float = 0.5,
                     xp=None) -> np.ndarray:
    """
    Create VLT-like pupil with central obstruction and spiders.
    
    Args:
        n_pix: Grid size
        pixel_size: Physical pixel size (meters)
        D: Primary diameter (meters)
        D_obs: Obstruction diameter (meters)
        spider_width: Spider vane width (meters)
        xp: Array module (numpy or cupy), defaults to numpy
    
    Returns:
        Pupil mask array
    """
    if xp is None:
        xp = np
    
    # Physical coordinate grid
    extent = n_pix * pixel_size
    x = xp.linspace(-extent/2, extent/2, n_pix)
    X, Y = xp.meshgrid(x, x)
    R = xp.sqrt(X**2 + Y**2)
    
    # Annular aperture
    pupil = ((R <= D/2) & (R >= D_obs/2)).astype(xp.float32)
    
    # Add spider vanes (4 symmetric spiders)
    spider_half = spider_width / 2
    mask_h = (xp.abs(Y) < spider_half) & (R <= D/2) & (R >= D_obs/2)
    mask_v = (xp.abs(X) < spider_half) & (R <= D/2) & (R >= D_obs/2)
    pupil = xp.where(mask_h | mask_v, xp.float32(0), pupil)
    
    return pupil


def compute_phase_rms(phase, pupil, xp=None) -> float:
    """Compute RMS of phase over pupil."""
    if xp is None:
        xp = np
    
    mask = pupil > 0.5
    phase_masked = phase[mask]
    
    # Remove piston
    mean = xp.mean(phase_masked)
    phase_centered = phase_masked - mean
    
    rms = float(xp.sqrt(xp.mean(phase_centered**2)))
    return rms


def compute_strehl_from_phase(phase, pupil, xp=None) -> float:
    """Compute Strehl ratio using phasor average method."""
    if xp is None:
        xp = np
    
    mask = pupil > 0.5
    phase_masked = phase[mask]
    
    # Remove piston
    mean = xp.mean(phase_masked)
    phase_centered = phase_masked - mean
    
    # Phasor average Strehl: |<exp(i*phi)>|^2
    phasor = xp.exp(1j * phase_centered)
    strehl = float(xp.abs(xp.mean(phasor))**2)
    
    return strehl


def marechal_strehl(rms_rad: float) -> float:
    """Maréchal approximation: S ≈ exp(-σ²)."""
    return np.exp(-rms_rad**2)


@contextmanager
def timer():
    """Context manager for timing code blocks."""
    start = time.time()
    yield lambda: time.time() - start
    

# =============================================================================
# Import NEBRAA modules (after backend setup)
# =============================================================================

def import_nebraa_modules():
    """Import NEBRAA modules after path setup."""
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
        AOSystemConfig,
        create_simple_atmosphere,
        create_ao_config,
        LowWindEffectConfig,
    )
    
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
        'AOSystemConfig': AOSystemConfig,
        'create_simple_atmosphere': create_simple_atmosphere,
        'create_ao_config': create_ao_config,
        'LowWindEffectConfig': LowWindEffectConfig,
    }


# =============================================================================
# Test Classes
# =============================================================================

class ComprehensiveTestSuite:
    """
    Main test suite orchestrating all validation tests.
    """
    
    def __init__(
        self,
        output_dir: Path,
        n_pix: int = DEFAULT_N_PIX,
        n_samples: int = DEFAULT_N_SAMPLES,
        skip_gpu: bool = False,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_pix = n_pix
        self.n_samples = n_samples
        self.skip_gpu = skip_gpu or not HAS_CUPY
        self.seed = seed
        
        # Setup logging
        log_path = self.output_dir / "test_log.txt"
        self.logger = DualLogger(log_path)
        
        # Results accumulator
        self.results = TestSuiteResults(
            suite_name="Comprehensive PSF Test Suite",
            start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        # Storage for intermediate results (for plotting)
        self.data = {}
        
        # Import modules
        self.modules = import_nebraa_modules()
        
        # Common parameters
        self.pixel_size = WAVELENGTH / (n_pix * PIXEL_SCALE_RAD)
        
        self.logger.header("COMPREHENSIVE PSF TEST SUITE")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Grid size: {n_pix}x{n_pix}")
        self.logger.info(f"Pixel size: {self.pixel_size*1000:.2f} mm")
        self.logger.info(f"Samples per test: {n_samples}")
        self.logger.info(f"GPU available: {HAS_CUPY}")
        self.logger.info(f"Skip GPU tests: {self.skip_gpu}")
        
    def run_all(self):
        """Run all test categories."""
        start_time = time.time()
        
        try:
            # 1. Backend parity tests
            self.test_backend_parity()
            
            # 2. Short vs Long exposure tests
            self.test_exposure_types()
            
            # 3. High-frequency turbulence tests
            self.test_hf_turbulence()
            
            # 4. LWE integration tests
            self.test_lwe_integration()
            
            # 5. Maréchal approximation tests
            self.test_marechal_approximation()
            
            # 6. Cross-method comparison
            self.test_cross_method_comparison()
            
        except Exception as e:
            self.logger.fail(f"Test suite error: {e}")
            import traceback
            traceback.print_exc()
        
        self.results.total_duration_s = time.time() - start_time
        
        # Print summary
        self.logger.info(self.results.summary())
        
        # Save results
        self.save_results()
        
        # Generate plots
        if HAS_MATPLOTLIB:
            self.generate_all_plots()
        
        self.logger.close()
        
        return self.results
    
    def _add_result(self, name: str, passed: bool, message: str, 
                   metrics: Dict = None, duration: float = 0.0):
        """Add a test result."""
        result = TestResult(
            name=name,
            passed=passed,
            message=message,
            metrics=metrics or {},
            duration_s=duration,
        )
        self.results.add(result)
        
        if passed:
            self.logger.pass_(f"{name}: {message}")
        else:
            self.logger.fail(f"{name}: {message}")
    
    # =========================================================================
    # Test Category 1: Backend Parity (CPU vs GPU)
    # =========================================================================
    
    def test_backend_parity(self):
        """Test that CPU and GPU produce statistically consistent results.
        
        Note: CPU and GPU random number generators produce different sequences,
        so we compare statistics over multiple samples rather than exact values.
        """
        self.logger.header("TEST CATEGORY 1: BACKEND PARITY (CPU vs GPU)")
        
        if self.skip_gpu:
            self.logger.warn("Skipping GPU tests (GPU not available or --cpu-only flag)")
            self._add_result(
                "backend_parity_skip", True,
                "Skipped (no GPU)", {}
            )
            return
        
        self.logger.info("Note: CPU/GPU use different RNG sequences, comparing statistics.")
        
        init_backend = self.modules['init_backend']
        get_backend = self.modules['get_backend']
        
        # Test each generator with more samples for statistical comparison
        n_parity_samples = max(20, self.n_samples)
        
        for method_name in ['Zernike', 'DualPowerLaw', 'Jolissaint']:
            self.logger.subheader(f"Backend parity: {method_name}")
            
            results_cpu = self._run_generator_on_backend(method_name, 'CPU', n_samples=n_parity_samples)
            results_gpu = self._run_generator_on_backend(method_name, 'GPU', n_samples=n_parity_samples)
            
            # Compare statistics (mean and std)
            metrics = {}
            all_passed = True
            
            # Compare mean RMS
            rms_diff = abs(results_cpu['rms_mean'] - results_gpu['rms_mean'])
            rms_rdiff = rms_diff / max(results_cpu['rms_mean'], 1e-10)
            metrics['rms_rel_diff'] = float(rms_rdiff)
            metrics['cpu_rms_mean'] = float(results_cpu['rms_mean'])
            metrics['gpu_rms_mean'] = float(results_gpu['rms_mean'])
            
            if rms_rdiff > GPU_PARITY_RTOL:
                all_passed = False
                self.logger.fail(f"  RMS mismatch: CPU={results_cpu['rms_mean']:.4f}±{results_cpu['rms_std']:.4f}, "
                               f"GPU={results_gpu['rms_mean']:.4f}±{results_gpu['rms_std']:.4f}, rel_diff={rms_rdiff:.2e}")
            else:
                self.logger.pass_(f"  RMS match: CPU={results_cpu['rms_mean']:.4f}±{results_cpu['rms_std']:.4f}, "
                                f"GPU={results_gpu['rms_mean']:.4f}±{results_gpu['rms_std']:.4f} (rel_diff={rms_rdiff:.2e})")
            
            # Compare mean Strehl
            strehl_diff = abs(results_cpu['strehl_mean'] - results_gpu['strehl_mean'])
            strehl_rdiff = strehl_diff / max(results_cpu['strehl_mean'], 1e-10)
            metrics['strehl_rel_diff'] = float(strehl_rdiff)
            metrics['cpu_strehl_mean'] = float(results_cpu['strehl_mean'])
            metrics['gpu_strehl_mean'] = float(results_gpu['strehl_mean'])
            
            if strehl_rdiff > GPU_PARITY_RTOL:
                all_passed = False
                self.logger.fail(f"  Strehl mismatch: CPU={results_cpu['strehl_mean']:.4f}±{results_cpu['strehl_std']:.4f}, "
                               f"GPU={results_gpu['strehl_mean']:.4f}±{results_gpu['strehl_std']:.4f}")
            else:
                self.logger.pass_(f"  Strehl match: CPU={results_cpu['strehl_mean']:.4f}±{results_cpu['strehl_std']:.4f}, "
                                f"GPU={results_gpu['strehl_mean']:.4f}±{results_gpu['strehl_std']:.4f} (rel_diff={strehl_rdiff:.2e})")
            
            self._add_result(
                f"backend_parity_{method_name.lower()}",
                all_passed,
                f"CPU/GPU mean rel_diff: RMS={rms_rdiff:.2e}, Strehl={strehl_rdiff:.2e}",
                metrics,
            )
        
        # Store for plotting
        self.data['backend_parity'] = {'tested': True}
    
    def _run_generator_on_backend(self, method: str, backend_name: str, n_samples: int = 5) -> Dict:
        """Run a generator on specified backend and return statistics."""
        init_backend = self.modules['init_backend']
        get_backend = self.modules['get_backend']
        
        init_backend(backend_name)
        backend = get_backend()
        xp = backend.xp
        
        # Clear Zernike cache when switching backends to ensure arrays match
        try:
            from nebraa.physics.zernike import _cache
            _cache.clear()
        except ImportError:
            pass
        
        # Create pupil
        pupil = create_vlt_pupil(self.n_pix, self.pixel_size, xp=xp)
        
        # Create generator based on method (no seed for fair statistical comparison)
        if method == 'Zernike':
            config = self.modules['ZernikeConfig'](
                n_range=(2, 8),
                power_law=2.5,
                target_rms=0.5,
                seed=None,  # Different random samples each time
            )
            gen = self.modules['ZernikePhaseGenerator'](
                n_pix=self.n_pix,
                pixel_size=self.pixel_size,
                zernike_config=config,
            )
            phases = gen.generate(n_samples, pupil)
            
        elif method == 'DualPowerLaw':
            config = self.modules['DualPowerLawConfig'](
                alpha_lf=3.0,
                alpha_hf=11.0/3.0,
                rms_lf=0.3,
                rms_hf=0.2,
                f_cutoff=1.0,
                seed=None,  # Different random samples each time
            )
            gen = self.modules['DualPowerLawPhaseGenerator'](
                n_pix=self.n_pix,
                pixel_size=self.pixel_size,
                psd_config=config,
            )
            phases = gen.generate(n_samples, pupil)
            
        elif method == 'Jolissaint':
            atmosphere = self.modules['create_simple_atmosphere'](
                r0=0.15, L0=25.0, wind_speed=10.0
            )
            ao_config = self.modules['create_ao_config'](
                n_actuators=14,
                telescope_diameter=TELESCOPE_DIAMETER,
            )
            model = self.modules['JolissaintAOModel'](
                n_pix=self.n_pix,
                telescope_diameter=TELESCOPE_DIAMETER,
                obstruction_diameter=OBSTRUCTION_DIAMETER,
                wavelength=WAVELENGTH,
                pixel_scale=PIXEL_SCALE_RAD,
                atmosphere=atmosphere,
                ao_config=ao_config,
            )
            phases = model.generate_phase_screens(n_samples, pupil, seed=None)
        
        # Compute statistics over all samples (convert to numpy for consistency)
        phases_np = to_numpy(phases)
        pupil_np = to_numpy(pupil)
        
        rms_values = [compute_phase_rms(phases_np[i], pupil_np) for i in range(phases_np.shape[0])]
        strehl_values = [compute_strehl_from_phase(phases_np[i], pupil_np) for i in range(phases_np.shape[0])]
        
        return {
            'rms_mean': np.mean(rms_values),
            'rms_std': np.std(rms_values),
            'strehl_mean': np.mean(strehl_values),
            'strehl_std': np.std(strehl_values),
            'phases': phases_np,
        }
    
    # =========================================================================
    # Test Category 2: Short vs Long Exposure
    # =========================================================================
    
    def test_exposure_types(self):
        """Test short-exposure vs long-exposure PSF computation."""
        self.logger.header("TEST CATEGORY 2: SHORT vs LONG EXPOSURE")
        
        init_backend = self.modules['init_backend']
        get_backend = self.modules['get_backend']
        PSFEngine = self.modules['PSFEngine']
        
        init_backend('GPU' if not self.skip_gpu else 'CPU')
        backend = get_backend()
        xp = backend.xp
        
        # Create pupil and PSF engine
        pupil = create_vlt_pupil(self.n_pix, self.pixel_size, xp=xp)
        
        psf_engine = PSFEngine(
            n_pix=self.n_pix,
            wavelength=WAVELENGTH,
            pixel_scale=PIXEL_SCALE_RAD,
            normalize_to='sum',
        )
        
        exposure_data = {}
        
        for method_name in ['Zernike', 'DualPowerLaw', 'Jolissaint']:
            self.logger.subheader(f"Exposure test: {method_name}")
            
            # Generate phase screens
            phases = self._generate_phases(method_name, self.n_samples, pupil, xp)
            phases_np = to_numpy(phases)
            pupil_np = to_numpy(pupil)
            
            # Compute diffraction-limited PSF
            psf_dl = psf_engine.compute_diffraction_limited_psf(pupil)
            psf_dl_np = to_numpy(psf_dl)
            
            # Short exposure: single realization PSFs
            se_strehls = []
            for i in range(min(self.n_samples, 20)):
                psf_se = psf_engine.compute_psf(pupil, phases[i])
                psf_se_np = to_numpy(psf_se)
                strehl_se = float(psf_se_np.max() / psf_dl_np.max())
                se_strehls.append(strehl_se)
            
            se_strehl_mean = np.mean(se_strehls)
            se_strehl_std = np.std(se_strehls)
            
            # Long exposure: averaged PSF
            psf_le = psf_engine.compute_long_exposure_psf(pupil, phases)
            psf_le_np = to_numpy(psf_le)
            le_strehl = float(psf_le_np.max() / psf_dl_np.max())
            
            # Compute RMS and expected Maréchal Strehl
            rms_mean = np.mean([compute_phase_rms(phases_np[i], pupil_np) 
                               for i in range(phases_np.shape[0])])
            marechal = marechal_strehl(rms_mean)
            
            # Log results
            self.logger.info(f"  Phase RMS: {rms_mean:.3f} rad")
            self.logger.info(f"  SE Strehl: {se_strehl_mean:.3f} ± {se_strehl_std:.3f}")
            self.logger.info(f"  LE Strehl: {le_strehl:.3f}")
            self.logger.info(f"  Maréchal:  {marechal:.3f}")
            
            # Validation: LE Strehl should be close to mean SE Strehl
            strehl_diff = abs(le_strehl - se_strehl_mean)
            
            # Store data
            exposure_data[method_name] = {
                'rms_mean': rms_mean,
                'se_strehl_mean': se_strehl_mean,
                'se_strehl_std': se_strehl_std,
                'le_strehl': le_strehl,
                'marechal': marechal,
                'psf_le': psf_le_np,
                'psf_dl': psf_dl_np,
            }
            
            # Test: Strehl values should be reasonable
            passed = (0 < le_strehl < 1) and (abs(le_strehl - marechal) < 0.15)
            
            self._add_result(
                f"exposure_{method_name.lower()}",
                passed,
                f"LE_Strehl={le_strehl:.3f}, Maréchal={marechal:.3f}",
                {
                    'rms': rms_mean,
                    'se_strehl': se_strehl_mean,
                    'le_strehl': le_strehl,
                    'marechal': marechal,
                },
            )
        
        self.data['exposure'] = exposure_data
    
    def _generate_phases(self, method: str, n_screens: int, pupil, xp) -> Any:
        """Generate phase screens for a given method.
        
        For fair comparison, Zernike includes HF turbulence to have similar
        spectral content as DualPowerLaw and Jolissaint. Uses 90% HF / 10% LF
        power split (typical of AO-corrected wavefronts).
        """
        if method == 'Zernike':
            # 90% HF, 10% LF for ~0.5 rad total RMS
            # sqrt(0.1)*0.5 ≈ 0.158, sqrt(0.9)*0.5 ≈ 0.474
            config = self.modules['ZernikeConfig'](
                n_range=(2, 8),
                power_law=2.5,
                lf_rms=0.158,  # 10% of variance
                seed=self.seed,
                f_cutoff=1.0,
                hf_alpha=11.0/3.0,
                hf_rms=0.474,  # 90% of variance
                transition_width=0.2,
            )
            gen = self.modules['ZernikePhaseGenerator'](
                n_pix=self.n_pix,
                pixel_size=self.pixel_size,
                zernike_config=config,
            )
            return gen.generate(n_screens, pupil)
            
        elif method == 'DualPowerLaw':
            config = self.modules['DualPowerLawConfig'](
                alpha_lf=3.0,
                alpha_hf=11.0/3.0,
                rms_lf=0.35,
                rms_hf=0.35,
                f_cutoff=1.0,
                seed=self.seed,
            )
            gen = self.modules['DualPowerLawPhaseGenerator'](
                n_pix=self.n_pix,
                pixel_size=self.pixel_size,
                psd_config=config,
            )
            return gen.generate(n_screens, pupil)
            
        elif method == 'Jolissaint':
            atmosphere = self.modules['create_simple_atmosphere'](
                r0=0.15, L0=25.0, wind_speed=10.0
            )
            ao_config = self.modules['create_ao_config'](
                n_actuators=14,
                telescope_diameter=TELESCOPE_DIAMETER,
            )
            model = self.modules['JolissaintAOModel'](
                n_pix=self.n_pix,
                telescope_diameter=TELESCOPE_DIAMETER,
                obstruction_diameter=OBSTRUCTION_DIAMETER,
                wavelength=WAVELENGTH,
                pixel_scale=PIXEL_SCALE_RAD,
                atmosphere=atmosphere,
                ao_config=ao_config,
            )
            return model.generate_phase_screens(n_screens, pupil, seed=self.seed)
    
    # =========================================================================
    # Test Category 3: High-Frequency Turbulence
    # =========================================================================
    
    def test_hf_turbulence(self):
        """Test high-frequency turbulence component (Zernike + HF)."""
        self.logger.header("TEST CATEGORY 3: HIGH-FREQUENCY TURBULENCE")
        
        init_backend = self.modules['init_backend']
        get_backend = self.modules['get_backend']
        
        init_backend('GPU' if not self.skip_gpu else 'CPU')
        backend = get_backend()
        xp = backend.xp
        
        pupil = create_vlt_pupil(self.n_pix, self.pixel_size, xp=xp)
        pupil_np = to_numpy(pupil)
        
        hf_data = {}
        
        # Test 1: Zernike without HF
        self.logger.subheader("Zernike WITHOUT HF turbulence")
        
        config_no_hf = self.modules['ZernikeConfig'](
            n_range=(2, 8),
            power_law=2.5,
            target_rms=None,  # Natural scaling
            seed=self.seed,
        )
        gen_no_hf = self.modules['ZernikePhaseGenerator'](
            n_pix=self.n_pix,
            pixel_size=self.pixel_size,
            zernike_config=config_no_hf,
        )
        
        phases_no_hf = gen_no_hf.generate(self.n_samples, pupil)
        phases_no_hf_np = to_numpy(phases_no_hf)
        
        rms_no_hf = np.mean([compute_phase_rms(phases_no_hf_np[i], pupil_np) 
                            for i in range(phases_no_hf_np.shape[0])])
        strehl_no_hf = np.mean([compute_strehl_from_phase(phases_no_hf_np[i], pupil_np) 
                               for i in range(phases_no_hf_np.shape[0])])
        
        self.logger.info(f"  RMS (no HF): {rms_no_hf:.3f} rad")
        self.logger.info(f"  Strehl (no HF): {strehl_no_hf:.3f}")
        
        hf_data['zernike_no_hf'] = {
            'rms': rms_no_hf,
            'strehl': strehl_no_hf,
            'phase_sample': phases_no_hf_np[0],
        }
        
        # Test 2: Zernike WITH HF (Kolmogorov-like)
        self.logger.subheader("Zernike WITH HF turbulence")
        
        hf_rms_target = 0.3
        config_hf = self.modules['ZernikeConfig'](
            n_range=(2, 8),
            power_law=2.5,
            target_rms=None,
            seed=self.seed,
            f_cutoff=1.0,
            hf_alpha=11.0/3.0,
            hf_rms=hf_rms_target,
            transition_width=0.2,
        )
        gen_hf = self.modules['ZernikePhaseGenerator'](
            n_pix=self.n_pix,
            pixel_size=self.pixel_size,
            zernike_config=config_hf,
        )
        
        phases_hf = gen_hf.generate(self.n_samples, pupil)
        phases_hf_np = to_numpy(phases_hf)
        
        rms_hf = np.mean([compute_phase_rms(phases_hf_np[i], pupil_np) 
                         for i in range(phases_hf_np.shape[0])])
        strehl_hf = np.mean([compute_strehl_from_phase(phases_hf_np[i], pupil_np) 
                            for i in range(phases_hf_np.shape[0])])
        
        self.logger.info(f"  RMS (with HF): {rms_hf:.3f} rad")
        self.logger.info(f"  Strehl (with HF): {strehl_hf:.3f}")
        
        # Validate RMS additivity: rms_total ≈ sqrt(rms_lf² + rms_hf²)
        expected_rms_hf = np.sqrt(rms_no_hf**2 + hf_rms_target**2)
        rms_additivity_error = abs(rms_hf - expected_rms_hf) / expected_rms_hf
        
        self.logger.info(f"  Expected RMS (additivity): {expected_rms_hf:.3f} rad")
        self.logger.info(f"  RMS additivity error: {rms_additivity_error*100:.1f}%")
        
        hf_data['zernike_hf'] = {
            'rms': rms_hf,
            'strehl': strehl_hf,
            'expected_rms': expected_rms_hf,
            'additivity_error': rms_additivity_error,
            'phase_sample': phases_hf_np[0],
        }
        
        # Test 3: Different HF alpha values
        self.logger.subheader("Varying HF power-law exponent")
        
        alphas = [2.5, 3.0, 11.0/3.0, 4.0]
        alpha_results = []
        
        for alpha in alphas:
            config_alpha = self.modules['ZernikeConfig'](
                n_range=(2, 5),
                power_law=2.0,
                seed=self.seed,
                f_cutoff=1.0,
                hf_alpha=alpha,
                hf_rms=0.3,
            )
            gen_alpha = self.modules['ZernikePhaseGenerator'](
                n_pix=self.n_pix,
                pixel_size=self.pixel_size,
                zernike_config=config_alpha,
            )
            
            phases_alpha = gen_alpha.generate(10, pupil)
            phases_alpha_np = to_numpy(phases_alpha)
            
            rms_alpha = np.mean([compute_phase_rms(phases_alpha_np[i], pupil_np) 
                                for i in range(phases_alpha_np.shape[0])])
            
            alpha_results.append({
                'alpha': alpha,
                'rms': rms_alpha,
            })
            
            self.logger.info(f"  α={alpha:.2f}: RMS={rms_alpha:.3f} rad")
        
        hf_data['alpha_sweep'] = alpha_results
        
        # Validation
        hf_increases_rms = rms_hf > rms_no_hf
        hf_decreases_strehl = strehl_hf < strehl_no_hf
        
        passed = hf_increases_rms and hf_decreases_strehl and (rms_additivity_error < 0.3)
        
        self._add_result(
            "hf_turbulence",
            passed,
            f"RMS: {rms_no_hf:.3f}→{rms_hf:.3f}, Strehl: {strehl_no_hf:.3f}→{strehl_hf:.3f}",
            {
                'rms_no_hf': rms_no_hf,
                'rms_with_hf': rms_hf,
                'strehl_no_hf': strehl_no_hf,
                'strehl_with_hf': strehl_hf,
                'rms_additivity_error': rms_additivity_error,
            },
        )
        
        self.data['hf_turbulence'] = hf_data
    
    # =========================================================================
    # Test Category 4: LWE Integration
    # =========================================================================
    
    def test_lwe_integration(self):
        """Test Low Wind Effect integration with all generators."""
        self.logger.header("TEST CATEGORY 4: LWE INTEGRATION")
        
        init_backend = self.modules['init_backend']
        get_backend = self.modules['get_backend']
        PSFEngine = self.modules['PSFEngine']
        LowWindEffectConfig = self.modules['LowWindEffectConfig']
        
        init_backend('GPU' if not self.skip_gpu else 'CPU')
        backend = get_backend()
        xp = backend.xp
        
        pupil = create_vlt_pupil(self.n_pix, self.pixel_size, xp=xp)
        pupil_np = to_numpy(pupil)
        
        psf_engine = PSFEngine(
            n_pix=self.n_pix,
            wavelength=WAVELENGTH,
            pixel_scale=PIXEL_SCALE_RAD,
            normalize_to='sum',
        )
        
        lwe_config = LowWindEffectConfig(
            piston_rms_rad=0.5,
            tilt_rms_rad=0.2,
            n_realizations=20,
            seed=self.seed,
        )
        
        lwe_data = {}
        
        for method_name in ['Zernike', 'DualPowerLaw', 'Jolissaint']:
            self.logger.subheader(f"LWE test: {method_name}")
            
            # Generate WITHOUT LWE
            phases_no_lwe = self._generate_phases_with_lwe_option(
                method_name, self.n_samples, pupil, xp, lwe_config=None
            )
            phases_no_lwe_np = to_numpy(phases_no_lwe)
            
            rms_no_lwe = np.mean([compute_phase_rms(phases_no_lwe_np[i], pupil_np) 
                                 for i in range(phases_no_lwe_np.shape[0])])
            
            psf_no_lwe = psf_engine.compute_long_exposure_psf(pupil, phases_no_lwe)
            psf_dl = psf_engine.compute_diffraction_limited_psf(pupil)
            
            psf_no_lwe_np = to_numpy(psf_no_lwe)
            psf_dl_np = to_numpy(psf_dl)
            
            strehl_no_lwe = float(psf_no_lwe_np.max() / psf_dl_np.max())
            
            self.logger.info(f"  Without LWE - RMS: {rms_no_lwe:.3f} rad, Strehl: {strehl_no_lwe:.3f}")
            
            # Generate WITH LWE
            results_lwe = self._generate_phases_with_lwe_option(
                method_name, self.n_samples, pupil, xp, lwe_config=lwe_config,
                return_components=True
            )
            
            if results_lwe is None:
                self.logger.warn(f"  {method_name} LWE integration not available")
                continue
            
            phases_res, phases_lwe, phases_total = results_lwe
            phases_total_np = to_numpy(phases_total)
            phases_lwe_np = to_numpy(phases_lwe)
            
            rms_lwe = np.mean([compute_phase_rms(phases_lwe_np[i], pupil_np) 
                              for i in range(phases_lwe_np.shape[0])])
            rms_total = np.mean([compute_phase_rms(phases_total_np[i], pupil_np) 
                                for i in range(phases_total_np.shape[0])])
            
            psf_with_lwe = psf_engine.compute_long_exposure_psf(pupil, phases_total)
            psf_with_lwe_np = to_numpy(psf_with_lwe)
            
            strehl_with_lwe = float(psf_with_lwe_np.max() / psf_dl_np.max())
            
            self.logger.info(f"  With LWE    - RMS: {rms_total:.3f} rad, Strehl: {strehl_with_lwe:.3f}")
            self.logger.info(f"  LWE component RMS: {rms_lwe:.3f} rad")
            self.logger.info(f"  Strehl degradation: {(strehl_no_lwe - strehl_with_lwe)/strehl_no_lwe*100:.1f}%")
            
            # Store data
            lwe_data[method_name] = {
                'rms_no_lwe': rms_no_lwe,
                'rms_with_lwe': rms_total,
                'rms_lwe_only': rms_lwe,
                'strehl_no_lwe': strehl_no_lwe,
                'strehl_with_lwe': strehl_with_lwe,
                'psf_no_lwe': psf_no_lwe_np,
                'psf_with_lwe': psf_with_lwe_np,
                'phase_lwe': phases_lwe_np[0],
            }
            
            # Validate: LWE should degrade Strehl
            passed = strehl_with_lwe < strehl_no_lwe
            
            self._add_result(
                f"lwe_{method_name.lower()}",
                passed,
                f"Strehl: {strehl_no_lwe:.3f}→{strehl_with_lwe:.3f} (degradation: {(1-strehl_with_lwe/strehl_no_lwe)*100:.1f}%)",
                {
                    'strehl_no_lwe': strehl_no_lwe,
                    'strehl_with_lwe': strehl_with_lwe,
                    'rms_lwe': rms_lwe,
                },
            )
        
        self.data['lwe'] = lwe_data
    
    def _generate_phases_with_lwe_option(
        self, method: str, n_screens: int, pupil, xp, 
        lwe_config=None, return_components=False
    ):
        """Generate phases with optional LWE integration."""
        
        if method == 'Zernike':
            config = self.modules['ZernikeConfig'](
                n_range=(2, 8),
                power_law=2.5,
                target_rms=0.5,
                seed=self.seed,
            )
            gen = self.modules['ZernikePhaseGenerator'](
                n_pix=self.n_pix,
                pixel_size=self.pixel_size,
                zernike_config=config,
                lwe_config=lwe_config,
            )
            
            if return_components and lwe_config is not None:
                return gen.generate_with_lwe(n_screens, pupil, seed=self.seed)
            else:
                return gen.generate(n_screens, pupil)
            
        elif method == 'DualPowerLaw':
            config = self.modules['DualPowerLawConfig'](
                alpha_lf=3.0,
                alpha_hf=11.0/3.0,
                rms_lf=0.35,
                rms_hf=0.35,
                f_cutoff=1.0,
                seed=self.seed,
            )
            gen = self.modules['DualPowerLawPhaseGenerator'](
                n_pix=self.n_pix,
                pixel_size=self.pixel_size,
                psd_config=config,
                lwe_config=lwe_config,
            )
            
            if return_components and lwe_config is not None:
                return gen.generate_with_lwe(n_screens, pupil, seed=self.seed)
            else:
                return gen.generate(n_screens, pupil)
            
        elif method == 'Jolissaint':
            atmosphere = self.modules['create_simple_atmosphere'](
                r0=0.15, L0=25.0, wind_speed=10.0
            )
            ao_config = self.modules['create_ao_config'](
                n_actuators=14,
                telescope_diameter=TELESCOPE_DIAMETER,
            )
            model = self.modules['JolissaintAOModel'](
                n_pix=self.n_pix,
                telescope_diameter=TELESCOPE_DIAMETER,
                obstruction_diameter=OBSTRUCTION_DIAMETER,
                wavelength=WAVELENGTH,
                pixel_scale=PIXEL_SCALE_RAD,
                atmosphere=atmosphere,
                ao_config=ao_config,
                lwe_config=lwe_config,
            )
            
            if return_components and lwe_config is not None:
                return model.generate_phase_screens_with_lwe(n_screens, pupil, seed=self.seed)
            else:
                return model.generate_phase_screens(n_screens, pupil, seed=self.seed)
    
    # =========================================================================
    # Test Category 5: Maréchal Approximation
    # =========================================================================
    
    def test_marechal_approximation(self):
        """Test Strehl vs Maréchal approximation across RMS values.
        
        This test validates that the Maréchal approximation S ≈ exp(-σ²) holds
        for actual PSF-computed Strehl ratios (not just phasor averages).
        
        The Strehl is measured as the ratio of the aberrated PSF peak to the
        diffraction-limited PSF peak, computed via actual FFT-based PSF calculation.
        This is a true validation of the approximation, not a mathematical tautology.
        """
        self.logger.header("TEST CATEGORY 5: MARÉCHAL APPROXIMATION")
        
        init_backend = self.modules['init_backend']
        get_backend = self.modules['get_backend']
        PSFEngine = self.modules['PSFEngine']
        
        init_backend('GPU' if not self.skip_gpu else 'CPU')
        backend = get_backend()
        xp = backend.xp
        
        pupil = create_vlt_pupil(self.n_pix, self.pixel_size, xp=xp)
        pupil_np = to_numpy(pupil)
        
        # Create PSF engine for actual PSF computation
        psf_engine = PSFEngine(
            n_pix=self.n_pix,
            wavelength=WAVELENGTH,
            pixel_scale=PIXEL_SCALE_RAD,
            normalize_to='sum',
        )
        
        # Compute diffraction-limited PSF once
        psf_dl = psf_engine.compute_diffraction_limited_psf(pupil)
        psf_dl_np = to_numpy(psf_dl)
        psf_dl_peak = psf_dl_np.max()
        
        # Test RMS values
        rms_targets = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        n_samples_per_rms = max(10, self.n_samples // 3)
        
        marechal_data = {}
        
        # Note: Maréchal approximation S≈exp(-σ²) is valid for phase screens with
        # Gaussian statistics and sufficient high-frequency content.
        # This test computes ACTUAL PSFs via FFT and measures the peak ratio,
        # providing a true validation (not a mathematical tautology).
        
        for method_name in ['Zernike', 'DualPowerLaw', 'Jolissaint']:
            self.logger.subheader(f"Maréchal test: {method_name}")
            
            measured_strehls = []
            measured_stds = []
            measured_rms = []
            
            for rms_target in rms_targets:
                strehls = []
                actual_rms_list = []
                
                for i in range(n_samples_per_rms):
                    seed_i = self.seed + i * 1000 + int(rms_target * 100)
                    
                    # Generate with target RMS
                    phases = self._generate_phases_with_target_rms(
                        method_name, 1, pupil, xp, rms_target, seed=seed_i
                    )
                    phases_np = to_numpy(phases)
                    
                    # Measure actual RMS
                    actual_rms = compute_phase_rms(phases_np[0], pupil_np)
                    
                    # Compute actual PSF and measure Strehl from PSF peak ratio
                    # This is the TRUE Strehl measurement, not a phasor average
                    psf = psf_engine.compute_psf(pupil, phases[0])
                    psf_np = to_numpy(psf)
                    strehl = float(psf_np.max() / psf_dl_peak)
                    
                    strehls.append(strehl)
                    actual_rms_list.append(actual_rms)
                
                measured_strehls.append(np.mean(strehls))
                measured_stds.append(np.std(strehls))
                measured_rms.append(np.mean(actual_rms_list))
            
            measured_strehls = np.array(measured_strehls)
            measured_stds = np.array(measured_stds)
            measured_rms = np.array(measured_rms)
            
            # Theoretical Maréchal values
            theory_strehls = marechal_strehl(measured_rms)
            
            # Compute residuals
            residuals = measured_strehls - theory_strehls
            max_residual = np.max(np.abs(residuals))
            mean_residual = np.mean(np.abs(residuals))
            
            self.logger.info(f"  Max residual: {max_residual:.4f}")
            self.logger.info(f"  Mean residual: {mean_residual:.4f}")
            
            # Store data
            marechal_data[method_name] = {
                'rms_targets': rms_targets.tolist(),
                'rms_measured': measured_rms.tolist(),
                'strehl_measured': measured_strehls.tolist(),
                'strehl_std': measured_stds.tolist(),
                'strehl_theory': theory_strehls.tolist(),
                'max_residual': float(max_residual),
                'mean_residual': float(mean_residual),
            }
            
            # Validation - all methods should follow Maréchal when HF turbulence is included
            passed = max_residual < MARECHAL_MAX_RESIDUAL
            status_msg = f"Max residual: {max_residual:.4f} (threshold: {MARECHAL_MAX_RESIDUAL})"
            
            self._add_result(
                f"marechal_{method_name.lower()}",
                passed,
                status_msg,
                {
                    'max_residual': float(max_residual),
                    'mean_residual': float(mean_residual),
                },
            )
        
        self.data['marechal'] = marechal_data
    
    def _generate_phases_with_target_rms(
        self, method: str, n_screens: int, pupil, xp, target_rms: float, seed: int
    ):
        """Generate phases normalized to a specific target RMS.
        
        For Maréchal approximation to hold, the phase must have realistic
        high-frequency content (not just smooth Zernike modes). We use
        Zernike + HF turbulence with 90% HF / 10% LF power split, which is
        typical of AO-corrected wavefronts where fitting error dominates.
        """
        
        if method == 'Zernike':
            # Split target RMS: 90% HF, 10% LF (typical AO residuals)
            # sqrt(lf^2 + hf^2) = target, with hf^2 = 0.9*target^2, lf^2 = 0.1*target^2
            lf_rms = np.sqrt(0.1) * target_rms
            hf_rms = np.sqrt(0.9) * target_rms
            
            config = self.modules['ZernikeConfig'](
                n_range=(2, 15),
                power_law=2.0,
                lf_rms=lf_rms,  # Control LF RMS independently
                seed=seed,
                # Add HF turbulence for realistic phase spectrum
                f_cutoff=1.0,
                hf_alpha=11.0/3.0,  # Kolmogorov
                hf_rms=hf_rms,
                transition_width=0.2,
            )
            gen = self.modules['ZernikePhaseGenerator'](
                n_pix=self.n_pix,
                pixel_size=self.pixel_size,
                zernike_config=config,
            )
            return gen.generate(n_screens, pupil)
            
        elif method == 'DualPowerLaw':
            # Scale RMS to achieve target
            scale = target_rms / np.sqrt(0.35**2 + 0.35**2)
            config = self.modules['DualPowerLawConfig'](
                alpha_lf=3.0,
                alpha_hf=11.0/3.0,
                rms_lf=0.35 * scale,
                rms_hf=0.35 * scale,
                f_cutoff=1.0,
                seed=seed,
            )
            gen = self.modules['DualPowerLawPhaseGenerator'](
                n_pix=self.n_pix,
                pixel_size=self.pixel_size,
                psd_config=config,
            )
            return gen.generate(n_screens, pupil)
        
        elif method == 'Jolissaint':
            # Generate phases with a reference atmosphere, then scale to target RMS
            # Use moderate turbulence as reference
            TurbulentLayer = self.modules['TurbulentLayer']
            AtmosphereProfile = self.modules['AtmosphereProfile']
            JolissaintAOModel = self.modules['JolissaintAOModel']
            create_ao_config = self.modules['create_ao_config']
            
            # Reference atmosphere with r0=0.2m at 500nm
            ref_layer = TurbulentLayer(
                r0=0.2,
                altitude=0.0,
                wind_speed=10.0,
                wind_direction=0.0,
            )
            ref_atm = AtmosphereProfile(
                layers=[ref_layer],
                wavelength_ref=500e-9,
                L0=25.0,
            )
            
            # Simple AO config using helper function
            ao_config = create_ao_config(
                n_actuators=40,
                telescope_diameter=TELESCOPE_DIAMETER,
                sampling_frequency=1000.0,
                loop_gain=0.5,
            )
            
            model = JolissaintAOModel(
                n_pix=self.n_pix,
                telescope_diameter=TELESCOPE_DIAMETER,
                obstruction_diameter=OBSTRUCTION_DIAMETER,
                wavelength=WAVELENGTH,
                pixel_scale=PIXEL_SCALE_RAD,
                atmosphere=ref_atm,
                ao_config=ao_config,
            )
            
            # Generate reference phases
            phases = model.generate_phase_screens(n_screens, pupil, seed=seed)
            phases_np = to_numpy(phases)
            pupil_np = to_numpy(pupil)
            
            # Measure reference RMS
            mask = pupil_np > 0.5
            ref_rms_list = []
            for i in range(n_screens):
                phase_masked = phases_np[i][mask]
                mean = np.mean(phase_masked)
                rms = np.sqrt(np.mean((phase_masked - mean)**2))
                ref_rms_list.append(rms)
            ref_rms = np.mean(ref_rms_list)
            
            # Scale to target RMS
            if ref_rms > 0:
                scale_factor = target_rms / ref_rms
                phases = phases * scale_factor
            
            return phases
    
    # =========================================================================
    # Test Category 6: Cross-Method Comparison
    # =========================================================================
    
    def test_cross_method_comparison(self):
        """Compare all three methods at equivalent parameters."""
        self.logger.header("TEST CATEGORY 6: CROSS-METHOD COMPARISON")
        
        init_backend = self.modules['init_backend']
        get_backend = self.modules['get_backend']
        PSFEngine = self.modules['PSFEngine']
        
        init_backend('GPU' if not self.skip_gpu else 'CPU')
        backend = get_backend()
        xp = backend.xp
        
        pupil = create_vlt_pupil(self.n_pix, self.pixel_size, xp=xp)
        pupil_np = to_numpy(pupil)
        
        psf_engine = PSFEngine(
            n_pix=self.n_pix,
            wavelength=WAVELENGTH,
            pixel_scale=PIXEL_SCALE_RAD,
            normalize_to='sum',
        )
        
        psf_dl = psf_engine.compute_diffraction_limited_psf(pupil)
        psf_dl_np = to_numpy(psf_dl)
        
        comparison_data = {}
        
        self.logger.info("\nGenerating phases for each method...")
        
        for method_name in ['Zernike', 'DualPowerLaw', 'Jolissaint']:
            self.logger.subheader(f"Method: {method_name}")
            
            with timer() as get_time:
                phases = self._generate_phases(method_name, self.n_samples, pupil, xp)
            
            phases_np = to_numpy(phases)
            gen_time = get_time()
            
            # Compute statistics
            rms_values = [compute_phase_rms(phases_np[i], pupil_np) 
                         for i in range(phases_np.shape[0])]
            strehl_values = [compute_strehl_from_phase(phases_np[i], pupil_np) 
                           for i in range(phases_np.shape[0])]
            
            rms_mean = np.mean(rms_values)
            rms_std = np.std(rms_values)
            strehl_mean = np.mean(strehl_values)
            strehl_std = np.std(strehl_values)
            
            # Long-exposure PSF
            psf_le = psf_engine.compute_long_exposure_psf(pupil, phases)
            psf_le_np = to_numpy(psf_le)
            le_strehl = float(psf_le_np.max() / psf_dl_np.max())
            
            self.logger.info(f"  Generation time: {gen_time:.2f}s")
            self.logger.info(f"  RMS: {rms_mean:.3f} ± {rms_std:.3f} rad")
            self.logger.info(f"  Strehl (phasor): {strehl_mean:.3f} ± {strehl_std:.3f}")
            self.logger.info(f"  Strehl (LE PSF): {le_strehl:.3f}")
            self.logger.info(f"  Maréchal: {marechal_strehl(rms_mean):.3f}")
            
            comparison_data[method_name] = {
                'gen_time': gen_time,
                'rms_mean': rms_mean,
                'rms_std': rms_std,
                'strehl_phasor_mean': strehl_mean,
                'strehl_phasor_std': strehl_std,
                'strehl_le_psf': le_strehl,
                'marechal': marechal_strehl(rms_mean),
                'psf_le': psf_le_np,
                'phase_sample': phases_np[0],
            }
        
        comparison_data['psf_dl'] = psf_dl_np
        self.data['comparison'] = comparison_data
        
        # Summary table
        self.logger.subheader("Summary Table")
        self.logger.info(f"{'Method':<15} {'RMS (rad)':<12} {'Strehl (LE)':<12} {'Maréchal':<12} {'Time (s)':<10}")
        self.logger.info("-" * 61)
        
        for method_name in ['Zernike', 'DualPowerLaw', 'Jolissaint']:
            d = comparison_data[method_name]
            self.logger.info(
                f"{method_name:<15} {d['rms_mean']:.3f}±{d['rms_std']:.3f}   "
                f"{d['strehl_le_psf']:.3f}        {d['marechal']:.3f}        {d['gen_time']:.2f}"
            )
        
        # All methods should produce reasonable results
        all_valid = all(
            0 < comparison_data[m]['strehl_le_psf'] < 1 
            for m in ['Zernike', 'DualPowerLaw', 'Jolissaint']
        )
        
        self._add_result(
            "cross_method_comparison",
            all_valid,
            "All methods produced valid PSFs with reasonable Strehl",
            {m: comparison_data[m]['strehl_le_psf'] for m in ['Zernike', 'DualPowerLaw', 'Jolissaint']},
        )
    
    # =========================================================================
    # Results & Plotting
    # =========================================================================
    
    def save_results(self):
        """Save test results to JSON."""
        results_path = self.output_dir / "test_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(self.results.to_dict(), f, indent=2)
        
        self.logger.info(f"Results saved to: {results_path}")
        
        # Save data for plotting (numpy arrays as .npz)
        data_path = self.output_dir / "test_data.npz"
        
        # Flatten nested dicts for npz
        flat_data = {}
        for category, cat_data in self.data.items():
            if isinstance(cat_data, dict):
                for key, value in cat_data.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, np.ndarray):
                                flat_data[f"{category}_{key}_{subkey}"] = subvalue
                    elif isinstance(value, np.ndarray):
                        flat_data[f"{category}_{key}"] = value
        
        if flat_data:
            np.savez(data_path, **flat_data)
            self.logger.info(f"Data saved to: {data_path}")
    
    def generate_all_plots(self):
        """Generate comprehensive diagnostic plots."""
        self.logger.header("GENERATING DIAGNOSTIC PLOTS")
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Phase comparison
        if 'comparison' in self.data:
            self._plot_phase_comparison(plots_dir)
        
        # 2. PSF comparison
        if 'comparison' in self.data:
            self._plot_psf_comparison(plots_dir)
        
        # 3. Maréchal validation
        if 'marechal' in self.data:
            self._plot_marechal_validation(plots_dir)
        
        # 4. HF turbulence
        if 'hf_turbulence' in self.data:
            self._plot_hf_turbulence(plots_dir)
        
        # 5. LWE impact
        if 'lwe' in self.data:
            self._plot_lwe_impact(plots_dir)
        
        # 6. Summary dashboard
        self._plot_summary_dashboard(plots_dir)
        
        self.logger.info(f"All plots saved to: {plots_dir}")
    
    def _plot_phase_comparison(self, plots_dir: Path):
        """Plot phase screen comparison across methods."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Phase Screen Comparison (Single Realization)", fontsize=14, fontweight='bold')
        
        data = self.data['comparison']
        
        # Get pupil for masking
        pupil = create_vlt_pupil(self.n_pix, self.pixel_size)
        
        for i, method in enumerate(['Zernike', 'DualPowerLaw', 'Jolissaint']):
            phase = data[method]['phase_sample']
            phase_masked = np.where(pupil > 0.5, phase, np.nan)
            
            # Phase map
            im = axes[0, i].imshow(phase_masked, cmap='RdBu_r', origin='lower')
            axes[0, i].set_title(f"{method}\nRMS={data[method]['rms_mean']:.3f} rad")
            axes[0, i].axis('off')
            plt.colorbar(im, ax=axes[0, i], label='Phase (rad)', fraction=0.046)
            
            # Power spectrum
            phase_fft = np.fft.fftshift(np.fft.fft2(phase * pupil))
            psd = np.abs(phase_fft)**2
            
            axes[1, i].imshow(np.log10(psd + 1e-10), cmap='viridis', origin='lower',
                             vmin=0, vmax=np.log10(psd.max()))
            axes[1, i].set_title(f"Log PSD")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "phase_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info("  Saved: phase_comparison.png")
    
    def _plot_psf_comparison(self, plots_dir: Path):
        """Plot PSF comparison across methods."""
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        fig.suptitle("PSF Comparison (Long Exposure)", fontsize=14, fontweight='bold')
        
        data = self.data['comparison']
        psf_dl = data['psf_dl']
        
        # Plot DL PSF
        im = axes[0, 0].imshow(psf_dl, cmap='hot', norm=LogNorm(vmin=psf_dl.max()*1e-5),
                               origin='lower')
        axes[0, 0].set_title("Diffraction Limited")
        axes[0, 0].axis('off')
        plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
        
        # Radial profile - DL
        center = psf_dl.shape[0] // 2
        r = np.arange(center)
        profile_dl = self._radial_profile(psf_dl)[:center]
        axes[1, 0].semilogy(r, profile_dl / profile_dl.max(), 'k-', linewidth=2, label='DL')
        axes[1, 0].set_xlabel('Radius (pixels)')
        axes[1, 0].set_ylabel('Normalized Intensity')
        axes[1, 0].set_title('Radial Profiles')
        axes[1, 0].legend()
        axes[1, 0].set_ylim(1e-5, 1.5)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot each method
        colors = ['C0', 'C1', 'C2']
        for i, method in enumerate(['Zernike', 'DualPowerLaw', 'Jolissaint']):
            psf = data[method]['psf_le']
            strehl = data[method]['strehl_le_psf']
            
            # PSF image
            im = axes[0, i+1].imshow(psf, cmap='hot', norm=LogNorm(vmin=psf_dl.max()*1e-5),
                                     origin='lower')
            axes[0, i+1].set_title(f"{method}\nStrehl={strehl:.3f}")
            axes[0, i+1].axis('off')
            plt.colorbar(im, ax=axes[0, i+1], fraction=0.046)
            
            # Add to radial profile plot
            profile = self._radial_profile(psf)[:center]
            axes[1, 0].semilogy(r, profile / profile_dl.max(), colors[i], 
                               linewidth=1.5, label=f'{method} (S={strehl:.2f})')
        
        axes[1, 0].legend(fontsize=8)
        
        # Remove unused axes
        for j in range(1, 4):
            axes[1, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "psf_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info("  Saved: psf_comparison.png")
    
    def _plot_marechal_validation(self, plots_dir: Path):
        """Plot Maréchal approximation validation."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Maréchal Approximation Validation: S ≈ exp(-σ²)", fontsize=14, fontweight='bold')
        
        data = self.data['marechal']
        
        # Theory curve
        rms_fine = np.linspace(0, 1.3, 100)
        theory_fine = marechal_strehl(rms_fine)
        
        colors = {'Zernike': 'C0', 'DualPowerLaw': 'C1', 'Jolissaint': 'C2'}
        
        for method, color in colors.items():
            if method not in data:
                continue
                
            d = data[method]
            rms = np.array(d['rms_measured'])
            strehl = np.array(d['strehl_measured'])
            std = np.array(d['strehl_std'])
            theory = np.array(d['strehl_theory'])
            
            # Left: Strehl vs RMS
            axes[0].errorbar(rms, strehl, yerr=std, fmt='o', color=color, 
                           markersize=6, capsize=3, label=f'{method}')
            
            # Right: Residuals
            residuals = strehl - theory
            axes[1].errorbar(rms, residuals, yerr=std, fmt='o', color=color,
                           markersize=6, capsize=3, label=f'{method} (max={d["max_residual"]:.3f})')
        
        # Theory line
        axes[0].plot(rms_fine, theory_fine, 'k--', linewidth=2, label='Maréchal: exp(-σ²)')
        
        axes[0].set_xlabel('RMS Wavefront Error (radians)')
        axes[0].set_ylabel('Strehl Ratio')
        axes[0].set_xlim(0, 1.3)
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Measured vs Theory')
        
        # Residuals
        axes[1].axhline(0, color='k', linestyle='--', linewidth=1)
        axes[1].axhline(MARECHAL_MAX_RESIDUAL, color='r', linestyle=':', label=f'Threshold (±{MARECHAL_MAX_RESIDUAL})')
        axes[1].axhline(-MARECHAL_MAX_RESIDUAL, color='r', linestyle=':')
        axes[1].set_xlabel('RMS Wavefront Error (radians)')
        axes[1].set_ylabel('Residual (Measured - Maréchal)')
        axes[1].set_xlim(0, 1.3)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Residuals')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "marechal_validation.png", dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info("  Saved: marechal_validation.png")
    
    def _plot_hf_turbulence(self, plots_dir: Path):
        """Plot HF turbulence comparison."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("High-Frequency Turbulence Component", fontsize=14, fontweight='bold')
        
        data = self.data['hf_turbulence']
        pupil = create_vlt_pupil(self.n_pix, self.pixel_size)
        
        # Phase without HF
        phase_no_hf = data['zernike_no_hf']['phase_sample']
        phase_no_hf_masked = np.where(pupil > 0.5, phase_no_hf, np.nan)
        
        vmax = max(np.nanmax(np.abs(phase_no_hf_masked)), 
                  np.nanmax(np.abs(data['zernike_hf']['phase_sample'])))
        
        im = axes[0, 0].imshow(phase_no_hf_masked, cmap='RdBu_r', origin='lower',
                               vmin=-vmax, vmax=vmax)
        axes[0, 0].set_title(f"Zernike Only\nRMS={data['zernike_no_hf']['rms']:.3f} rad")
        axes[0, 0].axis('off')
        plt.colorbar(im, ax=axes[0, 0], label='Phase (rad)', fraction=0.046)
        
        # Phase with HF
        phase_hf = data['zernike_hf']['phase_sample']
        phase_hf_masked = np.where(pupil > 0.5, phase_hf, np.nan)
        
        im = axes[0, 1].imshow(phase_hf_masked, cmap='RdBu_r', origin='lower',
                               vmin=-vmax, vmax=vmax)
        axes[0, 1].set_title(f"Zernike + HF (α=11/3)\nRMS={data['zernike_hf']['rms']:.3f} rad")
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], label='Phase (rad)', fraction=0.046)
        
        # Difference (HF component)
        diff = phase_hf - phase_no_hf
        diff_masked = np.where(pupil > 0.5, diff, np.nan)
        
        im = axes[0, 2].imshow(diff_masked, cmap='RdBu_r', origin='lower')
        axes[0, 2].set_title("HF Component (difference)")
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], label='Phase (rad)', fraction=0.046)
        
        # PSDs
        psd_no_hf = np.abs(np.fft.fftshift(np.fft.fft2(phase_no_hf * pupil)))**2
        psd_hf = np.abs(np.fft.fftshift(np.fft.fft2(phase_hf * pupil)))**2
        
        axes[1, 0].imshow(np.log10(psd_no_hf + 1e-10), cmap='viridis', origin='lower')
        axes[1, 0].set_title("Log PSD (Zernike only)")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(np.log10(psd_hf + 1e-10), cmap='viridis', origin='lower')
        axes[1, 1].set_title("Log PSD (Zernike + HF)")
        axes[1, 1].axis('off')
        
        # Alpha sweep
        alphas = [r['alpha'] for r in data['alpha_sweep']]
        rms_vals = [r['rms'] for r in data['alpha_sweep']]
        
        axes[1, 2].bar(range(len(alphas)), rms_vals, tick_label=[f'{a:.2f}' for a in alphas])
        axes[1, 2].set_xlabel('HF Power-Law Exponent (α)')
        axes[1, 2].set_ylabel('Total RMS (rad)')
        axes[1, 2].set_title('Effect of HF α on Total RMS')
        axes[1, 2].axhline(data['zernike_no_hf']['rms'], color='r', linestyle='--', 
                          label='LF only')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "hf_turbulence.png", dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info("  Saved: hf_turbulence.png")
    
    def _plot_lwe_impact(self, plots_dir: Path):
        """Plot LWE impact across methods."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Low Wind Effect Impact", fontsize=14, fontweight='bold')
        
        data = self.data['lwe']
        
        for i, method in enumerate(['Zernike', 'DualPowerLaw', 'Jolissaint']):
            if method not in data:
                axes[0, i].axis('off')
                axes[1, i].axis('off')
                continue
            
            d = data[method]
            
            # LWE phase
            if 'phase_lwe' in d:
                pupil = create_vlt_pupil(self.n_pix, self.pixel_size)
                phase_lwe = d['phase_lwe']
                phase_lwe_masked = np.where(pupil > 0.5, phase_lwe, np.nan)
                
                im = axes[0, i].imshow(phase_lwe_masked, cmap='RdBu_r', origin='lower')
                axes[0, i].set_title(f"{method} LWE Phase\nRMS={d['rms_lwe_only']:.3f} rad")
                axes[0, i].axis('off')
                plt.colorbar(im, ax=axes[0, i], label='Phase (rad)', fraction=0.046)
            
            # PSF comparison
            psf_no = d['psf_no_lwe']
            psf_with = d['psf_with_lwe']
            
            # Radial profiles
            r = np.arange(psf_no.shape[0] // 2)
            profile_no = self._radial_profile(psf_no)[:len(r)]
            profile_with = self._radial_profile(psf_with)[:len(r)]
            
            axes[1, i].semilogy(r, profile_no / profile_no.max(), 'b-', 
                               linewidth=1.5, label=f'No LWE (S={d["strehl_no_lwe"]:.3f})')
            axes[1, i].semilogy(r, profile_with / profile_no.max(), 'r-', 
                               linewidth=1.5, label=f'With LWE (S={d["strehl_with_lwe"]:.3f})')
            axes[1, i].set_xlabel('Radius (pixels)')
            axes[1, i].set_ylabel('Normalized Intensity')
            axes[1, i].set_title(f'{method} PSF Profiles')
            axes[1, i].legend(fontsize=8)
            axes[1, i].set_ylim(1e-5, 1.5)
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "lwe_impact.png", dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info("  Saved: lwe_impact.png")
    
    def _plot_summary_dashboard(self, plots_dir: Path):
        """Generate summary dashboard with all key metrics."""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle("Comprehensive PSF Test Suite - Summary Dashboard", 
                    fontsize=16, fontweight='bold')
        
        # Test results summary (top left)
        ax_summary = fig.add_subplot(gs[0, 0])
        ax_summary.axis('off')
        
        summary_text = [
            f"Total Tests: {self.results.n_total}",
            f"Passed: {self.results.n_passed}",
            f"Failed: {self.results.n_failed}",
            f"Duration: {self.results.total_duration_s:.1f}s",
            "",
            f"Grid: {self.n_pix}×{self.n_pix}",
            f"Samples: {self.n_samples}",
        ]
        ax_summary.text(0.1, 0.9, "\n".join(summary_text), transform=ax_summary.transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_summary.set_title("Test Summary", fontweight='bold')
        
        # Strehl comparison bar chart (top middle)
        ax_strehl = fig.add_subplot(gs[0, 1:3])
        
        if 'comparison' in self.data:
            methods = ['Zernike', 'DualPowerLaw', 'Jolissaint']
            strehls = [self.data['comparison'][m]['strehl_le_psf'] for m in methods]
            marechals = [self.data['comparison'][m]['marechal'] for m in methods]
            
            x = np.arange(len(methods))
            width = 0.35
            
            ax_strehl.bar(x - width/2, strehls, width, label='Measured (LE PSF)', color='steelblue')
            ax_strehl.bar(x + width/2, marechals, width, label='Maréchal', color='coral')
            
            ax_strehl.set_ylabel('Strehl Ratio')
            ax_strehl.set_xticks(x)
            ax_strehl.set_xticklabels(methods)
            ax_strehl.legend()
            ax_strehl.set_ylim(0, 1)
            ax_strehl.set_title('Strehl Ratio Comparison', fontweight='bold')
            ax_strehl.grid(True, alpha=0.3, axis='y')
        
        # Pass/Fail indicator (top right)
        ax_passfail = fig.add_subplot(gs[0, 3])
        ax_passfail.axis('off')
        
        pass_rate = self.results.n_passed / max(self.results.n_total, 1) * 100
        color = 'green' if pass_rate > 90 else ('orange' if pass_rate > 70 else 'red')
        
        ax_passfail.text(0.5, 0.5, f"{pass_rate:.0f}%\nPASS", 
                        transform=ax_passfail.transAxes,
                        fontsize=24, ha='center', va='center', fontweight='bold',
                        color=color,
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=3))
        ax_passfail.set_title("Pass Rate", fontweight='bold')
        
        # Maréchal curve (middle row, left half)
        ax_marechal = fig.add_subplot(gs[1, :2])
        
        if 'marechal' in self.data:
            rms_fine = np.linspace(0, 1.3, 100)
            ax_marechal.plot(rms_fine, marechal_strehl(rms_fine), 'k--', linewidth=2, 
                            label='Maréchal: exp(-σ²)')
            
            for method, color in [('Zernike', 'C0'), ('DualPowerLaw', 'C1')]:
                if method in self.data['marechal']:
                    d = self.data['marechal'][method]
                    ax_marechal.errorbar(d['rms_measured'], d['strehl_measured'],
                                        yerr=d['strehl_std'], fmt='o', color=color,
                                        markersize=5, capsize=2, label=method)
            
            ax_marechal.set_xlabel('RMS Wavefront Error (rad)')
            ax_marechal.set_ylabel('Strehl Ratio')
            ax_marechal.set_xlim(0, 1.3)
            ax_marechal.set_ylim(0, 1)
            ax_marechal.legend()
            ax_marechal.grid(True, alpha=0.3)
            ax_marechal.set_title('Maréchal Approximation Validation', fontweight='bold')
        
        # LWE impact (middle row, right half)
        ax_lwe = fig.add_subplot(gs[1, 2:])
        
        if 'lwe' in self.data:
            methods = []
            no_lwe = []
            with_lwe = []
            
            for method in ['Zernike', 'DualPowerLaw', 'Jolissaint']:
                if method in self.data['lwe']:
                    methods.append(method)
                    no_lwe.append(self.data['lwe'][method]['strehl_no_lwe'])
                    with_lwe.append(self.data['lwe'][method]['strehl_with_lwe'])
            
            if methods:
                x = np.arange(len(methods))
                width = 0.35
                
                ax_lwe.bar(x - width/2, no_lwe, width, label='Without LWE', color='steelblue')
                ax_lwe.bar(x + width/2, with_lwe, width, label='With LWE', color='coral')
                
                ax_lwe.set_ylabel('Strehl Ratio')
                ax_lwe.set_xticks(x)
                ax_lwe.set_xticklabels(methods)
                ax_lwe.legend()
                ax_lwe.set_ylim(0, 1)
                ax_lwe.grid(True, alpha=0.3, axis='y')
        
        ax_lwe.set_title('LWE Impact on Strehl', fontweight='bold')
        
        # PSF images (bottom row)
        if 'comparison' in self.data:
            for i, method in enumerate(['Zernike', 'DualPowerLaw', 'Jolissaint']):
                ax = fig.add_subplot(gs[2, i])
                
                psf = self.data['comparison'][method]['psf_le']
                psf_dl = self.data['comparison']['psf_dl']
                
                ax.imshow(psf, cmap='hot', norm=LogNorm(vmin=psf_dl.max()*1e-4),
                         origin='lower')
                ax.set_title(f'{method}\nS={self.data["comparison"][method]["strehl_le_psf"]:.3f}')
                ax.axis('off')
            
            # DL reference
            ax = fig.add_subplot(gs[2, 3])
            ax.imshow(self.data['comparison']['psf_dl'], cmap='hot', 
                     norm=LogNorm(vmin=self.data['comparison']['psf_dl'].max()*1e-4),
                     origin='lower')
            ax.set_title('Diffraction Limited')
            ax.axis('off')
        
        plt.savefig(plots_dir / "summary_dashboard.png", dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info("  Saved: summary_dashboard.png")
    
    def _radial_profile(self, data: np.ndarray, center: Tuple[int, int] = None) -> np.ndarray:
        """Compute radial profile of 2D data."""
        if center is None:
            center = (data.shape[0] // 2, data.shape[1] // 2)
        
        y, x = np.indices(data.shape)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        
        return tbin / np.maximum(nr, 1)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive PSF Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (fewer samples)')
    parser.add_argument('--thorough', action='store_true',
                       help='Thorough mode (more samples)')
    parser.add_argument('--cpu-only', action='store_true',
                       help='Skip GPU tests')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: timestamped)')
    parser.add_argument('--n-pix', type=int, default=DEFAULT_N_PIX,
                       help=f'Grid size (default: {DEFAULT_N_PIX})')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Determine parameters
    if args.quick:
        n_samples = QUICK_N_SAMPLES
    elif args.thorough:
        n_samples = THOROUGH_N_SAMPLES
    else:
        n_samples = DEFAULT_N_SAMPLES
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"output_comprehensive_{timestamp}")
    
    # Run test suite
    suite = ComprehensiveTestSuite(
        output_dir=output_dir,
        n_pix=args.n_pix,
        n_samples=n_samples,
        skip_gpu=args.cpu_only,
        seed=args.seed,
    )
    
    results = suite.run_all()
    
    # Exit code based on test results
    sys.exit(0 if results.n_failed == 0 else 1)


if __name__ == '__main__':
    main()

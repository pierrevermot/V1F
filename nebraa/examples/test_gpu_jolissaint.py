#!/usr/bin/env python3
"""Test GPU performance for Jolissaint AO model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np

# Check CuPy availability
HAS_CUPY = False
HAS_CUPY_J1 = False
HAS_GPU = False

try:
    import cupy as cp
    print(f'CuPy version: {cp.__version__}')
    try:
        device = cp.cuda.Device()
        print(f'Device: {device.name}')
        print(f'Memory: {device.mem_info[1] / 1e9:.1f} GB')
        HAS_CUPY = True
        HAS_GPU = True
        
        # Check cupyx.scipy.special.j1
        try:
            from cupyx.scipy.special import j1 as cupy_j1
            x = cp.linspace(0.1, 10, 100)
            _ = cupy_j1(x)
            print('cupyx.scipy.special.j1: Available')
            HAS_CUPY_J1 = True
        except Exception as e:
            print(f'cupyx.scipy.special.j1: NOT available ({e})')
            HAS_CUPY_J1 = False
    except Exception as e:
        print(f'No GPU detected: {e}')
        HAS_GPU = False
        
except ImportError as e:
    print(f'CuPy not available: {e}')

print()

# Now test the Jolissaint model
from nebraa.utils.compute import init_backend, get_backend
from nebraa.physics.jolissaint_ao import (
    JolissaintAOModel,
    create_simple_atmosphere,
    create_ao_config,
)


def benchmark_psf(n_pix, backend_name, n_runs=5):
    """Benchmark PSF generation."""
    # Force backend
    init_backend(backend_name)
    backend = get_backend()
    xp = backend.xp
    
    wavelength = 2.2e-6
    pixel_scale_rad = 13e-3 * (np.pi / 180 / 3600)
    
    # Create pupil
    x = np.linspace(-1, 1, n_pix)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    pupil = ((R <= 1.0) & (R >= 0.14)).astype(np.float64)
    pupil = xp.asarray(pupil)
    
    atmosphere = create_simple_atmosphere(r0=0.15, L0=25.0, wind_speed=10.0)
    ao_config = create_ao_config(n_actuators=14, telescope_diameter=8.2, sampling_frequency=500.0)
    
    model = JolissaintAOModel(
        n_pix=n_pix, telescope_diameter=8.2, obstruction_diameter=1.116,
        wavelength=wavelength, pixel_scale=pixel_scale_rad,
        atmosphere=atmosphere, ao_config=ao_config,
    )
    
    # Warm-up
    _ = model.compute_long_exposure_psf(pupil)
    if backend_name == 'cupy':
        cp.cuda.Stream.null.synchronize()
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.compute_long_exposure_psf(pupil)
        if backend_name == 'cupy':
            cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    return np.mean(times), np.std(times)


def profile_components(n_pix, backend_name):
    """Profile individual components."""
    init_backend(backend_name)
    backend = get_backend()
    xp = backend.xp
    
    wavelength = 2.2e-6
    pixel_scale_rad = 13e-3 * (np.pi / 180 / 3600)
    
    x = np.linspace(-1, 1, n_pix)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    pupil = ((R <= 1.0) & (R >= 0.14)).astype(np.float64)
    pupil = xp.asarray(pupil)
    
    atmosphere = create_simple_atmosphere(r0=0.15, L0=25.0, wind_speed=10.0)
    ao_config = create_ao_config(n_actuators=14, telescope_diameter=8.2, sampling_frequency=500.0)
    
    model = JolissaintAOModel(
        n_pix=n_pix, telescope_diameter=8.2, obstruction_diameter=1.116,
        wavelength=wavelength, pixel_scale=pixel_scale_rad,
        atmosphere=atmosphere, ao_config=ao_config,
    )
    
    def sync():
        if backend_name == 'cupy':
            cp.cuda.Stream.null.synchronize()
    
    n_runs = 3
    
    # Fitting PSD
    model.compute_fitting_psd(); sync()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.compute_fitting_psd(); sync()
    t_fit = (time.perf_counter() - t0) / n_runs * 1000
    
    # Aliasing PSD
    model.compute_aliasing_psd(); sync()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.compute_aliasing_psd(); sync()
    t_alias = (time.perf_counter() - t0) / n_runs * 1000
    
    # Full PSF
    model.compute_long_exposure_psf(pupil); sync()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.compute_long_exposure_psf(pupil); sync()
    t_psf = (time.perf_counter() - t0) / n_runs * 1000
    
    return {
        'fitting': t_fit,
        'aliasing': t_alias,
        'full_psf': t_psf,
    }


if __name__ == '__main__':
    print('=' * 60)
    print('Benchmarking Jolissaint AO Model: CPU vs GPU')
    print('=' * 60)
    
    # CPU benchmark
    print('\n--- CPU (NumPy) ---')
    for n_pix in [128, 256]:
        mean_t, std_t = benchmark_psf(n_pix, 'numpy', n_runs=3)
        print(f'  {n_pix}x{n_pix}: {mean_t*1000:.1f} ± {std_t*1000:.1f} ms')
    
    # GPU benchmark
    if HAS_GPU:
        print('\n--- GPU (CuPy) ---')
        for n_pix in [128, 256]:
            try:
                mean_t, std_t = benchmark_psf(n_pix, 'cupy', n_runs=3)
                print(f'  {n_pix}x{n_pix}: {mean_t*1000:.1f} ± {std_t*1000:.1f} ms')
            except Exception as e:
                print(f'  {n_pix}x{n_pix}: FAILED ({e})')
        
        # Profile components
        print('\n--- Component Profiling (256x256) ---')
        print('\nCPU:')
        cpu_profile = profile_components(256, 'numpy')
        for k, v in cpu_profile.items():
            print(f'  {k}: {v:.1f} ms')
        
        print('\nGPU:')
        try:
            gpu_profile = profile_components(256, 'cupy')
            for k, v in gpu_profile.items():
                print(f'  {k}: {v:.1f} ms')
            
            print('\nSpeedup (CPU/GPU):')
            for k in cpu_profile:
                speedup = cpu_profile[k] / gpu_profile[k]
                print(f'  {k}: {speedup:.2f}x')
        except Exception as e:
            print(f'  FAILED: {e}')
    else:
        print('\n--- GPU not available, skipping GPU benchmark ---')
    
    print('\n' + '=' * 60)
    print('Done.')

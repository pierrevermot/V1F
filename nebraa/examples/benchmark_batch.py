#!/usr/bin/env python
"""
Benchmark batch PSF generation to measure throughput and per-example overhead.

Tests:
1. Repeated PSF generation with same model (reuse model instance)
2. Repeated PSF generation with varying atmosphere parameters
3. Full model re-instantiation per PSF (worst case)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
    print(f'CuPy: {cp.__version__}')
    print(f'Device: {cp.cuda.Device(0).compute_capability}')
except ImportError:
    HAS_CUPY = False
    print('CuPy not available, GPU benchmarks will be skipped')

from nebraa.physics.jolissaint_ao import JolissaintAOModel, AOSystemConfig, AtmosphereProfile, TurbulentLayer
from nebraa.utils.compute import init_backend, get_backend

# Base configuration
D = 8.0
D_obs = 1.12
wavelength = 2.2e-6
pixel_scale = 10e-3 / 206265  # 10 mas in radians
Lambda = 0.5
d_act = 0.4
f_loop = 1000.0


def create_atmosphere(r0=0.15, wind_speed=10.0, L0=25.0):
    """Create atmosphere profile with given parameters."""
    atm_layers = [
        TurbulentLayer(altitude=0, r0=r0, wind_speed=wind_speed, wind_direction=0, Cn2_fraction=0.5),
        TurbulentLayer(altitude=10000, r0=r0, wind_speed=wind_speed * 2.24, wind_direction=0.46, Cn2_fraction=0.5),
    ]
    return AtmosphereProfile(layers=atm_layers, L0=L0)


def create_ao_config():
    """Create AO configuration."""
    return AOSystemConfig(
        wfs_subaperture_size=Lambda,
        actuator_pitch=d_act,
        integration_time=1.0/f_loop,
        noise_variance=0.1,
        include_aliasing=True,
    )


def create_pupil(n_pix, xp):
    """Create simple circular pupil."""
    x = xp.linspace(-1, 1, n_pix)
    X, Y = xp.meshgrid(x, x)
    R = xp.sqrt(X**2 + Y**2)
    pupil = ((R <= 1.0) & (R >= D_obs/D)).astype(xp.float64)
    return pupil


def benchmark_same_model(n_examples, n_pix, use_gpu=True):
    """
    Benchmark: Same model instance, same parameters, repeated PSF generation.
    
    This is the best case - only compute_long_exposure_psf() is called repeatedly.
    """
    backend_name = 'GPU' if use_gpu else 'CPU'
    init_backend(backend_name)
    backend = get_backend()
    xp = backend.xp
    
    atm = create_atmosphere()
    ao_cfg = create_ao_config()
    
    model = JolissaintAOModel(
        n_pix=n_pix,
        telescope_diameter=D,
        obstruction_diameter=D_obs,
        wavelength=wavelength,
        pixel_scale=pixel_scale,
        ao_config=ao_cfg,
        atmosphere=atm,
    )
    pupil = create_pupil(n_pix, xp)
    
    # Warmup
    for _ in range(5):
        _ = model.compute_long_exposure_psf(pupil)
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    if use_gpu:
        cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    
    for _ in range(n_examples):
        psf = model.compute_long_exposure_psf(pupil)
    
    if use_gpu:
        cp.cuda.Stream.null.synchronize()
    total_time = time.perf_counter() - t0
    
    return total_time, total_time / n_examples


def benchmark_varying_atmosphere(n_examples, n_pix, use_gpu=True):
    """
    Benchmark: Same model instance, but atmosphere parameters vary per example.
    
    This simulates generating PSFs for different seeing conditions.
    The model is reused but atmosphere.layers are updated.
    """
    backend_name = 'GPU' if use_gpu else 'CPU'
    init_backend(backend_name)
    backend = get_backend()
    xp = backend.xp
    
    # Create base model
    atm = create_atmosphere()
    ao_cfg = create_ao_config()
    
    model = JolissaintAOModel(
        n_pix=n_pix,
        telescope_diameter=D,
        obstruction_diameter=D_obs,
        wavelength=wavelength,
        pixel_scale=pixel_scale,
        ao_config=ao_cfg,
        atmosphere=atm,
    )
    pupil = create_pupil(n_pix, xp)
    
    # Precompute varying parameters
    np.random.seed(42)
    r0_values = np.random.uniform(0.10, 0.25, n_examples)
    wind_values = np.random.uniform(5, 20, n_examples)
    
    # Warmup
    for i in range(5):
        model.atmosphere.layers[0].r0 = r0_values[i % n_examples]
        model.atmosphere.layers[0].wind_speed = wind_values[i % n_examples]
        _ = model.compute_long_exposure_psf(pupil)
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    if use_gpu:
        cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    
    for i in range(n_examples):
        # Update atmosphere parameters
        model.atmosphere.layers[0].r0 = r0_values[i]
        model.atmosphere.layers[0].wind_speed = wind_values[i]
        model.atmosphere.layers[1].r0 = r0_values[i]
        model.atmosphere.layers[1].wind_speed = wind_values[i] * 2.24
        
        psf = model.compute_long_exposure_psf(pupil)
    
    if use_gpu:
        cp.cuda.Stream.null.synchronize()
    total_time = time.perf_counter() - t0
    
    return total_time, total_time / n_examples


def benchmark_new_model_per_example(n_examples, n_pix, use_gpu=True):
    """
    Benchmark: Create new model instance for each PSF (worst case).
    
    This measures the full overhead of model instantiation.
    """
    backend_name = 'GPU' if use_gpu else 'CPU'
    init_backend(backend_name)
    backend = get_backend()
    xp = backend.xp
    
    ao_cfg = create_ao_config()
    
    # Precompute varying parameters
    np.random.seed(42)
    r0_values = np.random.uniform(0.10, 0.25, n_examples)
    
    # Warmup
    for i in range(3):
        atm = create_atmosphere(r0=r0_values[i % n_examples])
        model = JolissaintAOModel(
            n_pix=n_pix,
            telescope_diameter=D,
            obstruction_diameter=D_obs,
            wavelength=wavelength,
            pixel_scale=pixel_scale,
            ao_config=ao_cfg,
            atmosphere=atm,
        )
        pupil = create_pupil(n_pix, xp)
        _ = model.compute_long_exposure_psf(pupil)
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    if use_gpu:
        cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    
    for i in range(n_examples):
        atm = create_atmosphere(r0=r0_values[i])
        model = JolissaintAOModel(
            n_pix=n_pix,
            telescope_diameter=D,
            obstruction_diameter=D_obs,
            wavelength=wavelength,
            pixel_scale=pixel_scale,
            ao_config=ao_cfg,
            atmosphere=atm,
        )
        pupil = create_pupil(n_pix, xp)
        psf = model.compute_long_exposure_psf(pupil)
    
    if use_gpu:
        cp.cuda.Stream.null.synchronize()
    total_time = time.perf_counter() - t0
    
    return total_time, total_time / n_examples


def benchmark_aliasing_only(n_examples, n_pix, use_gpu=True):
    """
    Benchmark: Only aliasing PSD computation (the main bottleneck).
    """
    backend_name = 'GPU' if use_gpu else 'CPU'
    init_backend(backend_name)
    backend = get_backend()
    xp = backend.xp
    
    atm = create_atmosphere()
    ao_cfg = create_ao_config()
    
    model = JolissaintAOModel(
        n_pix=n_pix,
        telescope_diameter=D,
        obstruction_diameter=D_obs,
        wavelength=wavelength,
        pixel_scale=pixel_scale,
        ao_config=ao_cfg,
        atmosphere=atm,
    )
    
    # Warmup
    for _ in range(5):
        _ = model.compute_aliasing_psd()
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    if use_gpu:
        cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    
    for _ in range(n_examples):
        psd = model.compute_aliasing_psd()
    
    if use_gpu:
        cp.cuda.Stream.null.synchronize()
    total_time = time.perf_counter() - t0
    
    return total_time, total_time / n_examples


def main():
    n_examples_gpu = 1000
    n_examples_cpu = 100  # Fewer examples for CPU due to speed
    n_pix = 256
    
    print()
    print('=' * 70)
    print(f'BATCH PSF GENERATION BENCHMARK ({n_pix}x{n_pix})')
    print('=' * 70)
    
    results = {}
    
    # CPU benchmarks (fewer examples)
    print(f'\n--- CPU Benchmarks ({n_examples_cpu} examples) ---')
    
    print('  [1/4] Same model, repeated PSF...', end=' ', flush=True)
    total, per_ex = benchmark_same_model(n_examples_cpu, n_pix, use_gpu=False)
    results['cpu_same_model'] = (total, per_ex)
    print(f'{total:.2f}s total, {per_ex*1000:.2f} ms/example')
    
    print('  [2/4] Varying atmosphere params...', end=' ', flush=True)
    total, per_ex = benchmark_varying_atmosphere(n_examples_cpu, n_pix, use_gpu=False)
    results['cpu_varying_atm'] = (total, per_ex)
    print(f'{total:.2f}s total, {per_ex*1000:.2f} ms/example')
    
    print('  [3/4] New model per example...', end=' ', flush=True)
    # Use even fewer examples for this slow benchmark
    total, per_ex = benchmark_new_model_per_example(20, n_pix, use_gpu=False)
    results['cpu_new_model'] = (total * 5, per_ex)  # Extrapolate to 100
    print(f'{total*5:.2f}s total (extrapolated), {per_ex*1000:.2f} ms/example')
    
    print('  [4/4] Aliasing PSD only...', end=' ', flush=True)
    total, per_ex = benchmark_aliasing_only(n_examples_cpu, n_pix, use_gpu=False)
    results['cpu_aliasing'] = (total, per_ex)
    print(f'{total:.2f}s total, {per_ex*1000:.2f} ms/example')
    
    # GPU benchmarks
    if HAS_CUPY:
        print(f'\n--- GPU Benchmarks ({n_examples_gpu} examples) ---')
        
        print('  [1/4] Same model, repeated PSF...', end=' ', flush=True)
        total, per_ex = benchmark_same_model(n_examples_gpu, n_pix, use_gpu=True)
        results['gpu_same_model'] = (total, per_ex)
        print(f'{total:.2f}s total, {per_ex*1000:.2f} ms/example')
        
        print('  [2/4] Varying atmosphere params...', end=' ', flush=True)
        total, per_ex = benchmark_varying_atmosphere(n_examples_gpu, n_pix, use_gpu=True)
        results['gpu_varying_atm'] = (total, per_ex)
        print(f'{total:.2f}s total, {per_ex*1000:.2f} ms/example')
        
        print('  [3/4] New model per example...', end=' ', flush=True)
        total, per_ex = benchmark_new_model_per_example(n_examples_gpu, n_pix, use_gpu=True)
        results['gpu_new_model'] = (total, per_ex)
        print(f'{total:.2f}s total, {per_ex*1000:.2f} ms/example')
        
        print('  [4/4] Aliasing PSD only...', end=' ', flush=True)
        total, per_ex = benchmark_aliasing_only(n_examples_gpu, n_pix, use_gpu=True)
        results['gpu_aliasing'] = (total, per_ex)
        print(f'{total:.2f}s total, {per_ex*1000:.2f} ms/example')
    
    # Summary
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    
    print(f'\n{"Scenario":<35} {"CPU (ms/ex)":<15} {"GPU (ms/ex)":<15} {"Speedup":<10}')
    print('-' * 75)
    
    scenarios = [
        ('Same model (best case)', 'same_model'),
        ('Varying atmosphere', 'varying_atm'),
        ('New model per example', 'new_model'),
        ('Aliasing PSD only', 'aliasing'),
    ]
    
    for name, key in scenarios:
        cpu_time = results[f'cpu_{key}'][1] * 1000
        if HAS_CUPY and f'gpu_{key}' in results:
            gpu_time = results[f'gpu_{key}'][1] * 1000
            speedup = cpu_time / gpu_time
            print(f'{name:<35} {cpu_time:>10.2f}     {gpu_time:>10.2f}     {speedup:>8.1f}x')
        else:
            print(f'{name:<35} {cpu_time:>10.2f}     {"N/A":>10}     {"N/A":>8}')
    
    if HAS_CUPY:
        print('\n' + '=' * 70)
        print('THROUGHPUT ANALYSIS')
        print('=' * 70)
        
        gpu_best = results['gpu_same_model'][1]
        gpu_varying = results['gpu_varying_atm'][1]
        gpu_worst = results['gpu_new_model'][1]
        
        print(f'\nGPU throughput @ {n_pix}x{n_pix}:')
        print(f'  Best case (model reuse):     {1.0/gpu_best:>8.1f} PSFs/second')
        print(f'  Varying atmosphere:          {1.0/gpu_varying:>8.1f} PSFs/second')
        print(f'  Worst case (new model/PSF):  {1.0/gpu_worst:>8.1f} PSFs/second')
        
        # Overhead analysis
        model_overhead = (gpu_worst - gpu_best) * 1000
        print(f'\nModel instantiation overhead: {model_overhead:.2f} ms')
        print(f'  (This includes: frequency grids, masks, LUT construction, precomputed arrays)')
        
        # Extrapolate to larger datasets
        print(f'\nEstimated time to generate 10,000 PSFs:')
        print(f'  With model reuse:     {10000 * gpu_best / 60:>6.1f} minutes')
        print(f'  With new model each:  {10000 * gpu_worst / 60:>6.1f} minutes')
    
    print('\nDone!')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Comprehensive benchmark suite for the Jolissaint AO model.

Includes:
1. CPU vs GPU performance comparison
2. Batch generation throughput analysis
3. Detailed aliasing PSD profiling
4. Mask geometry comparison

Usage:
    python benchmark_jolissaint.py              # Run all benchmarks
    python benchmark_jolissaint.py --quick      # Quick benchmark (smaller grids)
    python benchmark_jolissaint.py --profile    # Detailed profiling only
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
import numpy as np

# Check CuPy availability
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

from nebraa.physics.jolissaint_ao import JolissaintAOModel, AOSystemConfig, AtmosphereProfile, TurbulentLayer
from nebraa.utils.compute import init_backend, get_backend


# =============================================================================
# Setup Utilities
# =============================================================================

def create_atmosphere(r0=0.15, wind_speed=10.0, L0=25.0):
    """Create atmosphere profile with given parameters."""
    atm_layers = [
        TurbulentLayer(altitude=0, r0=r0, wind_speed=wind_speed, wind_direction=0, Cn2_fraction=0.5),
        TurbulentLayer(altitude=10000, r0=r0, wind_speed=wind_speed * 2.24, wind_direction=0.46, Cn2_fraction=0.5),
    ]
    return AtmosphereProfile(layers=atm_layers, L0=L0)


def create_ao_config(mask_geometry='circular', mask_rolloff=0.0):
    """Create AO configuration."""
    return AOSystemConfig(
        wfs_subaperture_size=0.5,
        actuator_pitch=0.4,
        integration_time=1.0/1000.0,
        noise_variance=0.1,
        include_aliasing=True,
        mask_geometry=mask_geometry,
        mask_rolloff=mask_rolloff,
    )


def create_pupil(n_pix, xp, D=8.0, D_obs=1.12):
    """Create simple circular pupil."""
    x = xp.linspace(-1, 1, n_pix)
    X, Y = xp.meshgrid(x, x)
    R = xp.sqrt(X**2 + Y**2)
    pupil = ((R <= 1.0) & (R >= D_obs/D)).astype(xp.float64)
    return pupil


def print_header(title):
    """Print section header."""
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


# =============================================================================
# 1. CPU vs GPU Benchmark
# =============================================================================

def benchmark_cpu_vs_gpu(grid_sizes=[128, 256, 512], n_runs=5):
    """Compare CPU and GPU performance across grid sizes."""
    print_header('CPU vs GPU PERFORMANCE BENCHMARK')
    
    D = 8.0
    D_obs = 1.12
    wavelength = 2.2e-6
    pixel_scale = 10e-3 / 206265
    
    atm = create_atmosphere()
    ao_cfg = create_ao_config()
    
    results = {}
    
    for n_pix in grid_sizes:
        print(f'\nGrid size: {n_pix}x{n_pix}')
        
        # CPU benchmark
        init_backend('CPU')
        model_cpu = JolissaintAOModel(
            n_pix=n_pix, telescope_diameter=D, obstruction_diameter=D_obs,
            wavelength=wavelength, pixel_scale=pixel_scale,
            ao_config=ao_cfg, atmosphere=atm,
        )
        
        # Warmup
        _ = model_cpu.compute_aliasing_psd()
        
        # Time CPU
        times_cpu = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model_cpu.compute_aliasing_psd()
            times_cpu.append(time.perf_counter() - t0)
        cpu_time = np.median(times_cpu) * 1000
        print(f'  CPU (loop-based): {cpu_time:.1f} ms')
        
        # GPU benchmark
        if HAS_CUPY:
            init_backend('GPU')
            model_gpu = JolissaintAOModel(
                n_pix=n_pix, telescope_diameter=D, obstruction_diameter=D_obs,
                wavelength=wavelength, pixel_scale=pixel_scale,
                ao_config=ao_cfg, atmosphere=atm,
            )
            
            # Warmup
            for _ in range(3):
                _ = model_gpu.compute_aliasing_psd()
                cp.cuda.Stream.null.synchronize()
            
            # Time GPU
            times_gpu = []
            for _ in range(n_runs):
                cp.cuda.Stream.null.synchronize()
                t0 = time.perf_counter()
                _ = model_gpu.compute_aliasing_psd()
                cp.cuda.Stream.null.synchronize()
                times_gpu.append(time.perf_counter() - t0)
            gpu_time = np.median(times_gpu) * 1000
            
            speedup = cpu_time / gpu_time
            print(f'  GPU (vectorized): {gpu_time:.1f} ms')
            print(f'  Speedup: {speedup:.2f}x')
            results[n_pix] = {'cpu': cpu_time, 'gpu': gpu_time, 'speedup': speedup}
        else:
            results[n_pix] = {'cpu': cpu_time, 'gpu': None, 'speedup': None}
    
    return results


# =============================================================================
# 2. Batch Generation Throughput
# =============================================================================

def benchmark_batch_throughput(n_examples_gpu=1000, n_examples_cpu=100, n_pix=256):
    """Benchmark batch PSF generation throughput."""
    print_header(f'BATCH PSF GENERATION THROUGHPUT ({n_pix}x{n_pix})')
    
    D = 8.0
    D_obs = 1.12
    wavelength = 2.2e-6
    pixel_scale = 10e-3 / 206265
    ao_cfg = create_ao_config()
    
    results = {}
    
    # Test scenarios
    scenarios = [
        ('Same model (best case)', 'same_model'),
        ('Varying atmosphere', 'varying_atm'),
        ('New model per PSF', 'new_model'),
    ]
    
    def run_same_model(n_examples, use_gpu):
        init_backend('GPU' if use_gpu else 'CPU')
        backend = get_backend()
        xp = backend.xp
        
        atm = create_atmosphere()
        model = JolissaintAOModel(
            n_pix=n_pix, telescope_diameter=D, obstruction_diameter=D_obs,
            wavelength=wavelength, pixel_scale=pixel_scale,
            ao_config=ao_cfg, atmosphere=atm,
        )
        pupil = create_pupil(n_pix, xp)
        
        # Warmup
        for _ in range(3):
            _ = model.compute_long_exposure_psf(pupil)
            if use_gpu:
                cp.cuda.Stream.null.synchronize()
        
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_examples):
            _ = model.compute_long_exposure_psf(pupil)
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        return time.perf_counter() - t0
    
    def run_varying_atm(n_examples, use_gpu):
        init_backend('GPU' if use_gpu else 'CPU')
        backend = get_backend()
        xp = backend.xp
        
        atm = create_atmosphere()
        model = JolissaintAOModel(
            n_pix=n_pix, telescope_diameter=D, obstruction_diameter=D_obs,
            wavelength=wavelength, pixel_scale=pixel_scale,
            ao_config=ao_cfg, atmosphere=atm,
        )
        pupil = create_pupil(n_pix, xp)
        
        np.random.seed(42)
        r0_values = np.random.uniform(0.10, 0.25, n_examples)
        
        # Warmup
        for _ in range(3):
            _ = model.compute_long_exposure_psf(pupil)
            if use_gpu:
                cp.cuda.Stream.null.synchronize()
        
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for i in range(n_examples):
            model.atmosphere.layers[0].r0 = r0_values[i]
            model.atmosphere.layers[1].r0 = r0_values[i]
            _ = model.compute_long_exposure_psf(pupil)
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        return time.perf_counter() - t0
    
    def run_new_model(n_examples, use_gpu):
        init_backend('GPU' if use_gpu else 'CPU')
        backend = get_backend()
        xp = backend.xp
        
        np.random.seed(42)
        r0_values = np.random.uniform(0.10, 0.25, n_examples)
        
        # Warmup
        atm = create_atmosphere(r0=r0_values[0])
        model = JolissaintAOModel(
            n_pix=n_pix, telescope_diameter=D, obstruction_diameter=D_obs,
            wavelength=wavelength, pixel_scale=pixel_scale,
            ao_config=ao_cfg, atmosphere=atm,
        )
        pupil = create_pupil(n_pix, xp)
        _ = model.compute_long_exposure_psf(pupil)
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for i in range(n_examples):
            atm = create_atmosphere(r0=r0_values[i])
            model = JolissaintAOModel(
                n_pix=n_pix, telescope_diameter=D, obstruction_diameter=D_obs,
                wavelength=wavelength, pixel_scale=pixel_scale,
                ao_config=ao_cfg, atmosphere=atm,
            )
            pupil = create_pupil(n_pix, xp)
            _ = model.compute_long_exposure_psf(pupil)
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        return time.perf_counter() - t0
    
    benchmark_funcs = {
        'same_model': run_same_model,
        'varying_atm': run_varying_atm,
        'new_model': run_new_model,
    }
    
    # CPU benchmarks
    print(f'\n--- CPU Benchmarks ({n_examples_cpu} examples) ---')
    for name, key in scenarios:
        print(f'  {name}...', end=' ', flush=True)
        n_ex = 20 if key == 'new_model' else n_examples_cpu
        total = benchmark_funcs[key](n_ex, use_gpu=False)
        per_ex = total / n_ex
        results[f'cpu_{key}'] = per_ex
        print(f'{per_ex*1000:.2f} ms/example')
    
    # GPU benchmarks
    if HAS_CUPY:
        print(f'\n--- GPU Benchmarks ({n_examples_gpu} examples) ---')
        for name, key in scenarios:
            print(f'  {name}...', end=' ', flush=True)
            total = benchmark_funcs[key](n_examples_gpu, use_gpu=True)
            per_ex = total / n_examples_gpu
            results[f'gpu_{key}'] = per_ex
            print(f'{per_ex*1000:.2f} ms/example')
    
    # Summary
    print('\n--- Summary ---')
    print(f'{"Scenario":<30} {"CPU (ms)":<12} {"GPU (ms)":<12} {"Speedup":<10}')
    print('-' * 64)
    for name, key in scenarios:
        cpu = results[f'cpu_{key}'] * 1000
        if HAS_CUPY:
            gpu = results[f'gpu_{key}'] * 1000
            print(f'{name:<30} {cpu:>8.1f}     {gpu:>8.1f}     {cpu/gpu:>6.1f}x')
        else:
            print(f'{name:<30} {cpu:>8.1f}     {"N/A":>8}     {"N/A":>6}')
    
    if HAS_CUPY:
        best = results['gpu_same_model']
        print(f'\nGPU throughput (best case): {1.0/best:.1f} PSFs/second')
        print(f'Estimated time for 10,000 PSFs: {10000*best/60:.1f} minutes')
    
    return results


# =============================================================================
# 3. Detailed Aliasing Profiling
# =============================================================================

def profile_aliasing(n_pix=256, n_runs=3):
    """Profile aliasing PSD computation step by step."""
    print_header(f'ALIASING PSD PROFILING ({n_pix}x{n_pix})')
    
    D = 8.0
    D_obs = 1.12
    wavelength = 2.2e-6
    pixel_scale = 10e-3 / 206265
    
    atm = create_atmosphere()
    ao_cfg = create_ao_config()
    
    def profile_method(model, use_vectorized, sync_gpu=False):
        """Time individual operations."""
        xp = model._xp
        Lambda = model.ao_config.wfs_subaperture_size
        f_wfs = model.ao_config.f_wfs
        dt = model.ao_config.integration_time
        
        timings = {}
        
        if sync_gpu:
            cp.cuda.Stream.null.synchronize()
        
        t_total = time.perf_counter()
        
        # Setup
        t0 = time.perf_counter()
        psd_alias = xp.zeros_like(model.F, dtype=xp.float64)
        if sync_gpu:
            cp.cuda.Stream.null.synchronize()
        timings['setup'] = time.perf_counter() - t0
        
        for layer in model.atmosphere.layers:
            r0_sci = layer.r0 * (model.wavelength / model.atmosphere.wavelength_ref) ** (6/5)
            vx, vy = layer.wind_velocity
            
            # Temporal factor
            t0 = time.perf_counter()
            f_dot_v = model.FX * vx + model.FY * vy
            sinc2_temporal = xp.sinc(dt * f_dot_v) ** 2
            if sync_gpu:
                cp.cuda.Stream.null.synchronize()
            timings['sinc2_temporal'] = time.perf_counter() - t0
            
            if use_vectorized:
                # Vectorized path
                k_arr = model._alias_k_arr
                l_arr = model._alias_l_arr
                sign_arr = model._alias_sign_arr
                
                t0 = time.perf_counter()
                FX_3d = model.FX[None, :, :]
                FY_3d = model.FY[None, :, :]
                fx_alias = FX_3d - k_arr / Lambda
                fy_alias = FY_3d - l_arr / Lambda
                f_alias = xp.sqrt(fx_alias**2 + fy_alias**2)
                if sync_gpu:
                    cp.cuda.Stream.null.synchronize()
                timings['aliased_freq'] = time.perf_counter() - t0
                
                t0 = time.perf_counter()
                if model.ao_config.mask_geometry == 'circular':
                    is_hf = f_alias >= f_wfs
                else:
                    is_hf = (xp.abs(fx_alias) >= f_wfs) | (xp.abs(fy_alias) >= f_wfs)
                if sync_gpu:
                    cp.cuda.Stream.null.synchronize()
                timings['hf_mask'] = time.perf_counter() - t0
                
                t0 = time.perf_counter()
                L0_inv2 = 1.0 / model.atmosphere.L0**2 if model.atmosphere.L0 else 0
                coeff = 0.023 * (r0_sci ** (-5/3))
                psd_alias_f = coeff * xp.power(f_alias**2 + L0_inv2, -11/6)
                if sync_gpu:
                    cp.cuda.Stream.null.synchronize()
                timings['psd_vonkarman'] = time.perf_counter() - t0
                
                t0 = time.perf_counter()
                Fp_alias = model._piston_lut(f_alias)
                if sync_gpu:
                    cp.cuda.Stream.null.synchronize()
                timings['piston_filter'] = time.perf_counter() - t0
                
                t0 = time.perf_counter()
                amplitude = xp.sqrt(xp.maximum(Fp_alias * psd_alias_f, 0.0))
                if sync_gpu:
                    cp.cuda.Stream.null.synchronize()
                timings['amplitude'] = time.perf_counter() - t0
                
                t0 = time.perf_counter()
                eps = 1e-12
                safe_denom_x = xp.where(xp.abs(fy_alias) < eps, eps, fy_alias)
                safe_denom_y = xp.where(xp.abs(fx_alias) < eps, eps, fx_alias)
                geom_factor = FX_3d / safe_denom_x + FY_3d / safe_denom_y
                if sync_gpu:
                    cp.cuda.Stream.null.synchronize()
                timings['geometry'] = time.perf_counter() - t0
                
                t0 = time.perf_counter()
                contrib = is_hf.astype(xp.float64) * sign_arr * geom_factor * amplitude
                complex_sum = xp.sum(contrib, axis=0)
                if sync_gpu:
                    cp.cuda.Stream.null.synchronize()
                timings['contrib_sum'] = time.perf_counter() - t0
        
        if sync_gpu:
            cp.cuda.Stream.null.synchronize()
        timings['total'] = time.perf_counter() - t_total
        
        return timings
    
    # Profile CPU
    print('\n[CPU Vectorized]')
    init_backend('CPU')
    model_cpu = JolissaintAOModel(
        n_pix=n_pix, telescope_diameter=D, obstruction_diameter=D_obs,
        wavelength=wavelength, pixel_scale=pixel_scale,
        ao_config=ao_cfg, atmosphere=atm,
    )
    
    # Warmup
    _ = model_cpu.compute_aliasing_psd()
    
    cpu_times = []
    for _ in range(n_runs):
        cpu_times.append(profile_method(model_cpu, use_vectorized=True, sync_gpu=False))
    
    # Average
    cpu_avg = {k: np.mean([t[k] for t in cpu_times]) * 1000 for k in cpu_times[0]}
    
    for op, t in sorted(cpu_avg.items(), key=lambda x: -x[1])[:8]:
        print(f'  {op:<20} {t:>8.2f} ms')
    
    # Profile GPU
    if HAS_CUPY:
        print('\n[GPU Vectorized]')
        init_backend('GPU')
        model_gpu = JolissaintAOModel(
            n_pix=n_pix, telescope_diameter=D, obstruction_diameter=D_obs,
            wavelength=wavelength, pixel_scale=pixel_scale,
            ao_config=ao_cfg, atmosphere=atm,
        )
        
        # Warmup
        for _ in range(3):
            _ = model_gpu.compute_aliasing_psd()
            cp.cuda.Stream.null.synchronize()
        
        gpu_times = []
        for _ in range(n_runs):
            gpu_times.append(profile_method(model_gpu, use_vectorized=True, sync_gpu=True))
        
        gpu_avg = {k: np.mean([t[k] for t in gpu_times]) * 1000 for k in gpu_times[0]}
        
        for op, t in sorted(gpu_avg.items(), key=lambda x: -x[1])[:8]:
            pct = t / gpu_avg['total'] * 100
            print(f'  {op:<20} {t:>8.2f} ms ({pct:>5.1f}%)')
        
        print(f'\nSpeedup: {cpu_avg["total"]/gpu_avg["total"]:.1f}x')


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark Jolissaint AO model')
    parser.add_argument('--quick', action='store_true', help='Quick benchmark with smaller grids')
    parser.add_argument('--profile', action='store_true', help='Run detailed profiling only')
    parser.add_argument('--batch', action='store_true', help='Run batch throughput benchmark only')
    args = parser.parse_args()
    
    print('=' * 70)
    print('JOLISSAINT AO MODEL BENCHMARK SUITE')
    print('=' * 70)
    
    if HAS_CUPY:
        print(f'CuPy: {cp.__version__}')
        print(f'Device: {cp.cuda.Device(0).compute_capability}')
    else:
        print('CuPy: Not available (CPU only)')
    print(f'NumPy: {np.__version__}')
    
    if args.profile:
        profile_aliasing()
    elif args.batch:
        benchmark_batch_throughput()
    elif args.quick:
        benchmark_cpu_vs_gpu(grid_sizes=[128, 256])
        benchmark_batch_throughput(n_examples_gpu=100, n_examples_cpu=20)
    else:
        benchmark_cpu_vs_gpu()
        benchmark_batch_throughput()
        profile_aliasing()
    
    print('\nDone!')


if __name__ == '__main__':
    main()

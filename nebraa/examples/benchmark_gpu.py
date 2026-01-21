#!/usr/bin/env python
"""Benchmark vectorized GPU vs loop-based CPU aliasing PSD computation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import cupy as cp

print('NumPy: ', np.__version__)
print(f'CuPy: {cp.__version__}')
print(f'Device: {cp.cuda.Device(0).compute_capability}')

from nebraa.physics.jolissaint_ao import JolissaintAOModel, AOSystemConfig, AtmosphereProfile, TurbulentLayer
from nebraa.utils.compute import init_backend

# Setup
D = 8.0
D_obs = 1.12  # Central obstruction
wavelength = 2.2e-6
pixel_scale = 10e-3 / 206265  # 10 mas in radians
Lambda = 0.5
d_act = 0.4
f_loop = 1000.0

atm_layers = [
    TurbulentLayer(altitude=0, r0=0.15, wind_speed=10, wind_direction=0, Cn2_fraction=0.5),
    TurbulentLayer(altitude=10000, r0=0.15, wind_speed=22.4, wind_direction=0.46, Cn2_fraction=0.5),
]
atm = AtmosphereProfile(layers=atm_layers, L0=25.0)
ao_cfg = AOSystemConfig(
    wfs_subaperture_size=Lambda,
    actuator_pitch=d_act,
    integration_time=1.0/f_loop,
    noise_variance=0.1,
    include_aliasing=True,
)

print()
print('=' * 60)
print('BENCHMARKING VECTORIZED GPU vs LOOP-BASED CPU')
print('=' * 60)

for n_pix in [128, 256, 512]:
    print(f'\nGrid size: {n_pix}x{n_pix}')
    
    # CPU benchmark
    init_backend('CPU')
    model_cpu = JolissaintAOModel(
        n_pix=n_pix,
        telescope_diameter=D,
        obstruction_diameter=D_obs,
        wavelength=wavelength,
        pixel_scale=pixel_scale,
        ao_config=ao_cfg,
        atmosphere=atm,
    )
    
    # Warm up
    _ = model_cpu.compute_aliasing_psd()
    
    # Time CPU
    n_runs = 5
    times_cpu = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model_cpu.compute_aliasing_psd()
        times_cpu.append(time.perf_counter() - t0)
    cpu_time = np.median(times_cpu) * 1000
    print(f'  CPU (loop-based): {cpu_time:.1f} ms')
    
    # GPU benchmark
    init_backend('GPU')
    model_gpu = JolissaintAOModel(
        n_pix=n_pix,
        telescope_diameter=D,
        obstruction_diameter=D_obs,
        wavelength=wavelength,
        pixel_scale=pixel_scale,
        ao_config=ao_cfg,
        atmosphere=atm,
    )
    
    # Warm up (includes JIT compilation)
    for _ in range(3):
        psd = model_gpu.compute_aliasing_psd()
        cp.cuda.Stream.null.synchronize()
    
    # Time GPU
    times_gpu = []
    for _ in range(n_runs):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        psd = model_gpu.compute_aliasing_psd()
        cp.cuda.Stream.null.synchronize()
        times_gpu.append(time.perf_counter() - t0)
    gpu_time = np.median(times_gpu) * 1000
    
    speedup = cpu_time / gpu_time
    print(f'  GPU (vectorized): {gpu_time:.1f} ms')
    print(f'  Speedup: {speedup:.2f}x')

print('\nDone!')

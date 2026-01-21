#!/usr/bin/env python
"""
Detailed profiling of aliasing PSD computation.
Compares CPU vs GPU, loop vs vectorized, operation by operation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import math
import numpy as np

# Check CuPy availability
try:
    import cupy as cp
    HAS_CUPY = True
    print(f"CuPy: {cp.__version__}")
except ImportError:
    HAS_CUPY = False
    print("CuPy not available")

from nebraa.physics.jolissaint_ao import JolissaintAOModel, AOSystemConfig, AtmosphereProfile, TurbulentLayer
from nebraa.utils.compute import init_backend, get_backend


def profile_loop_based(model, n_runs=3, sync_gpu=False):
    """Profile the loop-based aliasing computation step by step."""
    backend = get_backend()
    xp = backend.xp
    
    Lambda = model.ao_config.wfs_subaperture_size
    f_wfs = model.ao_config.f_wfs
    dt = model.ao_config.integration_time
    n_alias = 3
    
    timings = {
        'setup': [],
        'sinc2_temporal': [],
        'loop_aliased_freq': [],
        'loop_hf_mask': [],
        'loop_psd_vonkarman': [],
        'loop_piston_filter': [],
        'loop_amplitude': [],
        'loop_geometry': [],
        'loop_accumulate': [],
        'prefactor': [],
        'main_psd': [],
        'axis_singularities': [],
        'total': [],
    }
    
    for _ in range(n_runs):
        if sync_gpu and HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
        
        t_total_start = time.perf_counter()
        
        # Setup
        t0 = time.perf_counter()
        psd_alias = xp.zeros_like(model.F, dtype=xp.float64)
        if sync_gpu and HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
        timings['setup'].append(time.perf_counter() - t0)
        
        for layer in model.atmosphere.layers:
            r0_sci = layer.r0 * (model.wavelength / model.atmosphere.wavelength_ref) ** (6/5)
            vx, vy = layer.wind_velocity
            
            # Temporal averaging
            t0 = time.perf_counter()
            f_dot_v = model.FX * vx + model.FY * vy
            sinc2_temporal = xp.sinc(dt * f_dot_v) ** 2
            if sync_gpu and HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            timings['sinc2_temporal'].append(time.perf_counter() - t0)
            
            complex_sum = xp.zeros_like(model.F, dtype=xp.complex128)
            
            t_loop_aliased = 0
            t_loop_hf = 0
            t_loop_psd = 0
            t_loop_piston = 0
            t_loop_amp = 0
            t_loop_geom = 0
            t_loop_accum = 0
            
            for k in range(-n_alias, n_alias + 1):
                for l in range(-n_alias, n_alias + 1):
                    if k == 0 and l == 0:
                        continue
                    
                    # Aliased frequencies
                    t0 = time.perf_counter()
                    fx_alias = model.FX - k / Lambda
                    fy_alias = model.FY - l / Lambda
                    f_alias = xp.sqrt(fx_alias**2 + fy_alias**2)
                    if sync_gpu and HAS_CUPY:
                        cp.cuda.Stream.null.synchronize()
                    t_loop_aliased += time.perf_counter() - t0
                    
                    # HF mask
                    t0 = time.perf_counter()
                    is_hf = (xp.abs(fx_alias) >= f_wfs) | (xp.abs(fy_alias) >= f_wfs)
                    if sync_gpu and HAS_CUPY:
                        cp.cuda.Stream.null.synchronize()
                    t_loop_hf += time.perf_counter() - t0
                    
                    # Von Karman PSD
                    t0 = time.perf_counter()
                    L0 = model.atmosphere.L0
                    if L0 is not None:
                        L0_inv2 = 1.0 / L0**2
                        coeff = 0.023 * (r0_sci ** (-5/3))
                        psd_alias_f = coeff * xp.power(f_alias**2 + L0_inv2, -11/6)
                    else:
                        coeff = 0.023 * (r0_sci ** (-5/3))
                        f_safe = xp.maximum(f_alias, 1e-12)
                        psd_alias_f = coeff * xp.power(f_safe, -11/3)
                    if sync_gpu and HAS_CUPY:
                        cp.cuda.Stream.null.synchronize()
                    t_loop_psd += time.perf_counter() - t0
                    
                    # Piston filter (LUT)
                    t0 = time.perf_counter()
                    Fp_alias = model._piston_lut(f_alias)
                    if sync_gpu and HAS_CUPY:
                        cp.cuda.Stream.null.synchronize()
                    t_loop_piston += time.perf_counter() - t0
                    
                    # Amplitude
                    t0 = time.perf_counter()
                    amplitude = xp.sqrt(xp.maximum(Fp_alias * psd_alias_f, 0.0))
                    sign = (-1) ** (k + l)
                    if sync_gpu and HAS_CUPY:
                        cp.cuda.Stream.null.synchronize()
                    t_loop_amp += time.perf_counter() - t0
                    
                    # Geometry factor
                    t0 = time.perf_counter()
                    eps = 1e-12
                    safe_denom_x = xp.where(xp.abs(fy_alias) < eps, eps * xp.sign(fy_alias + eps), fy_alias)
                    safe_denom_y = xp.where(xp.abs(fx_alias) < eps, eps * xp.sign(fx_alias + eps), fx_alias)
                    term_x = model.FX / safe_denom_x
                    term_y = model.FY / safe_denom_y
                    geom_factor = term_x + term_y
                    if sync_gpu and HAS_CUPY:
                        cp.cuda.Stream.null.synchronize()
                    t_loop_geom += time.perf_counter() - t0
                    
                    # Accumulate
                    t0 = time.perf_counter()
                    complex_sum = complex_sum + is_hf.astype(xp.float64) * sign * geom_factor * amplitude
                    if sync_gpu and HAS_CUPY:
                        cp.cuda.Stream.null.synchronize()
                    t_loop_accum += time.perf_counter() - t0
            
            timings['loop_aliased_freq'].append(t_loop_aliased)
            timings['loop_hf_mask'].append(t_loop_hf)
            timings['loop_psd_vonkarman'].append(t_loop_psd)
            timings['loop_piston_filter'].append(t_loop_piston)
            timings['loop_amplitude'].append(t_loop_amp)
            timings['loop_geometry'].append(t_loop_geom)
            timings['loop_accumulate'].append(t_loop_accum)
            
            # Prefactor
            t0 = time.perf_counter()
            f4 = xp.maximum(model.F**4, 1e-40)
            prefactor = (model.FX**2 * model.FY**2) / f4
            if sync_gpu and HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            timings['prefactor'].append(time.perf_counter() - t0)
            
            # Main PSD
            t0 = time.perf_counter()
            psd_layer = prefactor * sinc2_temporal * xp.abs(complex_sum)**2
            if sync_gpu and HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            timings['main_psd'].append(time.perf_counter() - t0)
            
            # Axis singularities (simplified timing)
            t0 = time.perf_counter()
            fx_zero = xp.abs(model.FX) < 1e-10
            fy_zero = xp.abs(model.FY) < 1e-10
            # ... simplified, just measure overall time
            if sync_gpu and HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            timings['axis_singularities'].append(time.perf_counter() - t0)
        
        timings['total'].append(time.perf_counter() - t_total_start)
    
    return {k: np.median(v) * 1000 for k, v in timings.items()}


def profile_vectorized(model, n_runs=3, sync_gpu=False):
    """Profile the vectorized aliasing computation step by step."""
    backend = get_backend()
    xp = backend.xp
    
    Lambda = model.ao_config.wfs_subaperture_size
    f_wfs = model.ao_config.f_wfs
    dt = model.ao_config.integration_time
    n_alias = 3
    
    timings = {
        'setup': [],
        'build_kl_arrays': [],
        'sinc2_temporal': [],
        'broadcast_freq': [],
        'aliased_freq': [],
        'hf_mask': [],
        'psd_vonkarman': [],
        'piston_filter': [],
        'amplitude': [],
        'geometry': [],
        'contrib_sum': [],
        'prefactor': [],
        'main_psd': [],
        'axis_singularities': [],
        'total': [],
    }
    
    # Build kl pairs once
    kl_pairs = [(k, l) for k in range(-n_alias, n_alias + 1)
                for l in range(-n_alias, n_alias + 1) if not (k == 0 and l == 0)]
    n_terms = len(kl_pairs)
    
    for _ in range(n_runs):
        if sync_gpu and HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
        
        t_total_start = time.perf_counter()
        
        # Setup
        t0 = time.perf_counter()
        psd_alias = xp.zeros_like(model.F, dtype=xp.float64)
        if sync_gpu and HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
        timings['setup'].append(time.perf_counter() - t0)
        
        # Build k, l arrays
        t0 = time.perf_counter()
        k_arr = xp.array([kl[0] for kl in kl_pairs], dtype=xp.float64)[:, None, None]
        l_arr = xp.array([kl[1] for kl in kl_pairs], dtype=xp.float64)[:, None, None]
        sign_arr = xp.array([(-1) ** (kl[0] + kl[1]) for kl in kl_pairs], dtype=xp.float64)[:, None, None]
        if sync_gpu and HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
        timings['build_kl_arrays'].append(time.perf_counter() - t0)
        
        for layer in model.atmosphere.layers:
            r0_sci = layer.r0 * (model.wavelength / model.atmosphere.wavelength_ref) ** (6/5)
            vx, vy = layer.wind_velocity
            
            # Temporal averaging
            t0 = time.perf_counter()
            f_dot_v = model.FX * vx + model.FY * vy
            sinc2_temporal = xp.sinc(dt * f_dot_v) ** 2
            if sync_gpu and HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            timings['sinc2_temporal'].append(time.perf_counter() - t0)
            
            # Broadcast FX, FY to 3D
            t0 = time.perf_counter()
            FX_3d = model.FX[None, :, :]
            FY_3d = model.FY[None, :, :]
            if sync_gpu and HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            timings['broadcast_freq'].append(time.perf_counter() - t0)
            
            # Aliased frequencies (3D)
            t0 = time.perf_counter()
            fx_alias = FX_3d - k_arr / Lambda
            fy_alias = FY_3d - l_arr / Lambda
            f_alias = xp.sqrt(fx_alias**2 + fy_alias**2)
            if sync_gpu and HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            timings['aliased_freq'].append(time.perf_counter() - t0)
            
            # HF mask (3D)
            t0 = time.perf_counter()
            is_hf = (xp.abs(fx_alias) >= f_wfs) | (xp.abs(fy_alias) >= f_wfs)
            if sync_gpu and HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            timings['hf_mask'].append(time.perf_counter() - t0)
            
            # Von Karman PSD (3D)
            t0 = time.perf_counter()
            L0 = model.atmosphere.L0
            coeff = 0.023 * (r0_sci ** (-5/3))
            if L0 is not None:
                L0_inv2 = 1.0 / L0**2
                psd_alias_f = coeff * xp.power(f_alias**2 + L0_inv2, -11/6)
            else:
                f_safe = xp.maximum(f_alias, 1e-12)
                psd_alias_f = coeff * xp.power(f_safe, -11/3)
            if sync_gpu and HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            timings['psd_vonkarman'].append(time.perf_counter() - t0)
            
            # Piston filter (3D)
            t0 = time.perf_counter()
            Fp_alias = model._piston_lut(f_alias)
            if sync_gpu and HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            timings['piston_filter'].append(time.perf_counter() - t0)
            
            # Amplitude (3D)
            t0 = time.perf_counter()
            amplitude = xp.sqrt(xp.maximum(Fp_alias * psd_alias_f, 0.0))
            if sync_gpu and HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            timings['amplitude'].append(time.perf_counter() - t0)
            
            # Geometry factor (3D)
            t0 = time.perf_counter()
            eps = 1e-12
            safe_denom_x = xp.where(xp.abs(fy_alias) < eps, eps * xp.sign(fy_alias + eps), fy_alias)
            safe_denom_y = xp.where(xp.abs(fx_alias) < eps, eps * xp.sign(fx_alias + eps), fx_alias)
            term_x = FX_3d / safe_denom_x
            term_y = FY_3d / safe_denom_y
            geom_factor = term_x + term_y
            if sync_gpu and HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            timings['geometry'].append(time.perf_counter() - t0)
            
            # Contribution sum
            t0 = time.perf_counter()
            contrib = is_hf.astype(xp.float64) * sign_arr * geom_factor * amplitude
            complex_sum = xp.sum(contrib, axis=0)
            if sync_gpu and HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            timings['contrib_sum'].append(time.perf_counter() - t0)
            
            # Prefactor
            t0 = time.perf_counter()
            f4 = xp.maximum(model.F**4, 1e-40)
            prefactor = (model.FX**2 * model.FY**2) / f4
            if sync_gpu and HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            timings['prefactor'].append(time.perf_counter() - t0)
            
            # Main PSD
            t0 = time.perf_counter()
            psd_layer = prefactor * sinc2_temporal * complex_sum**2
            if sync_gpu and HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            timings['main_psd'].append(time.perf_counter() - t0)
            
            # Axis singularities - FULL VECTORIZED computation
            t0 = time.perf_counter()
            fx_zero = xp.abs(model.FX) < 1e-10
            fy_zero = xp.abs(model.FY) < 1e-10
            
            L0 = model.atmosphere.L0
            L0_inv2 = 1.0 / L0**2 if L0 else 0
            coeff = 0.023 * (r0_sci ** (-5/3))
            
            # fx=0 case: vectorized over l != 0
            l_vals = xp.array([l for l in range(-n_alias, n_alias + 1) if l != 0], dtype=xp.float64)
            FY_3d_l = model.FY[None, :, :]
            fy_al_3d = FY_3d_l - l_vals[:, None, None] / Lambda
            f_al_fx0 = xp.abs(fy_al_3d)
            is_hf_fx0 = xp.abs(fy_al_3d) >= f_wfs
            if L0 is not None:
                psd_fx0_3d = coeff * xp.power(f_al_fx0**2 + L0_inv2, -11/6)
            else:
                psd_fx0_3d = coeff * xp.power(xp.maximum(f_al_fx0, 1e-12), -11/3)
            Fp_fx0_3d = model._piston_lut(f_al_fx0)
            psd_fx0 = xp.sum(is_hf_fx0.astype(xp.float64) * Fp_fx0_3d * psd_fx0_3d, axis=0) * sinc2_temporal
            
            # fy=0 case: vectorized over k != 0
            k_vals = xp.array([k for k in range(-n_alias, n_alias + 1) if k != 0], dtype=xp.float64)
            FX_3d_k = model.FX[None, :, :]
            fx_al_3d = FX_3d_k - k_vals[:, None, None] / Lambda
            f_al_fy0 = xp.abs(fx_al_3d)
            is_hf_fy0 = xp.abs(fx_al_3d) >= f_wfs
            if L0 is not None:
                psd_fy0_3d = coeff * xp.power(f_al_fy0**2 + L0_inv2, -11/6)
            else:
                psd_fy0_3d = coeff * xp.power(xp.maximum(f_al_fy0, 1e-12), -11/3)
            Fp_fy0_3d = model._piston_lut(f_al_fy0)
            psd_fy0 = xp.sum(is_hf_fy0.astype(xp.float64) * Fp_fy0_3d * psd_fy0_3d, axis=0) * sinc2_temporal
            
            # (0,0) case: vectorized
            kl_00 = [(k, l) for k in range(-n_alias, n_alias + 1)
                     for l in range(-n_alias, n_alias + 1) if k != 0 and l != 0]
            k_00 = xp.array([kl[0] for kl in kl_00], dtype=xp.float64)
            l_00 = xp.array([kl[1] for kl in kl_00], dtype=xp.float64)
            f_00_arr = xp.sqrt((k_00 / Lambda)**2 + (l_00 / Lambda)**2)
            if L0 is not None:
                psd_00_arr = coeff * xp.power(f_00_arr**2 + L0_inv2, -11/6)
            else:
                psd_00_arr = coeff * xp.power(f_00_arr, -11/3)
            Fp_00_arr = model._piston_lut(f_00_arr)
            psd_00 = xp.sum(Fp_00_arr * psd_00_arr)
            
            # Combine
            both_zero = fx_zero & fy_zero
            psd_layer = xp.where(both_zero, psd_00, psd_layer)
            psd_layer = xp.where(fx_zero & ~both_zero, psd_fx0, psd_layer)
            psd_layer = xp.where(fy_zero & ~both_zero, psd_fy0, psd_layer)
            
            if sync_gpu and HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            timings['axis_singularities'].append(time.perf_counter() - t0)
        
        timings['total'].append(time.perf_counter() - t_total_start)
    
    return {k: np.median(v) * 1000 for k, v in timings.items()}


def main():
    # Setup
    D = 8.0
    D_obs = 1.12
    wavelength = 2.2e-6
    pixel_scale = 10e-3 / 206265
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
    
    n_pix = 256  # Fixed grid size for detailed profiling
    
    print("=" * 80)
    print(f"DETAILED PROFILING OF ALIASING PSD ({n_pix}x{n_pix})")
    print("=" * 80)
    
    results = {}
    
    # 1. CPU Loop-based
    print("\n[1] CPU + Loop-based")
    init_backend('CPU')
    model = JolissaintAOModel(n_pix=n_pix, telescope_diameter=D, obstruction_diameter=D_obs,
                               wavelength=wavelength, pixel_scale=pixel_scale, 
                               ao_config=ao_cfg, atmosphere=atm)
    # Warmup
    _ = model._compute_aliasing_psd_loop()
    results['cpu_loop'] = profile_loop_based(model, n_runs=3, sync_gpu=False)
    
    # 2. CPU Vectorized
    print("[2] CPU + Vectorized")
    _ = model._compute_aliasing_psd_vectorized()
    results['cpu_vec'] = profile_vectorized(model, n_runs=3, sync_gpu=False)
    
    if HAS_CUPY:
        # 3. GPU Loop-based
        print("[3] GPU + Loop-based")
        init_backend('GPU')
        model_gpu = JolissaintAOModel(n_pix=n_pix, telescope_diameter=D, obstruction_diameter=D_obs,
                                       wavelength=wavelength, pixel_scale=pixel_scale,
                                       ao_config=ao_cfg, atmosphere=atm)
        # Warmup
        for _ in range(3):
            _ = model_gpu._compute_aliasing_psd_loop()
            cp.cuda.Stream.null.synchronize()
        results['gpu_loop'] = profile_loop_based(model_gpu, n_runs=3, sync_gpu=True)
        
        # 4. GPU Vectorized
        print("[4] GPU + Vectorized")
        for _ in range(3):
            _ = model_gpu._compute_aliasing_psd_vectorized()
            cp.cuda.Stream.null.synchronize()
        results['gpu_vec'] = profile_vectorized(model_gpu, n_runs=3, sync_gpu=True)
    
    # Print results
    print("\n" + "=" * 80)
    print("TIMING BREAKDOWN (ms)")
    print("=" * 80)
    
    # Loop-based operations
    loop_ops = ['setup', 'sinc2_temporal', 'loop_aliased_freq', 'loop_hf_mask', 
                'loop_psd_vonkarman', 'loop_piston_filter', 'loop_amplitude',
                'loop_geometry', 'loop_accumulate', 'prefactor', 'main_psd', 
                'axis_singularities', 'total']
    
    print("\n--- LOOP-BASED ---")
    print(f"{'Operation':<25} {'CPU':>10} {'GPU':>10} {'Speedup':>10}")
    print("-" * 55)
    for op in loop_ops:
        cpu_t = results['cpu_loop'].get(op, 0)
        gpu_t = results.get('gpu_loop', {}).get(op, 0) if HAS_CUPY else 0
        speedup = cpu_t / gpu_t if gpu_t > 0 else 0
        print(f"{op:<25} {cpu_t:>10.2f} {gpu_t:>10.2f} {speedup:>10.2f}x")
    
    # Vectorized operations
    vec_ops = ['setup', 'build_kl_arrays', 'sinc2_temporal', 'broadcast_freq',
               'aliased_freq', 'hf_mask', 'psd_vonkarman', 'piston_filter',
               'amplitude', 'geometry', 'contrib_sum', 'prefactor', 'main_psd',
               'axis_singularities', 'total']
    
    print("\n--- VECTORIZED ---")
    print(f"{'Operation':<25} {'CPU':>10} {'GPU':>10} {'Speedup':>10}")
    print("-" * 55)
    for op in vec_ops:
        cpu_t = results['cpu_vec'].get(op, 0)
        gpu_t = results.get('gpu_vec', {}).get(op, 0) if HAS_CUPY else 0
        speedup = cpu_t / gpu_t if gpu_t > 0 else 0
        print(f"{op:<25} {cpu_t:>10.2f} {gpu_t:>10.2f} {speedup:>10.2f}x")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"{'Method':<25} {'Time (ms)':>12} {'vs CPU Loop':>15}")
    print("-" * 55)
    
    cpu_loop_total = results['cpu_loop']['total']
    print(f"{'CPU + Loop':<25} {cpu_loop_total:>12.1f} {'(baseline)':>15}")
    print(f"{'CPU + Vectorized':<25} {results['cpu_vec']['total']:>12.1f} {cpu_loop_total/results['cpu_vec']['total']:>14.2f}x")
    
    if HAS_CUPY:
        print(f"{'GPU + Loop':<25} {results['gpu_loop']['total']:>12.1f} {cpu_loop_total/results['gpu_loop']['total']:>14.2f}x")
        print(f"{'GPU + Vectorized':<25} {results['gpu_vec']['total']:>12.1f} {cpu_loop_total/results['gpu_vec']['total']:>14.2f}x")
    
    # Identify bottlenecks
    print("\n" + "=" * 80)
    print("BOTTLENECK ANALYSIS")
    print("=" * 80)
    
    print("\n[CPU Loop-based] Top 5 operations:")
    sorted_cpu = sorted([(k, v) for k, v in results['cpu_loop'].items() if k != 'total'], 
                        key=lambda x: x[1], reverse=True)[:5]
    for op, t in sorted_cpu:
        pct = t / results['cpu_loop']['total'] * 100
        print(f"  {op:<25} {t:>8.2f} ms ({pct:>5.1f}%)")
    
    print("\n[CPU Vectorized] Top 5 operations:")
    sorted_cpu_vec = sorted([(k, v) for k, v in results['cpu_vec'].items() if k != 'total'],
                            key=lambda x: x[1], reverse=True)[:5]
    for op, t in sorted_cpu_vec:
        pct = t / results['cpu_vec']['total'] * 100
        print(f"  {op:<25} {t:>8.2f} ms ({pct:>5.1f}%)")
    
    if HAS_CUPY:
        print("\n[GPU Loop-based] Top 5 operations:")
        sorted_gpu = sorted([(k, v) for k, v in results['gpu_loop'].items() if k != 'total'],
                            key=lambda x: x[1], reverse=True)[:5]
        for op, t in sorted_gpu:
            pct = t / results['gpu_loop']['total'] * 100
            print(f"  {op:<25} {t:>8.2f} ms ({pct:>5.1f}%)")
        
        print("\n[GPU Vectorized] Top 5 operations:")
        sorted_gpu_vec = sorted([(k, v) for k, v in results['gpu_vec'].items() if k != 'total'],
                                key=lambda x: x[1], reverse=True)[:5]
        for op, t in sorted_gpu_vec:
            pct = t / results['gpu_vec']['total'] * 100
            print(f"  {op:<25} {t:>8.2f} ms ({pct:>5.1f}%)")
    
    # Validation: benchmark actual public method compute_aliasing_psd()
    print("\n" + "=" * 80)
    print("VALIDATION: PUBLIC METHOD compute_aliasing_psd()")
    print("=" * 80)
    
    # CPU
    init_backend('CPU')
    model_cpu = JolissaintAOModel(n_pix=n_pix, telescope_diameter=D, obstruction_diameter=D_obs,
                                   wavelength=wavelength, pixel_scale=pixel_scale,
                                   ao_config=ao_cfg, atmosphere=atm)
    _ = model_cpu.compute_aliasing_psd()  # warmup
    times_cpu = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = model_cpu.compute_aliasing_psd()
        times_cpu.append(time.perf_counter() - t0)
    cpu_public = np.median(times_cpu) * 1000
    
    if HAS_CUPY:
        # GPU
        init_backend('GPU')
        model_gpu2 = JolissaintAOModel(n_pix=n_pix, telescope_diameter=D, obstruction_diameter=D_obs,
                                        wavelength=wavelength, pixel_scale=pixel_scale,
                                        ao_config=ao_cfg, atmosphere=atm)
        for _ in range(3):  # warmup
            _ = model_gpu2.compute_aliasing_psd()
            cp.cuda.Stream.null.synchronize()
        
        times_gpu = []
        for _ in range(5):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            _ = model_gpu2.compute_aliasing_psd()
            cp.cuda.Stream.null.synchronize()
            times_gpu.append(time.perf_counter() - t0)
        gpu_public = np.median(times_gpu) * 1000
        
        print(f"\nPublic method compute_aliasing_psd() @ {n_pix}x{n_pix}:")
        print(f"  CPU (uses loop):       {cpu_public:>8.1f} ms")
        print(f"  GPU (uses vectorized): {gpu_public:>8.1f} ms")
        print(f"  Speedup:               {cpu_public/gpu_public:>8.2f}x")
    else:
        print(f"\nPublic method compute_aliasing_psd() @ {n_pix}x{n_pix}:")
        print(f"  CPU (uses loop):       {cpu_public:>8.1f} ms")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Jolissaint AO Model Test & Validation Suite
============================================

Comprehensive tests for the Jolissaint analytical AO model including:
1. Basic PSF computation
2. Error budget validation (fitting, aliasing, servo-lag, anisoplanatism)
3. Parameter sensitivity tests (actuator density, sampling frequency)
4. Mask geometry comparison (circular vs square)
5. GPU correctness verification

Based on: Jolissaint, Véran & Conan (2006)
"Analytical modeling of adaptive optics: foundations of the 
 phase spatial power spectrum approach"
J. Opt. Soc. Am. A, Vol. 23, No. 2, pp. 382-394

Usage:
    python test_jolissaint.py              # Run all tests
    python test_jolissaint.py --quick      # Quick tests only
    python test_jolissaint.py --plot       # Generate diagnostic plots
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np

# Check plotting availability
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Check CuPy availability
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

from nebraa.physics.jolissaint_ao import (
    JolissaintAOModel,
    TurbulentLayer,
    AtmosphereProfile,
    AOSystemConfig,
    create_simple_atmosphere,
    create_ao_config,
)
from nebraa.utils.compute import init_backend, get_backend


# =============================================================================
# Utilities
# =============================================================================

def create_pupil(n_pix, xp, D=8.2, D_obs=1.116):
    """Create simple VLT-like circular pupil."""
    x = xp.linspace(-1, 1, n_pix)
    X, Y = xp.meshgrid(x, x)
    R = xp.sqrt(X**2 + Y**2)
    pupil = ((R <= 1.0) & (R >= D_obs/D)).astype(xp.float64)
    return pupil


def print_header(title):
    """Print section header."""
    print('\n' + '=' * 60)
    print(title)
    print('=' * 60)


def print_pass(msg):
    """Print pass message."""
    print(f'  ✓ {msg}')


def print_fail(msg):
    """Print fail message."""
    print(f'  ✗ {msg}')


def print_info(msg):
    """Print info message."""
    print(f'  • {msg}')


# =============================================================================
# Test 1: Basic PSF Computation
# =============================================================================

def test_basic_psf(n_pix=256):
    """Test basic PSF computation and verify output properties."""
    print_header('Test 1: Basic PSF Computation')
    
    init_backend('CPU')
    backend = get_backend()
    xp = backend.xp
    
    wavelength = 2.2e-6  # K-band
    pixel_scale = 13e-3 / 206265  # 13 mas in radians
    
    pupil = create_pupil(n_pix, xp)
    
    atmosphere = create_simple_atmosphere(r0=0.15, L0=25.0, wind_speed=10.0)
    ao_config = create_ao_config(n_actuators=14, telescope_diameter=8.2)
    
    print_info(f'r0 at 500nm: {atmosphere.r0_total:.3f} m')
    print_info(f'r0 at {wavelength*1e6:.1f}μm: {atmosphere.r0_at_wavelength(wavelength):.3f} m')
    print_info(f'Actuator pitch: {ao_config.actuator_pitch:.3f} m')
    
    model = JolissaintAOModel(
        n_pix=n_pix, telescope_diameter=8.2, obstruction_diameter=1.116,
        wavelength=wavelength, pixel_scale=pixel_scale,
        atmosphere=atmosphere, ao_config=ao_config,
    )
    
    # Compute PSF
    result = model.compute_long_exposure_psf(pupil, return_components=True)
    psf = backend.to_numpy(result['psf'])
    
    # Verify PSF properties
    tests_passed = True
    
    # Check normalization
    psf_sum = np.sum(psf)
    if abs(psf_sum - 1.0) < 1e-6:
        print_pass(f'PSF normalized to 1.0 (sum = {psf_sum:.6f})')
    else:
        print_fail(f'PSF not normalized (sum = {psf_sum:.6f})')
        tests_passed = False
    
    # Check non-negative
    if np.all(psf >= 0):
        print_pass('PSF non-negative')
    else:
        print_fail('PSF has negative values')
        tests_passed = False
    
    # Check peak at center
    peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
    center = n_pix // 2
    if abs(peak_idx[0] - center) <= 1 and abs(peak_idx[1] - center) <= 1:
        print_pass(f'Peak at center ({peak_idx})')
    else:
        print_fail(f'Peak not at center ({peak_idx})')
        tests_passed = False
    
    # Check Strehl ratio is reasonable
    strehl = model.compute_strehl_ratio(pupil)
    if 0.1 < strehl < 0.95:
        print_pass(f'Strehl ratio reasonable: {strehl:.3f}')
    else:
        print_fail(f'Strehl ratio suspicious: {strehl:.3f}')
        tests_passed = False
    
    # Error breakdown
    errors = model.get_error_breakdown()
    print_info(f'Total phase variance: {errors["total"]:.4f} rad²')
    print_info(f'RMS wavefront error: {errors["rms_rad"]:.3f} rad')
    
    return tests_passed, model


# =============================================================================
# Test 2: Error Budget Validation
# =============================================================================

def test_error_budget(n_pix=256):
    """Validate individual error contributions."""
    print_header('Test 2: Error Budget Validation')
    
    init_backend('CPU')
    backend = get_backend()
    xp = backend.xp
    
    wavelength = 2.2e-6
    pixel_scale = 13e-3 / 206265
    pupil = create_pupil(n_pix, xp)
    
    atmosphere = create_simple_atmosphere(r0=0.15, L0=25.0, wind_speed=10.0)
    
    tests_passed = True
    
    # Test fitting error only
    print('\n  Fitting error:')
    ao_cfg = AOSystemConfig(
        actuator_pitch=8.2/14, integration_time=0.001,
        include_fitting=True, include_aliasing=False,
        include_servo_lag=False, include_anisoplanatism=False, include_noise=False,
    )
    model = JolissaintAOModel(
        n_pix=n_pix, telescope_diameter=8.2, obstruction_diameter=1.116,
        wavelength=wavelength, pixel_scale=pixel_scale,
        atmosphere=atmosphere, ao_config=ao_cfg,
    )
    errors = model.get_error_breakdown()
    fit_var = errors['fitting']
    
    # Hardy approximation: σ² ≈ 0.28 (d/r0)^(5/3)
    r0_sci = atmosphere.r0_at_wavelength(wavelength)
    hardy_fit = 0.28 * (ao_cfg.actuator_pitch / r0_sci) ** (5/3)
    ratio = fit_var / hardy_fit
    
    print_info(f'Model: {fit_var:.4f} rad², Hardy approx: {hardy_fit:.4f} rad²')
    if 0.5 < ratio < 2.0:
        print_pass(f'Within expected range (ratio = {ratio:.2f})')
    else:
        print_fail(f'Outside expected range (ratio = {ratio:.2f})')
        tests_passed = False
    
    # Test aliasing error
    print('\n  Aliasing error:')
    ao_cfg = AOSystemConfig(
        actuator_pitch=8.2/14, integration_time=0.001,
        include_fitting=False, include_aliasing=True,
        include_servo_lag=False, include_anisoplanatism=False, include_noise=False,
    )
    model = JolissaintAOModel(
        n_pix=n_pix, telescope_diameter=8.2, obstruction_diameter=1.116,
        wavelength=wavelength, pixel_scale=pixel_scale,
        atmosphere=atmosphere, ao_config=ao_cfg,
    )
    errors = model.get_error_breakdown()
    alias_var = errors['aliasing']
    
    # Aliasing should be smaller than fitting for well-sampled WFS
    if alias_var < fit_var:
        print_pass(f'Aliasing ({alias_var:.4f}) < Fitting ({fit_var:.4f})')
    else:
        print_info(f'Aliasing ({alias_var:.4f}) ≥ Fitting ({fit_var:.4f}) - may be okay')
    
    # Test servo-lag
    print('\n  Servo-lag error:')
    ao_cfg = AOSystemConfig(
        actuator_pitch=8.2/14, integration_time=0.002,  # 500 Hz
        include_fitting=False, include_aliasing=False,
        include_servo_lag=True, include_anisoplanatism=False, include_noise=False,
    )
    model = JolissaintAOModel(
        n_pix=n_pix, telescope_diameter=8.2, obstruction_diameter=1.116,
        wavelength=wavelength, pixel_scale=pixel_scale,
        atmosphere=atmosphere, ao_config=ao_cfg,
    )
    errors = model.get_error_breakdown()
    servo_var = errors['servo_lag']
    print_info(f'Servo-lag variance: {servo_var:.4f} rad²')
    if servo_var > 0:
        print_pass('Servo-lag error computed')
    else:
        print_fail('Servo-lag error is zero')
        tests_passed = False
    
    return tests_passed


# =============================================================================
# Test 3: Parameter Sensitivity
# =============================================================================

def test_actuator_density(n_pix=256):
    """Test Strehl vs actuator density (should increase with more actuators)."""
    print_header('Test 3: Actuator Density Sensitivity')
    
    init_backend('CPU')
    backend = get_backend()
    xp = backend.xp
    
    wavelength = 2.2e-6
    pixel_scale = 13e-3 / 206265
    pupil = create_pupil(n_pix, xp)
    
    atmosphere = create_simple_atmosphere(r0=0.15, L0=25.0, wind_speed=10.0)
    
    n_acts = [8, 12, 16, 20, 24]
    strehls = []
    
    for n in n_acts:
        ao_cfg = create_ao_config(n_actuators=n, telescope_diameter=8.2)
        model = JolissaintAOModel(
            n_pix=n_pix, telescope_diameter=8.2, obstruction_diameter=1.116,
            wavelength=wavelength, pixel_scale=pixel_scale,
            atmosphere=atmosphere, ao_config=ao_cfg,
        )
        s = model.compute_strehl_ratio(pupil)
        strehls.append(s)
        print_info(f'n_act={n:2d}, pitch={ao_cfg.actuator_pitch:.3f}m, Strehl={s:.3f}')
    
    # Strehl should increase with actuator count
    if all(strehls[i] <= strehls[i+1] for i in range(len(strehls)-1)):
        print_pass('Strehl increases monotonically with actuator count')
        return True
    else:
        print_fail('Strehl does not increase monotonically')
        return False


def test_sampling_frequency(n_pix=256):
    """Test Strehl vs sampling frequency (should increase with higher freq)."""
    print_header('Test 4: Sampling Frequency Sensitivity')
    
    init_backend('CPU')
    backend = get_backend()
    xp = backend.xp
    
    wavelength = 2.2e-6
    pixel_scale = 13e-3 / 206265
    pupil = create_pupil(n_pix, xp)
    
    # Higher wind speed to make servo-lag visible
    atmosphere = create_simple_atmosphere(r0=0.15, L0=25.0, wind_speed=15.0)
    
    freqs = [100, 250, 500, 1000, 2000]
    strehls = []
    
    for f in freqs:
        ao_cfg = AOSystemConfig(
            actuator_pitch=8.2/14, integration_time=1.0/f, loop_delay=0.5/f,
        )
        model = JolissaintAOModel(
            n_pix=n_pix, telescope_diameter=8.2, obstruction_diameter=1.116,
            wavelength=wavelength, pixel_scale=pixel_scale,
            atmosphere=atmosphere, ao_config=ao_cfg,
        )
        s = model.compute_strehl_ratio(pupil)
        strehls.append(s)
        print_info(f'freq={f:4d}Hz, dt={1000/f:.1f}ms, Strehl={s:.3f}')
    
    # Strehl should generally increase with frequency
    if strehls[-1] > strehls[0]:
        print_pass('Strehl improves with higher sampling frequency')
        return True
    else:
        print_fail('Strehl does not improve with frequency')
        return False


# =============================================================================
# Test 4: Mask Geometry
# =============================================================================

def test_mask_geometry(n_pix=256):
    """Test circular vs square mask geometry."""
    print_header('Test 5: Mask Geometry Comparison')
    
    init_backend('CPU')
    backend = get_backend()
    xp = backend.xp
    
    wavelength = 2.2e-6
    pixel_scale = 13e-3 / 206265
    pupil = create_pupil(n_pix, xp)
    
    atmosphere = create_simple_atmosphere(r0=0.15, L0=25.0, wind_speed=10.0)
    
    geometries = [
        ('circular', 0.0),
        ('circular', 0.1 * (1.0 / (2 * 0.4))),  # 10% rolloff
        ('square', 0.0),
    ]
    
    strehls = []
    for geom, rolloff in geometries:
        ao_cfg = AOSystemConfig(
            actuator_pitch=8.2/14, integration_time=0.001,
            mask_geometry=geom, mask_rolloff=rolloff,
        )
        model = JolissaintAOModel(
            n_pix=n_pix, telescope_diameter=8.2, obstruction_diameter=1.116,
            wavelength=wavelength, pixel_scale=pixel_scale,
            atmosphere=atmosphere, ao_config=ao_cfg,
        )
        s = model.compute_strehl_ratio(pupil)
        strehls.append(s)
        rolloff_str = f'rolloff={rolloff:.4f}' if rolloff > 0 else 'hard'
        print_info(f'{geom}, {rolloff_str}: Strehl={s:.3f}')
    
    # All should produce valid PSFs
    if all(0.1 < s < 0.95 for s in strehls):
        print_pass('All mask geometries produce valid Strehls')
        return True
    else:
        print_fail('Some mask geometries produce invalid Strehls')
        return False


# =============================================================================
# Test 5: GPU Correctness
# =============================================================================

def test_gpu_correctness(n_pix=256):
    """Verify GPU results match CPU results."""
    print_header('Test 6: GPU Correctness')
    
    if not HAS_CUPY:
        print_info('CuPy not available, skipping GPU test')
        return True
    
    wavelength = 2.2e-6
    pixel_scale = 13e-3 / 206265
    
    atmosphere = create_simple_atmosphere(r0=0.15, L0=25.0, wind_speed=10.0)
    ao_cfg = create_ao_config(n_actuators=14, telescope_diameter=8.2)
    
    # CPU computation
    init_backend('CPU')
    backend_cpu = get_backend()
    pupil_cpu = create_pupil(n_pix, backend_cpu.xp)
    
    model_cpu = JolissaintAOModel(
        n_pix=n_pix, telescope_diameter=8.2, obstruction_diameter=1.116,
        wavelength=wavelength, pixel_scale=pixel_scale,
        atmosphere=atmosphere, ao_config=ao_cfg,
    )
    psd_cpu = backend_cpu.to_numpy(model_cpu.compute_aliasing_psd())
    psf_cpu = backend_cpu.to_numpy(model_cpu.compute_long_exposure_psf(pupil_cpu))
    
    # GPU computation
    init_backend('GPU')
    backend_gpu = get_backend()
    pupil_gpu = create_pupil(n_pix, backend_gpu.xp)
    
    model_gpu = JolissaintAOModel(
        n_pix=n_pix, telescope_diameter=8.2, obstruction_diameter=1.116,
        wavelength=wavelength, pixel_scale=pixel_scale,
        atmosphere=atmosphere, ao_config=ao_cfg,
    )
    psd_gpu = backend_gpu.to_numpy(model_gpu.compute_aliasing_psd())
    psf_gpu = backend_gpu.to_numpy(model_gpu.compute_long_exposure_psf(pupil_gpu))
    
    # Compare
    tests_passed = True
    
    psd_diff = np.max(np.abs(psd_cpu - psd_gpu))
    psd_rel = psd_diff / np.max(psd_cpu)
    if psd_rel < 1e-5:
        print_pass(f'Aliasing PSD matches (rel diff = {psd_rel:.2e})')
    else:
        print_fail(f'Aliasing PSD mismatch (rel diff = {psd_rel:.2e})')
        tests_passed = False
    
    psf_diff = np.max(np.abs(psf_cpu - psf_gpu))
    psf_rel = psf_diff / np.max(psf_cpu)
    if psf_rel < 1e-5:
        print_pass(f'PSF matches (rel diff = {psf_rel:.2e})')
    else:
        print_fail(f'PSF mismatch (rel diff = {psf_rel:.2e})')
        tests_passed = False
    
    return tests_passed


# =============================================================================
# Diagnostic Plots
# =============================================================================

def generate_plots(n_pix=256):
    """Generate diagnostic plots."""
    if not HAS_MATPLOTLIB:
        print('Matplotlib not available, skipping plots')
        return
    
    print_header('Generating Diagnostic Plots')
    
    init_backend('CPU')
    backend = get_backend()
    xp = backend.xp
    
    wavelength = 2.2e-6
    pixel_scale = 13e-3 / 206265
    pupil = create_pupil(n_pix, xp)
    
    atmosphere = create_simple_atmosphere(r0=0.15, L0=25.0, wind_speed=10.0)
    ao_cfg = create_ao_config(n_actuators=14, telescope_diameter=8.2)
    
    model = JolissaintAOModel(
        n_pix=n_pix, telescope_diameter=8.2, obstruction_diameter=1.116,
        wavelength=wavelength, pixel_scale=pixel_scale,
        atmosphere=atmosphere, ao_config=ao_cfg,
    )
    
    # Compute PSDs
    psd_fit = backend.to_numpy(model.compute_fitting_psd())
    psd_alias = backend.to_numpy(model.compute_aliasing_psd())
    psd_servo = backend.to_numpy(model.compute_servo_lag_psd())
    psd_total = backend.to_numpy(model.compute_total_residual_psd())
    psf = backend.to_numpy(model.compute_long_exposure_psf(pupil))
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    f_max = 1 / (2 * model.pupil_pixel_size)
    extent = [-f_max, f_max, -f_max, f_max]
    
    # Fitting PSD
    im = axes[0, 0].imshow(np.fft.fftshift(psd_fit) + 1e-20, norm=LogNorm(),
                           extent=extent, cmap='viridis')
    axes[0, 0].set_title('Fitting Error PSD')
    plt.colorbar(im, ax=axes[0, 0])
    
    # Aliasing PSD
    im = axes[0, 1].imshow(np.fft.fftshift(psd_alias) + 1e-20, norm=LogNorm(),
                           extent=extent, cmap='viridis')
    axes[0, 1].set_title('Aliasing Error PSD')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Servo-lag PSD
    im = axes[0, 2].imshow(np.fft.fftshift(psd_servo) + 1e-20, norm=LogNorm(),
                           extent=extent, cmap='viridis')
    axes[0, 2].set_title('Servo-lag Error PSD')
    plt.colorbar(im, ax=axes[0, 2])
    
    # Total PSD
    im = axes[1, 0].imshow(np.fft.fftshift(psd_total) + 1e-20, norm=LogNorm(),
                           extent=extent, cmap='viridis')
    axes[1, 0].set_title('Total Residual PSD')
    plt.colorbar(im, ax=axes[1, 0])
    
    # PSF
    center = n_pix // 2
    zoom = 40
    psf_zoom = psf[center-zoom:center+zoom, center-zoom:center+zoom]
    im = axes[1, 1].imshow(np.log10(psf_zoom + 1e-8), vmin=-6, vmax=0, cmap='inferno')
    axes[1, 1].set_title('log₁₀(PSF)')
    plt.colorbar(im, ax=axes[1, 1])
    
    # Error budget bar chart
    errors = model.get_error_breakdown()
    labels = ['Fitting', 'Aliasing', 'Servo-lag', 'Aniso', 'Noise']
    values = [errors['fitting'], errors['aliasing'], errors['servo_lag'],
              errors['anisoplanatism'], errors['noise']]
    axes[1, 2].bar(labels, values)
    axes[1, 2].set_ylabel('Variance (rad²)')
    axes[1, 2].set_title(f'Error Budget\nTotal: {errors["total"]:.3f} rad², Strehl: {model.compute_strehl_ratio(pupil):.3f}')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('jolissaint_diagnostics.png', dpi=150)
    print_pass('Saved jolissaint_diagnostics.png')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test Jolissaint AO model')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--plot', action='store_true', help='Generate diagnostic plots')
    args = parser.parse_args()
    
    print('=' * 60)
    print('JOLISSAINT AO MODEL TEST SUITE')
    print('=' * 60)
    
    if HAS_CUPY:
        print(f'CuPy: {cp.__version__}')
    else:
        print('CuPy: Not available')
    print(f'NumPy: {np.__version__}')
    
    n_pix = 128 if args.quick else 256
    all_passed = True
    
    # Run tests
    passed, _ = test_basic_psf(n_pix)
    all_passed &= passed
    
    all_passed &= test_error_budget(n_pix)
    all_passed &= test_actuator_density(n_pix)
    all_passed &= test_sampling_frequency(n_pix)
    all_passed &= test_mask_geometry(n_pix)
    all_passed &= test_gpu_correctness(n_pix)
    
    if args.plot:
        generate_plots(n_pix)
    
    # Summary
    print_header('TEST SUMMARY')
    if all_passed:
        print('  All tests PASSED ✓')
    else:
        print('  Some tests FAILED ✗')
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

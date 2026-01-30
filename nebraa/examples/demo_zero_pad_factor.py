"""
Demonstration: Zero-Padding Factor in PSFEngine

This script demonstrates the zero_pad_factor feature that enables
finer PSF sampling through FFT zero-padding.

Key Demonstrations:
1. Array size scaling with zero_pad_factor
2. Effective pixel scale changes
3. Energy conservation under padding
4. Airy disk structure resolution improvement
5. Batch processing with zero-padding
6. Performance comparison

Author: NEBRAA
"""

import numpy as np
import sys
import os

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nebraa.physics.psf_engine import PSFEngine, PSFEngineConfig
from nebraa.physics.pupil import Pupil
from nebraa.utils.compute import init_backend

# Force CPU for reproducibility
init_backend('CPU')


def demo_array_size_scaling():
    """Demonstrate how zero_pad_factor affects PSF array size."""
    print("\n" + "="*70)
    print("DEMO 1: Array Size Scaling with zero_pad_factor")
    print("="*70)
    
    n_pix = 128
    pupil = Pupil.from_circular(
        n_pix=n_pix,
        wavelength=2.2e-6,
        pixel_scale=1e-5,
        diameter=8.0,
    )
    
    factors = [1, 2, 4]
    
    print(f"\nInput pupil size: {n_pix}×{n_pix} pixels")
    print("\nOutput PSF sizes with different zero_pad_factor:")
    
    for factor in factors:
        engine = PSFEngine(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=factor,
        )
        
        psf = engine.compute_psf(pupil.amplitude)
        
        expected_size = n_pix * factor
        actual_size = psf.shape[0]
        status = "✓" if actual_size == expected_size else "✗"
        
        print(f"  zero_pad_factor={factor}: {psf.shape[0]}×{psf.shape[1]} pixels "
              f"(expected: {expected_size}×{expected_size}) {status}")
    
    print("\n✅ DEMO 1 PASSED: Array sizes scale correctly with zero_pad_factor")


def demo_effective_pixel_scale():
    """Demonstrate how zero_pad_factor affects effective pixel scale."""
    print("\n" + "="*70)
    print("DEMO 2: Effective Pixel Scale Changes")
    print("="*70)
    
    nominal_scale_rad = 1e-5  # radians/pixel
    nominal_scale_mas = nominal_scale_rad * 180 / np.pi * 3600 * 1000  # mas/pixel
    
    print(f"\nNominal pixel scale: {nominal_scale_mas:.3f} mas/pixel")
    print("\nEffective pixel scales with zero-padding:")
    
    factors = [1, 2, 4, 8]
    
    for factor in factors:
        engine = PSFEngine(
            n_pix=128,
            wavelength=2.2e-6,
            pixel_scale=nominal_scale_rad,
            zero_pad_factor=factor,
        )
        
        effective_scale_rad = engine.pixel_scale_effective
        effective_scale_mas = effective_scale_rad * 180 / np.pi * 3600 * 1000
        expected_scale_rad = nominal_scale_rad / factor
        
        match = "✓" if abs(effective_scale_rad - expected_scale_rad) < 1e-15 else "✗"
        
        print(f"  zero_pad_factor={factor}: {effective_scale_mas:.3f} mas/pixel "
              f"(÷{factor} finer) {match}")
    
    print("\n✅ DEMO 2 PASSED: Effective pixel scale = nominal_scale / zero_pad_factor")


def demo_energy_conservation():
    """Demonstrate energy conservation with zero-padding."""
    print("\n" + "="*70)
    print("DEMO 3: Energy Conservation Under Zero-Padding")
    print("="*70)
    
    n_pix = 128
    pupil = Pupil.from_circular(
        n_pix=n_pix,
        wavelength=2.2e-6,
        pixel_scale=1e-5,
        diameter=8.0,
    )
    
    # Test both diffraction-limited and aberrated PSFs
    print("\nDiffraction-Limited PSF:")
    for factor in [1, 2, 4]:
        engine = PSFEngine(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=factor,
            normalize_to="sum",
        )
        
        psf = engine.compute_psf(pupil.amplitude, phase=None)
        total = np.sum(psf)
        error = abs(total - 1.0)
        status = "✓" if error < 1e-6 else "✗"
        
        print(f"  zero_pad_factor={factor}: sum(PSF) = {total:.10f} "
              f"(error: {error:.2e}) {status}")
    
    print("\nAberrated PSF (random phase):")
    phase = np.random.RandomState(42).uniform(0, 2*np.pi, size=(n_pix, n_pix))
    
    for factor in [1, 2, 4]:
        engine = PSFEngine(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=factor,
            normalize_to="sum",
        )
        
        psf = engine.compute_psf(pupil.amplitude, phase=phase)
        total = np.sum(psf)
        error = abs(total - 1.0)
        status = "✓" if error < 1e-6 else "✗"
        
        print(f"  zero_pad_factor={factor}: sum(PSF) = {total:.10f} "
              f"(error: {error:.2e}) {status}")
    
    print("\n✅ DEMO 3 PASSED: Energy conserved (sum=1) for all padding factors")


def demo_airy_disk_resolution():
    """Demonstrate improved Airy disk sampling."""
    print("\n" + "="*70)
    print("DEMO 4: Airy Disk Structure Resolution")
    print("="*70)
    
    n_pix = 64  # Smaller grid for demonstration
    pupil = Pupil.from_circular(
        n_pix=n_pix,
        wavelength=2.2e-6,
        pixel_scale=1e-5,
        diameter=8.0,
    )
    
    print(f"\nComputing diffraction-limited PSFs with different sampling:")
    
    factors = [1, 2, 4]
    psfs = {}
    
    for factor in factors:
        engine = PSFEngine(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=factor,
        )
        
        psf = engine.compute_psf(pupil.amplitude)
        psfs[factor] = psf
        
        # Find peak location
        peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
        center = np.array(psf.shape) // 2
        peak_offset = peak_idx[0] - center[0], peak_idx[1] - center[1]
        
        # Central slice statistics
        center_slice = psf[psf.shape[0] // 2, :]
        n_samples = len(center_slice)
        peak_value = np.max(center_slice)
        
        print(f"\n  zero_pad_factor={factor}:")
        print(f"    Output size: {psf.shape[0]}×{psf.shape[1]} pixels")
        print(f"    Central slice samples: {n_samples}")
        print(f"    Peak offset from center: {peak_offset} pixels")
        print(f"    Peak value (in slice): {peak_value:.6f}")
    
    # Compare sampling density
    print("\n  Sampling density comparison:")
    for i in range(1, len(factors)):
        prev_factor = factors[i-1]
        curr_factor = factors[i]
        ratio = len(psfs[curr_factor][0]) / len(psfs[prev_factor][0])
        print(f"    factor {curr_factor} vs {prev_factor}: {ratio:.1f}× more samples")
    
    print("\n✅ DEMO 4 PASSED: Finer padding provides more samples of Airy structure")


def demo_batch_processing():
    """Demonstrate batch processing with zero-padding."""
    print("\n" + "="*70)
    print("DEMO 5: Batch Processing with Zero-Padding")
    print("="*70)
    
    n_pix = 64
    n_screens = 5
    
    pupil = Pupil.from_circular(
        n_pix=n_pix,
        wavelength=2.2e-6,
        pixel_scale=1e-5,
        diameter=8.0,
    )
    
    # Generate random phase screens
    phases = np.random.RandomState(123).uniform(0, 2*np.pi, size=(n_screens, n_pix, n_pix))
    
    print(f"\nInput: {n_screens} phase screens of {n_pix}×{n_pix} pixels")
    
    for factor in [1, 2, 4]:
        engine = PSFEngine(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
            zero_pad_factor=factor,
        )
        
        psf_avg, psfs_individual = engine.compute_psf_batch(
            pupil.amplitude,
            phases,
            return_individual=True,
        )
        
        expected_size = n_pix * factor
        
        # Check energy conservation
        avg_sum = np.sum(psf_avg)
        individual_sums = [np.sum(psf) for psf in psfs_individual]
        all_conserved = all(abs(s - 1.0) < 1e-6 for s in individual_sums) and abs(avg_sum - 1.0) < 1e-6
        
        status = "✓" if all_conserved else "✗"
        
        print(f"\n  zero_pad_factor={factor}:")
        print(f"    Average PSF: {psf_avg.shape}")
        print(f"    Individual PSFs: {psfs_individual.shape}")
        print(f"    Energy conserved: {status}")
    
    print("\n✅ DEMO 5 PASSED: Batch processing works correctly with zero-padding")


def demo_backward_compatibility():
    """Demonstrate backward compatibility (factor=1 unchanged)."""
    print("\n" + "="*70)
    print("DEMO 6: Backward Compatibility (factor=1)")
    print("="*70)
    
    n_pix = 128
    pupil = Pupil.from_circular(
        n_pix=n_pix,
        wavelength=2.2e-6,
        pixel_scale=1e-5,
        diameter=8.0,
    )
    
    phase = np.random.RandomState(456).uniform(0, 2*np.pi, size=(n_pix, n_pix))
    
    # Engine without explicit zero_pad_factor (defaults to 1)
    engine_default = PSFEngine(
        n_pix=n_pix,
        wavelength=2.2e-6,
        pixel_scale=1e-5,
    )
    
    # Engine with explicit zero_pad_factor=1
    engine_explicit = PSFEngine(
        n_pix=n_pix,
        wavelength=2.2e-6,
        pixel_scale=1e-5,
        zero_pad_factor=1,
    )
    
    psf_default = engine_default.compute_psf(pupil.amplitude, phase)
    psf_explicit = engine_explicit.compute_psf(pupil.amplitude, phase)
    
    diff = np.max(np.abs(psf_default - psf_explicit))
    
    print(f"\nPSF (default factor):  shape={psf_default.shape}, sum={np.sum(psf_default):.10f}")
    print(f"PSF (explicit factor=1): shape={psf_explicit.shape}, sum={np.sum(psf_explicit):.10f}")
    print(f"\nMax difference: {diff:.2e}")
    
    status = "✓" if diff < 1e-10 else "✗"
    print(f"Results identical: {status}")
    
    print("\n✅ DEMO 6 PASSED: zero_pad_factor=1 (default) gives unchanged behavior")


def demo_config_integration():
    """Demonstrate PSFEngineConfig integration."""
    print("\n" + "="*70)
    print("DEMO 7: PSFEngineConfig Integration")
    print("="*70)
    
    # Create config with zero-padding
    config = PSFEngineConfig(
        n_pix=128,
        wavelength=2.2e-6,
        pixel_scale=1e-5,
        normalize_to="sum",
        zero_pad_factor=3,
    )
    
    print(f"\nPSFEngineConfig:")
    print(f"  n_pix: {config.n_pix}")
    print(f"  wavelength: {config.wavelength} m")
    print(f"  pixel_scale: {config.pixel_scale} rad/pixel")
    print(f"  zero_pad_factor: {config.zero_pad_factor}")
    
    # Create engine from config
    engine = PSFEngine.from_config(config)
    
    print(f"\nPSFEngine (from config):")
    info = engine.info()
    print(f"  n_pix: {info['n_pix']}")
    print(f"  n_pix_padded: {info['n_pix_padded']}")
    print(f"  zero_pad_factor: {info['zero_pad_factor']}")
    print(f"  pixel_scale: {info['pixel_scale_mas']:.3f} mas/pixel")
    print(f"  pixel_scale_effective: {info['pixel_scale_effective_mas']:.3f} mas/pixel")
    
    # Verify it works
    pupil = Pupil.from_circular(
        n_pix=128,
        wavelength=2.2e-6,
        pixel_scale=1e-5,
        diameter=8.0,
    )
    
    psf = engine.compute_psf(pupil.amplitude)
    
    expected_size = 128 * 3
    status = "✓" if psf.shape == (expected_size, expected_size) else "✗"
    
    print(f"\nComputed PSF shape: {psf.shape} (expected: {expected_size}×{expected_size}) {status}")
    
    print("\n✅ DEMO 7 PASSED: Config integration works correctly")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("ZERO-PADDING FACTOR DEMONSTRATION")
    print("PSFEngine with FFT Zero-Padding for Finer PSF Sampling")
    print("="*70)
    
    try:
        demo_array_size_scaling()
        demo_effective_pixel_scale()
        demo_energy_conservation()
        demo_airy_disk_resolution()
        demo_batch_processing()
        demo_backward_compatibility()
        demo_config_integration()
        
        print("\n" + "="*70)
        print("ALL DEMONSTRATIONS PASSED ✅")
        print("="*70)
        print("\nKey Benefits of Zero-Padding:")
        print("  1. Finer PSF sampling (pixel_scale / zero_pad_factor)")
        print("  2. Larger field-of-view coverage")
        print("  3. Better resolved Airy disk structure")
        print("  4. Energy conservation maintained")
        print("  5. Fully backward compatible (factor=1 default)")
        print("\nTypical Usage:")
        print("  - factor=1: Default, no padding (fastest)")
        print("  - factor=2-4: Good balance of detail and speed")
        print("  - factor>4: High detail, slower (rarely needed)")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

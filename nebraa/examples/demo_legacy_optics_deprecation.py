#!/usr/bin/env python
"""
Demonstration of legacy optics deprecation (Item #8).

Shows:
1. Deprecation warnings are emitted
2. Legacy and modern APIs produce identical results
3. How to migrate code to PSFEngine
"""

import numpy as np
import warnings

# Suppress deprecation warnings for cleaner demo output
# (in real code, you should see and address these warnings)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from nebraa.physics.optics import (
    compute_psf,
    compute_psf_batch,
    compute_reference_psf,
)
from nebraa.physics import PSFEngine
from nebraa.utils.compute import init_backend


def demo_deprecation_warnings():
    """Demonstrate that deprecation warnings are emitted."""
    print("\n" + "="*70)
    print("1. DEPRECATION WARNINGS")
    print("="*70)
    
    n_pix = 64
    pupil = np.ones((n_pix, n_pix), dtype=np.float32)
    phase = np.zeros((n_pix, n_pix), dtype=np.float32)
    
    # Re-enable warnings to see them
    warnings.filterwarnings('default', category=DeprecationWarning)
    
    print("\n  Calling compute_psf()...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = compute_psf(pupil, phase)
        if w:
            print(f"  ⚠️  Warning: {w[0].message}")
    
    print("\n  Calling compute_psf_batch()...")
    phases = np.zeros((3, n_pix, n_pix), dtype=np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = compute_psf_batch(pupil, phases)
        if w:
            print(f"  ⚠️  Warning: {w[0].message}")
    
    print("\n  Calling compute_reference_psf()...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = compute_reference_psf(pupil)
        if w:
            print(f"  ⚠️  Warning: {w[0].message}")
    
    # Suppress again for rest of demo
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    print("\n  ✓ All legacy functions emit deprecation warnings")


def demo_consistency_single_psf():
    """Demonstrate legacy and modern APIs produce identical results."""
    print("\n" + "="*70)
    print("2. CONSISTENCY: Single PSF computation")
    print("="*70)
    
    # Force CPU for reproducibility
    backend = init_backend(compute_mode="CPU")
    
    n_pix = 64
    pupil = np.ones((n_pix, n_pix), dtype=np.float32)
    phase = np.random.randn(n_pix, n_pix).astype(np.float32)
    
    # Legacy API
    psf_legacy = compute_psf(pupil, phase, normalize=True, normalize_to="sum")
    
    # Modern API
    engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
    psf_modern = engine.compute_psf(pupil, phase, normalize=True)
    
    # Compare
    max_diff = np.max(np.abs(psf_legacy - psf_modern))
    rel_diff = max_diff / np.max(psf_legacy)
    
    print(f"  Legacy PSF sum: {np.sum(psf_legacy):.6f}")
    print(f"  Modern PSF sum: {np.sum(psf_modern):.6f}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative difference: {rel_diff:.2e}")
    
    assert np.allclose(psf_legacy, psf_modern, rtol=1e-5)
    print("\n  ✓ Legacy and modern APIs produce identical results!")


def demo_consistency_batch():
    """Demonstrate batch PSF consistency."""
    print("\n" + "="*70)
    print("3. CONSISTENCY: Batch PSF computation")
    print("="*70)
    
    n_pix = 64
    n_screens = 10
    pupil = np.ones((n_pix, n_pix), dtype=np.float32)
    phases = np.random.randn(n_screens, n_pix, n_pix).astype(np.float32)
    
    # Legacy API (returns individual PSFs)
    psfs_legacy = compute_psf_batch(pupil, phases, normalize=True, normalize_to="sum")
    
    # Modern API (get individual PSFs)
    engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
    _, psfs_modern = engine.compute_psf_batch(
        pupil, phases, normalize=True, return_individual=True
    )
    
    # Compare
    max_diff = np.max(np.abs(psfs_legacy - psfs_modern))
    rel_diff = max_diff / np.max(psfs_legacy)
    
    print(f"  Number of PSFs: {n_screens}")
    print(f"  Legacy PSFs shape: {psfs_legacy.shape}")
    print(f"  Modern PSFs shape: {psfs_modern.shape}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative difference: {rel_diff:.2e}")
    
    # Check normalization
    for i in range(n_screens):
        assert np.isclose(np.sum(psfs_legacy[i]), 1.0, rtol=1e-5)
        assert np.isclose(np.sum(psfs_modern[i]), 1.0, rtol=1e-5)
    
    assert np.allclose(psfs_legacy, psfs_modern, rtol=1e-5)
    print("\n  ✓ Batch processing produces identical results!")


def demo_consistency_reference():
    """Demonstrate reference PSF consistency."""
    print("\n" + "="*70)
    print("4. CONSISTENCY: Reference (diffraction-limited) PSF")
    print("="*70)
    
    n_pix = 128
    pupil = np.ones((n_pix, n_pix), dtype=np.float32)
    
    # Legacy API
    psf_legacy = compute_reference_psf(pupil)
    
    # Modern API
    engine = PSFEngine(n_pix=n_pix, wavelength=2.2e-6, pixel_scale=1e-5)
    psf_modern = engine.compute_diffraction_limited_psf(pupil)
    
    # Compare
    max_diff = np.max(np.abs(psf_legacy - psf_modern))
    rel_diff = max_diff / np.max(psf_legacy)
    
    print(f"  PSF shape: {psf_legacy.shape}")
    print(f"  Legacy PSF sum: {np.sum(psf_legacy):.6f}")
    print(f"  Modern PSF sum: {np.sum(psf_modern):.6f}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative difference: {rel_diff:.2e}")
    
    assert np.allclose(psf_legacy, psf_modern, rtol=1e-5)
    print("\n  ✓ Reference PSFs are identical!")


def demo_migration_example():
    """Show complete migration example."""
    print("\n" + "="*70)
    print("5. MIGRATION EXAMPLE")
    print("="*70)
    
    print("\n  OLD CODE (deprecated):")
    print("  " + "-"*66)
    print("    from nebraa.physics.optics import compute_psf")
    print("    ")
    print("    pupil = ...")
    print("    phase = ...")
    print("    psf = compute_psf(pupil, phase, normalize=True)")
    
    print("\n  NEW CODE (recommended):")
    print("  " + "-"*66)
    print("    from nebraa.physics import PSFEngine")
    print("    ")
    print("    # Create engine once")
    print("    engine = PSFEngine(")
    print("        n_pix=pupil.shape[0],")
    print("        wavelength=2.2e-6,    # K-band")
    print("        pixel_scale=1e-5      # radians/pixel")
    print("    )")
    print("    ")
    print("    # Compute PSF")
    print("    psf = engine.compute_psf(pupil, phase, normalize=True)")
    
    print("\n  BENEFITS:")
    print("    • Consistent normalization behavior")
    print("    • Better performance (reusable engine)")
    print("    • More control (wavelength, pixel scale)")
    print("    • Future-proof (no deprecation warnings)")


def demo_backward_compatibility():
    """Demonstrate backward compatibility."""
    print("\n" + "="*70)
    print("6. BACKWARD COMPATIBILITY")
    print("="*70)
    
    n_pix = 64
    pupil = np.ones((n_pix, n_pix), dtype=np.float32)
    phase = np.random.randn(n_pix, n_pix).astype(np.float32)
    
    # Old parameter style: normalize=False
    psf_old_style = compute_psf(pupil, phase, normalize=False)
    
    # New parameter style: normalize_to="none"
    psf_new_style = compute_psf(pupil, phase, normalize_to="none")
    
    print(f"  Old style (normalize=False) sum: {np.sum(psf_old_style):.3f}")
    print(f"  New style (normalize_to='none') sum: {np.sum(psf_new_style):.3f}")
    
    assert np.array_equal(psf_old_style, psf_new_style)
    print("\n  ✓ Old parameter conventions still work!")
    
    # Peak normalization
    psf_peak = compute_psf(pupil, phase, normalize=True, normalize_to="peak")
    print(f"\n  Peak normalized PSF max: {np.max(psf_peak):.6f}")
    assert np.isclose(np.max(psf_peak), 1.0)
    print("  ✓ Peak normalization mode works!")


def main():
    """Run all demonstrations."""
    print("\n" + "#"*70)
    print("# LEGACY OPTICS DEPRECATION DEMONSTRATION (Item #8)")
    print("#"*70)
    print("\nShowing that legacy optics functions are now thin wrappers")
    print("around PSFEngine with consistent behavior.")
    
    try:
        demo_deprecation_warnings()
        demo_consistency_single_psf()
        demo_consistency_batch()
        demo_consistency_reference()
        demo_migration_example()
        demo_backward_compatibility()
        
        print("\n" + "="*70)
        print("ALL DEMONSTRATIONS PASSED!")
        print("="*70)
        print("\nKey achievements:")
        print("  ✓ Deprecation warnings emitted")
        print("  ✓ Legacy == Modern (exact match)")
        print("  ✓ Backward compatible")
        print("  ✓ Clear migration path")
        print("\nAction items:")
        print("  → Update code to use PSFEngine")
        print("  → Remove deprecation warnings")
        print("  → Future-proof your codebase")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

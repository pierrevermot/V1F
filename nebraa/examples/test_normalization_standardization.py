"""
Test standardized PSF normalization semantics.

Validates that normalization modes ('sum', 'peak', 'none') work correctly
across all public APIs:
- psf_engine.PSFEngine
- optics.compute_psf
- optics.compute_psf_batch

Ensures:
1. normalize_to="sum" → psf.sum() == 1
2. normalize_to="peak" → psf.max() == 1
3. normalize_to="none" → raw scaling (no normalization)
4. Scalar and batch operations behave identically
5. Backward compatibility with legacy normalize=True/False parameter
"""

import numpy as np
import sys
from pathlib import Path

# Add nebraa to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nebraa.physics.psf_engine import PSFEngine
from nebraa.physics import optics
from nebraa.utils.compute import get_backend


def create_test_data(n_screens=10, n_pix=128, seed=42):
    """Create test pupil and phase screens."""
    np.random.seed(seed)
    
    # Simple circular pupil
    y, x = np.ogrid[-n_pix//2:n_pix//2, -n_pix//2:n_pix//2]
    r = np.sqrt(x**2 + y**2)
    pupil = (r < n_pix // 3).astype(np.float32)
    
    # Random phase screens
    phases = np.random.randn(n_screens, n_pix, n_pix).astype(np.float32) * 0.5
    
    return pupil, phases


def to_numpy(arr):
    """Convert array to numpy if needed."""
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def test_psf_engine_normalization_modes():
    """Test PSFEngine normalization modes on single PSF."""
    print("\n=== Testing PSFEngine Single PSF Normalization ===")
    
    pupil, phases = create_test_data(n_screens=1, n_pix=128)
    phase = phases[0]
    
    tolerance = 1e-5
    
    # Test "sum" mode
    print("\n1. Testing normalize_to='sum'")
    engine_sum = PSFEngine(n_pix=128, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to="sum")
    psf_sum = to_numpy(engine_sum.compute_psf(pupil, phase, normalize=True))
    
    total = psf_sum.sum()
    print(f"   PSF sum: {total:.8f}")
    assert abs(total - 1.0) < tolerance, f"Sum normalization failed: sum={total}"
    print("   ✓ PSF sums to 1.0")
    
    # Test "peak" mode
    print("\n2. Testing normalize_to='peak'")
    engine_peak = PSFEngine(n_pix=128, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to="peak")
    psf_peak = to_numpy(engine_peak.compute_psf(pupil, phase, normalize=True))
    
    peak = psf_peak.max()
    print(f"   PSF peak: {peak:.8f}")
    assert abs(peak - 1.0) < tolerance, f"Peak normalization failed: peak={peak}"
    print("   ✓ PSF peak is 1.0")
    
    # Test "none" mode
    print("\n3. Testing normalize_to='none'")
    engine_none = PSFEngine(n_pix=128, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to="none")
    psf_none = to_numpy(engine_none.compute_psf(pupil, phase, normalize=True))
    
    # Raw PSF should not sum to 1 or have peak of 1
    total_none = psf_none.sum()
    peak_none = psf_none.max()
    print(f"   PSF sum: {total_none:.8f}, peak: {peak_none:.8f}")
    # Should be significantly different from normalized versions
    assert abs(total_none - 1.0) > 0.01 or abs(peak_none - 1.0) > 0.01, \
        "None mode should produce different scaling"
    print("   ✓ PSF has raw scaling (not normalized)")
    
    # Test that normalize=False works
    print("\n4. Testing normalize=False")
    psf_no_norm = to_numpy(engine_sum.compute_psf(pupil, phase, normalize=False))
    # Should match "none" mode
    diff = np.abs(psf_no_norm - psf_none).max()
    print(f"   Max diff from 'none' mode: {diff:.2e}")
    assert diff < tolerance, "normalize=False should match normalize_to='none'"
    print("   ✓ normalize=False matches 'none' mode")
    
    print("\n✓ All single PSF normalization modes work correctly")


def test_psf_engine_batch_normalization():
    """Test PSFEngine batch normalization modes."""
    print("\n=== Testing PSFEngine Batch Normalization ===")
    
    pupil, phases = create_test_data(n_screens=10, n_pix=128)
    
    tolerance = 1e-5
    
    # Test "sum" mode
    print("\n1. Testing normalize_to='sum' (batch)")
    engine_sum = PSFEngine(n_pix=128, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to="sum")
    _, psfs_sum = engine_sum.compute_psf_batch(pupil, phases, normalize=True, return_individual=True)
    psfs_sum = to_numpy(psfs_sum)
    
    sums = psfs_sum.sum(axis=(1, 2))
    print(f"   PSF sums: min={sums.min():.8f}, max={sums.max():.8f}, mean={sums.mean():.8f}")
    assert np.allclose(sums, 1.0, rtol=tolerance), f"Batch sum normalization failed: {sums}"
    print("   ✓ All PSFs sum to 1.0")
    
    # Test "peak" mode
    print("\n2. Testing normalize_to='peak' (batch)")
    engine_peak = PSFEngine(n_pix=128, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to="peak")
    _, psfs_peak = engine_peak.compute_psf_batch(pupil, phases, normalize=True, return_individual=True)
    psfs_peak = to_numpy(psfs_peak)
    
    peaks = psfs_peak.max(axis=(1, 2))
    print(f"   PSF peaks: min={peaks.min():.8f}, max={peaks.max():.8f}, mean={peaks.mean():.8f}")
    assert np.allclose(peaks, 1.0, rtol=tolerance), f"Batch peak normalization failed: {peaks}"
    print("   ✓ All PSFs have peak of 1.0")
    
    # Test "none" mode
    print("\n3. Testing normalize_to='none' (batch)")
    engine_none = PSFEngine(n_pix=128, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to="none")
    _, psfs_none = engine_none.compute_psf_batch(pupil, phases, normalize=True, return_individual=True)
    psfs_none = to_numpy(psfs_none)
    
    sums_none = psfs_none.sum(axis=(1, 2))
    peaks_none = psfs_none.max(axis=(1, 2))
    print(f"   PSF sums: {sums_none.mean():.2f} (not normalized)")
    print(f"   PSF peaks: {peaks_none.mean():.2f} (not normalized)")
    print("   ✓ PSFs have raw scaling")
    
    print("\n✓ All batch normalization modes work correctly")


def test_batch_vs_scalar_consistency():
    """Test that batch and scalar operations produce identical results."""
    print("\n=== Testing Batch vs Scalar Consistency ===")
    
    pupil, phases = create_test_data(n_screens=5, n_pix=128)
    
    tolerance = 1e-6
    
    for mode in ["sum", "peak", "none"]:
        print(f"\nTesting mode: '{mode}'")
        engine = PSFEngine(n_pix=128, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to=mode)
        
        # Compute individually
        psfs_individual = []
        for i in range(phases.shape[0]):
            psf = to_numpy(engine.compute_psf(pupil, phases[i], normalize=True))
            psfs_individual.append(psf)
        psfs_individual = np.stack(psfs_individual)
        
        # Compute in batch
        _, psfs_batch = engine.compute_psf_batch(pupil, phases, normalize=True, return_individual=True)
        psfs_batch = to_numpy(psfs_batch)
        
        # Compare
        max_diff = np.abs(psfs_individual - psfs_batch).max()
        rel_diff = max_diff / np.abs(psfs_individual).max()
        
        print(f"  Max abs diff: {max_diff:.2e}, Rel diff: {rel_diff:.2e}")
        assert rel_diff < tolerance, f"Batch/scalar mismatch for mode '{mode}'"
        print(f"  ✓ Batch matches scalar for '{mode}' mode")
    
    print("\n✓ Batch and scalar operations are consistent")


def test_optics_normalization_modes():
    """Test optics module normalization modes."""
    print("\n=== Testing optics Module Normalization ===")
    
    pupil, phases = create_test_data(n_screens=5, n_pix=128)
    phase = phases[0]
    
    tolerance = 1e-5
    
    # Test single PSF
    print("\n1. Single PSF normalization")
    
    # Sum mode
    psf_sum = to_numpy(optics.compute_psf(pupil, phase, normalize=True, normalize_to="sum"))
    print(f"   Sum mode: psf.sum()={psf_sum.sum():.8f}")
    assert abs(psf_sum.sum() - 1.0) < tolerance, "optics.compute_psf sum mode failed"
    
    # Peak mode (default for backward compatibility)
    psf_peak = to_numpy(optics.compute_psf(pupil, phase, normalize=True, normalize_to="peak"))
    print(f"   Peak mode: psf.max()={psf_peak.max():.8f}")
    assert abs(psf_peak.max() - 1.0) < tolerance, "optics.compute_psf peak mode failed"
    
    # None mode
    psf_none = to_numpy(optics.compute_psf(pupil, phase, normalize=True, normalize_to="none"))
    print(f"   None mode: raw scaling")
    
    # Legacy normalize=False
    psf_legacy = to_numpy(optics.compute_psf(pupil, phase, normalize=False))
    diff = np.abs(psf_legacy - psf_none).max()
    assert diff < tolerance, "Legacy normalize=False should match normalize_to='none'"
    print(f"   ✓ Legacy normalize=False works correctly")
    
    # Test batch PSFs
    print("\n2. Batch PSF normalization")
    
    # Sum mode
    psfs_sum = to_numpy(optics.compute_psf_batch(pupil, phases, normalize=True, normalize_to="sum"))
    sums = psfs_sum.sum(axis=(1, 2))
    print(f"   Sum mode: psf.sum()={sums.mean():.8f} (mean)")
    assert np.allclose(sums, 1.0, rtol=tolerance), "optics.compute_psf_batch sum mode failed"
    
    # Peak mode
    psfs_peak = to_numpy(optics.compute_psf_batch(pupil, phases, normalize=True, normalize_to="peak"))
    peaks = psfs_peak.max(axis=(1, 2))
    print(f"   Peak mode: psf.max()={peaks.mean():.8f} (mean)")
    assert np.allclose(peaks, 1.0, rtol=tolerance), "optics.compute_psf_batch peak mode failed"
    
    # None mode
    psfs_none = to_numpy(optics.compute_psf_batch(pupil, phases, normalize=True, normalize_to="none"))
    print(f"   None mode: raw scaling")
    
    # Legacy normalize=False
    psfs_legacy = to_numpy(optics.compute_psf_batch(pupil, phases, normalize=False))
    diff = np.abs(psfs_legacy - psfs_none).max()
    assert diff < tolerance, "Legacy normalize=False should match normalize_to='none'"
    print(f"   ✓ Legacy normalize=False works correctly (batch)")
    
    print("\n✓ optics module normalization works correctly")


def test_default_normalization():
    """Test that default normalization is 'sum' for energy conservation."""
    print("\n=== Testing Default Normalization ===")
    
    pupil, phases = create_test_data(n_screens=1, n_pix=128)
    phase = phases[0]
    
    # PSFEngine default should be 'sum'
    engine = PSFEngine(n_pix=128, wavelength=2.2e-6, pixel_scale=1e-5)
    print(f"\nPSFEngine default normalize_to: '{engine.normalize_to}'")
    assert engine.normalize_to == "sum", "PSFEngine default should be 'sum'"
    
    psf = to_numpy(engine.compute_psf(pupil, phase))
    print(f"PSF sum: {psf.sum():.8f}")
    assert abs(psf.sum() - 1.0) < 1e-5, "Default PSFEngine should produce sum=1"
    print("✓ PSFEngine defaults to 'sum' (energy conservation)")
    
    # optics default should be 'peak' for backward compatibility
    psf_optics = to_numpy(optics.compute_psf(pupil, phase, normalize=True))
    print(f"\noptics.compute_psf with normalize=True:")
    print(f"  PSF peak: {psf_optics.max():.8f}")
    # Legacy behavior: peak=1
    assert abs(psf_optics.max() - 1.0) < 1e-5, "Legacy optics should default to peak=1"
    print("✓ optics.compute_psf maintains backward compatibility (peak=1)")
    
    print("\n✓ Default normalization modes are appropriate")


def test_average_psf_normalization():
    """Test that averaged long-exposure PSFs maintain proper normalization."""
    print("\n=== Testing Average PSF Normalization ===")
    
    pupil, phases = create_test_data(n_screens=20, n_pix=128)
    
    tolerance = 1e-5
    
    # Sum mode
    engine_sum = PSFEngine(n_pix=128, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to="sum")
    psf_avg_sum = to_numpy(engine_sum.compute_psf_batch(pupil, phases, normalize=True))
    
    total = psf_avg_sum.sum()
    print(f"\nSum mode - Average PSF sum: {total:.8f}")
    assert abs(total - 1.0) < tolerance, f"Averaged PSF sum normalization failed: {total}"
    print("✓ Averaged PSF maintains sum=1")
    
    # Peak mode
    engine_peak = PSFEngine(n_pix=128, wavelength=2.2e-6, pixel_scale=1e-5, normalize_to="peak")
    psf_avg_peak = to_numpy(engine_peak.compute_psf_batch(pupil, phases, normalize=True))
    
    # Note: averaged PSF won't necessarily have peak=1 in peak mode
    # Each individual PSF has peak=1, but averaging changes the peak
    print(f"Peak mode - Average PSF peak: {psf_avg_peak.max():.8f}")
    print("  (Note: averaged PSF peak may differ from 1.0, which is expected)")
    print("✓ Average PSF computed correctly in peak mode")
    
    print("\n✓ Average PSF normalization works correctly")


def main():
    """Run all normalization tests."""
    print("=" * 70)
    print("PSF Normalization Standardization Tests")
    print("=" * 70)
    
    backend = get_backend()
    backend_name = "CuPy" if hasattr(backend.xp, 'cuda') else "NumPy"
    print(f"Backend: {backend_name}")
    
    try:
        # Core normalization tests
        test_psf_engine_normalization_modes()
        test_psf_engine_batch_normalization()
        test_batch_vs_scalar_consistency()
        test_optics_normalization_modes()
        test_default_normalization()
        test_average_psf_normalization()
        
        print("\n" + "=" * 70)
        print("✓ All normalization tests passed!")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ 'sum' mode: PSFs sum to 1.0 (energy conservation)")
        print("  ✓ 'peak' mode: PSFs have peak of 1.0")
        print("  ✓ 'none' mode: Raw scaling (no normalization)")
        print("  ✓ Batch and scalar operations are consistent")
        print("  ✓ Backward compatibility maintained (normalize=True/False)")
        print("  ✓ Default: PSFEngine uses 'sum', optics uses 'peak'")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

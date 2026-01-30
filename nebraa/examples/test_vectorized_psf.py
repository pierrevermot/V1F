"""
Test vectorized PSF batch computation.

Verifies that the vectorized implementation:
1. Produces identical results to the loop-based version
2. Works on both NumPy and CuPy backends
3. Handles normalization correctly (sum and peak modes)
"""

import numpy as np
import time
import sys
import os
from pathlib import Path

# Add nebraa to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Allow forcing CPU mode via command line or environment
if len(sys.argv) > 1 and sys.argv[1] == "--cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("Forcing CPU mode")

from nebraa.physics.psf_engine import PSFEngine
from nebraa.physics import optics
from nebraa.utils.compute import get_backend


def create_test_data(n_screens=100, n_pix=256, seed=42):
    """Create test pupil and phase screens."""
    np.random.seed(seed)
    
    # Simple circular pupil
    y, x = np.ogrid[-n_pix//2:n_pix//2, -n_pix//2:n_pix//2]
    r = np.sqrt(x**2 + y**2)
    pupil = (r < n_pix // 3).astype(np.float32)
    
    # Random phase screens
    phases = np.random.randn(n_screens, n_pix, n_pix).astype(np.float32) * 0.5
    
    return pupil, phases


def test_psf_engine_batch_consistency():
    """Test that batch computation matches single-PSF computation."""
    print("\n=== Testing PSFEngine Batch Consistency ===")
    
    pupil, phases = create_test_data(n_screens=10, n_pix=128)
    
    # Create engine
    engine = PSFEngine(
        n_pix=128,
        wavelength=2.2e-6,
        pixel_scale=1e-5,
        normalize_to="sum"
    )
    
    # Compute individually (reference)
    psfs_individual = []
    for i in range(phases.shape[0]):
        psf = engine.compute_psf(pupil, phases[i], normalize=True)
        psfs_individual.append(psf)
    
    backend = get_backend()
    xp = backend.xp
    psfs_individual = xp.stack(psfs_individual)
    psf_avg_ref = xp.mean(psfs_individual, axis=0)
    
    # Compute in batch (vectorized)
    psf_avg_batch, psfs_batch = engine.compute_psf_batch(
        pupil, phases, normalize=True, return_individual=True
    )
    
    # Convert to numpy for comparison
    if hasattr(psfs_individual, 'get'):
        psfs_individual = psfs_individual.get()
        psf_avg_ref = psf_avg_ref.get()
    if hasattr(psfs_batch, 'get'):
        psfs_batch = psfs_batch.get()
        psf_avg_batch = psf_avg_batch.get()
    
    # Check individual PSFs match
    max_diff = np.max(np.abs(psfs_individual - psfs_batch))
    rel_diff = max_diff / np.max(psfs_individual)
    
    print(f"Individual PSFs - Max abs diff: {max_diff:.2e}, Rel diff: {rel_diff:.2e}")
    assert rel_diff < 1e-5, f"Individual PSFs don't match: rel_diff={rel_diff}"
    
    # Check averages match
    avg_diff = np.max(np.abs(psf_avg_ref - psf_avg_batch))
    avg_rel_diff = avg_diff / np.max(psf_avg_ref)
    
    print(f"Average PSF - Max abs diff: {avg_diff:.2e}, Rel diff: {avg_rel_diff:.2e}")
    assert avg_rel_diff < 1e-5, f"Average PSFs don't match: rel_diff={avg_rel_diff}"
    
    # Check normalization (each PSF should sum to 1)
    sums = np.sum(psfs_batch, axis=(1, 2))
    print(f"PSF sums: min={sums.min():.6f}, max={sums.max():.6f}, mean={sums.mean():.6f}")
    assert np.allclose(sums, 1.0, rtol=1e-5), f"PSFs not properly normalized: {sums}"
    
    print("✓ Batch computation produces identical results")


def test_optics_batch_consistency():
    """Test optics.compute_psf_batch vectorization."""
    print("\n=== Testing optics.compute_psf_batch Consistency ===")
    
    pupil, phases = create_test_data(n_screens=10, n_pix=128)
    
    backend = get_backend()
    xp = backend.xp
    
    # Compute individually (reference)
    psfs_ref = []
    for i in range(phases.shape[0]):
        psf = optics.compute_psf(pupil, phases[i], normalize=True)
        psfs_ref.append(psf)
    psfs_ref = xp.stack(psfs_ref)
    
    # Compute in batch (vectorized)
    psfs_batch = optics.compute_psf_batch(pupil, phases, normalize=True)
    
    # Convert to numpy for comparison
    if hasattr(psfs_ref, 'get'):
        psfs_ref = psfs_ref.get()
    if hasattr(psfs_batch, 'get'):
        psfs_batch = psfs_batch.get()
    
    # Check match
    max_diff = np.max(np.abs(psfs_ref - psfs_batch))
    rel_diff = max_diff / np.max(psfs_ref)
    
    print(f"Max abs diff: {max_diff:.2e}, Rel diff: {rel_diff:.2e}")
    assert rel_diff < 1e-5, f"Batch results don't match: rel_diff={rel_diff}"
    
    print("✓ optics batch computation produces identical results")


def test_peak_normalization():
    """Test peak normalization mode."""
    print("\n=== Testing Peak Normalization ===")
    
    pupil, phases = create_test_data(n_screens=10, n_pix=128)
    
    engine = PSFEngine(
        n_pix=128,
        wavelength=2.2e-6,
        pixel_scale=1e-5,
        normalize_to="peak"
    )
    
    _, psfs = engine.compute_psf_batch(
        pupil, phases, normalize=True, return_individual=True
    )
    
    backend = get_backend()
    xp = backend.xp
    
    # Convert to numpy
    if hasattr(psfs, 'get'):
        psfs = psfs.get()
    
    # Check that each PSF has peak of 1
    peaks = np.max(psfs, axis=(1, 2))
    print(f"PSF peaks: min={peaks.min():.6f}, max={peaks.max():.6f}, mean={peaks.mean():.6f}")
    assert np.allclose(peaks, 1.0, rtol=1e-5), f"PSFs not normalized to peak: {peaks}"
    
    print("✓ Peak normalization works correctly")


def benchmark_speedup():
    """Benchmark speedup from vectorization."""
    print("\n=== Benchmarking Speedup ===")
    
    n_screens_list = [50, 100, 200]
    n_pix = 256
    
    for n_screens in n_screens_list:
        print(f"\nBatch size: {n_screens}, Grid: {n_pix}x{n_pix}")
        
        pupil, phases = create_test_data(n_screens=n_screens, n_pix=n_pix)
        
        engine = PSFEngine(
            n_pix=n_pix,
            wavelength=2.2e-6,
            pixel_scale=1e-5,
        )
        
        backend = get_backend()
        xp = backend.xp
        
        # Warm up
        _ = engine.compute_psf_batch(pupil, phases[:5], normalize=True)
        
        # Ensure data is on backend
        pupil_backend = xp.asarray(pupil)
        phases_backend = xp.asarray(phases)
        
        # Time batch computation
        n_runs = 5
        times = []
        for _ in range(n_runs):
            if hasattr(xp, 'cuda'):
                xp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            _ = engine.compute_psf_batch(pupil_backend, phases_backend, normalize=True)
            if hasattr(xp, 'cuda'):
                xp.cuda.Device().synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        time_per_psf = mean_time / n_screens * 1000  # ms per PSF
        throughput = n_screens / mean_time
        
        print(f"  Time: {mean_time:.3f} ± {std_time:.3f} s")
        print(f"  Per PSF: {time_per_psf:.2f} ms")
        print(f"  Throughput: {throughput:.1f} PSF/s")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Vectorized PSF Batch Computation Tests")
    print("=" * 60)
    
    backend = get_backend()
    backend_name = "CuPy" if hasattr(backend.xp, 'cuda') else "NumPy"
    print(f"Backend: {backend_name}")
    
    try:
        # Correctness tests
        test_psf_engine_batch_consistency()
        test_optics_batch_consistency()
        test_peak_normalization()
        
        # Performance benchmark
        benchmark_speedup()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
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

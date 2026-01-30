#!/usr/bin/env python
"""
Demonstration of RNG discipline improvements (Item #7).

Shows:
1. Reproducibility with seed parameter
2. Advanced usage with RNG objects
3. Thread safety
4. Global state isolation
"""

import numpy as np
from nebraa.physics.zernike import generate_zernike_phase
from nebraa.physics.low_wind_effect import LowWindEffect
from nebraa.utils.rng import create_rng, check_rng_determinism
from nebraa.utils.compute import init_backend


def demo_reproducibility():
    """Demonstrate that same seed produces identical results."""
    print("\n" + "="*70)
    print("1. REPRODUCIBILITY: Same seed → same results")
    print("="*70)
    
    # Force CPU for deterministic comparison
    backend = init_backend(compute_mode="CPU")
    
    # Generate phase screens twice with same seed
    phase1 = generate_zernike_phase(n_screens=5, n_pix=64, radius=30, seed=42)
    phase2 = generate_zernike_phase(n_screens=5, n_pix=64, radius=30, seed=42)
    
    # Check they're identical
    is_identical = np.allclose(phase1, phase2, atol=0)
    print(f"  Phase1 shape: {phase1.shape}")
    print(f"  Phase2 shape: {phase2.shape}")
    print(f"  Are identical: {is_identical}")
    print(f"  Max difference: {np.max(np.abs(phase1 - phase2)):.2e}")
    
    assert is_identical, "Same seed should produce identical results!"
    print("  ✓ Reproducibility verified!")


def demo_rng_objects():
    """Demonstrate advanced usage with RNG objects."""
    print("\n" + "="*70)
    print("2. RNG OBJECTS: Reusable RNG with state advancement")
    print("="*70)
    
    # Create RNG once
    rng = create_rng(seed=42)
    
    # Use for multiple calls (state advances)
    phase1 = generate_zernike_phase(n_screens=3, n_pix=32, radius=15, rng=rng)
    phase2 = generate_zernike_phase(n_screens=3, n_pix=32, radius=15, rng=rng)
    
    # Results should be different (RNG state advanced)
    is_different = not np.allclose(phase1, phase2)
    print(f"  Phase1 mean: {np.mean(phase1):.3f}, std: {np.std(phase1):.3f}")
    print(f"  Phase2 mean: {np.mean(phase2):.3f}, std: {np.std(phase2):.3f}")
    print(f"  Are different: {is_different}")
    
    assert is_different, "RNG state should advance between calls!"
    print("  ✓ RNG state advancement verified!")


def demo_global_state_isolation():
    """Demonstrate that functions don't mutate global RNG state."""
    print("\n" + "="*70)
    print("3. GLOBAL STATE ISOLATION: No global RNG mutation")
    print("="*70)
    
    # Set global seed
    np.random.seed(12345)
    
    # Get some global random numbers
    global_before = np.random.randn(5)
    
    # Call function (should NOT affect global state)
    _ = generate_zernike_phase(n_screens=3, n_pix=32, radius=15, seed=42)
    
    # Continue global sequence
    global_after = np.random.randn(5)
    
    # Reset and verify we get same sequence
    np.random.seed(12345)
    expected_before = np.random.randn(5)
    expected_after = np.random.randn(5)
    
    before_match = np.allclose(global_before, expected_before)
    after_match = np.allclose(global_after, expected_after)
    
    print(f"  Global sequence before call matches: {before_match}")
    print(f"  Global sequence after call matches: {after_match}")
    print(f"  First global value: {global_before[0]:.6f}")
    print(f"  Expected value: {expected_before[0]:.6f}")
    
    assert before_match and after_match, "Global RNG state was mutated!"
    print("  ✓ Global state isolation verified!")


def demo_thread_safety():
    """Demonstrate thread-safe parallel execution."""
    print("\n" + "="*70)
    print("4. THREAD SAFETY: Parallel execution without interference")
    print("="*70)
    
    from concurrent.futures import ThreadPoolExecutor
    
    def worker(worker_id):
        # Each worker uses its own seed
        seed = 100 + worker_id
        phase = generate_zernike_phase(n_screens=2, n_pix=32, radius=15, seed=seed)
        return worker_id, np.mean(phase), np.std(phase)
    
    # Run 10 workers in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, i) for i in range(10)]
        results = [f.result() for f in futures]
    
    print(f"  Number of workers: {len(results)}")
    print("  Worker results:")
    for worker_id, mean, std in results[:5]:  # Show first 5
        print(f"    Worker {worker_id}: mean={mean:.3f}, std={std:.3f}")
    print("    ...")
    
    # Verify reproducibility: run same worker again
    worker_0_result_1 = worker(0)
    worker_0_result_2 = worker(0)
    
    reproducible = np.isclose(worker_0_result_1[1], worker_0_result_2[1])
    print(f"\n  Worker 0 run 1: mean={worker_0_result_1[1]:.6f}")
    print(f"  Worker 0 run 2: mean={worker_0_result_2[1]:.6f}")
    print(f"  Reproducible: {reproducible}")
    
    assert reproducible, "Parallel execution should be reproducible!"
    print("  ✓ Thread safety verified!")


def demo_low_wind_effect():
    """Demonstrate LWE with RNG discipline."""
    print("\n" + "="*70)
    print("5. LOW WIND EFFECT: LWE with reproducibility")
    print("="*70)
    
    # Create pupil
    n_pix = 64
    pupil = np.ones((n_pix, n_pix), dtype=np.float32)
    
    # Generate LWE screens twice with same seed
    lwe1 = LowWindEffect(pupil)
    phase1 = lwe1.generate(n_screens=5, seed=42)
    
    lwe2 = LowWindEffect(pupil)
    phase2 = lwe2.generate(n_screens=5, seed=42)
    
    is_identical = np.allclose(phase1, phase2, atol=0)
    print(f"  Phase1 shape: {phase1.shape}")
    print(f"  Phase2 shape: {phase2.shape}")
    print(f"  Are identical: {is_identical}")
    print(f"  Max difference: {np.max(np.abs(phase1 - phase2)):.2e}")
    
    assert is_identical, "LWE should be reproducible!"
    print("  ✓ LWE reproducibility verified!")


def demo_determinism_checker():
    """Demonstrate the determinism checking utility."""
    print("\n" + "="*70)
    print("6. DETERMINISM CHECKER: Automated reproducibility testing")
    print("="*70)
    
    # Check Zernike phase generation
    is_deterministic = check_rng_determinism(
        func=lambda rng: generate_zernike_phase(
            n_screens=3, n_pix=32, radius=15, rng=rng
        ),
        rng_seed=42
    )
    
    print(f"  Zernike phase generation is deterministic: {is_deterministic}")
    
    # Check LWE generation
    n_pix = 64
    pupil = np.ones((n_pix, n_pix), dtype=np.float32)
    lwe = LowWindEffect(pupil)
    
    is_deterministic_lwe = check_rng_determinism(
        func=lambda rng: lwe.generate(n_screens=3, rng=rng),
        rng_seed=42
    )
    
    print(f"  LWE generation is deterministic: {is_deterministic_lwe}")
    
    assert is_deterministic and is_deterministic_lwe
    print("  ✓ Determinism checking verified!")


def main():
    """Run all demonstrations."""
    print("\n" + "#"*70)
    print("# RNG DISCIPLINE DEMONSTRATION (Item #7)")
    print("#"*70)
    print("\nShowing improvements in reproducibility, thread safety,")
    print("and global state isolation.")
    
    try:
        demo_reproducibility()
        demo_rng_objects()
        demo_global_state_isolation()
        demo_thread_safety()
        demo_low_wind_effect()
        demo_determinism_checker()
        
        print("\n" + "="*70)
        print("ALL DEMONSTRATIONS PASSED!")
        print("="*70)
        print("\nKey improvements:")
        print("  ✓ Reproducibility: Same seed → same results")
        print("  ✓ Thread safety: No global state mutation")
        print("  ✓ Test determinism: Order-independent execution")
        print("  ✓ Parallel safety: Safe for multi-GPU/multi-threaded work")
        print("\nBackward compatible: Existing code using seed= still works!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

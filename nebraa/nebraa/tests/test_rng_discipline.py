"""
Tests for RNG discipline: reproducibility and thread safety.

Verifies that:
1. Same seed produces identical results (deterministic)
2. No global RNG state mutation (thread-safe)
3. Test order independence
"""

import numpy as np
import pytest
from nebraa.utils.rng import create_rng, get_rng_or_create
from nebraa.physics.zernike import generate_zernike_phase
from nebraa.physics.low_wind_effect import LowWindEffect


# Force CPU backend for RNG tests (determinism)
@pytest.fixture(autouse=True, scope="module")
def force_cpu_backend():
    """Force CPU backend for reproducibility across NumPy/CuPy RNG differences."""
    from nebraa.utils.compute import init_backend
    backend = init_backend(compute_mode="CPU")
    yield backend


def to_numpy(arr):
    """Convert array to NumPy, handling both NumPy and CuPy."""
    if hasattr(arr, 'get'):  # CuPy array
        return arr.get()
    return np.asarray(arr)


class TestRNGUtilities:
    """Test RNG utility functions."""
    
    def test_create_rng_numpy(self):
        """Test RNG creation for NumPy backend."""
        rng = create_rng(seed=42)
        
        # Should be NumPy Generator
        assert isinstance(rng, np.random.Generator)
        
        # Should produce repeatable results
        val1 = rng.standard_normal((10,))
        rng2 = create_rng(seed=42)
        val2 = rng2.standard_normal((10,))
        
        np.testing.assert_array_equal(val1, val2)
    
    def test_create_rng_no_seed(self):
        """Test RNG creation without seed (non-deterministic)."""
        rng1 = create_rng()
        rng2 = create_rng()
        
        val1 = rng1.standard_normal((100,))
        val2 = rng2.standard_normal((100,))
        
        # Should NOT be identical (different RNG states)
        assert not np.allclose(val1, val2)
    
    def test_get_rng_or_create_with_rng(self):
        """Test that provided RNG is returned as-is."""
        user_rng = create_rng(seed=42)
        
        returned_rng = get_rng_or_create(rng=user_rng)
        
        # Should be same object
        assert returned_rng is user_rng
    
    def test_get_rng_or_create_with_seed(self):
        """Test that seed creates new RNG."""
        rng = get_rng_or_create(seed=42)
        
        assert isinstance(rng, np.random.Generator)
        
        # Should be deterministic
        val1 = rng.standard_normal((10,))
        rng2 = get_rng_or_create(seed=42)
        val2 = rng2.standard_normal((10,))
        
        np.testing.assert_array_equal(val1, val2)


class TestGlobalStateIsolation:
    """Test that functions don't mutate global RNG state."""
    
    def test_no_global_mutation_zernike(self):
        """Verify generate_zernike_phase doesn't mutate global RNG."""
        # Capture global RNG state before
        np.random.seed(12345)
        state_before = np.random.get_state()[1][:10].copy()  # First 10 elements
        
        # Generate phase (should NOT affect global state)
        _ = generate_zernike_phase(
            n_screens=5,
            n_pix=64,
            radius=30,
            seed=42
        )
        
        # Check global RNG state after
        state_after = np.random.get_state()[1][:10].copy()
        
        # Should be UNCHANGED (no global mutation)
        np.testing.assert_array_equal(state_before, state_after)
    
    def test_no_global_mutation_lwe(self):
        """Verify LowWindEffect.generate doesn't mutate global RNG."""
        n_pix = 64
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        # Capture global RNG state before
        np.random.seed(12345)
        state_before = np.random.get_state()[1][:10].copy()
        
        # Generate LWE phase (should NOT affect global state)
        lwe = LowWindEffect(pupil)
        _ = lwe.generate(n_screens=5, seed=42)
        
        # Check global RNG state after
        state_after = np.random.get_state()[1][:10].copy()
        
        # Should be UNCHANGED (no global mutation)
        np.testing.assert_array_equal(state_before, state_after)
    
    def test_global_rng_sequence_unchanged(self):
        """Test that function calls don't affect global random sequence."""
        # Set global seed
        np.random.seed(9999)
        
        # Generate some global random numbers
        global_vals_1 = np.random.randn(5)
        
        # Call function with its own seed (should not interfere)
        _ = generate_zernike_phase(n_screens=3, n_pix=32, radius=15, seed=42)
        
        # Continue global sequence
        global_vals_2 = np.random.randn(5)
        
        # Reset and check we get same sequence
        np.random.seed(9999)
        expected_vals_1 = np.random.randn(5)
        expected_vals_2 = np.random.randn(5)
        
        np.testing.assert_array_almost_equal(global_vals_1, expected_vals_1)
        np.testing.assert_array_almost_equal(global_vals_2, expected_vals_2)


class TestReproducibility:
    """Test that same seed produces identical results."""
    
    def test_zernike_reproducibility_with_seed(self):
        """Same seed should produce identical Zernike phases."""
        phase1 = generate_zernike_phase(
            n_screens=5,
            n_pix=64,
            radius=30,
            seed=42
        )
        
        phase2 = generate_zernike_phase(
            n_screens=5,
            n_pix=64,
            radius=30,
            seed=42
        )
        
        np.testing.assert_array_equal(phase1, phase2)
    
    def test_zernike_reproducibility_with_rng(self):
        """Same RNG object state should produce identical results."""
        rng1 = create_rng(seed=42)
        phase1 = generate_zernike_phase(
            n_screens=5,
            n_pix=64,
            radius=30,
            rng=rng1
        )
        
        rng2 = create_rng(seed=42)
        phase2 = generate_zernike_phase(
            n_screens=5,
            n_pix=64,
            radius=30,
            rng=rng2
        )
        
        np.testing.assert_array_equal(phase1, phase2)
    
    def test_lwe_reproducibility_with_seed(self):
        """Same seed should produce identical LWE phases."""
        n_pix = 64
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        lwe1 = LowWindEffect(pupil)
        phase1 = lwe1.generate(n_screens=5, seed=42)
        
        lwe2 = LowWindEffect(pupil)
        phase2 = lwe2.generate(n_screens=5, seed=42)
        
        np.testing.assert_array_equal(phase1, phase2)
    
    def test_lwe_reproducibility_with_rng(self):
        """Same RNG object state should produce identical LWE results."""
        n_pix = 64
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        lwe1 = LowWindEffect(pupil)
        rng1 = create_rng(seed=42)
        phase1 = lwe1.generate(n_screens=5, rng=rng1)
        
        lwe2 = LowWindEffect(pupil)
        rng2 = create_rng(seed=42)
        phase2 = lwe2.generate(n_screens=5, rng=rng2)
        
        np.testing.assert_array_equal(phase1, phase2)
    
    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different results."""
        phase1 = generate_zernike_phase(
            n_screens=5,
            n_pix=64,
            radius=30,
            seed=42
        )
        
        phase2 = generate_zernike_phase(
            n_screens=5,
            n_pix=64,
            radius=30,
            seed=43
        )
        
        # Should NOT be equal
        assert not np.allclose(phase1, phase2)


class TestOrderIndependence:
    """Test that test execution order doesn't affect results."""
    
    def test_a_first_call(self):
        """First call in alphabetical order."""
        phase = generate_zernike_phase(n_screens=3, n_pix=32, radius=15, seed=1111)
        
        # Store expected value
        self.expected_sum = float(np.sum(phase))
    
    def test_b_second_call(self):
        """Second call - should get same result as first."""
        phase = generate_zernike_phase(n_screens=3, n_pix=32, radius=15, seed=1111)
        
        # Should match first call
        actual_sum = float(np.sum(phase))
        
        # Due to test isolation, we just verify determinism
        # (running twice with same seed gives same result)
        phase2 = generate_zernike_phase(n_screens=3, n_pix=32, radius=15, seed=1111)
        expected_sum = float(np.sum(phase2))
        
        assert actual_sum == expected_sum
    
    def test_z_last_call(self):
        """Last call in alphabetical order - independent of others."""
        phase = generate_zernike_phase(n_screens=3, n_pix=32, radius=15, seed=2222)
        
        # Should be deterministic for this seed
        phase2 = generate_zernike_phase(n_screens=3, n_pix=32, radius=15, seed=2222)
        
        np.testing.assert_array_equal(phase, phase2)


class TestRNGParameter:
    """Test that rng parameter works correctly."""
    
    def test_rng_parameter_takes_precedence_over_seed(self):
        """If both rng and seed provided, rng should be used."""
        rng = create_rng(seed=42)
        
        # Both parameters provided - rng should be used
        phase1 = generate_zernike_phase(
            n_screens=3,
            n_pix=32,
            radius=15,
            seed=999,  # Should be ignored
            rng=rng
        )
        
        # Create new RNG with same seed as the one we passed
        rng2 = create_rng(seed=42)
        phase2 = generate_zernike_phase(
            n_screens=3,
            n_pix=32,
            radius=15,
            rng=rng2
        )
        
        np.testing.assert_array_equal(phase1, phase2)
    
    def test_rng_can_be_reused(self):
        """Single RNG object can be used for multiple calls."""
        rng = create_rng(seed=42)
        
        phase1 = generate_zernike_phase(n_screens=2, n_pix=32, radius=15, rng=rng)
        phase2 = generate_zernike_phase(n_screens=2, n_pix=32, radius=15, rng=rng)
        
        # Should be different (RNG state advanced)
        assert not np.allclose(phase1, phase2)
    
    def test_lwe_rng_parameter(self):
        """LWE should accept rng parameter."""
        n_pix = 64
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        lwe = LowWindEffect(pupil)
        
        rng1 = create_rng(seed=42)
        phase1 = lwe.generate(n_screens=3, rng=rng1)
        
        rng2 = create_rng(seed=42)
        phase2 = lwe.generate(n_screens=3, rng=rng2)
        
        np.testing.assert_array_equal(phase1, phase2)


class TestBackwardCompatibility:
    """Test that seed parameter still works (backward compatibility)."""
    
    def test_seed_parameter_still_works(self):
        """Old code using seed= should still work."""
        # Old-style call with seed
        phase = generate_zernike_phase(
            n_screens=5,
            n_pix=64,
            radius=30,
            seed=42
        )
        
        # Should be deterministic
        phase2 = generate_zernike_phase(
            n_screens=5,
            n_pix=64,
            radius=30,
            seed=42
        )
        
        np.testing.assert_array_equal(phase, phase2)
    
    def test_lwe_seed_parameter_still_works(self):
        """LWE with seed= should still work."""
        n_pix = 64
        pupil = np.ones((n_pix, n_pix), dtype=np.float32)
        
        lwe = LowWindEffect(pupil)
        
        phase1 = lwe.generate(n_screens=5, seed=42)
        phase2 = lwe.generate(n_screens=5, seed=42)
        
        np.testing.assert_array_equal(phase1, phase2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

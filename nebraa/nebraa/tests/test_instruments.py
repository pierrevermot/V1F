"""
Tests for instrument implementations.
"""

import numpy as np
import pytest


class TestInstrumentRegistry:
    """Tests for instrument registry."""
    
    def test_list_instruments(self):
        """Test listing available instruments."""
        from nebraa.instruments import list_instruments
        
        instruments = list_instruments()
        
        assert 'vlt' in instruments
        assert 'lbti' in instruments
    
    def test_get_instrument_vlt(self):
        """Test getting VLT instrument."""
        from nebraa.instruments import get_instrument
        
        inst = get_instrument('vlt')
        
        assert inst is not None
        assert hasattr(inst, 'generate_psf')
        assert hasattr(inst, 'generate_psfs')
        assert hasattr(inst, 'pupil')
    
    def test_get_instrument_lbti(self):
        """Test getting LBTI instrument."""
        from nebraa.instruments import get_instrument
        
        inst = get_instrument('lbti')
        
        assert inst is not None
        assert hasattr(inst, 'generate_psf')
    
    def test_unknown_instrument(self):
        """Test error on unknown instrument."""
        from nebraa.instruments import get_instrument
        
        with pytest.raises(ValueError):
            get_instrument('unknown_telescope')


class TestVLTInstrument:
    """Tests for VLT instrument."""
    
    def test_pupil_shape(self, backend):
        """Test pupil generation."""
        from nebraa.instruments import get_instrument
        
        inst = get_instrument('vlt')
        pupil = inst.pupil
        
        # Should be 2D array
        assert pupil.ndim == 2
        assert pupil.shape[0] == pupil.shape[1]
    
    def test_pupil_values(self, backend):
        """Test pupil values are in valid range."""
        from nebraa.instruments import get_instrument
        
        inst = get_instrument('vlt')
        pupil = backend.to_numpy(inst.pupil)
        
        # Values should be between 0 and 1
        assert pupil.min() >= 0
        assert pupil.max() <= 1
    
    def test_pupil_has_obstruction(self, backend):
        """Test that pupil has central obstruction."""
        from nebraa.instruments import get_instrument
        
        inst = get_instrument('vlt')
        pupil = backend.to_numpy(inst.pupil)
        
        # Center should be zero (obscured)
        center = pupil.shape[0] // 2
        center_val = pupil[center, center]
        assert center_val < 0.5  # Central obstruction
    
    def test_psf_generation(self, backend):
        """Test PSF generation."""
        from nebraa.instruments import get_instrument
        
        inst = get_instrument('vlt')
        psf = inst.generate_psf()
        
        # Should be 2D
        assert psf.ndim == 2
        
        # Should be positive
        psf_np = backend.to_numpy(psf)
        assert psf_np.min() >= 0
    
    def test_multiple_psfs(self, backend):
        """Test generating multiple PSFs."""
        from nebraa.instruments import get_instrument
        
        inst = get_instrument('vlt')
        psfs = inst.generate_psfs(n=3)
        
        # Should have batch dimension
        assert psfs.shape[0] == 3


class TestLBTIInstrument:
    """Tests for LBTI instrument."""
    
    def test_pupil_shape(self, backend):
        """Test LBTI pupil shape."""
        from nebraa.instruments import get_instrument
        
        inst = get_instrument('lbti')
        pupil = inst.pupil
        
        assert pupil.ndim == 2
    
    def test_dual_aperture_pupil(self, backend):
        """Test that LBTI has two apertures."""
        from nebraa.instruments import get_instrument
        
        inst = get_instrument('lbti')
        pupil = backend.to_numpy(inst.pupil)
        
        # Check horizontal profile through center
        center_row = pupil.shape[0] // 2
        profile = pupil[center_row, :]
        
        # Should have gaps (zeros between apertures)
        assert np.min(profile) == 0  # Some zeros
        assert np.max(profile) > 0   # Some nonzero
    
    def test_psf_generation(self, backend):
        """Test LBTI PSF generation."""
        from nebraa.instruments import get_instrument
        
        inst = get_instrument('lbti')
        psf = backend.to_numpy(inst.generate_psf())
        
        # Should be positive
        assert psf.min() >= 0
        assert psf.max() > 0

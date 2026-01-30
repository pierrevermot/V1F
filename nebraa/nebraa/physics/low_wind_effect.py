"""
Low Wind Effect (LWE) Model

General-purpose implementation of quasi-static aberrations caused by low wind conditions.
Automatically detects disconnected regions ("islands" or "petals") in a pupil mask and 
applies differential piston, tip, and tilt to each region.

This implementation is pupil-agnostic and can be used with any telescope geometry
(VLT, ELT, segmented mirrors, etc.) without requiring manual sector definition.

The LWE model generates phase screens representing slowly-varying aberrations that
are not corrected by the AO system. These are typically used in long-exposure PSF
modeling by averaging over multiple realizations.

Author: NEBRAA
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from ..utils.compute import get_backend
from ..utils.rng import create_rng, get_rng_or_create, RNGType


@dataclass
class LowWindEffectConfig:
    """
    Configuration for Low Wind Effect model.
    
    Attributes:
        piston_rms_rad: RMS of differential piston between regions (radians)
        tilt_rms_rad: RMS of differential tip/tilt between regions (radians)
        ar_coeff: AR(1) coefficient for temporal correlation (not yet used)
        n_realizations: Number of realizations to average for long-exposure
        seed: Random seed for reproducibility. If None, random each time.
        pupil_threshold: Threshold for binarizing pupil mask (0-1). Lower values
                        allow detection of apodized/soft-edge pupils.
        connectivity: Connectivity for island detection. 1 for 4-connected,
                     2 for 8-connected (diagonal connections).
    """
    piston_rms_rad: float = 0.5
    tilt_rms_rad: float = 0.3
    ar_coeff: float = 0.95
    n_realizations: int = 10
    seed: int = None
    pupil_threshold: float = 0.99
    connectivity: int = 1


class LowWindEffect:
    """
    General Low Wind Effect (LWE) model.
    
    Automatically detects disconnected regions ("islands" or "petals") in a
    pupil mask and applies differential piston, tip, and tilt to each region.
    
    This is a general implementation that works with any pupil geometry
    (VLT, ELT, segmented mirrors, etc.) without requiring manual sector
    definition.
    
    Usage:
        lwe = LowWindEffect(pupil, piston_rms_rad=0.5, tilt_rms_rad=0.3)
        phase_screens = lwe.generate(n_screens=10)
        # phase_screens.shape = (10, n_pix, n_pix)
    """
    
    # Class-level cache for island masks
    _mask_cache = {}
    
    def __init__(
        self,
        pupil: np.ndarray,
        piston_rms_rad: float = 0.5,
        tilt_rms_rad: float = 0.3,
        ar_coeff: float = 0.95,
        pupil_threshold: float = 0.99,
        connectivity: int = 1,
        precomputed_island_masks: Optional[np.ndarray] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize LWE model from a pupil mask.
        
        Args:
            pupil: 2D pupil amplitude mask (values > threshold are considered part of pupil)
            piston_rms_rad: RMS of differential piston in radians
            tilt_rms_rad: RMS of differential tip/tilt in radians
            ar_coeff: AR(1) coefficient for temporal correlation (not yet used)
            pupil_threshold: Threshold for binarizing pupil (0-1). Lower values allow
                           detection of apodized/soft-edge pupils. Default 0.99.
            connectivity: Connectivity for island detection (1=4-connected, 2=8-connected)
            precomputed_island_masks: Optional precomputed (n_islands, n_pix, n_pix) masks.
                                     If provided, skips island detection.
            enable_cache: Whether to cache island masks for future use. Default True.
        """
        # Detect backend from the pupil array
        try:
            import cupy as cp
            self._is_gpu = isinstance(pupil, cp.ndarray)
        except ImportError:
            self._is_gpu = False
        
        backend = get_backend()
        xp = backend.xp
        
        self.piston_rms = piston_rms_rad
        self.tilt_rms = tilt_rms_rad
        self.ar_coeff = ar_coeff
        self.pupil_threshold = pupil_threshold
        self.connectivity = connectivity
        self.enable_cache = enable_cache
        
        # Convert pupil to numpy for scipy labeling (scipy only works on CPU)
        if self._is_gpu:
            import cupy as cp
            pupil_np = cp.asnumpy(pupil)
        else:
            pupil_np = np.asarray(pupil)
        self.n_pix = pupil_np.shape[0]
        
        # Use precomputed masks if provided, otherwise detect islands
        if precomputed_island_masks is not None:
            if self._is_gpu:
                import cupy as cp
                self.island_masks = cp.asarray(precomputed_island_masks)
            else:
                self.island_masks = np.asarray(precomputed_island_masks)
            self.n_islands = self.island_masks.shape[0]
        else:
            # Check cache first
            cache_key = self._get_cache_key(pupil_np)
            if enable_cache and cache_key in LowWindEffect._mask_cache:
                cached_masks, cached_n_islands = LowWindEffect._mask_cache[cache_key]
                self.island_masks = cached_masks.copy()
                self.n_islands = cached_n_islands
            else:
                # Detect islands using connected component labeling
                self.island_masks, self.n_islands = self._detect_islands(
                    pupil_np, pupil_threshold, connectivity
                )
                
                # Cache the result
                if enable_cache:
                    LowWindEffect._mask_cache[cache_key] = (
                        self.island_masks.copy(), self.n_islands
                    )
        
        # Early return guard: if no islands detected, we can't apply LWE
        if self.n_islands == 0:
            # Set empty coordinate grids to avoid NaNs
            if self._is_gpu:
                import cupy as cp
                self.X_norm = cp.zeros((0, self.n_pix, self.n_pix), dtype=cp.float32)
                self.Y_norm = cp.zeros((0, self.n_pix, self.n_pix), dtype=cp.float32)
            else:
                self.X_norm = np.zeros((0, self.n_pix, self.n_pix), dtype=np.float32)
                self.Y_norm = np.zeros((0, self.n_pix, self.n_pix), dtype=np.float32)
            return  # Skip coordinate building
        
        # Build normalized tip/tilt coordinates for each island
        self._build_tilt_coords()
    
    def _get_cache_key(self, pupil: np.ndarray) -> tuple:
        """
        Generate cache key for island masks.
        
        Args:
            pupil: Pupil mask (numpy array)
            
        Returns:
            Cache key tuple: (pupil_id, threshold, connectivity, n_pix)
        """
        # Use hash of pupil array as ID (robust to array equality)
        pupil_bytes = pupil.tobytes()
        pupil_id = hash(pupil_bytes)
        return (pupil_id, self.pupil_threshold, self.connectivity, self.n_pix)
    
    def _detect_islands(self, pupil: np.ndarray, threshold: float, connectivity: int) -> Tuple[np.ndarray, int]:
        """
        Detect disconnected regions in the pupil using connected component labeling.
        
        GPU-accelerated when pupil is a CuPy array (uses cupyx.scipy.ndimage).
        Falls back to scipy.ndimage for NumPy arrays.
        
        Args:
            pupil: 2D pupil mask (numpy or cupy array)
            threshold: Threshold for binarizing pupil (0-1)
            connectivity: Connectivity for island detection (1 or 2)
            
        Returns:
            island_masks: (n_islands, n_pix, n_pix) array of island masks
            n_islands: number of detected islands
        """
        # Determine if input is GPU array (check module name, not just device attr)
        # NumPy 2.x arrays also have a 'device' attribute, so we need a better check
        is_gpu_input = hasattr(pupil, '__module__') and 'cupy' in pupil.__module__
        
        # Create structure for connectivity
        if connectivity == 1:
            structure = None  # Default 4-connected
        elif connectivity == 2:
            structure = np.ones((3, 3), dtype=np.int32)  # 8-connected
        else:
            raise ValueError(f"connectivity must be 1 or 2, got {connectivity}")
        
        # Binarize pupil with configurable threshold
        if is_gpu_input:
            import cupy as cp
            binary_pupil = (pupil > threshold).astype(cp.int32)
            
            # Try GPU-accelerated labeling
            try:
                from cupyx.scipy import ndimage as ndimage_gpu
                if structure is not None:
                    structure_gpu = cp.asarray(structure)
                    labeled, n_islands = ndimage_gpu.label(binary_pupil, structure=structure_gpu)
                else:
                    labeled, n_islands = ndimage_gpu.label(binary_pupil)
                
                # Create individual masks for each island (on GPU)
                island_masks = cp.zeros((n_islands, self.n_pix, self.n_pix), dtype=cp.float32)
                for i in range(n_islands):
                    island_masks[i] = (labeled == (i + 1)).astype(cp.float32)
                
                # Convert to numpy for return (will be converted back if needed)
                island_masks = cp.asnumpy(island_masks)
                
                return island_masks, n_islands
                
            except ImportError:
                # cupyx.scipy not available - fall back to CPU
                pupil = cp.asnumpy(pupil)
                binary_pupil = (pupil > threshold).astype(np.int32)
        else:
            binary_pupil = (pupil > threshold).astype(np.int32)
        
        # CPU path using scipy
        try:
            from scipy import ndimage
            
            # Label connected components
            labeled, n_islands = ndimage.label(binary_pupil, structure=structure)
            
            # Create individual masks for each island
            island_masks = np.zeros((n_islands, self.n_pix, self.n_pix), dtype=np.float32)
            for i in range(n_islands):
                island_masks[i] = (labeled == (i + 1)).astype(np.float32)
            
            return island_masks, n_islands
            
        except ImportError:
            raise ImportError(
                "Neither scipy.ndimage nor cupyx.scipy.ndimage available. "
                "Install scipy to use LowWindEffect: pip install scipy"
            )
    
    def _build_tilt_coords(self):
        """Build normalized tip/tilt coordinate grids for each island."""
        backend = get_backend()
        xp = backend.xp
        
        # Global coordinate grid (pixel-centered)
        center = (self.n_pix - 1) / 2.0
        idx = np.arange(self.n_pix, dtype=np.float32) - center
        X, Y = np.meshgrid(idx, idx)
        
        # For each island, compute normalized coordinates relative to island center
        self.X_norm = np.zeros((self.n_islands, self.n_pix, self.n_pix), dtype=np.float32)
        self.Y_norm = np.zeros((self.n_islands, self.n_pix, self.n_pix), dtype=np.float32)
        
        for i in range(self.n_islands):
            mask = self.island_masks[i]
            
            # Find island centroid
            total = mask.sum()
            if total > 0:
                cx = (X * mask).sum() / total
                cy = (Y * mask).sum() / total
                
                # Find island extent for normalization
                island_pixels = np.where(mask > 0.5)
                if len(island_pixels[0]) > 0:
                    extent = max(
                        island_pixels[0].max() - island_pixels[0].min(),
                        island_pixels[1].max() - island_pixels[1].min()
                    )
                    extent = max(extent, 1)  # Avoid division by zero
                    
                    # Normalized coordinates (centered on island, scaled by extent)
                    self.X_norm[i] = (X - cx) / (extent / 2)
                    self.Y_norm[i] = (Y - cy) / (extent / 2)
        
        # Convert to backend array type
        if self._is_gpu:
            import cupy as cp
            self.island_masks = cp.asarray(self.island_masks)
            self.X_norm = cp.asarray(self.X_norm)
            self.Y_norm = cp.asarray(self.Y_norm)
        else:
            self.island_masks = np.asarray(self.island_masks)
            self.X_norm = np.asarray(self.X_norm)
            self.Y_norm = np.asarray(self.Y_norm)
    
    def generate(self, n_screens: int, seed: Optional[int] = None, rng: Optional[RNGType] = None):
        """
        Generate LWE phase screens.
        
        Args:
            n_screens: Number of phase screens to generate
            seed: Random seed for reproducibility (ignored if rng provided)
            rng: Optional RNG object for reproducibility (preferred over seed)
            
        Returns:
            phase: (n_screens, n_pix, n_pix) array of phase screens in radians
                  Returns zeros if no islands were detected.
        """
        # Early return if no islands detected
        if self.n_islands == 0:
            # Return zero phase screens (no LWE to apply)
            if self._is_gpu:
                import cupy as cp
                return cp.zeros((n_screens, self.n_pix, self.n_pix), dtype=cp.float32)
            else:
                return np.zeros((n_screens, self.n_pix, self.n_pix), dtype=np.float32)
        
        # Determine backend
        backend = get_backend()
        
        # Use the backend determined at init from the pupil
        if self._is_gpu:
            import cupy as cp
            xp = cp
        else:
            xp = np
        
        # Get RNG object (never mutates global state)
        rng = get_rng_or_create(rng=rng, seed=seed, backend=backend)
        
        # Random differential coefficients for each island
        # Note: We make these differential (zero mean across islands)
        pistons = rng.standard_normal((self.n_islands, n_screens)).astype(xp.float32) * self.piston_rms
        tips_x = rng.standard_normal((self.n_islands, n_screens)).astype(xp.float32) * self.tilt_rms
        tips_y = rng.standard_normal((self.n_islands, n_screens)).astype(xp.float32) * self.tilt_rms
        
        # Remove mean to make truly differential
        pistons = pistons - pistons.mean(axis=0, keepdims=True)
        tips_x = tips_x - tips_x.mean(axis=0, keepdims=True)
        tips_y = tips_y - tips_y.mean(axis=0, keepdims=True)
        
        # Build phase screens
        phase = xp.zeros((n_screens, self.n_pix, self.n_pix), dtype=xp.float32)
        
        for i in range(self.n_islands):
            mask = self.island_masks[i]
            x_norm = self.X_norm[i]
            y_norm = self.Y_norm[i]
            
            # Add piston
            phase += pistons[i, :, None, None] * mask[None, :, :]
            
            # Add tip (X tilt)
            phase += tips_x[i, :, None, None] * x_norm[None, :, :] * mask[None, :, :]
            
            # Add tilt (Y tilt)
            phase += tips_y[i, :, None, None] * y_norm[None, :, :] * mask[None, :, :]
        
        return phase
    
    @property 
    def masks(self):
        """Return island masks for visualization/debugging."""
        return self.island_masks
    
    @property
    def info(self) -> dict:
        """Return information about detected islands and configuration."""
        backend = get_backend()
        masks_np = backend.to_numpy(self.island_masks)
        
        info = {
            'n_islands': self.n_islands,
            'island_sizes': [int(mask.sum()) for mask in masks_np] if self.n_islands > 0 else [],
            'piston_rms_rad': self.piston_rms,
            'tilt_rms_rad': self.tilt_rms,
            'pupil_threshold': self.pupil_threshold,
            'connectivity': self.connectivity,
            'cache_enabled': self.enable_cache,
            'cache_size': len(LowWindEffect._mask_cache),
        }
        return info
    
    @classmethod
    def clear_cache(cls):
        """Clear the island mask cache."""
        cls._mask_cache.clear()
    
    @classmethod
    def get_cache_size(cls) -> int:
        """Return the current size of the island mask cache."""
        return len(cls._mask_cache)

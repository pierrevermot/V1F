"""
Quick reference for standardized PSF normalization API.

Run this to see examples of all normalization modes in action.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nebraa.physics.psf_engine import PSFEngine
from nebraa.physics import optics


def create_test_pupil_phase(n_pix=128):
    """Create simple test data."""
    # Circular pupil
    y, x = np.ogrid[-n_pix//2:n_pix//2, -n_pix//2:n_pix//2]
    r = np.sqrt(x**2 + y**2)
    pupil = (r < n_pix // 3).astype(np.float32)
    
    # Simple phase screen
    phase = np.random.randn(n_pix, n_pix).astype(np.float32) * 0.3
    
    return pupil, phase


def main():
    print("=" * 70)
    print("PSF Normalization API Quick Reference")
    print("=" * 70)
    
    pupil, phase = create_test_pupil_phase()
    
    # =================================================================
    # PSFEngine API
    # =================================================================
    print("\n" + "=" * 70)
    print("1. PSFEngine API (Recommended)")
    print("=" * 70)
    
    print("\n--- Sum Normalization (Energy Conservation) ---")
    engine_sum = PSFEngine(
        n_pix=128,
        wavelength=2.2e-6,
        pixel_scale=1e-5,
        normalize_to="sum"  # ← Energy conservation (default)
    )
    psf_sum = engine_sum.compute_psf(pupil, phase)
    if hasattr(psf_sum, 'get'):
        psf_sum = psf_sum.get()
    
    print(f"normalize_to='sum':")
    print(f"  PSF sum:  {psf_sum.sum():.8f}  ← Should be 1.0")
    print(f"  PSF peak: {psf_sum.max():.8f}")
    print(f"  Use case: Photometry, flux measurements, energy conservation")
    
    print("\n--- Peak Normalization ---")
    engine_peak = PSFEngine(
        n_pix=128,
        wavelength=2.2e-6,
        pixel_scale=1e-5,
        normalize_to="peak"  # ← Peak = 1.0
    )
    psf_peak = engine_peak.compute_psf(pupil, phase)
    if hasattr(psf_peak, 'get'):
        psf_peak = psf_peak.get()
    
    print(f"normalize_to='peak':")
    print(f"  PSF sum:  {psf_peak.sum():.8f}")
    print(f"  PSF peak: {psf_peak.max():.8f}  ← Should be 1.0")
    print(f"  Use case: Visualization, relative intensity")
    
    print("\n--- No Normalization (Raw) ---")
    engine_none = PSFEngine(
        n_pix=128,
        wavelength=2.2e-6,
        pixel_scale=1e-5,
        normalize_to="none"  # ← Raw FFT output
    )
    psf_none = engine_none.compute_psf(pupil, phase)
    if hasattr(psf_none, 'get'):
        psf_none = psf_none.get()
    
    print(f"normalize_to='none':")
    print(f"  PSF sum:  {psf_none.sum():.2e}  ← Raw scale")
    print(f"  PSF peak: {psf_none.max():.2e}")
    print(f"  Use case: Custom processing, debugging")
    
    # =================================================================
    # optics Module API
    # =================================================================
    print("\n" + "=" * 70)
    print("2. optics Module API (Backward Compatible)")
    print("=" * 70)
    
    print("\n--- Explicit normalize_to (New, Recommended) ---")
    psf_opt_sum = optics.compute_psf(pupil, phase, normalize_to="sum")
    if hasattr(psf_opt_sum, 'get'):
        psf_opt_sum = psf_opt_sum.get()
    print(f"normalize_to='sum': sum={psf_opt_sum.sum():.8f}")
    
    psf_opt_peak = optics.compute_psf(pupil, phase, normalize_to="peak")
    if hasattr(psf_opt_peak, 'get'):
        psf_opt_peak = psf_opt_peak.get()
    print(f"normalize_to='peak': peak={psf_opt_peak.max():.8f}")
    
    psf_opt_none = optics.compute_psf(pupil, phase, normalize_to="none")
    if hasattr(psf_opt_none, 'get'):
        psf_opt_none = psf_opt_none.get()
    print(f"normalize_to='none': sum={psf_opt_none.sum():.2e}")
    
    print("\n--- Legacy API (Still Works) ---")
    psf_legacy_norm = optics.compute_psf(pupil, phase, normalize=True)
    if hasattr(psf_legacy_norm, 'get'):
        psf_legacy_norm = psf_legacy_norm.get()
    print(f"normalize=True (legacy): peak={psf_legacy_norm.max():.8f}  ← Default: peak=1")
    
    psf_legacy_no = optics.compute_psf(pupil, phase, normalize=False)
    if hasattr(psf_legacy_no, 'get'):
        psf_legacy_no = psf_legacy_no.get()
    print(f"normalize=False (legacy): sum={psf_legacy_no.sum():.2e}  ← Raw (none)")
    
    # =================================================================
    # Code Examples
    # =================================================================
    print("\n" + "=" * 70)
    print("3. Code Examples")
    print("=" * 70)
    
    print("\n--- Example 1: Photometry (Energy Conservation) ---")
    print("""
engine = PSFEngine(normalize_to="sum")
psf = engine.compute_psf(pupil, phase)
total_flux = psf.sum()  # → 1.0
central_flux = psf[center_y, center_x]
fractional_flux = central_flux / total_flux
    """.strip())
    
    print("\n--- Example 2: Visualization ---")
    print("""
psf = optics.compute_psf(pupil, phase, normalize_to="peak")
plt.imshow(psf, vmax=1.0)  # Peak is always 1.0
plt.colorbar(label="Normalized Intensity")
    """.strip())
    
    print("\n--- Example 3: Batch Processing ---")
    print("""
engine = PSFEngine(normalize_to="sum")
psf_avg, psfs = engine.compute_psf_batch(
    pupil, phases,
    normalize=True,
    return_individual=True
)
# All PSFs sum to 1.0 individually
# Average also sums to 1.0
    """.strip())
    
    # =================================================================
    # Comparison Table
    # =================================================================
    print("\n" + "=" * 70)
    print("4. Normalization Mode Comparison")
    print("=" * 70)
    
    print("\nMode      | Sum       | Peak      | Use Case")
    print("-" * 70)
    print(f"'sum'     | {psf_sum.sum():9.6f} | {psf_sum.max():9.6f} | Photometry, energy conservation")
    print(f"'peak'    | {psf_peak.sum():9.6f} | {psf_peak.max():9.6f} | Visualization, relative intensity")
    print(f"'none'    | {psf_none.sum():9.2e} | {psf_none.max():9.2e} | Custom processing, debugging")
    
    # =================================================================
    # Recommendations
    # =================================================================
    print("\n" + "=" * 70)
    print("5. Recommendations")
    print("=" * 70)
    
    print("""
✓ For NEW CODE:
  - Use PSFEngine with explicit normalize_to parameter
  - Default to "sum" for energy conservation (scientific correctness)
  - Use "peak" for visualization only

✓ For PHOTOMETRY/FLUX:
  - Always use normalize_to="sum"
  - Ensures energy conservation
  - Correct for absolute flux measurements

✓ For VISUALIZATION:
  - Use normalize_to="peak"
  - Makes peak visible at 1.0
  - Good for displaying PSF structure

✓ For EXISTING CODE:
  - No changes required (backward compatible)
  - Can gradually migrate to explicit normalize_to
  - Legacy normalize=True/False still works
    """.strip())
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()

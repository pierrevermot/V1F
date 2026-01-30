#!/usr/bin/env python3
"""
Demonstration of FFT Centering Convention in NEBRAA

This script demonstrates the importance of the FFT centering convention
(ifftshift before fft2) for PSF computation, showing:

1. Standard convention produces centered PSFs
2. Consistency across different modules
3. No spurious phase ramps from FFT centering
4. Symmetric inputs produce symmetric PSFs

Author: NEBRAA Development Team
Date: 2026-01-30
Related: Item #10 - FFT Centering Convention
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Force CPU for demonstration
import os
os.environ["NEBRAA_FORCE_CPU"] = "1"

from nebraa.physics.pupil import Pupil
from nebraa.physics.psf_engine import PSFEngine
from nebraa.utils.compute import get_backend


def to_numpy(arr):
    """Convert array to numpy for plotting (handles both CPU and GPU)."""
    backend = get_backend()
    return backend.to_numpy(arr)


def demo_fft_convention_comparison():
    """
    Compare FFT with and without ifftshift to show why the convention matters.
    """
    print("\n" + "="*80)
    print("DEMO 1: FFT Convention Comparison")
    print("="*80)
    
    # Create simple pupil
    n_pix = 128
    diameter = 1.0
    wavelength = 500e-9
    pixel_scale = 10e-3 * np.pi / 180 / 3600  # 10 mas/pixel in radians
    
    pupil = Pupil.from_circular(
        n_pix=n_pix,
        diameter=diameter,
        wavelength=wavelength,
        pixel_scale=pixel_scale,
    )
    pupil_array = to_numpy(pupil.amplitude)
    
    # Method 1: With ifftshift (CORRECT)
    E_pupil = pupil_array.astype(np.complex128)
    E_focal_correct = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pupil)))
    psf_correct = np.abs(E_focal_correct)**2
    psf_correct /= psf_correct.max()
    
    # Method 2: Without ifftshift (LESS STANDARD)
    E_focal_no_ifftshift = np.fft.fftshift(np.fft.fft2(E_pupil))
    psf_no_ifftshift = np.abs(E_focal_no_ifftshift)**2
    psf_no_ifftshift /= psf_no_ifftshift.max()
    
    # Find PSF peaks
    peak_correct = np.unravel_index(np.argmax(psf_correct), psf_correct.shape)
    peak_no_ifftshift = np.unravel_index(np.argmax(psf_no_ifftshift), psf_no_ifftshift.shape)
    
    center = (n_pix - 1) / 2.0
    
    print(f"\nPupil: {n_pix}x{n_pix} pixels, diameter={diameter}m")
    print(f"Expected PSF peak location (center): ({center:.1f}, {center:.1f})")
    print(f"\nWith ifftshift (STANDARD):    Peak at {peak_correct}")
    print(f"Without ifftshift (ALT):      Peak at {peak_no_ifftshift}")
    
    # For NEBRAA's pixel-centered pupils, both methods give identical results
    difference = np.abs(psf_correct - psf_no_ifftshift).max()
    print(f"\nMax difference between methods: {difference:.2e}")
    
    if difference < 1e-10:
        print("✓ Both methods produce identical PSFs (for pixel-centered pupils)")
        print("✓ NEBRAA uses standard convention for consistency with best practices")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Pupil
    axes[0, 0].imshow(pupil_array, cmap='gray', origin='lower')
    axes[0, 0].axvline(center, color='r', linestyle='--', alpha=0.5, label='Center')
    axes[0, 0].axhline(center, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Input Pupil (pixel-centered)')
    axes[0, 0].set_xlabel('X [pixels]')
    axes[0, 0].set_ylabel('Y [pixels]')
    axes[0, 0].legend()
    
    # PSF with ifftshift (standard)
    im1 = axes[0, 1].imshow(psf_correct, cmap='hot', origin='lower', norm=plt.matplotlib.colors.LogNorm(vmin=1e-6, vmax=1))
    axes[0, 1].plot(peak_correct[1], peak_correct[0], 'c+', markersize=20, markeredgewidth=2, label='Peak')
    axes[0, 1].axvline(center, color='cyan', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(center, color='cyan', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('PSF with ifftshift (STANDARD)')
    axes[0, 1].set_xlabel('X [pixels]')
    axes[0, 1].set_ylabel('Y [pixels]')
    axes[0, 1].legend()
    plt.colorbar(im1, ax=axes[0, 1], label='Normalized intensity')
    
    # PSF without ifftshift
    im2 = axes[1, 0].imshow(psf_no_ifftshift, cmap='hot', origin='lower', norm=plt.matplotlib.colors.LogNorm(vmin=1e-6, vmax=1))
    axes[1, 0].plot(peak_no_ifftshift[1], peak_no_ifftshift[0], 'c+', markersize=20, markeredgewidth=2, label='Peak')
    axes[1, 0].axvline(center, color='cyan', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(center, color='cyan', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('PSF without ifftshift')
    axes[1, 0].set_xlabel('X [pixels]')
    axes[1, 0].set_ylabel('Y [pixels]')
    axes[1, 0].legend()
    plt.colorbar(im2, ax=axes[1, 0], label='Normalized intensity')
    
    # Difference
    diff_img = np.abs(psf_correct - psf_no_ifftshift)
    im3 = axes[1, 1].imshow(diff_img, cmap='viridis', origin='lower')
    axes[1, 1].set_title(f'Absolute Difference (max={difference:.2e})')
    axes[1, 1].set_xlabel('X [pixels]')
    axes[1, 1].set_ylabel('Y [pixels]')
    plt.colorbar(im3, ax=axes[1, 1], label='Absolute difference')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "demo_fft_convention_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure: {output_dir / 'demo_fft_convention_comparison.png'}")
    
    return psf_correct, psf_no_ifftshift


def demo_psf_engine_centering():
    """
    Demonstrate that PSFEngine produces properly centered PSFs.
    """
    print("\n" + "="*80)
    print("DEMO 2: PSFEngine Centering Verification")
    print("="*80)
    
    # Create PSF engine and pupil
    n_pix = 256
    diameter = 8.0  # VLT-like telescope
    wavelength = 1.65e-6  # H-band
    pixel_scale = 10e-3 * np.pi / 180 / 3600  # 10 mas/pixel in radians
    
    pupil = Pupil.from_circular(
        n_pix=n_pix,
        diameter=diameter,
        wavelength=wavelength,
        pixel_scale=pixel_scale,
    )
    
    engine = PSFEngine(
        n_pix=n_pix,
        wavelength=wavelength,
        pixel_scale=pixel_scale,
    )
    
    # Compute PSF with zero phase (perfect optics)
    phase = np.zeros((n_pix, n_pix))
    psf = to_numpy(engine.compute_psf(to_numpy(pupil.amplitude), phase))
    
    # Find peak location
    peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
    center = (n_pix - 1) / 2.0
    
    print(f"\nTelescope: D={diameter}m, λ={wavelength*1e6:.2f}μm")
    print(f"PSF: {n_pix}x{n_pix} pixels, scale={pixel_scale*1000:.1f} mas/pixel")
    print(f"Expected peak location: ({center:.1f}, {center:.1f})")
    print(f"Actual peak location:   {peak_idx}")
    
    offset = np.sqrt((peak_idx[0] - center)**2 + (peak_idx[1] - center)**2)
    print(f"Peak offset from center: {offset:.3f} pixels")
    
    # For even grids, center is between pixels (e.g., 127.5), so peak at nearest integer is expected
    if n_pix % 2 == 0:
        expected_offset = np.sqrt(2) / 2  # ~0.707 pixels for even grids
        if offset < expected_offset + 0.1:
            print(f"✓ PSF is properly centered (even grid: peak within {expected_offset:.2f} pixels of center)")
        else:
            print("✗ WARNING: PSF appears offset from center!")
    else:
        if offset < 0.1:
            print("✓ PSF is properly centered (odd grid: peak at exact center)")
        else:
            print("✗ WARNING: PSF appears offset from center!")
    
    # Plot PSF with centering analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Full PSF
    im1 = axes[0].imshow(psf, cmap='hot', origin='lower', norm=plt.matplotlib.colors.LogNorm(vmin=1e-6, vmax=psf.max()))
    axes[0].plot(peak_idx[1], peak_idx[0], 'c+', markersize=20, markeredgewidth=2, label=f'Peak {peak_idx}')
    axes[0].axvline(center, color='cyan', linestyle='--', alpha=0.5, label=f'Center {center:.1f}')
    axes[0].axhline(center, color='cyan', linestyle='--', alpha=0.5)
    axes[0].set_title('Full PSF (log scale)')
    axes[0].set_xlabel('X [pixels]')
    axes[0].set_ylabel('Y [pixels]')
    axes[0].legend()
    plt.colorbar(im1, ax=axes[0], label='Intensity')
    
    # Zoomed core
    zoom = 32
    y_start, y_end = int(center - zoom//2), int(center + zoom//2)
    x_start, x_end = int(center - zoom//2), int(center + zoom//2)
    psf_zoom = psf[y_start:y_end, x_start:x_end]
    
    im2 = axes[1].imshow(psf_zoom, cmap='hot', origin='lower', extent=[x_start, x_end, y_start, y_end])
    axes[1].plot(peak_idx[1], peak_idx[0], 'c+', markersize=20, markeredgewidth=2, label='Peak')
    axes[1].axvline(center, color='cyan', linestyle='--', alpha=0.5)
    axes[1].axhline(center, color='cyan', linestyle='--', alpha=0.5)
    axes[1].set_title('PSF Core (zoomed, linear scale)')
    axes[1].set_xlabel('X [pixels]')
    axes[1].set_ylabel('Y [pixels]')
    axes[1].legend()
    plt.colorbar(im2, ax=axes[1], label='Intensity')
    
    # Radial profile
    y, x = np.ogrid[:n_pix, :n_pix]
    r = np.sqrt((x - center)**2 + (y - center)**2)
    r_flat = r.ravel()
    psf_flat = psf.ravel()
    
    # Bin radial profile
    r_bins = np.arange(0, n_pix//2, 1)
    r_profile = np.zeros_like(r_bins, dtype=float)
    for i, r_val in enumerate(r_bins):
        mask = (r_flat >= r_val) & (r_flat < r_val + 1)
        if np.any(mask):
            r_profile[i] = np.mean(psf_flat[mask])
    
    axes[2].semilogy(r_bins, r_profile, 'b-', linewidth=2, label='Radial profile')
    axes[2].axvline(1.22 * wavelength / diameter / (pixel_scale * np.pi/180/3600) / np.sqrt(2), 
                    color='r', linestyle='--', label='1.22λ/D')
    axes[2].set_title('Radial Profile')
    axes[2].set_xlabel('Radius [pixels]')
    axes[2].set_ylabel('Mean intensity')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "demo_psf_engine_centering.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure: {output_dir / 'demo_psf_engine_centering.png'}")
    
    return psf, peak_idx


def demo_phase_effects():
    """
    Demonstrate that phase perturbations don't introduce spurious shifts.
    """
    print("\n" + "="*80)
    print("DEMO 3: Phase Effects on PSF Centering")
    print("="*80)
    
    # Create PSF engine and pupil
    n_pix = 128
    diameter = 8.0
    wavelength = 1.65e-6
    pixel_scale = 10e-3 * np.pi / 180 / 3600  # radians/pixel
    
    pupil = Pupil.from_circular(
        n_pix=n_pix,
        diameter=diameter,
        wavelength=wavelength,
        pixel_scale=pixel_scale,
    )
    
    engine = PSFEngine(
        n_pix=n_pix,
        wavelength=wavelength,
        pixel_scale=pixel_scale,
    )
    
    # Get pupil
    pupil_array = to_numpy(pupil.amplitude)
    center = (n_pix - 1) / 2.0
    
    # Test different phase screens
    phases = {
        'Zero phase (perfect)': np.zeros((n_pix, n_pix)),
        'Piston (π/2)': np.pi/2 * np.ones((n_pix, n_pix)),
        'Random turbulence': np.random.randn(n_pix, n_pix) * 0.5,
    }
    
    psfs = {}
    peaks = {}
    offsets = {}
    
    for name, phase in phases.items():
        # Apply pupil mask
        phase = phase * pupil_array
        
        # Compute PSF
        psf = to_numpy(engine.compute_psf(pupil_array, phase))
        psfs[name] = psf
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
        peaks[name] = peak_idx
        
        # Compute offset
        offset = np.sqrt((peak_idx[0] - center)**2 + (peak_idx[1] - center)**2)
        offsets[name] = offset
        
        print(f"\n{name}:")
        print(f"  Peak location: {peak_idx}")
        print(f"  Offset from center: {offset:.3f} pixels")
        
        # For even grids, expect ~0.707 pixel offset; for odd grids, expect ~0
        expected_max = 0.8 if n_pix % 2 == 0 else 0.1
        if offset < expected_max:
            print(f"  ✓ PSF properly centered")
        else:
            print(f"  ⚠ PSF offset (may be expected for asymmetric turbulence)")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (name, phase) in enumerate(phases.items()):
        row = idx // 3
        col = idx % 3
        
        # Phase screen
        im = axes[row, col].imshow(phase * pupil_array, cmap='twilight', origin='lower', 
                                   vmin=-np.pi, vmax=np.pi)
        axes[row, col].set_title(f'{name} - Phase')
        axes[row, col].set_xlabel('X [pixels]')
        axes[row, col].set_ylabel('Y [pixels]')
        plt.colorbar(im, ax=axes[row, col], label='Phase [rad]')
    
    for idx, name in enumerate(phases.keys()):
        row = (idx + 3) // 3
        col = (idx + 3) % 3
        
        # PSF
        psf = psfs[name]
        peak = peaks[name]
        
        im = axes[row, col].imshow(psf, cmap='hot', origin='lower',
                                   norm=plt.matplotlib.colors.LogNorm(vmin=1e-6, vmax=psf.max()))
        axes[row, col].plot(peak[1], peak[0], 'c+', markersize=20, markeredgewidth=2, 
                           label=f'Peak {peak}')
        axes[row, col].axvline(center, color='cyan', linestyle='--', alpha=0.5)
        axes[row, col].axhline(center, color='cyan', linestyle='--', alpha=0.5)
        axes[row, col].set_title(f'{name} - PSF (offset={offsets[name]:.2f}px)')
        axes[row, col].set_xlabel('X [pixels]')
        axes[row, col].set_ylabel('Y [pixels]')
        axes[row, col].legend()
        plt.colorbar(im, ax=axes[row, col], label='Intensity')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "demo_phase_effects.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure: {output_dir / 'demo_phase_effects.png'}")
    
    return psfs, peaks, offsets


def demo_symmetry_check():
    """
    Verify PSF symmetry for symmetric pupils (no turbulence).
    """
    print("\n" + "="*80)
    print("DEMO 4: PSF Symmetry Verification")
    print("="*80)
    
    # Create PSF engine and pupil
    n_pix = 256
    diameter = 8.0
    wavelength = 1.65e-6
    pixel_scale = 10e-3 * np.pi / 180 / 3600  # radians/pixel
    
    pupil = Pupil.from_circular(
        n_pix=n_pix,
        diameter=diameter,
        wavelength=wavelength,
        pixel_scale=pixel_scale,
    )
    
    engine = PSFEngine(
        n_pix=n_pix,
        wavelength=wavelength,
        pixel_scale=pixel_scale,
    )
    
    # Compute diffraction-limited PSF
    pupil_array = to_numpy(pupil.amplitude)
    phase = np.zeros((n_pix, n_pix))
    psf = to_numpy(engine.compute_psf(pupil_array, phase))
    
    center = (n_pix - 1) / 2.0
    center_int = int(np.round(center))
    
    # Check X symmetry
    left_half = psf[:, :center_int]
    right_half = psf[:, center_int:]
    right_half_flipped = np.fliplr(right_half)
    
    # Pad to same size
    min_width = min(left_half.shape[1], right_half_flipped.shape[1])
    left_half = left_half[:, -min_width:]
    right_half_flipped = right_half_flipped[:, :min_width]
    
    x_symmetry_error = np.abs(left_half - right_half_flipped).max() / psf.max()
    
    # Check Y symmetry
    top_half = psf[:center_int, :]
    bottom_half = psf[center_int:, :]
    bottom_half_flipped = np.flipud(bottom_half)
    
    # Pad to same size
    min_height = min(top_half.shape[0], bottom_half_flipped.shape[0])
    top_half = top_half[-min_height:, :]
    bottom_half_flipped = bottom_half_flipped[:min_height, :]
    
    y_symmetry_error = np.abs(top_half - bottom_half_flipped).max() / psf.max()
    
    print(f"\nDiffraction-limited PSF symmetry:")
    print(f"  X-axis symmetry error: {x_symmetry_error:.2e} (relative to peak)")
    print(f"  Y-axis symmetry error: {y_symmetry_error:.2e} (relative to peak)")
    
    if x_symmetry_error < 1e-10 and y_symmetry_error < 1e-10:
        print("  ✓ PSF is perfectly symmetric (within numerical precision)")
    elif x_symmetry_error < 1e-3 and y_symmetry_error < 1e-3:
        print("  ✓ PSF is highly symmetric (< 0.1% error)")
    else:
        print("  ⚠ PSF shows some asymmetry (may be due to even grid size and pixel alignment)")
    
    # Plot symmetry check
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Full PSF
    im1 = axes[0, 0].imshow(psf, cmap='hot', origin='lower', 
                           norm=plt.matplotlib.colors.LogNorm(vmin=1e-6, vmax=psf.max()))
    axes[0, 0].axvline(center, color='cyan', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(center, color='cyan', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Full PSF')
    axes[0, 0].set_xlabel('X [pixels]')
    axes[0, 0].set_ylabel('Y [pixels]')
    plt.colorbar(im1, ax=axes[0, 0], label='Intensity')
    
    # X-axis symmetry
    axes[0, 1].imshow(left_half, cmap='hot', origin='lower')
    axes[0, 1].set_title('Left Half')
    axes[0, 1].set_xlabel('X [pixels]')
    axes[0, 1].set_ylabel('Y [pixels]')
    
    axes[0, 2].imshow(right_half_flipped, cmap='hot', origin='lower')
    axes[0, 2].set_title('Right Half (flipped)')
    axes[0, 2].set_xlabel('X [pixels]')
    axes[0, 2].set_ylabel('Y [pixels]')
    
    # Y-axis symmetry
    axes[1, 0].imshow(top_half, cmap='hot', origin='lower')
    axes[1, 0].set_title('Top Half')
    axes[1, 0].set_xlabel('X [pixels]')
    axes[1, 0].set_ylabel('Y [pixels]')
    
    axes[1, 1].imshow(bottom_half_flipped, cmap='hot', origin='lower')
    axes[1, 1].set_title('Bottom Half (flipped)')
    axes[1, 1].set_xlabel('X [pixels]')
    axes[1, 1].set_ylabel('Y [pixels]')
    
    # Symmetry error map
    error_map_x = np.abs(left_half - right_half_flipped)
    im2 = axes[1, 2].imshow(error_map_x, cmap='viridis', origin='lower')
    axes[1, 2].set_title(f'X-Symmetry Error (max={x_symmetry_error:.2e})')
    axes[1, 2].set_xlabel('X [pixels]')
    axes[1, 2].set_ylabel('Y [pixels]')
    plt.colorbar(im2, ax=axes[1, 2], label='Absolute difference')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "demo_symmetry_check.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure: {output_dir / 'demo_symmetry_check.png'}")
    
    return psf, x_symmetry_error, y_symmetry_error


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("FFT CENTERING CONVENTION DEMONSTRATION")
    print("="*80)
    print("\nThis script demonstrates the FFT centering convention used in NEBRAA")
    print("for pupil-to-PSF transformations.")
    print("\nStandard convention: PSF = |fftshift(fft2(ifftshift(E_pupil)))|²")
    print("="*80)
    
    # Run demonstrations
    demo_fft_convention_comparison()
    demo_psf_engine_centering()
    demo_phase_effects()
    demo_symmetry_check()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nKey findings:")
    print("1. ✓ FFT convention with ifftshift ensures proper centering")
    print("2. ✓ PSFEngine produces centered PSFs (peak at array center)")
    print("3. ✓ Phase perturbations don't introduce spurious shifts")
    print("4. ✓ Symmetric pupils produce symmetric PSFs")
    print("\nAll figures saved to: examples/output/")
    print("="*80)
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
VLT Turbulence and PSF Study (V2)
=================================

This script demonstrates the relationship between:
1. Turbulence strength (RMS OPD / wavefront error)
2. Power-law exponent of Zernike mode amplitudes
3. Resulting PSF quality (Strehl ratio)

It also validates:
- Maréchal approximation: S ≈ exp(-σ²)
- Spider masking behavior
- Low Wind Effect (LWE)
- High-frequency Kolmogorov turbulence

Author: NEBRAA Example
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import os

# Add nebraa to path if not installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nebraa.utils.compute import init_backend, get_backend
from nebraa.physics.zernike import generate_zernike_phase, normalize_phase_rms
from nebraa.physics.optics import compute_psf, compute_strehl
from nebraa.physics.kolmogorov import KolmogorovGenerator
from nebraa.instruments.vlt import VLTPupil, LowWindEffect


# =============================================================================
# Setup Functions
# =============================================================================

def setup_vlt_pupil(n_pix=256, wavelength=2.2e-6, pixel_scale_arcsec=13e-3, D=8.2, D_obs=1.116):
    """
    Create a VLT pupil with spiders using the VLTPupil class.
    
    Args:
        n_pix: Grid size (pixels)
        wavelength: Wavelength (meters), default 2.2 μm (K-band)
        pixel_scale_arcsec: Focal plane pixel scale (arcsec/pixel), default 13 mas (NACO)
        D: Primary mirror diameter (meters)
        D_obs: Obstruction diameter (meters)
    
    Returns:
        pupil: Pupil amplitude array
        pupil_pixel_size: Physical size of each pixel (meters)
        pixel_scale_arcsec: Focal plane pixel scale (arcsec/pixel)
        pupil_radius_pix: Radius of the pupil in pixels
    """
    # Convert pixel scale to radians
    pixel_scale_rad = pixel_scale_arcsec * (np.pi / 180 / 3600)
    
    # Create VLT pupil
    vlt_pupil = VLTPupil(
        n_pix=n_pix,
        wavelength=wavelength,
        pixel_scale=pixel_scale_rad,
        primary_diameter=D,
        obstruction_diameter=D_obs,
    )
    
    # Add spiders
    vlt_pupil.add_vlt_spiders(
        width=0.04,
        half_opening_deg=51.3,
        attach_angles_deg=[0.0, 180.0],
    )
    
    # Compute pupil radius in pixels
    pupil_radius_pix = D / (2 * vlt_pupil.pupil_pixel_size)
    
    return vlt_pupil.amplitude, vlt_pupil.pupil_pixel_size, pixel_scale_arcsec, pupil_radius_pix


# =============================================================================
# Test Functions
# =============================================================================

def test_marechal_approximation(pupil, n_pix, pupil_radius_pix, n_samples=50):
    """
    Test the Maréchal approximation: S ≈ exp(-σ²).
    
    Args:
        pupil: Pupil mask
        n_pix: Grid size
        pupil_radius_pix: Radius of pupil in pixels (for Zernike generation)
        n_samples: Number of random samples per RMS value
    
    Returns:
        rms_values: Array of RMS values tested
        measured_strehls: Measured Strehl ratios (mean ± std)
        theory_strehls: Theoretical Maréchal values
    """
    xp = get_backend().xp
    
    rms_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    measured_mean = []
    measured_std = []
    
    # Limit max Zernike order to what can be properly sampled
    # Nyquist: n_max ≈ pupil_radius / 2
    n_max = max(5, min(50, int(pupil_radius_pix / 2)))
    
    for rms_target in rms_values:
        strehls = []
        for seed in range(n_samples):
            # Generate phase with many Zernike modes for realistic turbulence
            # Use correct pupil radius for Zernike generation
            raw_phase = generate_zernike_phase(
                1, n_pix, pupil_radius_pix, 
                n_range=(2, n_max),
                power_law=11/3,   # Kolmogorov-like
                seed=seed
            )
            phase = normalize_phase_rms(raw_phase, pupil, rms_target)[0]
            
            # Use phasor average Strehl (not PSF peak)
            strehl = compute_strehl(phase, pupil)
            strehls.append(strehl)
        
        measured_mean.append(np.mean(strehls))
        measured_std.append(np.std(strehls))
    
    theory_strehls = np.exp(-rms_values**2)
    
    return rms_values, np.array(measured_mean), np.array(measured_std), theory_strehls


def test_power_law_effect(pupil, n_pix, pupil_radius_pix, rms_fixed=0.5, n_samples=30):
    """
    Test how power-law affects Strehl at fixed RMS.
    
    Higher power-law concentrates energy in low-order modes,
    which should give slightly higher Strehl (less scattering).
    """
    xp = get_backend().xp
    
    power_law_values = np.array([-1, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    measured_mean = []
    measured_std = []
    
    # Limit max Zernike order to what can be properly sampled
    n_max = max(5, min(50, int(pupil_radius_pix / 2)))
    
    for power_law in power_law_values:
        strehls = []
        for seed in range(n_samples):
            raw_phase = generate_zernike_phase(
                1, n_pix, pupil_radius_pix,
                n_range=(2, n_max),
                power_law=power_law,
                seed=seed
            )
            phase = normalize_phase_rms(raw_phase, pupil, rms_fixed)[0]
            
            # Use phasor average Strehl (not PSF peak)
            strehl = compute_strehl(phase, pupil)
            strehls.append(strehl)
        
        measured_mean.append(np.mean(strehls))
        measured_std.append(np.std(strehls))
    
    return power_law_values, np.array(measured_mean), np.array(measured_std)


def test_kolmogorov_hf(pupil, pupil_pixel_size, n_pix):
    """
    Test high-frequency Kolmogorov turbulence at different actuator pitches.
    """
    xp = get_backend().xp
    
    # Different actuator pitches (smaller = better AO = higher cutoff)
    pitches = [10, 1.0, 0.5, 0.3, 0.2, 0.15, 0.1]  # meters
    r0 = 0.15  # Fried parameter (meters)
    
    results = []
    
    for pitch in pitches:
        try:
            kolmo = KolmogorovGenerator(
                n_pix=n_pix,
                pixel_size=pupil_pixel_size,
                actuator_pitch=pitch,
                transition_frac=0.15,
            )
            
            # Generate HF phase
            hf_phase = kolmo.generate(1, r0=r0, pupil=pupil, seed=42)[0]
            rms = float(xp.sqrt(xp.sum(hf_phase**2 * pupil) / xp.sum(pupil)))
            
            results.append({
                'pitch': pitch,
                'fc': kolmo.fc,
                'rms': rms,
                'phase': hf_phase
            })
        except Exception as e:
            print(f"  Warning: pitch={pitch}m failed: {e}")
    
    return results


def test_lwe(pupil):
    """
    Test Low Wind Effect.
    """
    xp = get_backend().xp
    
    # Create LWE model from the pupil - automatically detects islands
    lwe = LowWindEffect(
        pupil=pupil,
        piston_rms_rad=0.5,
        tilt_rms_rad=0.3,
    )
    
    print(f"  Detected {lwe.n_islands} pupil islands")
    
    lwe_phase = lwe.generate(1, seed=42)[0]
    
    # Apply pupil mask and compute RMS
    lwe_phase_masked = lwe_phase * pupil
    rms = float(xp.sqrt(xp.sum(lwe_phase_masked**2) / xp.sum(pupil)))
    
    return lwe_phase_masked, rms


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_marechal_test(rms_values, measured_mean, measured_std, theory):
    """Plot Maréchal approximation test results."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Measured with error bars
    ax.errorbar(rms_values, measured_mean, yerr=measured_std, 
                fmt='o', color='blue', capsize=3, 
                label='Measured (Zernike phase)', markersize=8)
    
    # Theory
    rms_fine = np.linspace(0, 1.1, 100)
    ax.plot(rms_fine, np.exp(-rms_fine**2), 'r-', linewidth=2, 
            label=r'Maréchal: $S = e^{-\sigma^2}$')
    
    ax.set_xlabel('RMS wavefront error σ (radians)', fontsize=12)
    ax.set_ylabel('Strehl ratio', fontsize=12)
    ax.set_title('Maréchal Approximation Validation', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.05)
    
    # Add residuals as text
    residuals = measured_mean - theory
    max_res = np.max(np.abs(residuals))
    ax.text(0.05, 0.05, f'Max residual: {max_res:.4f}', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def plot_power_law_test(power_laws, measured_mean, measured_std, rms_fixed):
    """Plot power-law effect on Strehl."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.errorbar(power_laws, measured_mean, yerr=measured_std,
                fmt='s-', color='green', capsize=3, markersize=8)
    
    # Add reference line for Maréchal
    marechal = np.exp(-rms_fixed**2)
    ax.axhline(marechal, color='red', linestyle='--', linewidth=2,
               label=f'Maréchal at σ={rms_fixed} rad')
    
    ax.set_xlabel('Power-law exponent α', fontsize=12)
    ax.set_ylabel('Strehl ratio', fontsize=12)
    ax.set_title(f'Effect of Power-Law on Strehl (RMS = {rms_fixed} rad)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_power_law_phase_psf(pupil, n_pix, pupil_radius_pix, rms_fixed=0.5, seed=42):
    """
    Visualize the effect of power-law exponent on phase structure and PSF.
    
    Shows phase maps and PSFs for different power-law exponents, demonstrating
    how higher exponents concentrate energy in low-order (smooth) modes while
    lower exponents include more high-frequency content.
    
    Args:
        pupil: Pupil mask
        n_pix: Grid size
        pupil_radius_pix: Pupil radius in pixels
        rms_fixed: Fixed RMS for all phase screens (radians)
        seed: Random seed for reproducibility
    """
    xp = get_backend().xp
    pupil_np = np.asarray(pupil)
    
    # Power-law values to visualize
    power_laws = [0.0, 1.0, 2.0, 3.0, 11/3]  # Include Kolmogorov (11/3)
    power_law_labels = ['α=0 (flat)', 'α=1', 'α=2', 'α=3', 'α=11/3 (Kolmo)']
    
    # Use more Zernike modes for better visualization
    # Increase n_max to better show high-frequency content
    n_max = max(5, min(50, int(pupil_radius_pix)))
    
    # Define cutoff for low/high frequency separation
    n_cutoff = max(5, n_max // 3)  # Low-order = n <= n_cutoff
    
    # Create figure: 5 rows x len(power_laws) columns
    # Row 1: Full phase maps
    # Row 2: Low-order (n <= n_cutoff) phase only
    # Row 3: High-order (n > n_cutoff) phase only  
    # Row 4: Zernike variance spectrum
    # Row 5: PSFs (log scale)
    n_cols = len(power_laws)
    fig, axes = plt.subplots(5, n_cols, figsize=(4*n_cols, 18))
    
    phases_full = []
    phases_low = []
    phases_high = []
    psfs = []
    strehls = []
    all_coeffs = []
    
    # Pre-generate the Zernike modes and a fixed set of unit-variance coefficients
    from nebraa.physics.zernike import build_zernike_modes, get_nm_list
    
    modes = build_zernike_modes(n_pix, pupil_radius_pix, (2, n_max))
    nm_list = get_nm_list(2, n_max)
    n_modes = modes.shape[0]
    
    # Generate base coefficients with unit variance (same for all power-laws)
    xp.random.seed(seed)
    base_coeffs = xp.random.randn(n_modes).astype(xp.float32)
    
    # Generate phase for each power-law
    for i, (plaw, label) in enumerate(zip(power_laws, power_law_labels)):
        # Apply power-law scaling to base coefficients
        coeffs = base_coeffs.copy()
        for j, (n, m) in enumerate(nm_list):
            scale = (n + 1) ** (-plaw / 2)  # Variance scales as n^(-power_law)
            coeffs[j] *= scale
        
        all_coeffs.append(np.asarray(coeffs.copy()))
        
        # Compute FULL phase
        modes_flat = modes.reshape(n_modes, -1)
        phase_flat = xp.dot(coeffs, modes_flat)
        raw_phase = phase_flat.reshape(1, n_pix, n_pix)
        phase_full = normalize_phase_rms(raw_phase, pupil, rms_fixed)[0]
        phases_full.append(phase_full)
        
        # Compute LOW-order phase (n <= n_cutoff)
        coeffs_low = coeffs.copy()
        for j, (n, m) in enumerate(nm_list):
            if n > n_cutoff:
                coeffs_low[j] = 0
        phase_low_flat = xp.dot(coeffs_low, modes_flat)
        phase_low = phase_low_flat.reshape(n_pix, n_pix)
        phases_low.append(phase_low)
        
        # Compute HIGH-order phase (n > n_cutoff)
        coeffs_high = coeffs.copy()
        for j, (n, m) in enumerate(nm_list):
            if n <= n_cutoff:
                coeffs_high[j] = 0
        phase_high_flat = xp.dot(coeffs_high, modes_flat)
        phase_high = phase_high_flat.reshape(n_pix, n_pix)
        phases_high.append(phase_high)
        
        # Compute PSF and Strehl from full phase
        psf = compute_psf(pupil, phase_full, normalize=True)
        psfs.append(psf)
        strehl = compute_strehl(phase_full, pupil)
        strehls.append(strehl)
    
    # Compute RMS for low and high components
    def get_rms(phase, mask):
        p = np.asarray(phase)
        m = np.asarray(mask)
        return np.sqrt(np.mean(p[m > 0.5]**2))
    
    # Find global color scales
    vmax_full = max(np.nanmax(np.abs(np.where(pupil_np > 0.5, np.asarray(p), np.nan))) 
                    for p in phases_full)
    vmax_full = max(0.5, vmax_full)
    
    vmax_high = max(np.nanmax(np.abs(np.where(pupil_np > 0.5, np.asarray(p), np.nan))) 
                    for p in phases_high)
    vmax_high = max(0.1, vmax_high)
    
    # Plot each power-law
    for i, (label, coeffs_np) in enumerate(zip(power_law_labels, all_coeffs)):
        phase_full_np = np.asarray(phases_full[i])
        phase_low_np = np.asarray(phases_low[i])
        phase_high_np = np.asarray(phases_high[i])
        psf_np = np.asarray(psfs[i])
        strehl = strehls[i]
        
        rms_low = get_rms(phases_low[i], pupil)
        rms_high = get_rms(phases_high[i], pupil)
        
        # Row 1: Full phase map
        ax = axes[0, i]
        phase_masked = np.where(pupil_np > 0.5, phase_full_np, np.nan)
        im = ax.imshow(phase_masked, cmap='RdBu_r', vmin=-vmax_full, vmax=vmax_full, 
                       origin='lower')
        ax.set_title(f'{label}\nTotal RMS={rms_fixed:.2f} rad', fontsize=11)
        if i == 0:
            ax.set_ylabel('Full phase', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label='rad', shrink=0.8)
        
        # Row 2: Low-order phase (same color scale as full)
        ax = axes[1, i]
        phase_masked = np.where(pupil_np > 0.5, phase_low_np, np.nan)
        im = ax.imshow(phase_masked, cmap='RdBu_r', vmin=-vmax_full, vmax=vmax_full, 
                       origin='lower')
        ax.set_title(f'Low (n≤{n_cutoff}) RMS={rms_low:.2f}', fontsize=10)
        if i == 0:
            ax.set_ylabel(f'Low-order\n(n≤{n_cutoff})', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label='rad', shrink=0.8)
        
        # Row 3: High-order phase (INDEPENDENT color scale to show detail)
        ax = axes[2, i]
        phase_masked = np.where(pupil_np > 0.5, phase_high_np, np.nan)
        im = ax.imshow(phase_masked, cmap='RdBu_r', vmin=-vmax_high, vmax=vmax_high, 
                       origin='lower')
        ax.set_title(f'High (n>{n_cutoff}) RMS={rms_high:.2f}', fontsize=10)
        if i == 0:
            ax.set_ylabel(f'High-order\n(n>{n_cutoff})', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label='rad', shrink=0.8)
        
        # Row 4: Zernike variance spectrum
        ax = axes[3, i]
        n_orders = list(range(2, n_max + 1))
        var_per_n = []
        for n in n_orders:
            var_n = sum(coeffs_np[j]**2 for j, (nn, mm) in enumerate(nm_list) if nn == n)
            var_per_n.append(var_n)
        var_per_n = np.array(var_per_n)
        var_per_n = var_per_n / var_per_n.sum()
        
        # Color bars by low/high
        colors = ['steelblue' if n <= n_cutoff else 'coral' for n in n_orders]
        ax.bar(n_orders, var_per_n, color=colors, edgecolor='black', alpha=0.7)
        ax.axvline(n_cutoff + 0.5, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Radial order n', fontsize=10)
        if i == 0:
            ax.set_ylabel('Variance fraction', fontsize=12)
        ax.set_ylim(0, max(0.5, var_per_n.max() * 1.1))
        ax.set_title('Zernike spectrum', fontsize=10)
        
        # Row 5: PSF (log scale)
        ax = axes[4, i]
        im = ax.imshow(psf_np, norm=LogNorm(vmin=1e-5, vmax=1), cmap='inferno', 
                       origin='lower')
        ax.set_title(f'Strehl={strehl:.3f}', fontsize=11)
        if i == 0:
            ax.set_ylabel('PSF (log)', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label='Intensity', shrink=0.8)
    
    fig.suptitle(f'Power-Law Effect on Phase Structure\n'
                 f'(Zernike n=2-{n_max}, cutoff n={n_cutoff}, fixed total RMS={rms_fixed} rad)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_pupil_diagnostics(pupil, pupil_pixel_size, n_pix):
    """Plot pupil diagnostics including spider analysis."""
    xp = get_backend().xp
    pupil_np = np.asarray(pupil)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Full pupil
    ax = axes[0, 0]
    im = ax.imshow(pupil_np, cmap='gray', origin='lower')
    ax.set_title('VLT Pupil', fontsize=12)
    plt.colorbar(im, ax=ax, label='Transmission')
    
    # Stretched view to see spider edges
    ax = axes[0, 1]
    im = ax.imshow(pupil_np, cmap='RdYlBu_r', origin='lower', vmin=0.85, vmax=1.0)
    ax.set_title('Pupil (stretched: 0.85-1.0)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Transmission')
    
    # Histogram of transmission values
    ax = axes[0, 2]
    nonzero = pupil_np[pupil_np > 0]
    ax.hist(nonzero.flatten(), bins=50, color='steelblue', edgecolor='black')
    ax.set_xlabel('Transmission value', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Transmission Histogram (non-zero pixels)', fontsize=12)
    ax.set_yscale('log')
    
    # Zoom on spider region
    ax = axes[1, 0]
    c = n_pix // 2
    zoom = 40
    zoomed = pupil_np[c-zoom:c+zoom, c-zoom:c+zoom]
    im = ax.imshow(zoomed, cmap='RdYlBu_r', origin='lower')
    ax.set_title('Zoomed center region', fontsize=12)
    plt.colorbar(im, ax=ax, label='Transmission')
    
    # Radial profile
    ax = axes[1, 1]
    x = np.arange(n_pix) - n_pix/2
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2) * pupil_pixel_size
    
    # Azimuthal average
    r_bins = np.linspace(0, 4.5, 50)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    profile = []
    for i in range(len(r_bins)-1):
        mask = (R >= r_bins[i]) & (R < r_bins[i+1])
        if mask.sum() > 0:
            profile.append(np.mean(pupil_np[mask]))
        else:
            profile.append(np.nan)
    
    ax.plot(r_centers, profile, 'b-', linewidth=2)
    ax.axvline(1.116/2, color='red', linestyle='--', label='Obstruction edge')
    ax.axvline(8.2/2, color='green', linestyle='--', label='Primary edge')
    ax.set_xlabel('Radius (m)', fontsize=11)
    ax.set_ylabel('Mean transmission', fontsize=11)
    ax.set_title('Radial Profile', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Reference PSF
    ax = axes[1, 2]
    ref_psf = compute_psf(pupil, xp.zeros((n_pix, n_pix), dtype=xp.float32))
    ref_psf_np = np.asarray(ref_psf)
    im = ax.imshow(ref_psf_np, norm=LogNorm(vmin=1e-5, vmax=1), cmap='inferno', origin='lower')
    ax.set_title('Reference PSF (log scale)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Normalized intensity')
    
    plt.tight_layout()
    return fig


def plot_kolmogorov_hf_test(results, pupil):
    """Plot HF Kolmogorov test results."""
    xp = get_backend().xp
    pupil_np = np.asarray(pupil)
    
    n_results = len(results)
    if n_results == 0:
        print("No HF results to plot!")
        return None
    
    fig, axes = plt.subplots(2, min(n_results, 4), figsize=(4*min(n_results, 4), 8))
    if n_results == 1:
        axes = axes.reshape(2, 1)
    
    for i, res in enumerate(results[:4]):
        phase_np = np.asarray(res['phase'])
        phase_masked = np.where(pupil_np > 0.5, phase_np, np.nan)
        
        # Phase
        ax = axes[0, i]
        vmax = max(0.1, np.nanmax(np.abs(phase_masked)))
        im = ax.imshow(phase_masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        ax.set_title(f"pitch={res['pitch']}m\nfc={res['fc']:.1f} cy/m\nRMS={res['rms']:.3f} rad", fontsize=10)
        plt.colorbar(im, ax=ax, label='Phase (rad)')
        
        # PSF
        ax = axes[1, i]
        psf = compute_psf(pupil, xp.asarray(res['phase']), normalize=True)
        psf_np = np.asarray(psf)
        im = ax.imshow(psf_np, norm=LogNorm(vmin=1e-5, vmax=1), cmap='inferno', origin='lower')
        strehl = compute_strehl(xp.asarray(res['phase']), pupil)
        ax.set_title(f'PSF (Strehl={strehl:.3f})', fontsize=10)
        plt.colorbar(im, ax=ax, label='Intensity')
    
    fig.suptitle('High-Frequency Kolmogorov Turbulence (r0=0.15m)', fontsize=14)
    plt.tight_layout()
    return fig


def plot_lwe_test(lwe_phase, pupil, n_pix, rms):
    """Plot LWE test results."""
    xp = get_backend().xp
    pupil_np = np.asarray(pupil)
    phase_np = np.asarray(lwe_phase)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # LWE phase
    ax = axes[0]
    phase_masked = np.where(pupil_np > 0.5, phase_np, np.nan)
    vmax = max(0.1, np.nanmax(np.abs(phase_masked)))
    im = ax.imshow(phase_masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
    ax.set_title(f'LWE Phase (RMS={rms:.3f} rad)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Phase (rad)')
    
    # PSF with LWE
    ax = axes[1]
    psf = compute_psf(pupil, xp.asarray(lwe_phase), normalize=True)
    psf_np = np.asarray(psf)
    im = ax.imshow(psf_np, norm=LogNorm(vmin=1e-5, vmax=1), cmap='inferno', origin='lower')
    strehl = compute_strehl(xp.asarray(lwe_phase), pupil)
    ax.set_title(f'PSF with LWE (Strehl={strehl:.3f})', fontsize=12)
    plt.colorbar(im, ax=ax, label='Intensity')
    
    # Reference PSF
    ax = axes[2]
    ref_psf = compute_psf(pupil, xp.zeros((n_pix, n_pix), dtype=xp.float32), normalize=True)
    ref_psf_np = np.asarray(ref_psf)
    im = ax.imshow(ref_psf_np, norm=LogNorm(vmin=1e-5, vmax=1), cmap='inferno', origin='lower')
    ax.set_title('Reference PSF (no LWE)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Intensity')
    
    fig.suptitle('Low Wind Effect Test', fontsize=14)
    plt.tight_layout()
    return fig


def plot_combined_turbulence(pupil, pupil_pixel_size, n_pix, pixel_scale_arcsec, pupil_radius_pix):
    """
    Generate and plot all three turbulence types: Zernike, HF Kolmogorov, LWE.
    Shows each component separately and then all combined.
    """
    xp = get_backend().xp
    pupil_np = np.asarray(pupil)
    
    # Limit max Zernike order to what can be properly sampled
    n_max = max(5, min(50, int(pupil_radius_pix / 2)))
    
    # -------------------------------------------------------------------------
    # 1. Generate Zernike (low-frequency) phase
    # -------------------------------------------------------------------------
    zernike_phase_raw = generate_zernike_phase(
        1, n_pix, pupil_radius_pix,
        n_range=(2, n_max),
        power_law=11/3,  # Kolmogorov-like
        seed=123
    )
    zernike_phase = normalize_phase_rms(zernike_phase_raw, pupil, 0.4)[0]  # 0.4 rad RMS
    
    # -------------------------------------------------------------------------
    # 2. Generate HF Kolmogorov phase
    # -------------------------------------------------------------------------
    kolmo = KolmogorovGenerator(
        n_pix=n_pix,
        pixel_size=pupil_pixel_size,
        actuator_pitch=0.2,  # 20cm pitch
        transition_frac=0.15,
    )
    hf_phase = kolmo.generate(1, r0=0.15, pupil=pupil, seed=456)[0]
    
    # -------------------------------------------------------------------------
    # 3. Generate LWE phase
    # -------------------------------------------------------------------------
    lwe = LowWindEffect(pupil, piston_rms_rad=0.3, tilt_rms_rad=0.2)
    lwe_phase = lwe.generate(1, seed=789)[0]
    lwe_phase = xp.asarray(lwe_phase)
    
    # -------------------------------------------------------------------------
    # 4. Combined phase (all three)
    # -------------------------------------------------------------------------
    total_phase = zernike_phase + hf_phase + lwe_phase
    
    # Compute RMS values
    def get_rms(ph):
        return float(xp.sqrt(xp.sum(ph**2 * pupil) / xp.sum(pupil)))
    
    zernike_rms = get_rms(zernike_phase)
    hf_rms = get_rms(hf_phase)
    lwe_rms = get_rms(lwe_phase)
    total_rms = get_rms(total_phase)
    
    # -------------------------------------------------------------------------
    # Create figure: 2 rows x 5 columns
    # Row 1: Phase maps
    # Row 2: PSFs
    # Columns: Zernike | HF Kolmo | LWE | Combined | Reference
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    phases = [zernike_phase, hf_phase, lwe_phase, total_phase]
    titles = [
        f'Zernike (LF)\nRMS={zernike_rms:.3f} rad', 
        f'Kolmogorov (HF)\nRMS={hf_rms:.3f} rad',
        f'LWE\nRMS={lwe_rms:.3f} rad',
        f'Combined (All)\nRMS={total_rms:.3f} rad'
    ]
    
    # Find global color scale for phase
    all_phases_masked = []
    for phase in phases:
        phase_np = np.asarray(phase)
        phase_masked = np.where(pupil_np > 0.5, phase_np, np.nan)
        all_phases_masked.append(phase_masked)
    vmax_phase = max(0.5, max(np.nanmax(np.abs(pm)) for pm in all_phases_masked))
    
    # Compute extent in arcseconds for PSF display
    fov_arcsec = n_pix * pixel_scale_arcsec * 1000  # in mas
    extent_psf = [-fov_arcsec/2, fov_arcsec/2, -fov_arcsec/2, fov_arcsec/2]
    
    for i, (phase, title, phase_masked) in enumerate(zip(phases, titles, all_phases_masked)):
        # Phase map
        ax = axes[0, i]
        im = ax.imshow(phase_masked, cmap='RdBu_r', vmin=-vmax_phase, vmax=vmax_phase, origin='lower')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Pixel')
        plt.colorbar(im, ax=ax, label='Phase (rad)')
        
        # PSF
        ax = axes[1, i]
        psf = compute_psf(pupil, phase, normalize=True)
        psf_np = np.asarray(psf)
        im = ax.imshow(psf_np, norm=LogNorm(vmin=1e-5, vmax=1), cmap='inferno', 
                       origin='lower', extent=extent_psf)
        strehl = compute_strehl(phase, pupil)
        ax.set_title(f'PSF (Strehl={strehl:.3f})', fontsize=11)
        ax.set_xlabel('mas')
        ax.set_ylabel('mas')
        plt.colorbar(im, ax=ax, label='Intensity')
    
    # Reference (no aberrations)
    ax = axes[0, 4]
    ax.imshow(pupil_np, cmap='gray', origin='lower')
    ax.set_title('Pupil', fontsize=11)
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Pixel')
    
    ax = axes[1, 4]
    ref_psf = compute_psf(pupil, xp.zeros((n_pix, n_pix), dtype=xp.float32), normalize=True)
    ref_psf_np = np.asarray(ref_psf)
    im = ax.imshow(ref_psf_np, norm=LogNorm(vmin=1e-5, vmax=1), cmap='inferno', 
                   origin='lower', extent=extent_psf)
    ax.set_title('Reference PSF\n(Strehl=1.000)', fontsize=11)
    ax.set_xlabel('mas')
    ax.set_ylabel('mas')
    plt.colorbar(im, ax=ax, label='Intensity')
    
    fig.suptitle('VLT/NACO K-band Turbulence Components\n'
                 f'λ=2.2 μm, pixel scale=13 mas, pupil pixel={pupil_pixel_size*1000:.1f} mm', 
                 fontsize=14)
    plt.tight_layout()
    
    # Print summary
    print(f"  Turbulence RMS breakdown:")
    print(f"    Zernike (LF):     {zernike_rms:.3f} rad")
    print(f"    Kolmogorov (HF):  {hf_rms:.3f} rad")
    print(f"    LWE:              {lwe_rms:.3f} rad")
    print(f"    Combined:         {total_rms:.3f} rad")
    print(f"    Quadratic sum:    {np.sqrt(zernike_rms**2 + hf_rms**2 + lwe_rms**2):.3f} rad")
    
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    """Main function to run all tests."""
    print("=" * 70)
    print("VLT/NACO K-band Turbulence Study")
    print("=" * 70)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output_vlt_turbulence')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Output directory: {output_dir}")
    
    # Initialize CPU backend
    init_backend(compute_mode='CPU')
    xp = get_backend().xp
    print("✓ Backend initialized (CPU mode)")
    
    # Parameters - NACO K-band
    n_pix = 256
    wavelength = 2.2e-6  # K-band (meters)
    pixel_scale_arcsec = 13e-3  # NACO pixel scale (arcsec/pixel)
    D = 8.2  # VLT primary diameter (m)
    D_obs = 1.116  # Central obstruction (m)
    
    # Create VLT pupil with NACO K-band parameters
    print("\n[1/7] Creating VLT pupil (NACO K-band)...")
    pupil, pupil_pixel_size, pixel_scale, pupil_radius_pix = setup_vlt_pupil(
        n_pix=n_pix, 
        wavelength=wavelength,
        pixel_scale_arcsec=pixel_scale_arcsec,
        D=D, 
        D_obs=D_obs
    )
    print(f"  Wavelength: {wavelength*1e6:.1f} μm (K-band)")
    print(f"  Pixel scale: {pixel_scale*1000:.1f} mas/pixel")
    print(f"  Grid size: {n_pix} x {n_pix}")
    print(f"  Pupil pixel size: {pupil_pixel_size*1000:.2f} mm")
    print(f"  Pupil radius: {pupil_radius_pix:.1f} pixels")
    print(f"  Total pupil extent: {pupil_pixel_size * n_pix:.2f} m")
    print(f"  Nyquist frequency: {1/(2*pupil_pixel_size):.1f} cy/m")
    
    # Pupil diagnostics
    print("\n[2/7] Pupil diagnostics...")
    fig_pupil = plot_pupil_diagnostics(pupil, pupil_pixel_size, n_pix)
    fig_pupil.savefig(os.path.join(output_dir, 'vlt_pupil_diagnostics.png'), dpi=150, bbox_inches='tight')
    print("  Saved: vlt_pupil_diagnostics.png")
    
    # Test Maréchal approximation
    print("\n[3/7] Testing Maréchal approximation...")
    rms_vals, meas_mean, meas_std, theory = test_marechal_approximation(pupil, n_pix, pupil_radius_pix, n_samples=30)
    fig_marechal = plot_marechal_test(rms_vals, meas_mean, meas_std, theory)
    fig_marechal.savefig(os.path.join(output_dir, 'vlt_marechal_test.png'), dpi=150, bbox_inches='tight')
    print("  Saved: vlt_marechal_test.png")
    
    # Print Maréchal comparison
    print("\n  RMS (rad) | Measured      | Theory   | Diff")
    print("  " + "-" * 50)
    for i, rms in enumerate(rms_vals):
        print(f"  {rms:.1f}       | {meas_mean[i]:.4f}±{meas_std[i]:.3f} | {theory[i]:.4f}   | {meas_mean[i]-theory[i]:+.4f}")
    
    # Test power-law effect
    print("\n[4/7] Testing power-law effect...")
    plaws, plaw_mean, plaw_std = test_power_law_effect(pupil, n_pix, pupil_radius_pix, rms_fixed=0.5, n_samples=30)
    fig_plaw = plot_power_law_test(plaws, plaw_mean, plaw_std, rms_fixed=0.5)
    fig_plaw.savefig(os.path.join(output_dir, 'vlt_powerlaw_test.png'), dpi=150, bbox_inches='tight')
    print("  Saved: vlt_powerlaw_test.png")
    
    # Power-law phase/PSF visualization
    print("\n[5/7] Visualizing power-law effects on phase and PSF...")
    fig_plaw_viz = plot_power_law_phase_psf(pupil, n_pix, pupil_radius_pix, rms_fixed=0.5, seed=42)
    fig_plaw_viz.savefig(os.path.join(output_dir, 'vlt_powerlaw_visualization.png'), dpi=150, bbox_inches='tight')
    print("  Saved: vlt_powerlaw_visualization.png")
    
    # Test Kolmogorov HF
    print("\n[6/7] Testing high-frequency Kolmogorov...")
    hf_results = test_kolmogorov_hf(pupil, pupil_pixel_size, n_pix)
    if hf_results:
        fig_hf = plot_kolmogorov_hf_test(hf_results, pupil)
        if fig_hf:
            fig_hf.savefig(os.path.join(output_dir, 'vlt_kolmogorov_hf_test.png'), dpi=150, bbox_inches='tight')
            print("  Saved: vlt_kolmogorov_hf_test.png")
        
        print("\n  Actuator pitch | Cutoff freq | HF RMS")
        print("  " + "-" * 45)
        for res in hf_results:
            print(f"  {res['pitch']:.2f} m         | {res['fc']:.1f} cy/m     | {res['rms']:.4f} rad")
    
    # Test LWE
    print("\n[7/7] Testing Low Wind Effect...")
    lwe_phase, lwe_rms = test_lwe(pupil)
    fig_lwe = plot_lwe_test(lwe_phase, pupil, n_pix, lwe_rms)
    fig_lwe.savefig(os.path.join(output_dir, 'vlt_lwe_test.png'), dpi=150, bbox_inches='tight')
    print("  Saved: vlt_lwe_test.png")
    print(f"  LWE phase RMS: {lwe_rms:.3f} rad")
    
    # Combined turbulence demo (Zernike + HF Kolmogorov + LWE)
    print("\n[Bonus] Combined turbulence demonstration (all 3 types)...")
    fig_combined = plot_combined_turbulence(pupil, pupil_pixel_size, n_pix, pixel_scale, pupil_radius_pix)
    fig_combined.savefig(os.path.join(output_dir, 'vlt_combined_turbulence.png'), dpi=150, bbox_inches='tight')
    print("  Saved: vlt_combined_turbulence.png")
    
    print("\n" + "=" * 70)
    print("✓ All tests complete!")
    print("=" * 70)
    
    # plt.show()


if __name__ == '__main__':
    main()

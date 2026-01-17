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
from matplotlib import cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os

# Add nebraa to path if not installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nebraa.utils.compute import init_backend, get_backend
from nebraa.physics.zernike import (
    generate_zernike_phase, 
    normalize_phase_rms,
    build_zernike_modes,
    get_nm_list
)
from nebraa.physics.optics import compute_psf
from nebraa.physics.kolmogorov import KolmogorovGenerator, AtmosphereModel
from nebraa.instruments.vlt import VLTPupil, VLTLowWindEffect


# =============================================================================
# Setup Functions
# =============================================================================

def setup_vlt_pupil(n_pix=256, D=8.2, D_obs=1.116, fill_frac=0.9):
    """
    Create a VLT-like pupil with spiders.
    
    Args:
        n_pix: Grid size (pixels)
        D: Primary mirror diameter (meters)
        D_obs: Obstruction diameter (meters)
        fill_frac: Fraction of grid filled by pupil
    
    Returns:
        pupil: Pupil amplitude array
        pupil_pixel_size: Physical size of each pixel (meters)
    """
    backend = get_backend()
    xp = backend.xp
    
    # Physical pupil pixel size
    pupil_pixel_size = D / (fill_frac * n_pix)
    
    # Build annular pupil at high resolution
    upscaling = 16
    n_large = upscaling * n_pix
    
    x = xp.arange(n_large, dtype=xp.float32) * pupil_pixel_size / upscaling
    x = x - xp.mean(x)
    X, Y = xp.meshgrid(x, x)
    R = xp.sqrt(X**2 + Y**2)
    
    # Annular pupil
    pupil_large = ((R > D_obs/2) & (R < D/2)).astype(xp.float32)
    
    # Build spider mask
    spider_width = 0.04  # meters
    half_w = spider_width / 2
    half_opening = np.radians(51.3)
    
    # VLT spider geometry: 2 attachment points at 0° and 180°
    # Each has 2 vanes at ±half_opening from radial
    spider_mask = xp.ones((n_large, n_large), dtype=xp.float32)
    
    for attach_angle in [0.0, np.pi]:
        for sign in [-1, 1]:
            vane_angle = attach_angle + sign * half_opening
            # Direction of vane
            ux = np.cos(vane_angle)
            uy = np.sin(vane_angle)
            
            # For each point, compute perpendicular distance to vane line
            # Line from center along direction (ux, uy)
            perp_dist = xp.abs(-X * uy + Y * ux)
            
            # Points along the positive direction only (from center outward)
            along_dist = X * ux + Y * uy
            
            # Within spider width and in the right half-plane
            in_spider = (perp_dist < half_w) & (along_dist > 0) & (R > D_obs/2) & (R < D/2)
            spider_mask = spider_mask * (~in_spider).astype(xp.float32)
    
    # Apply spider to pupil
    pupil_large = pupil_large * spider_mask
    
    # Rebin to target size
    sh = (n_pix, n_large // n_pix, n_pix, n_large // n_pix)
    pupil = pupil_large.reshape(sh).mean(-1).mean(1)
    
    return pupil, pupil_pixel_size


def compute_strehl(pupil, phase, xp):
    """Compute Strehl ratio from pupil and phase."""
    # Reference PSF (no aberrations)
    ref_psf = compute_psf(pupil, xp.zeros_like(phase), normalize=False)
    ref_peak = float(xp.max(ref_psf))
    
    # Aberrated PSF
    psf = compute_psf(pupil, phase, normalize=False)
    peak = float(xp.max(psf))
    
    return peak / ref_peak


# =============================================================================
# Test Functions
# =============================================================================

def test_marechal_approximation(pupil, n_pix, n_samples=50):
    """
    Test the Maréchal approximation: S ≈ exp(-σ²).
    
    Returns:
        rms_values: Array of RMS values tested
        measured_strehls: Measured Strehl ratios (mean ± std)
        theory_strehls: Theoretical Maréchal values
    """
    xp = get_backend().xp
    
    rms_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    measured_mean = []
    measured_std = []
    
    # Reference PSF
    ref_psf = compute_psf(pupil, xp.zeros((n_pix, n_pix), dtype=xp.float32), normalize=False)
    ref_peak = float(xp.max(ref_psf))
    
    for rms_target in rms_values:
        strehls = []
        for seed in range(n_samples):
            # Generate phase with many Zernike modes for realistic turbulence
            raw_phase = generate_zernike_phase(
                1, n_pix, n_pix / 2, 
                n_range=(2, 20),  # Wide range of modes
                power_law=11/3,   # Kolmogorov-like
                seed=seed
            )
            phase = normalize_phase_rms(raw_phase, pupil, rms_target)[0]
            
            psf = compute_psf(pupil, phase, normalize=False)
            strehls.append(float(xp.max(psf)) / ref_peak)
        
        measured_mean.append(np.mean(strehls))
        measured_std.append(np.std(strehls))
    
    theory_strehls = np.exp(-rms_values**2)
    
    return rms_values, np.array(measured_mean), np.array(measured_std), theory_strehls


def test_power_law_effect(pupil, n_pix, rms_fixed=0.5, n_samples=30):
    """
    Test how power-law affects Strehl at fixed RMS.
    
    Higher power-law concentrates energy in low-order modes,
    which should give slightly higher Strehl (less scattering).
    """
    xp = get_backend().xp
    
    power_law_values = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    measured_mean = []
    measured_std = []
    
    ref_psf = compute_psf(pupil, xp.zeros((n_pix, n_pix), dtype=xp.float32), normalize=False)
    ref_peak = float(xp.max(ref_psf))
    
    for power_law in power_law_values:
        strehls = []
        for seed in range(n_samples):
            raw_phase = generate_zernike_phase(
                1, n_pix, n_pix / 2,
                n_range=(2, 20),
                power_law=power_law,
                seed=seed
            )
            phase = normalize_phase_rms(raw_phase, pupil, rms_fixed)[0]
            
            psf = compute_psf(pupil, phase, normalize=False)
            strehls.append(float(xp.max(psf)) / ref_peak)
        
        measured_mean.append(np.mean(strehls))
        measured_std.append(np.std(strehls))
    
    return power_law_values, np.array(measured_mean), np.array(measured_std)


def test_kolmogorov_hf(pupil, pupil_pixel_size, n_pix):
    """
    Test high-frequency Kolmogorov turbulence at different actuator pitches.
    """
    xp = get_backend().xp
    
    # Different actuator pitches (smaller = better AO = higher cutoff)
    pitches = [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]  # meters
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


def test_lwe(pupil, pupil_pixel_size, n_pix, D=8.2):
    """
    Test Low Wind Effect.
    """
    xp = get_backend().xp
    
    # Need to compute wavelength and pixel_scale from pupil_pixel_size
    # For LWE we just need consistent coordinates, so use nominal values
    wavelength = 2.2e-6
    pixel_scale = wavelength / (n_pix * pupil_pixel_size)
    
    lwe = VLTLowWindEffect(
        n_pix=n_pix,
        wavelength=wavelength,
        pixel_scale=pixel_scale,
        D=D,
        piston_rms_rad=0.5,
        tilt_rms_rad=0.3,
        ar_coeff=0.0,
        half_opening_deg=51.3,
    )
    
    lwe_phase = lwe.generate(1, pupil, seed=42)[0]
    rms = float(xp.sqrt(xp.sum(lwe_phase**2 * pupil) / xp.sum(pupil)))
    
    return lwe_phase, rms


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


def plot_kolmogorov_hf_test(results, pupil, n_pix):
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
        strehl = compute_strehl(pupil, xp.asarray(res['phase']), xp)
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
    strehl = compute_strehl(pupil, xp.asarray(lwe_phase), xp)
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


def plot_combined_turbulence(pupil, pupil_pixel_size, n_pix):
    """
    Generate and plot combined turbulence: LF Zernike + HF Kolmogorov.
    """
    xp = get_backend().xp
    pupil_np = np.asarray(pupil)
    
    # Generate LF phase (Zernike)
    lf_phase_raw = generate_zernike_phase(
        1, n_pix, n_pix / 2,
        n_range=(2, 15),
        power_law=11/3,  # Kolmogorov-like
        seed=123
    )
    lf_phase = normalize_phase_rms(lf_phase_raw, pupil, 0.4)[0]  # 0.4 rad RMS
    
    # Generate HF phase (Kolmogorov)
    kolmo = KolmogorovGenerator(
        n_pix=n_pix,
        pixel_size=pupil_pixel_size,
        actuator_pitch=0.2,  # 20cm pitch
        transition_frac=0.15,
    )
    hf_phase = kolmo.generate(1, r0=0.15, pupil=pupil, seed=456)[0]
    
    # Combined
    total_phase = lf_phase + hf_phase
    
    # Compute RMS values
    def get_rms(ph):
        return float(xp.sqrt(xp.sum(ph**2 * pupil) / xp.sum(pupil)))
    
    lf_rms = get_rms(lf_phase)
    hf_rms = get_rms(hf_phase)
    total_rms = get_rms(total_phase)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    phases = [lf_phase, hf_phase, total_phase]
    titles = [f'LF (Zernike)\nRMS={lf_rms:.3f} rad', 
              f'HF (Kolmogorov)\nRMS={hf_rms:.3f} rad',
              f'Combined\nRMS={total_rms:.3f} rad']
    
    for i, (phase, title) in enumerate(zip(phases, titles)):
        phase_np = np.asarray(phase)
        phase_masked = np.where(pupil_np > 0.5, phase_np, np.nan)
        vmax = max(0.5, np.nanmax(np.abs(phase_masked)))
        
        ax = axes[0, i]
        im = ax.imshow(phase_masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, label='Phase (rad)')
        
        ax = axes[1, i]
        psf = compute_psf(pupil, phase, normalize=True)
        psf_np = np.asarray(psf)
        im = ax.imshow(psf_np, norm=LogNorm(vmin=1e-5, vmax=1), cmap='inferno', origin='lower')
        strehl = compute_strehl(pupil, phase, xp)
        ax.set_title(f'PSF (Strehl={strehl:.3f})', fontsize=11)
        plt.colorbar(im, ax=ax, label='Intensity')
    
    # Reference
    ax = axes[0, 3]
    ax.imshow(pupil_np, cmap='gray', origin='lower')
    ax.set_title('Pupil', fontsize=11)
    
    ax = axes[1, 3]
    ref_psf = compute_psf(pupil, xp.zeros((n_pix, n_pix), dtype=xp.float32), normalize=True)
    ref_psf_np = np.asarray(ref_psf)
    im = ax.imshow(ref_psf_np, norm=LogNorm(vmin=1e-5, vmax=1), cmap='inferno', origin='lower')
    ax.set_title('Reference PSF', fontsize=11)
    plt.colorbar(im, ax=ax, label='Intensity')
    
    fig.suptitle('Combined Turbulence: Low-Frequency (Zernike) + High-Frequency (Kolmogorov)', fontsize=14)
    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    """Main function to run all tests."""
    print("=" * 70)
    print("VLT Turbulence and PSF Study (V2)")
    print("=" * 70)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output_vlt_turbulence')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Output directory: {output_dir}")
    
    # Initialize CPU backend
    init_backend(compute_mode='CPU')
    xp = get_backend().xp
    print("✓ Backend initialized (CPU mode)")
    
    # Parameters
    n_pix = 256
    D = 8.2  # VLT primary diameter (m)
    D_obs = 1.116  # Central obstruction (m)
    
    # Create VLT pupil with correct physical scaling
    print("\n[1/6] Creating VLT pupil...")
    pupil, pupil_pixel_size = setup_vlt_pupil(n_pix, D, D_obs)
    print(f"  Grid size: {n_pix} x {n_pix}")
    print(f"  Pupil pixel size: {pupil_pixel_size*1000:.2f} mm")
    print(f"  Total pupil extent: {pupil_pixel_size * n_pix:.2f} m")
    print(f"  Nyquist frequency: {1/(2*pupil_pixel_size):.1f} cy/m")
    
    # Pupil diagnostics
    print("\n[2/6] Pupil diagnostics...")
    fig_pupil = plot_pupil_diagnostics(pupil, pupil_pixel_size, n_pix)
    fig_pupil.savefig(os.path.join(output_dir, 'vlt_pupil_diagnostics.png'), dpi=150, bbox_inches='tight')
    print("  Saved: vlt_pupil_diagnostics.png")
    
    # Test Maréchal approximation
    print("\n[3/6] Testing Maréchal approximation...")
    rms_vals, meas_mean, meas_std, theory = test_marechal_approximation(pupil, n_pix, n_samples=30)
    fig_marechal = plot_marechal_test(rms_vals, meas_mean, meas_std, theory)
    fig_marechal.savefig(os.path.join(output_dir, 'vlt_marechal_test.png'), dpi=150, bbox_inches='tight')
    print("  Saved: vlt_marechal_test.png")
    
    # Print Maréchal comparison
    print("\n  RMS (rad) | Measured      | Theory   | Diff")
    print("  " + "-" * 50)
    for i, rms in enumerate(rms_vals):
        print(f"  {rms:.1f}       | {meas_mean[i]:.4f}±{meas_std[i]:.3f} | {theory[i]:.4f}   | {meas_mean[i]-theory[i]:+.4f}")
    
    # Test power-law effect
    print("\n[4/6] Testing power-law effect...")
    plaws, plaw_mean, plaw_std = test_power_law_effect(pupil, n_pix, rms_fixed=0.5, n_samples=30)
    fig_plaw = plot_power_law_test(plaws, plaw_mean, plaw_std, rms_fixed=0.5)
    fig_plaw.savefig(os.path.join(output_dir, 'vlt_powerlaw_test.png'), dpi=150, bbox_inches='tight')
    print("  Saved: vlt_powerlaw_test.png")
    
    # Test Kolmogorov HF
    print("\n[5/6] Testing high-frequency Kolmogorov...")
    hf_results = test_kolmogorov_hf(pupil, pupil_pixel_size, n_pix)
    if hf_results:
        fig_hf = plot_kolmogorov_hf_test(hf_results, pupil, n_pix)
        if fig_hf:
            fig_hf.savefig(os.path.join(output_dir, 'vlt_kolmogorov_hf_test.png'), dpi=150, bbox_inches='tight')
            print("  Saved: vlt_kolmogorov_hf_test.png")
        
        print("\n  Actuator pitch | Cutoff freq | HF RMS")
        print("  " + "-" * 45)
        for res in hf_results:
            print(f"  {res['pitch']:.2f} m         | {res['fc']:.1f} cy/m     | {res['rms']:.4f} rad")
    
    # Test LWE
    print("\n[6/6] Testing Low Wind Effect...")
    lwe_phase, lwe_rms = test_lwe(pupil, pupil_pixel_size, n_pix, D)
    fig_lwe = plot_lwe_test(lwe_phase, pupil, n_pix, lwe_rms)
    fig_lwe.savefig(os.path.join(output_dir, 'vlt_lwe_test.png'), dpi=150, bbox_inches='tight')
    print("  Saved: vlt_lwe_test.png")
    print(f"  LWE phase RMS: {lwe_rms:.3f} rad")
    
    # Combined turbulence demo
    print("\n[Bonus] Combined turbulence demonstration...")
    fig_combined = plot_combined_turbulence(pupil, pupil_pixel_size, n_pix)
    fig_combined.savefig(os.path.join(output_dir, 'vlt_combined_turbulence.png'), dpi=150, bbox_inches='tight')
    print("  Saved: vlt_combined_turbulence.png")
    
    print("\n" + "=" * 70)
    print("✓ All tests complete!")
    print("=" * 70)
    
    plt.show()


if __name__ == '__main__':
    main()

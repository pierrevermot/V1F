#!/usr/bin/env python3
"""
VLT Turbulence and PSF Study
============================

This script demonstrates the relationship between:
1. Turbulence strength (RMS OPD / wavefront error)
2. Power-law exponent of Zernike mode amplitudes
3. Resulting PSF quality (Strehl ratio, FWHM)

It generates phase screens and PSFs for various turbulence conditions
and visualizes the results with diagnostic plots.

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
from nebraa.physics.optics import compute_psf, compute_strehl
from nebraa.instruments.vlt import VLTPupil


def setup_vlt_pupil(n_pix=256, wavelength=2.2e-6, pixel_scale=0.0125/206265):
    """
    Create a VLT-like pupil with spiders.
    
    Args:
        n_pix: Grid size (pixels)
        wavelength: Observing wavelength (meters)
        pixel_scale: Pixel scale (radians/pixel)
    
    Returns:
        VLTPupil object
    """
    pupil = VLTPupil(
        n_pix=n_pix,
        wavelength=wavelength,
        pixel_scale=pixel_scale,
        primary_diameter=8.2,
        obstruction_diameter=1.116,
    )
    pupil.add_vlt_spiders(width=0.04)
    return pupil


def generate_turbulence_grid(
    pupil, 
    rms_values, 
    power_law_values, 
    n_pix, 
    n_range=(2, 8),
    seed=42
):
    """
    Generate phase screens for a grid of turbulence parameters.
    
    For each power-law, we generate multiple random phase realizations
    and compute the mean Strehl to get statistically meaningful results.
    
    Args:
        pupil: Pupil mask array
        rms_values: List of RMS wavefront errors (radians)
        power_law_values: List of power-law exponents
        n_pix: Grid size
        n_range: Zernike radial order range
        seed: Random seed
    
    Returns:
        phases: dict mapping (rms, plaw) -> phase array (one example)
        psfs: dict mapping (rms, plaw) -> PSF array (one example)
        strehls: dict mapping (rms, plaw) -> Strehl ratio (averaged)
    """
    xp = get_backend().xp
    
    phases = {}
    psfs = {}
    strehls = {}
    
    # Generate reference PSF (no aberrations)
    zero_phase = xp.zeros((n_pix, n_pix), dtype=xp.float32)
    ref_psf = compute_psf(pupil, zero_phase, normalize=False)
    ref_peak = float(xp.max(ref_psf))
    
    n_samples = 10  # Number of realizations for averaging Strehl
    
    for idx_plaw, power_law in enumerate(power_law_values):
        for idx_rms, rms in enumerate(rms_values):
            strehl_samples = []
            example_phase = None
            example_psf = None
            
            for sample in range(n_samples):
                # Generate phase with unique seed
                sample_seed = seed + idx_plaw * 1000 + idx_rms * 100 + sample
                raw_phase = generate_zernike_phase(
                    n_screens=1,
                    n_pix=n_pix,
                    radius=n_pix / 2,
                    n_range=n_range,
                    power_law=power_law,
                    seed=sample_seed
                )
                
                # Normalize to target RMS
                phase = normalize_phase_rms(raw_phase, pupil, xp.array([rms]))[0]
                
                # Compute PSF
                psf = compute_psf(pupil, phase, normalize=False)
                
                # Compute Strehl ratio
                strehl = float(xp.max(psf)) / ref_peak
                strehl_samples.append(strehl)
                
                # Keep first sample as example for plotting
                if sample == 0:
                    example_phase = phase
                    example_psf = psf / xp.max(psf)
            
            phases[(rms, power_law)] = example_phase
            psfs[(rms, power_law)] = example_psf
            strehls[(rms, power_law)] = np.mean(strehl_samples)
    
    return phases, psfs, strehls


def plot_phase_psf_grid(phases, psfs, strehls, rms_values, power_law_values, pupil):
    """
    Create a grid plot of phase screens and PSFs.
    """
    xp = get_backend().xp
    
    n_rms = len(rms_values)
    n_plaw = len(power_law_values)
    
    fig, axes = plt.subplots(
        n_rms * 2, n_plaw, 
        figsize=(3 * n_plaw, 3 * n_rms),
        squeeze=False
    )
    
    for j, power_law in enumerate(power_law_values):
        for i, rms in enumerate(rms_values):
            phase = phases[(rms, power_law)]
            psf = psfs[(rms, power_law)]
            strehl = strehls[(rms, power_law)]
            
            # Convert to numpy for plotting
            if hasattr(phase, 'get'):
                phase = phase.get()
                psf = psf.get()
                pupil_np = pupil.get()
            else:
                phase = np.asarray(phase)
                psf = np.asarray(psf)
                pupil_np = np.asarray(pupil)
            
            # Phase subplot (top row for this rms)
            ax_phase = axes[2*i, j]
            phase_masked = np.where(pupil_np > 0.5, phase, np.nan)
            vmax = max(abs(np.nanmin(phase_masked)), abs(np.nanmax(phase_masked)))
            im1 = ax_phase.imshow(phase_masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax_phase.set_xticks([])
            ax_phase.set_yticks([])
            
            if i == 0:
                ax_phase.set_title(f'Power law = {power_law:.1f}', fontsize=10)
            if j == 0:
                ax_phase.set_ylabel(f'RMS = {rms:.2f} rad\nPhase', fontsize=9)
            
            # PSF subplot (bottom row for this rms)
            ax_psf = axes[2*i + 1, j]
            # Use log scale for PSF
            psf_plot = np.maximum(psf, 1e-6)
            im2 = ax_psf.imshow(psf_plot, norm=LogNorm(vmin=1e-4, vmax=1), cmap='inferno')
            ax_psf.set_xticks([])
            ax_psf.set_yticks([])
            
            # Add Strehl annotation
            ax_psf.text(
                0.05, 0.95, f'S={strehl:.3f}', 
                transform=ax_psf.transAxes,
                fontsize=8, color='white', va='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
            )
            
            if j == 0:
                ax_psf.set_ylabel('PSF', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_strehl_heatmap(strehls, rms_values, power_law_values):
    """
    Create a heatmap of Strehl ratios.
    """
    n_rms = len(rms_values)
    n_plaw = len(power_law_values)
    
    # Build Strehl matrix
    strehl_matrix = np.zeros((n_rms, n_plaw))
    for i, rms in enumerate(rms_values):
        for j, plaw in enumerate(power_law_values):
            strehl_matrix[i, j] = strehls[(rms, plaw)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(
        strehl_matrix, 
        aspect='auto', 
        cmap='viridis',
        origin='lower',
        extent=[
            power_law_values[0] - 0.25, power_law_values[-1] + 0.25,
            rms_values[0] - 0.125, rms_values[-1] + 0.125
        ]
    )
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Strehl Ratio', fontsize=11)
    
    # Labels
    ax.set_xlabel('Power-law exponent', fontsize=12)
    ax.set_ylabel('RMS wavefront error (rad)', fontsize=12)
    ax.set_title('Strehl Ratio vs Turbulence Parameters', fontsize=14)
    
    # Add contour lines
    X, Y = np.meshgrid(power_law_values, rms_values)
    contours = ax.contour(X, Y, strehl_matrix, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='white', linewidths=0.8)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    
    # Add Maréchal approximation line (S ≈ exp(-σ²) where σ is RMS in radians)
    # σ² = -ln(S) -> σ = sqrt(-ln(S))
    rms_marechal = np.linspace(0.1, 2.0, 100)
    strehl_marechal = np.exp(-rms_marechal**2)
    
    plt.tight_layout()
    return fig


def plot_strehl_vs_rms(strehls, rms_values, power_law_values):
    """
    Plot Strehl ratio vs RMS for different power laws.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = cm.coolwarm(np.linspace(0, 1, len(power_law_values)))
    
    for j, (plaw, color) in enumerate(zip(power_law_values, colors)):
        strehls_plaw = [strehls[(rms, plaw)] for rms in rms_values]
        ax.plot(rms_values, strehls_plaw, 'o-', color=color, 
                label=f'α = {plaw:.1f}', markersize=6, linewidth=1.5)
    
    # Add Maréchal approximation
    rms_theory = np.linspace(0.1, max(rms_values), 100)
    strehl_theory = np.exp(-rms_theory**2)
    ax.plot(rms_theory, strehl_theory, 'k--', linewidth=2, alpha=0.7, 
            label='Maréchal: $S = e^{-\\sigma^2}$')
    
    ax.set_xlabel('RMS wavefront error (radians)', fontsize=12)
    ax.set_ylabel('Strehl Ratio', fontsize=12)
    ax.set_title('Strehl Ratio vs Turbulence Strength', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(rms_values) * 1.1)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    return fig


def plot_zernike_amplitude_distribution(power_law_values, n_range=(2, 8)):
    """
    Plot the amplitude distribution of Zernike modes for different power laws.
    """
    nm_list = get_nm_list(n_range[0], n_range[1])
    n_modes = len(nm_list)
    radial_orders = [n for n, m in nm_list]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = cm.coolwarm(np.linspace(0, 1, len(power_law_values)))
    
    for plaw, color in zip(power_law_values, colors):
        amplitudes = [(n + 1) ** (-plaw / 2) for n, m in nm_list]
        ax1.plot(range(n_modes), amplitudes, 'o-', color=color, 
                 label=f'α = {plaw:.1f}', markersize=4, linewidth=1)
    
    ax1.set_xlabel('Mode index', fontsize=11)
    ax1.set_ylabel('Relative amplitude (a.u.)', fontsize=11)
    ax1.set_title('Zernike Mode Amplitudes vs Mode Index', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot vs radial order
    unique_n = sorted(set(radial_orders))
    for plaw, color in zip(power_law_values, colors):
        amp_per_n = [(n + 1) ** (-plaw / 2) for n in unique_n]
        ax2.plot(unique_n, amp_per_n, 'o-', color=color, 
                 label=f'α = {plaw:.1f}', markersize=6, linewidth=1.5)
    
    ax2.set_xlabel('Radial order n', fontsize=11)
    ax2.set_ylabel('Amplitude ~ $(n+1)^{-\\alpha/2}$', fontsize=11)
    ax2.set_title('Amplitude Scaling with Radial Order', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    return fig


def plot_example_psf_cuts(psfs, rms_values, power_law_values):
    """
    Plot 1D cuts through PSFs for comparison.
    """
    xp = get_backend().xp
    
    # Pick a few representative cases
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Fixed power law, varying RMS
    ax1 = axes[0]
    plaw_fixed = power_law_values[len(power_law_values)//2]  # Middle power law
    colors = cm.plasma(np.linspace(0.1, 0.9, len(rms_values)))
    
    for rms, color in zip(rms_values, colors):
        psf = psfs[(rms, plaw_fixed)]
        if hasattr(psf, 'get'):
            psf = psf.get()
        n = psf.shape[0]
        cut = psf[n//2, :]
        x = np.arange(n) - n//2
        ax1.semilogy(x, cut, '-', color=color, label=f'RMS = {rms:.1f} rad', linewidth=1.5)
    
    ax1.set_xlabel('Position (pixels)', fontsize=11)
    ax1.set_ylabel('Normalized intensity', fontsize=11)
    ax1.set_title(f'PSF cuts (power law = {plaw_fixed:.1f})', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(1e-5, 1.5)
    
    # Fixed RMS, varying power law
    ax2 = axes[1]
    rms_fixed = rms_values[len(rms_values)//2]  # Middle RMS
    colors = cm.coolwarm(np.linspace(0, 1, len(power_law_values)))
    
    for plaw, color in zip(power_law_values, colors):
        psf = psfs[(rms_fixed, plaw)]
        if hasattr(psf, 'get'):
            psf = psf.get()
        n = psf.shape[0]
        cut = psf[n//2, :]
        x = np.arange(n) - n//2
        ax2.semilogy(x, cut, '-', color=color, label=f'α = {plaw:.1f}', linewidth=1.5)
    
    ax2.set_xlabel('Position (pixels)', fontsize=11)
    ax2.set_ylabel('Normalized intensity', fontsize=11)
    ax2.set_title(f'PSF cuts (RMS = {rms_fixed:.1f} rad)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-50, 50)
    ax2.set_ylim(1e-5, 1.5)
    
    plt.tight_layout()
    return fig


def main():
    """Main function to run the turbulence study."""
    print("=" * 60)
    print("VLT Turbulence and PSF Study")
    print("=" * 60)
    
    # Initialize CPU backend
    init_backend(compute_mode='CPU')
    print("\n✓ Backend initialized (CPU mode)")
    
    # Parameters
    n_pix = 256
    wavelength = 2.2e-6  # K-band
    pixel_scale = 0.0125 / 206265  # 12.5 mas in radians
    
    # Turbulence parameter ranges
    rms_values = [0.25, 0.5, 0.75, 1.0, 1.5]  # radians
    power_law_values = [1.0, 1.5, 2.0, 2.5, 3.0]  # exponents
    
    # Zernike modes range (excluding piston and tip-tilt)
    n_range = (2, 10)  # radial orders 2 to 9
    
    print(f"\nSimulation parameters:")
    print(f"  Grid size: {n_pix} x {n_pix}")
    print(f"  Wavelength: {wavelength*1e6:.1f} μm")
    print(f"  Pixel scale: {pixel_scale*206265*1000:.1f} mas")
    print(f"  Zernike orders: {n_range[0]} to {n_range[1]-1}")
    print(f"  RMS values: {rms_values} rad")
    print(f"  Power-law exponents: {power_law_values}")
    
    # Create VLT pupil
    print("\n[1/4] Creating VLT pupil...")
    vlt_pupil = setup_vlt_pupil(n_pix, wavelength, pixel_scale)
    pupil = vlt_pupil.amplitude
    print(f"  Pupil shape: {pupil.shape}")
    
    # Generate turbulence grid
    print("\n[2/4] Generating phase screens and PSFs...")
    phases, psfs, strehls = generate_turbulence_grid(
        pupil, rms_values, power_law_values, n_pix, n_range, seed=42
    )
    print(f"  Generated {len(phases)} phase/PSF pairs")
    
    # Print Strehl summary
    print("\n[3/4] Strehl ratio summary:")
    print(f"  {'RMS (rad)':<12} " + " ".join([f"α={p:.1f}" for p in power_law_values]))
    print("  " + "-" * (12 + 8 * len(power_law_values)))
    for rms in rms_values:
        row = f"  {rms:<12.2f} "
        for plaw in power_law_values:
            row += f"{strehls[(rms, plaw)]:.4f}  "
        print(row)
    
    # Create plots
    print("\n[4/4] Creating diagnostic plots...")
    
    # Plot 1: Phase and PSF grid
    fig1 = plot_phase_psf_grid(phases, psfs, strehls, rms_values, power_law_values, pupil)
    fig1.suptitle('Phase Screens and PSFs\n(VLT-like, K-band)', fontsize=14, y=1.02)
    fig1.savefig('vlt_phase_psf_grid.png', dpi=150, bbox_inches='tight')
    print("  Saved: vlt_phase_psf_grid.png")
    
    # Plot 2: Strehl heatmap
    fig2 = plot_strehl_heatmap(strehls, rms_values, power_law_values)
    fig2.savefig('vlt_strehl_heatmap.png', dpi=150, bbox_inches='tight')
    print("  Saved: vlt_strehl_heatmap.png")
    
    # Plot 3: Strehl vs RMS
    fig3 = plot_strehl_vs_rms(strehls, rms_values, power_law_values)
    fig3.savefig('vlt_strehl_vs_rms.png', dpi=150, bbox_inches='tight')
    print("  Saved: vlt_strehl_vs_rms.png")
    
    # Plot 4: Zernike amplitude distribution
    fig4 = plot_zernike_amplitude_distribution(power_law_values, n_range)
    fig4.savefig('vlt_zernike_amplitudes.png', dpi=150, bbox_inches='tight')
    print("  Saved: vlt_zernike_amplitudes.png")
    
    # Plot 5: PSF cuts
    fig5 = plot_example_psf_cuts(psfs, rms_values, power_law_values)
    fig5.savefig('vlt_psf_cuts.png', dpi=150, bbox_inches='tight')
    print("  Saved: vlt_psf_cuts.png")
    
    # Show all plots
    print("\n✓ Done! Displaying plots...")
    plt.show()
    
    return phases, psfs, strehls


if __name__ == '__main__':
    phases, psfs, strehls = main()

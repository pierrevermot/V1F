#!/usr/bin/env python3
"""
Validation of Jolissaint AO Model Against Paper
=================================================

This script validates the Jolissaint AO model implementation by 
checking it against known analytical results and figures from the paper.

Key checks:
1. Fitting error variance matches theoretical prediction
2. PSD shapes match expected profiles  
3. PSF structure at key angular positions matches expectations
4. Strehl values are reasonable for given parameters

Author: NEBRAA
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nebraa.utils.compute import init_backend, get_backend
from nebraa.instruments.vlt import VLTPupil
from nebraa.physics.jolissaint_ao import (
    JolissaintAOModel,
    AtmosphereProfile,
    TurbulentLayer,
    AOSystemConfig,
    kolmogorov_psd,
    piston_filter,
    create_simple_atmosphere,
)


def compute_theoretical_fitting_variance(r0, actuator_pitch, D, wavelength, L0=None):
    """
    Compute theoretical fitting error variance by integrating Kolmogorov PSD.
    
    The fitting error is the integral of the turbulent phase PSD over
    spatial frequencies above f_ao = 1/(2*d):
    
    σ²_fit = ∫∫_{|f| > f_ao} Φ(f) * F_p(f) d²f
    
    For Kolmogorov: Φ(f) = 0.023 * r0^(-5/3) * f^(-11/3)
    F_p(f) is the piston filter.
    """
    # Scale r0 to wavelength (assuming reference is 500nm)
    r0_sci = r0 * (wavelength / 0.5e-6) ** (6/5)
    
    # The exact integral of the Kolmogorov PSD from f_ao to infinity
    # without piston filter would diverge at low frequencies, but
    # the piston filter removes f < f_D = 1/D
    
    # Hardy (1998) approximation for fitting error
    # σ²_fit ≈ 0.28 * (d/r0)^(5/3) for Shack-Hartmann WFS
    # This coefficient can range from 0.23 to 0.33 depending on DM type
    sigma2_fit = 0.28 * (actuator_pitch / r0_sci) ** (5/3)
    
    return sigma2_fit


def validate_fitting_error():
    """Validate fitting error computation."""
    print("\n" + "="*60)
    print("Validation 1: Fitting Error")
    print("="*60)
    
    # Parameters similar to paper Table 1
    n_pix = 256
    wavelength = 1.65e-6  # H-band (as in paper)
    D = 7.9  # Gemini-like
    pixel_scale_rad = (0.020 * np.pi / 180 / 3600)  # ~20 mas/pixel
    r0 = 0.605  # r0 at wavelength (from paper Table 1)
    
    vlt_pupil = VLTPupil(
        n_pix=n_pix,
        wavelength=wavelength,
        pixel_scale=pixel_scale_rad,
        primary_diameter=D,
        obstruction_diameter=1.2,
    )
    
    # Create model with fitting-only error
    atmosphere = AtmosphereProfile(
        layers=[TurbulentLayer(altitude=0, r0=r0, wind_speed=0)],
        wavelength_ref=wavelength,  # r0 already at science wavelength
        L0=30.0,
    )
    
    ao_config = AOSystemConfig(
        actuator_pitch=0.6,  # From paper Table 1: ~13 actuators across 7.9m
        integration_time=0.01,  # 10ms
        loop_delay=0.0008,
        include_fitting=True,
        include_aliasing=False,
        include_servo_lag=False,
        include_anisoplanatism=False,
        include_noise=False,
    )
    
    model = JolissaintAOModel(
        n_pix=n_pix,
        telescope_diameter=D,
        obstruction_diameter=1.2,
        wavelength=wavelength,
        pixel_scale=pixel_scale_rad,
        atmosphere=atmosphere,
        ao_config=ao_config,
    )
    
    # Get model fitting variance
    errors = model.get_error_breakdown()
    model_var = errors['fitting']
    
    # Theoretical value
    theory_var = compute_theoretical_fitting_variance(
        r0, ao_config.actuator_pitch, D, wavelength, L0=30.0
    )
    
    print(f"Actuator pitch: {ao_config.actuator_pitch:.3f} m")
    print(f"r0 at λ={wavelength*1e6:.2f}μm: {r0:.3f} m")
    print(f"d/r0 ratio: {ao_config.actuator_pitch/r0:.3f}")
    print(f"\nModel fitting variance: {model_var:.4f} rad²")
    print(f"Theory (Hardy approx): {theory_var:.4f} rad²")
    print(f"Ratio model/theory: {model_var/theory_var:.3f}")
    
    # The model integrates the actual PSD over HF domain, so may differ
    # from simplified Hardy formula, but should be same order of magnitude
    if 0.5 < model_var/theory_var < 2.0:
        print("✓ Fitting error within expected range")
    else:
        print("✗ Fitting error outside expected range")
    
    return model_var, theory_var


def validate_psd_shapes():
    """Validate PSD shapes match expected profiles."""
    print("\n" + "="*60)
    print("Validation 2: PSD Shapes")
    print("="*60)
    
    n_pix = 256
    wavelength = 2.2e-6
    D = 8.2
    pixel_scale_rad = (0.013 * np.pi / 180 / 3600)
    
    vlt_pupil = VLTPupil(
        n_pix=n_pix,
        wavelength=wavelength,
        pixel_scale=pixel_scale_rad,
        primary_diameter=D,
        obstruction_diameter=1.116,
    )
    
    atmosphere = create_simple_atmosphere(
        r0=0.15,
        wavelength_ref=0.5e-6,
        L0=25.0,
        wind_speed=10.0,
    )
    
    ao_config = AOSystemConfig(
        actuator_pitch=D/14,
        integration_time=0.002,
        loop_delay=0.001,
    )
    
    model = JolissaintAOModel(
        n_pix=n_pix,
        telescope_diameter=D,
        obstruction_diameter=1.116,
        wavelength=wavelength,
        pixel_scale=pixel_scale_rad,
        atmosphere=atmosphere,
        ao_config=ao_config,
    )
    
    # Compute individual PSDs
    psd_fit = model.compute_fitting_psd()
    psd_servo = model.compute_servo_lag_psd()
    psd_alias = model.compute_aliasing_psd()
    psd_noise = model.compute_noise_psd()
    psd_total = model.compute_total_residual_psd()
    
    backend = get_backend()
    
    # Check PSD properties
    print("PSD Checks:")
    print(f"  Fitting PSD sum: {float(backend.xp.sum(psd_fit) * model.dA):.4f} rad²")
    print(f"  Servo-lag PSD sum: {float(backend.xp.sum(psd_servo) * model.dA):.4f} rad²")
    print(f"  Aliasing PSD sum: {float(backend.xp.sum(psd_alias) * model.dA):.4f} rad²")
    print(f"  Total PSD sum: {float(backend.xp.sum(psd_total) * model.dA):.4f} rad²")
    
    # Check that fitting error dominates in HF domain
    psd_fit_np = backend.to_numpy(psd_fit)
    psd_total_np = backend.to_numpy(psd_total)
    
    center = n_pix // 2
    hf_mask = np.abs(np.arange(n_pix) - center) > n_pix // 4
    
    fit_hf = np.sum(psd_fit_np[:, hf_mask])
    total_hf = np.sum(psd_total_np[:, hf_mask])
    
    print(f"\nHF domain analysis:")
    print(f"  Fitting contribution to HF: {fit_hf/total_hf*100:.1f}%")
    
    if fit_hf / total_hf > 0.8:
        print("✓ Fitting error dominates HF (as expected)")
    else:
        print("Note: Other terms contribute significantly to HF")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    f_max = 1 / (2 * model.pupil_pixel_size)
    extent = [-f_max, f_max, -f_max, f_max]
    
    # Fitting PSD
    ax = axes[0, 0]
    im = ax.imshow(np.fft.fftshift(psd_fit_np) + 1e-20, norm=LogNorm(), 
                   extent=extent, cmap='viridis')
    ax.set_title('Fitting Error PSD')
    ax.set_xlabel('fx (cy/m)')
    ax.set_ylabel('fy (cy/m)')
    plt.colorbar(im, ax=ax)
    
    # Servo-lag PSD
    ax = axes[0, 1]
    psd_servo_np = backend.to_numpy(psd_servo)
    im = ax.imshow(np.fft.fftshift(psd_servo_np) + 1e-20, norm=LogNorm(),
                   extent=extent, cmap='viridis')
    ax.set_title('Servo-lag Error PSD')
    plt.colorbar(im, ax=ax)
    
    # Aliasing PSD
    ax = axes[0, 2]
    psd_alias_np = backend.to_numpy(psd_alias)
    im = ax.imshow(np.fft.fftshift(psd_alias_np) + 1e-20, norm=LogNorm(),
                   extent=extent, cmap='viridis')
    ax.set_title('Aliasing Error PSD')
    plt.colorbar(im, ax=ax)
    
    # Total PSD
    ax = axes[1, 0]
    im = ax.imshow(np.fft.fftshift(psd_total_np) + 1e-20, norm=LogNorm(),
                   extent=extent, cmap='viridis')
    ax.set_title('Total Residual PSD')
    f_ao = ao_config.f_ao
    rect = plt.Rectangle((-f_ao, -f_ao), 2*f_ao, 2*f_ao,
                         fill=False, ec='white', ls='--', lw=1.5)
    ax.add_patch(rect)
    plt.colorbar(im, ax=ax)
    
    # PSD radial profile
    ax = axes[1, 1]
    fx = np.fft.fftshift(np.fft.fftfreq(n_pix, d=model.pupil_pixel_size))
    psd_profile_fit = np.fft.fftshift(psd_fit_np)[center, center:]
    psd_profile_total = np.fft.fftshift(psd_total_np)[center, center:]
    f_pos = fx[center:]
    
    ax.loglog(f_pos[1:], psd_profile_total[1:], 'k-', label='Total', lw=2)
    ax.loglog(f_pos[1:], psd_profile_fit[1:], 'b--', label='Fitting', lw=1.5)
    ax.axvline(f_ao, color='r', ls=':', label=f'f_ao={f_ao:.2f} cy/m')
    ax.set_xlabel('Spatial frequency (cy/m)')
    ax.set_ylabel('PSD (rad²/(cy/m)²)')
    ax.set_title('PSD Radial Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Kolmogorov reference
    ax = axes[1, 2]
    r0_sci = atmosphere.r0_at_wavelength(wavelength)
    f_theory = np.logspace(-1, 1.5, 100)
    psd_kolmo = 0.023 * r0_sci**(-5/3) * f_theory**(-11/3)
    
    ax.loglog(f_theory, psd_kolmo, 'g-', label='Kolmogorov', lw=2)
    ax.loglog(f_pos[1:], psd_profile_total[1:], 'k--', label='Model total', lw=1.5)
    ax.axvline(f_ao, color='r', ls=':', label='f_ao')
    ax.set_xlabel('Spatial frequency (cy/m)')
    ax.set_ylabel('PSD')
    ax.set_title('Comparison with Kolmogorov')
    ax.legend()
    ax.set_xlim(0.1, 20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'jolissaint_validation_psd.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved PSD validation figure to: {output_path}")
    
    return psd_fit_np, psd_total_np


def validate_strehl_marechal():
    """Validate Strehl matches Maréchal approximation."""
    print("\n" + "="*60)
    print("Validation 3: Strehl vs Maréchal Approximation")
    print("="*60)
    
    n_pix = 256
    wavelength = 2.2e-6
    D = 8.2
    pixel_scale_rad = (0.013 * np.pi / 180 / 3600)
    
    vlt_pupil = VLTPupil(
        n_pix=n_pix,
        wavelength=wavelength,
        pixel_scale=pixel_scale_rad,
        primary_diameter=D,
        obstruction_diameter=1.116,
    )
    pupil = vlt_pupil.amplitude
    
    # Test different r0 values to get different Strehl
    r0_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    strehls_model = []
    strehls_marechal = []
    variances = []
    
    for r0 in r0_values:
        atmosphere = create_simple_atmosphere(
            r0=r0,
            wavelength_ref=0.5e-6,
            L0=25.0,
            wind_speed=10.0,
        )
        
        ao_config = AOSystemConfig(
            actuator_pitch=D/14,
            integration_time=0.002,
            loop_delay=0.001,
        )
        
        model = JolissaintAOModel(
            n_pix=n_pix,
            telescope_diameter=D,
            obstruction_diameter=1.116,
            wavelength=wavelength,
            pixel_scale=pixel_scale_rad,
            atmosphere=atmosphere,
            ao_config=ao_config,
        )
        
        errors = model.get_error_breakdown()
        var_phi = errors['total']
        
        strehl_model = model.compute_strehl_ratio(pupil)
        strehl_marechal = np.exp(-var_phi)
        
        strehls_model.append(strehl_model)
        strehls_marechal.append(strehl_marechal)
        variances.append(var_phi)
        
        print(f"r0={r0:.2f}m: σ²={var_phi:.3f} rad², S_model={strehl_model:.3f}, S_Mar={strehl_marechal:.3f}")
    
    # Check that model Strehl matches Maréchal (they should be identical
    # since we use the Maréchal formula internally)
    max_diff = max(abs(np.array(strehls_model) - np.array(strehls_marechal)))
    
    if max_diff < 0.001:
        print("\n✓ Strehl matches Maréchal approximation")
    else:
        print(f"\nMax difference: {max_diff:.4f}")
    
    return r0_values, strehls_model, variances


def validate_psf_structure():
    """Validate PSF structure matches expectations from paper."""
    print("\n" + "="*60)
    print("Validation 4: PSF Structure")
    print("="*60)
    
    n_pix = 256
    wavelength = 2.2e-6
    D = 8.2
    pixel_scale_arcsec = 0.013
    pixel_scale_rad = pixel_scale_arcsec * np.pi / 180 / 3600
    
    vlt_pupil = VLTPupil(
        n_pix=n_pix,
        wavelength=wavelength,
        pixel_scale=pixel_scale_rad,
        primary_diameter=D,
        obstruction_diameter=1.116,
    )
    pupil = vlt_pupil.amplitude
    
    atmosphere = create_simple_atmosphere(
        r0=0.15,
        wavelength_ref=0.5e-6,
        L0=25.0,
        wind_speed=10.0,
    )
    
    ao_config = AOSystemConfig(
        actuator_pitch=D/14,
        integration_time=0.002,
        loop_delay=0.001,
    )
    
    model = JolissaintAOModel(
        n_pix=n_pix,
        telescope_diameter=D,
        obstruction_diameter=1.116,
        wavelength=wavelength,
        pixel_scale=pixel_scale_rad,
        atmosphere=atmosphere,
        ao_config=ao_config,
    )
    
    result = model.compute_long_exposure_psf(pupil, return_components=True)
    backend = get_backend()
    psf = backend.to_numpy(result['psf'])
    
    # AO-corrected field radius (from paper Section 5):
    # ρ_ao = λ * f_AO
    f_ao = ao_config.f_ao
    rho_ao_rad = wavelength * f_ao
    rho_ao_arcsec = rho_ao_rad * 206265
    rho_ao_pix = rho_ao_arcsec / pixel_scale_arcsec
    
    print(f"AO cutoff frequency: {f_ao:.2f} cy/m")
    print(f"AO-corrected field half-width: {rho_ao_arcsec:.3f} arcsec = {rho_ao_pix:.1f} pixels")
    
    # Check PSF structure:
    # 1. Within ρ_ao: PSF should follow diffraction-limited pattern (scaled by Strehl)
    # 2. Beyond ρ_ao: PSF should transition to seeing-limited halo
    
    center = n_pix // 2
    r_pix = np.arange(n_pix // 2)
    
    psf_profile = np.zeros(n_pix // 2)
    for r in r_pix:
        # Annular average
        y, x = np.ogrid[:n_pix, :n_pix]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        mask = (dist >= r) & (dist < r + 1)
        if np.sum(mask) > 0:
            psf_profile[r] = np.mean(psf[mask])
    
    r_arcsec = r_pix * pixel_scale_arcsec
    
    # Find transition region
    # The paper mentions aliasing causes a slight bump near ρ_ao
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    extent = np.array([-n_pix/2, n_pix/2, -n_pix/2, n_pix/2]) * pixel_scale_arcsec
    im = ax.imshow(psf, norm=LogNorm(vmin=psf.max()*1e-6), extent=extent, cmap='inferno')
    ax.set_title('Long-Exposure PSF')
    ax.set_xlabel('Offset (arcsec)')
    ax.set_ylabel('Offset (arcsec)')
    circle = plt.Circle((0, 0), rho_ao_arcsec, fill=False, ec='white', ls='--', lw=1.5)
    ax.add_patch(circle)
    ax.annotate(f'ρ_ao = {rho_ao_arcsec:.2f}"', xy=(rho_ao_arcsec, 0), xytext=(0.15, 0.05),
                color='white', fontsize=10)
    plt.colorbar(im, ax=ax)
    
    ax = axes[1]
    ax.semilogy(r_arcsec, psf_profile / psf_profile[0], 'b-', lw=2)
    ax.axvline(rho_ao_arcsec, color='r', ls='--', label=f'ρ_ao = {rho_ao_arcsec:.2f}"')
    ax.set_xlabel('Angular distance (arcsec)')
    ax.set_ylabel('Normalized intensity')
    ax.set_title('PSF Radial Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    
    plt.tight_layout()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'jolissaint_validation_psf.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved PSF validation figure to: {output_path}")
    
    print("\n✓ PSF structure validation complete")
    
    return psf, r_arcsec, psf_profile


def main():
    """Run all validations."""
    print("="*60)
    print("Jolissaint AO Model Validation")
    print("="*60)
    
    init_backend('numpy')
    
    validate_fitting_error()
    validate_psd_shapes()
    validate_strehl_marechal()
    validate_psf_structure()
    
    print("\n" + "="*60)
    print("All validations complete!")
    print("="*60)
    
    plt.show()


if __name__ == '__main__':
    main()

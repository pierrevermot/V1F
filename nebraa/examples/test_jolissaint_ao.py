#!/usr/bin/env python3
"""
Test Jolissaint AO Model
========================

This script tests and demonstrates the Jolissaint analytical AO model
for computing long-exposure, AO-corrected PSFs.

Based on: Jolissaint, Véran & Conan (2006)
"Analytical modeling of adaptive optics: foundations of the 
 phase spatial power spectrum approach"
J. Opt. Soc. Am. A, Vol. 23, No. 2, pp. 382-394

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
from nebraa.instruments.vlt import VLTPupil
from nebraa.physics.jolissaint_ao import (
    JolissaintAOModel,
    TurbulentLayer,
    AtmosphereProfile,
    AOSystemConfig,
    create_simple_atmosphere,
    create_mauna_kea_atmosphere,
    create_ao_config,
)


def setup_vlt_pupil(n_pix=256, wavelength=2.2e-6, pixel_scale_arcsec=13e-3):
    """Create VLT pupil for testing."""
    pixel_scale_rad = pixel_scale_arcsec * (np.pi / 180 / 3600)
    
    vlt_pupil = VLTPupil(
        n_pix=n_pix,
        wavelength=wavelength,
        pixel_scale=pixel_scale_rad,
        primary_diameter=8.2,
        obstruction_diameter=1.116,
    )
    vlt_pupil.add_vlt_spiders(
        width=0.04,
        half_opening_deg=51.3,
        attach_angles_deg=[0.0, 180.0],
    )
    
    return vlt_pupil


def test_basic_psf_computation():
    """Test basic PSF computation with Jolissaint model."""
    print("\n" + "="*60)
    print("Test 1: Basic PSF Computation")
    print("="*60)
    
    # Setup
    n_pix = 256
    wavelength = 2.2e-6  # K-band
    pixel_scale_arcsec = 13e-3
    pixel_scale_rad = pixel_scale_arcsec * (np.pi / 180 / 3600)
    
    # Create VLT pupil
    vlt_pupil = setup_vlt_pupil(n_pix, wavelength, pixel_scale_arcsec)
    pupil = vlt_pupil.amplitude
    
    # Simple atmosphere: r0 = 0.15 m at 500 nm
    atmosphere = create_simple_atmosphere(
        r0=0.15,
        wavelength_ref=0.5e-6,
        L0=25.0,
        wind_speed=10.0,
        altitude=0.0,
    )
    
    print(f"Total r0 at 500nm: {atmosphere.r0_total:.3f} m")
    print(f"Total r0 at {wavelength*1e6:.2f} μm: {atmosphere.r0_at_wavelength(wavelength):.3f} m")
    
    # AO configuration: 14x14 actuators (NAOS-like)
    ao_config = create_ao_config(
        n_actuators=14,
        telescope_diameter=8.2,
        sampling_frequency=500.0,
        noise_variance=0.0,
        science_field_offset_arcsec=0.0,
    )
    
    print(f"AO cutoff frequency: {ao_config.f_ao:.2f} cy/m")
    print(f"Actuator pitch: {ao_config.actuator_pitch:.3f} m")
    
    # Create model
    model = JolissaintAOModel(
        n_pix=n_pix,
        telescope_diameter=8.2,
        obstruction_diameter=1.116,
        wavelength=wavelength,
        pixel_scale=pixel_scale_rad,
        atmosphere=atmosphere,
        ao_config=ao_config,
    )
    
    # Compute PSF with components
    result = model.compute_long_exposure_psf(pupil, return_components=True)
    
    # Get Strehl
    strehl = model.compute_strehl_ratio(pupil)
    print(f"\nComputed Strehl ratio: {strehl:.3f}")
    
    # Error breakdown
    errors = model.get_error_breakdown()
    print("\nError breakdown (variance in rad²):")
    print(f"  Fitting:        {errors['fitting']:.4f}")
    print(f"  Anisoplanatism: {errors['anisoplanatism']:.4f}")
    print(f"  Servo-lag:      {errors['servo_lag']:.4f}")
    print(f"  Aliasing:       {errors['aliasing']:.4f}")
    print(f"  Noise:          {errors['noise']:.4f}")
    print(f"  Total:          {errors['total']:.4f}")
    print(f"  RMS phase:      {errors['rms_rad']:.4f} rad")
    
    print("\n✓ Basic PSF computation completed successfully")
    
    return result, model, vlt_pupil


def test_actuator_density_effect():
    """Test effect of actuator density on Strehl (Fig. 10 in paper)."""
    print("\n" + "="*60)
    print("Test 2: Actuator Density Effect")
    print("="*60)
    
    n_pix = 256
    wavelength = 2.2e-6
    pixel_scale_arcsec = 13e-3
    pixel_scale_rad = pixel_scale_arcsec * (np.pi / 180 / 3600)
    
    vlt_pupil = setup_vlt_pupil(n_pix, wavelength, pixel_scale_arcsec)
    pupil = vlt_pupil.amplitude
    
    # Atmosphere
    atmosphere = create_simple_atmosphere(
        r0=0.15,
        wavelength_ref=0.5e-6,
        L0=25.0,
        wind_speed=10.0,
    )
    
    # Test different actuator counts
    n_actuators_list = [8, 10, 12, 14, 16, 20, 24, 30]
    strehls = []
    
    for n_act in n_actuators_list:
        ao_config = create_ao_config(
            n_actuators=n_act,
            telescope_diameter=8.2,
            sampling_frequency=500.0,
        )
        
        model = JolissaintAOModel(
            n_pix=n_pix,
            telescope_diameter=8.2,
            obstruction_diameter=1.116,
            wavelength=wavelength,
            pixel_scale=pixel_scale_rad,
            atmosphere=atmosphere,
            ao_config=ao_config,
        )
        
        strehl = model.compute_strehl_ratio(pupil)
        strehls.append(strehl)
        print(f"  n_actuators={n_act:2d}, pitch={ao_config.actuator_pitch:.3f}m, Strehl={strehl:.3f}")
    
    print("\n✓ Actuator density test completed")
    
    return n_actuators_list, strehls


def test_sampling_frequency_effect():
    """Test effect of WFS sampling frequency on Strehl (Fig. 9 in paper)."""
    print("\n" + "="*60)
    print("Test 3: Sampling Frequency Effect (Servo-lag)")
    print("="*60)
    
    n_pix = 256
    wavelength = 2.2e-6
    pixel_scale_arcsec = 13e-3
    pixel_scale_rad = pixel_scale_arcsec * (np.pi / 180 / 3600)
    
    vlt_pupil = setup_vlt_pupil(n_pix, wavelength, pixel_scale_arcsec)
    pupil = vlt_pupil.amplitude
    
    # Atmosphere with wind
    atmosphere = create_simple_atmosphere(
        r0=0.15,
        wavelength_ref=0.5e-6,
        L0=25.0,
        wind_speed=15.0,  # Faster wind to show servo-lag effect
    )
    
    # Fixed AO actuators
    n_actuators = 14
    
    # Test different sampling frequencies
    frequencies = [50, 100, 200, 300, 500, 700, 1000, 1500, 2000]
    strehls = []
    
    for freq in frequencies:
        ao_config = AOSystemConfig(
            actuator_pitch=8.2 / n_actuators,
            integration_time=1.0 / freq,
            loop_delay=1.0 / freq,
        )
        
        model = JolissaintAOModel(
            n_pix=n_pix,
            telescope_diameter=8.2,
            obstruction_diameter=1.116,
            wavelength=wavelength,
            pixel_scale=pixel_scale_rad,
            atmosphere=atmosphere,
            ao_config=ao_config,
        )
        
        strehl = model.compute_strehl_ratio(pupil)
        strehls.append(strehl)
        print(f"  freq={freq:4d} Hz, dt={1000/freq:.2f} ms, Strehl={strehl:.3f}")
    
    print("\n✓ Sampling frequency test completed")
    
    return frequencies, strehls


def test_anisoplanatism_effect():
    """Test effect of off-axis angle on Strehl."""
    print("\n" + "="*60)
    print("Test 4: Anisoplanatism Effect (Off-axis NGS)")
    print("="*60)
    
    n_pix = 256
    wavelength = 2.2e-6
    pixel_scale_arcsec = 13e-3
    pixel_scale_rad = pixel_scale_arcsec * (np.pi / 180 / 3600)
    
    vlt_pupil = setup_vlt_pupil(n_pix, wavelength, pixel_scale_arcsec)
    pupil = vlt_pupil.amplitude
    
    # Multi-layer atmosphere (anisoplanatism needs altitude)
    layers = [
        TurbulentLayer(altitude=0, r0=0.20, wind_speed=5.0),
        TurbulentLayer(altitude=5000, r0=0.25, wind_speed=15.0),
        TurbulentLayer(altitude=10000, r0=0.40, wind_speed=25.0),
    ]
    atmosphere = AtmosphereProfile(layers=layers, wavelength_ref=0.5e-6, L0=25.0)
    
    print(f"Total r0 at 500nm: {atmosphere.r0_total:.3f} m")
    print(f"Mean altitude: {atmosphere.mean_altitude:.0f} m")
    print(f"Isoplanatic angle at {wavelength*1e6:.2f} μm: {atmosphere.isoplanatic_angle(wavelength)*206265:.2f} arcsec")
    
    # Test different off-axis angles
    angles_arcsec = [0, 2, 5, 10, 15, 20, 30, 45, 60]
    strehls = []
    
    for angle in angles_arcsec:
        ao_config = create_ao_config(
            n_actuators=14,
            telescope_diameter=8.2,
            sampling_frequency=500.0,
            science_field_offset_arcsec=float(angle),
        )
        
        model = JolissaintAOModel(
            n_pix=n_pix,
            telescope_diameter=8.2,
            obstruction_diameter=1.116,
            wavelength=wavelength,
            pixel_scale=pixel_scale_rad,
            atmosphere=atmosphere,
            ao_config=ao_config,
        )
        
        strehl = model.compute_strehl_ratio(pupil)
        strehls.append(strehl)
        print(f"  angle={angle:2d} arcsec, Strehl={strehl:.3f}")
    
    print("\n✓ Anisoplanatism test completed")
    
    return angles_arcsec, strehls


def test_mauna_kea_atmosphere():
    """Test with realistic Mauna Kea atmosphere."""
    print("\n" + "="*60)
    print("Test 5: Mauna Kea 7-Layer Atmosphere")
    print("="*60)
    
    n_pix = 256
    wavelength = 2.2e-6
    pixel_scale_arcsec = 13e-3
    pixel_scale_rad = pixel_scale_arcsec * (np.pi / 180 / 3600)
    
    vlt_pupil = setup_vlt_pupil(n_pix, wavelength, pixel_scale_arcsec)
    pupil = vlt_pupil.amplitude
    
    # Create Mauna Kea atmosphere
    atmosphere = create_mauna_kea_atmosphere(r0_total=0.15, L0=25.0)
    
    print(f"Number of layers: {len(atmosphere.layers)}")
    print(f"Total r0 at 500nm: {atmosphere.r0_total:.3f} m")
    print(f"Mean altitude: {atmosphere.mean_altitude:.0f} m")
    print(f"Mean wind speed: {atmosphere.mean_wind_speed:.1f} m/s")
    print(f"Coherence time at K-band: {atmosphere.coherence_time(wavelength)*1000:.1f} ms")
    
    # AO configuration
    ao_config = create_ao_config(
        n_actuators=14,
        telescope_diameter=8.2,
        sampling_frequency=500.0,
    )
    
    model = JolissaintAOModel(
        n_pix=n_pix,
        telescope_diameter=8.2,
        obstruction_diameter=1.116,
        wavelength=wavelength,
        pixel_scale=pixel_scale_rad,
        atmosphere=atmosphere,
        ao_config=ao_config,
    )
    
    strehl = model.compute_strehl_ratio(pupil)
    print(f"\nStrehl ratio: {strehl:.3f}")
    
    errors = model.get_error_breakdown()
    print("\nError breakdown (variance in rad²):")
    print(f"  Fitting:        {errors['fitting']:.4f}")
    print(f"  Anisoplanatism: {errors['anisoplanatism']:.4f}")
    print(f"  Servo-lag:      {errors['servo_lag']:.4f}")
    print(f"  Aliasing:       {errors['aliasing']:.4f}")
    print(f"  Noise:          {errors['noise']:.4f}")
    print(f"  Total:          {errors['total']:.4f}")
    
    print("\n✓ Mauna Kea atmosphere test completed")
    
    return model


def plot_results(result, model, vlt_pupil):
    """Create visualization of Jolissaint model results."""
    print("\n" + "="*60)
    print("Creating Visualization")
    print("="*60)
    
    backend = get_backend()
    xp = backend.xp
    
    # Convert arrays to numpy for plotting
    psf = backend.to_numpy(result['psf'])
    otf_total = backend.to_numpy(result['otf_total'])
    otf_ao = backend.to_numpy(result['otf_ao'])
    otf_tsc = backend.to_numpy(result['otf_tsc'])
    D_phi = backend.to_numpy(result['structure_function'])
    psd = backend.to_numpy(result['psd_total'])
    pupil = backend.to_numpy(vlt_pupil.amplitude)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # PSF (log scale)
    ax = axes[0, 0]
    n = psf.shape[0]
    extent = np.array([-n/2, n/2, -n/2, n/2]) * 13  # in mas
    im = ax.imshow(psf, norm=LogNorm(vmin=psf.max()*1e-6, vmax=psf.max()),
                   extent=extent, cmap='inferno')
    ax.set_title('Long-Exposure PSF')
    ax.set_xlabel('Position (mas)')
    ax.set_ylabel('Position (mas)')
    plt.colorbar(im, ax=ax, label='Intensity')
    
    # PSF profile
    ax = axes[0, 1]
    center = n // 2
    profile = psf[center, :]
    x_mas = (np.arange(n) - center) * 13
    ax.semilogy(x_mas, profile / profile.max(), 'b-', lw=1.5)
    ax.axhline(1e-2, color='gray', ls='--', alpha=0.5)
    ax.axhline(1e-4, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Position (mas)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('PSF Profile')
    ax.set_xlim(-500, 500)
    ax.set_ylim(1e-6, 2)
    ax.grid(True, alpha=0.3)
    
    # Residual Phase PSD
    ax = axes[0, 2]
    psd_shifted = np.fft.fftshift(psd)
    f_max = 1 / (2 * model.pupil_pixel_size)
    extent_f = [-f_max, f_max, -f_max, f_max]
    im = ax.imshow(psd_shifted + 1e-20, norm=LogNorm(), extent=extent_f, cmap='viridis')
    ax.set_title('Residual Phase PSD')
    ax.set_xlabel('fx (cy/m)')
    ax.set_ylabel('fy (cy/m)')
    plt.colorbar(im, ax=ax, label='PSD (rad²/(cy/m)²)')
    
    # Mark AO cutoff
    f_ao = model.ao_config.f_ao
    rect = plt.Rectangle((-f_ao, -f_ao), 2*f_ao, 2*f_ao, 
                         fill=False, ec='white', ls='--', lw=1.5)
    ax.add_patch(rect)
    
    # Structure Function
    ax = axes[1, 0]
    extent_rho = np.array([-n/2, n/2, -n/2, n/2]) * model.pupil_pixel_size
    im = ax.imshow(D_phi, extent=extent_rho, cmap='plasma')
    ax.set_title('Phase Structure Function')
    ax.set_xlabel('ρx (m)')
    ax.set_ylabel('ρy (m)')
    plt.colorbar(im, ax=ax, label='D_φ (rad²)')
    
    # OTF comparison
    ax = axes[1, 1]
    # 1D cut through OTFs
    otf_ao_1d = otf_ao[center, center:]
    otf_tsc_1d = otf_tsc[center, center:]
    otf_total_1d = otf_total[center, center:]
    f_norm = np.arange(len(otf_ao_1d)) / len(otf_ao_1d)
    
    ax.plot(f_norm, otf_tsc_1d, 'b-', label='Telescope OTF', lw=1.5)
    ax.plot(f_norm, otf_ao_1d, 'r-', label='AO OTF', lw=1.5)
    ax.plot(f_norm, otf_total_1d, 'k--', label='Total OTF', lw=1.5)
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('OTF')
    ax.set_title('OTF Comparison')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Error breakdown
    ax = axes[1, 2]
    errors = model.get_error_breakdown()
    labels = ['Fitting', 'Aniso', 'Servo', 'Alias', 'Noise']
    values = [errors['fitting'], errors['anisoplanatism'], 
              errors['servo_lag'], errors['aliasing'], errors['noise']]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('Variance (rad²)')
    ax.set_title(f'Error Breakdown\nStrehl = {model.compute_strehl_ratio(pupil):.3f}')
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 0.1)
    
    # Add values on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'jolissaint_ao_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    
    plt.show()


def plot_parameter_study(n_actuators_list, strehls_act, frequencies, strehls_freq,
                         angles, strehls_angle):
    """Plot parameter study results."""
    print("\n" + "="*60)
    print("Creating Parameter Study Plot")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Actuator density
    ax = axes[0]
    ax.plot(n_actuators_list, strehls_act, 'bo-', lw=2, markersize=8)
    ax.set_xlabel('Number of Actuators')
    ax.set_ylabel('Strehl Ratio')
    ax.set_title('Strehl vs Actuator Density')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Sampling frequency
    ax = axes[1]
    ax.semilogx(frequencies, strehls_freq, 'go-', lw=2, markersize=8)
    ax.set_xlabel('Sampling Frequency (Hz)')
    ax.set_ylabel('Strehl Ratio')
    ax.set_title('Strehl vs Sampling Frequency\n(Servo-lag effect)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Anisoplanatism
    ax = axes[2]
    ax.plot(angles, strehls_angle, 'ro-', lw=2, markersize=8)
    ax.set_xlabel('Off-axis Angle (arcsec)')
    ax.set_ylabel('Strehl Ratio')
    ax.set_title('Strehl vs Off-axis Angle\n(Anisoplanatism effect)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'jolissaint_ao_params.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    
    plt.show()


def main():
    """Run all tests."""
    print("="*60)
    print("Jolissaint AO Model Tests")
    print("="*60)
    print("\nImplementation of analytical AO PSF model from:")
    print("  Jolissaint, Véran & Conan (2006)")
    print("  J. Opt. Soc. Am. A, Vol. 23, No. 2")
    print()
    
    # Initialize backend
    init_backend('numpy')
    
    # Run tests
    result, model, vlt_pupil = test_basic_psf_computation()
    n_act, strehls_act = test_actuator_density_effect()
    freqs, strehls_freq = test_sampling_frequency_effect()
    angles, strehls_angle = test_anisoplanatism_effect()
    test_mauna_kea_atmosphere()
    
    # Create visualizations
    plot_results(result, model, vlt_pupil)
    plot_parameter_study(n_act, strehls_act, freqs, strehls_freq, angles, strehls_angle)
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()

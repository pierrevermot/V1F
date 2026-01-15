import os
import glob
import sys
import numpy as np
from joblib import Parallel, delayed
import tensorflow as tf
import config

pi = config.instrument

# Choose computation modules based on compute mode (CPU or GPU)
if pi.compute_mode == "CPU":
    import numpy as xp  # type: ignore
    from scipy.signal import fftconvolve  # type: ignore
    gpu = False
else:
    import cupy as xp  # type: ignore
    from cupyx.scipy.signal import fftconvolve  # type: ignore
    gpu = True

    
# Initializing global variables
x_np = np.arange(config.n_pix)
freqs_np = np.fft.fftfreq(config.n_pix)
mesh_np = np.meshgrid(freqs_np, freqs_np)
r_np = (mesh_np[0]**2 + mesh_np[1]**2)**0.5
r_np[r_np == 0] = sys.float_info.epsilon  # Avoid division by zero

# Function to rebin an array
def rebin(a, shape):
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

# Definitions related to the pupil
pupil_pixel_size = pi.lam/config.n_pix/pi.pixel_scale
x_pupil_large_np = np.arange(pi.upscaling*config.n_pix) * pupil_pixel_size / pi.upscaling
x_pupil_large_np -= np.mean(x_pupil_large_np)
mesh_pupil_large_np = np.meshgrid(x_pupil_large_np, x_pupil_large_np)
r_pupil_large_np = (mesh_pupil_large_np[0]**2 + mesh_pupil_large_np[1]**2)**0.5
pupil_large_np = 1. * (r_pupil_large_np > (pi.D2/2)) * (r_pupil_large_np < (pi.D/2))
pupil_rebin_np = rebin(pupil_large_np, (config.n_pix, config.n_pix))  # Rebinning the pupil
pupil_np = pupil_rebin_np  # Setting pupil
pupil_bool_np = np.array(pupil_np, dtype=bool)  # Boolean array of the pupil

# Function to generate power-law phase screens for atmospheric turbulence simulation
def get_powerlaw_phase_screens(power):
    """
    Generates a phase screen using a given power scale factor.
    """
    # Ensuring the power input is in array format
    if isinstance(power, float) or isinstance(power, int):
        power = xp.array([power])
    if isinstance(power, list):
        power = xp.array(power)

    rb = xp.array(r_np)  # Creating a copy of the radius array
    rb[0, 0] = rb[1, 1]  # Preventing division by zero
    power_spectrum = (rb/xp.mean(rb))**power  # Calculate the power spectrum
    phase = 2 * xp.pi * xp.random.random(xp.shape(power_spectrum))  # Generating random phase
    complex_fft = power_spectrum * xp.exp(1j * phase)  # Compute the complex FFT with the generated phase
    complex_screen = xp.fft.fft2(complex_fft)  # Compute the 2D FFT of the complex FFT
    # Splitting the generated screen into its real and imaginary components
    screen_a, screen_b = complex_screen.real, complex_screen.imag  
    return xp.concatenate([screen_a, screen_b])  # Returning the concatenated real and imaginary parts


# Function to calculate the Point Spread Function (PSF) from the phase screen
def get_psf(phase_screen, rms):
    """
    Calculate the Point Spread Function (PSF) from a given phase screen and RMS values.
    The function computes PSF by applying various transformations and normalization to the input phase screen.
    
    Parameters:
    - phase_screen: array representing the phase screen.
    - rms: array of RMS values used for scaling the phase screen.
    
    Returns:
    - psf: array representing the Point Spread Function (PSF).
    """
    # Adjusting the phase screen by the pupil and normalizing
    pupil = xp.array(pupil_np)
    ps = phase_screen - (xp.sum(phase_screen * pupil, (1, 2)) / xp.sum(pupil * (1+0*phase_screen), (1, 2)))[:, None, None]
    ps *= pupil  # Apply the pupil mask
    sum_ps = xp.sum(ps, axis=(1, 2))  # Summing phase screen values
    count_ps = xp.sum(pupil)  # Counting valid (pupil) pixels
    mean_ps = sum_ps / count_ps  # Calculating mean phase screen value
    # Computing variance and standard deviation of the phase screen
    var_ps = xp.sum((ps - mean_ps[:, None, None])**2 * pupil, axis=(1, 2)) / count_ps
    std = xp.sqrt(var_ps)[:, None, None]  # Calculating the standard deviation
    ps_scaled = ps * 2 * xp.pi * rms[:, None, None] / pi.lam / std  # Scaling the phase screen
    psf = abs(xp.fft.fft2(pupil * xp.exp(1j * ps_scaled)))**2  # Computing the PSF from the scaled phase screen
    return psf  # Returning the calculated PSF


# Function to create a set of Point Spread Functions (PSFs)
def make_psfs():
    """
    Create a set of Point Spread Functions (PSFs) by generating random parameters for turbulence power and RMS values.
    The function considers various random fluctuations and averages the computed PSFs for the output.

    Returns:
    - psfs: Fourier-shifted array of averaged Point Spread Functions (PSFs).
    """
    # Generating random parameters for the turbulence power and RMS values
    n_tau = 1+int(xp.random.random() * pi.max_n_tau)
    std_power = xp.random.random() * pi.std_max_power
    std_power_interframe = xp.random.random() * pi.std_max_power_interframe
    std_power_intraframe = xp.random.random() * pi.std_max_power_intraframe
    power_mean = pi.mean_power + xp.random.randn() * std_power
    
    # Generating the power array with random fluctuations
    power = xp.ones(n_tau) * power_mean + xp.random.randn(n_tau) * std_power_interframe
    power = power[:, None] * (xp.ones(config.n_frames_in)[None, :] + xp.random.randn(n_tau, config.n_frames_in) * std_power_intraframe)
    power = xp.reshape(power, n_tau * config.n_frames_in)
    
    # Generating the power-law screens with the calculated power
    pls = get_powerlaw_phase_screens(power[:, None, None] * xp.ones((n_tau * config.n_frames_in, config.n_pix, config.n_pix)))[:n_tau * config.n_frames_in]
    
    # Generating random RMS values
    rms_mean = xp.random.random() * pi.rms_max
    rms_std_interframe = xp.random.random() * pi.std_max_rms_interframe
    rms_std_intraframe = xp.random.random() * pi.std_max_rms_intraframe
    
    # Generating the RMS array with random fluctuations
    rms = xp.ones(n_tau) * rms_mean + xp.random.randn(n_tau) * rms_std_interframe
    rms = rms[:, None] * (xp.ones(config.n_frames_in)[None, :] + xp.random.randn(n_tau, config.n_frames_in) * rms_std_intraframe)
    rms = xp.reshape(rms, n_tau * config.n_frames_in)
    
    # Generating the PSFs from the power-law screens and RMS values
    psfs = get_psf(pls, rms)
    psfs_rs = xp.reshape(psfs, (n_tau, config.n_frames_in, config.n_pix, config.n_pix))
    psfs = xp.mean(psfs_rs, 0)  # Averaging the PSFs
    
    return xp.fft.fftshift(psfs)  # Returning the Fourier-shifted PSFs


# Function to recenter images in a cube based on their auto-correlations
def recenter_ims(cube):
    """
    Recenter images in a cube by cross-correlating each image with the mean image.
    The function then applies a phase shift to recenter the images.
    """
    mesh = xp.array(mesh_np)
    # Calculating the auto-correlation of each image in the cube with the mean image
    auto_corrs = fftconvolve(cube, xp.mean(cube, 0)[None, :, :], mode='same')
    
    # Finding the shifts necessary to recenter the images by locating the max in the auto-correlations
    shifts = xp.array(xp.unravel_index(xp.argmax(auto_corrs, (1,2)), auto_corrs[0].shape), dtype=int) - int(config.n_pix/2) - 1
    
    # Calculating the phase shift necessary to apply to each image based on the found shifts
    phase_shift = shifts[1, :, None, None]*mesh[0][None, :, :] + shifts[0, :, None, None]*mesh[1][None, :, :]
    
    # Applying the calculated phase shifts to the Fourier Transforms of the images
    cubeb = xp.fft.fft2(xp.fft.fft2(cube) * xp.exp(1j*2*xp.pi*phase_shift))
    
    return cubeb  # Returning the recentered cube of images

# Function to normalize and add noise to a data cube
def normalize_and_noise(cube):
    cube = cube/xp.std(cube, (1,2))[:, None, None]
    
    noise_ratio = xp.random.random()
    
    uniform_noise = xp.random.randn(*cube.shape)
    poisson_noise = xp.random.randn(*cube.shape)*cube**0.5
    poisson_noise = poisson_noise/xp.std(poisson_noise, (1,2))[:, None, None]
    
    noise = uniform_noise*noise_ratio+poisson_noise*(1-noise_ratio)
    snr = pi.snr_min+xp.random.random()*(pi.snr_max-pi.snr_min)
    
    cube += noise/snr
    
    return xp.nan_to_num(cube/xp.mean(cube, (1,2))[:, None, None])

# Function to make observations by convolving recentered PSFs with an image
def make_obs(im):
    """
    Make observations by convolving the image with recentered Point Spread Functions (PSFs).
    This function simulates the observed images as they would appear through a telescope with
    atmospheric turbulence.
    """
    psfs = make_psfs()  # Generating the PSFs
    # psfs_recentered = recenter_ims(psfs)  # Recentering the generated PSFs
    psfs_recentered = psfs.copy()
    # Convolving the recentered PSFs with the input image to generate the observed images
    cube_recentered = fftconvolve(psfs_recentered, im[None, :, :], mode='same')
    final_cube = normalize_and_noise(cube_recentered)
    
    return final_cube  # Returning the cube of observed images


def split(a, n):
    k, m = divmod(len(a), n)
    return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def npy_to_tfrecords(inputs, outputs, filename):
    # @Vijay Mariappan on stackoverflow
  with tf.io.TFRecordWriter(filename) as writer:
    for X, y in zip(inputs, outputs):
        # Feature contains a map of string to feature proto objects
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        feature = {}
        feature['X'] = tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten()))
        feature['y'] = tf.train.Feature(float_list=tf.train.FloatList(value=y.flatten()))

        # Construct the Example proto object
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize the example to a string
        serialized = example.SerializeToString()

        # write the serialized objec to the disk
        writer.write(serialized)

# Function to process and save images
def process(rep_bulk_ims, rep_bulk_obs):
    """
    Processes and saves images to a specified directory.
    """
    def process_ind(device_number, n_file):
        if gpu:        
            with xp.cuda.Device(device_number):
                ims = xp.load(f'{rep_bulk_ims}/IMS_{str(n_file)}.npy')
                obss = []
                for n_im in range(len(ims)):
                    obss.append(make_obs(ims[n_im]))  # Append generated images
                obss = xp.array(obss)
#                npy_to_tfrecords(obss, ims, f'{rep_bulk}/{str(n_file)}.tfrecords')
                xp.save(f'{rep_bulk_obs}/OBS_{str(n_file)}.npy', obss)  # Save the images as .npy files
        else:
            ims = xp.load(f'{rep_bulk_ims}/IMS_{str(n_file)}.npy')
            obss = []
            for n_im in range(len(ims)):
                obss.append(make_obs(ims[n_im]))  # Append generated images
            obss = xp.array(obss)
#            npy_to_tfrecors(obss, ims, f'{rep_bulk}/{str(n_file)}.tfrecords')
            xp.save(f'{rep_bulk_obs}/OBS_{str(n_file).npy}', obss)  # Save the images as .npy files

    im_files = glob.glob(rep_bulk_ims+'/IMS_*')
    n_files = []
    for f in im_files:
        n_file = int(f.split('IMS_')[-1].split('.')[0])
        n_files.append(n_file)
    if pi.compute_mode == 'CPU':
        n_cores = os.cpu_count()
    elif pi.compute_mode == 'GPU':
        n_cores = xp.cuda.runtime.getDeviceCount()
    n_filess = split(n_files, n_cores)
    for k in range(len(n_filess[0])):
        ns = []
        for kt in range(n_cores):
            try:
                ns.append(n_filess[kt][k])
            except:
                break
        Parallel(n_jobs=n_cores, backend="threading")(delayed(process_ind)(device_number, n_file) for device_number, n_file in enumerate(ns))


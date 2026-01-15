import os
import glob
import math
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
x_np = xp.arange(config.n_pix)
freqs_np = xp.fft.fftfreq(config.n_pix)
mesh_np = xp.meshgrid(freqs_np, freqs_np)
r_np = (mesh_np[0]**2 + mesh_np[1]**2)**0.5
r_np[r_np == 0] = sys.float_info.epsilon  # Avoid division by zero

# Function to rebin an array
def rebin(a, shape):
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

# Definitions related to the pupil
pupil_pixel_size = pi.lam/config.n_pix/pi.pixel_scale
x_pupil_large_np = xp.arange(pi.upscaling*config.n_pix) * pupil_pixel_size / pi.upscaling
x_pupil_large_np -= xp.mean(x_pupil_large_np)
mesh_pupil_large_np = xp.meshgrid(x_pupil_large_np, x_pupil_large_np)
xa = pi.c_to_c/2
r_pupil_large_np_a = ((mesh_pupil_large_np[0]-xa)**2 + mesh_pupil_large_np[1]**2)**0.5
r_pupil_large_np_b = ((mesh_pupil_large_np[0]+xa)**2 + mesh_pupil_large_np[1]**2)**0.5
pupil_large_np = 1. * ((r_pupil_large_np_a > (pi.D2/2)) * (r_pupil_large_np_a < (pi.D/2))+(r_pupil_large_np_b > (pi.D2/2)) * (r_pupil_large_np_b < (pi.D/2)))
pupil_rebin_np = rebin(pupil_large_np, (config.n_pix, config.n_pix))  # Rebinning the pupil
pupil_np = pupil_rebin_np  # Setting pupil
pupil_bool_np = xp.array(pupil_np, dtype=bool)  # Boolean array of the pupil

mask_left = pupil_np*0+(xp.arange(len(pupil_np))[None, :]<(len(pupil_np)/2))
mask_right = pupil_np*0+(xp.arange(len(pupil_np))[None, :]>(len(pupil_np)/2))


def factorial(k):
    if k >= 0:
        return math.factorial(k)
    else:
        return 1e100

def radial(rho, n, m):
    rho[rho==0] = sys.float_info.epsilon
    rad = rho*0
    if (n-m)%2 == 0:
        for k in range(int((n-m)/2)+1):
            num = (-1)**k*factorial(n-k)
            den = factorial(k)*factorial(int((n+m)/2)-k)*factorial(int((n-m)/2)-k)
            rad += rho**(n-2*k)*num/den
    return rad
    
def azimuthal(phi, m):
    if m >= 0:
        return xp.cos(m*phi)
    else:
        return xp.sin(m*phi)

def zernike(rho, phi, n, m):
    return radial(rho, n, m)*azimuthal(phi, m)

rad = config.n_pix*pi.lam/config.n_pix/pi.pixel_scale/2
x_left = xp.linspace(-rad, rad, config.n_pix)+pi.c_to_c/2
x_right = xp.linspace(-rad, rad, config.n_pix)-pi.c_to_c/2
y = xp.linspace(-rad, rad, config.n_pix)
mesh_left = xp.meshgrid(x_left, y)
mesh_right = xp.meshgrid(x_right, y)
rho_left = (mesh_left[0]**2+mesh_left[1]**2)**0.5/((pi.D)/2)
rho_right = (mesh_right[0]**2+mesh_right[1]**2)**0.5/((pi.D)/2)
phi_left = xp.arctan2(mesh_left[0], mesh_left[1])
phi_right = xp.arctan2(mesh_right[0], mesh_right[1])

modes_left = []
modes_right = []
ns = []
ms = []
for n in xp.arange(2, 5, 1):
    n = int(n)
    for m in xp.arange(-n, n+1, 2):
        mode_left = zernike(rho_left, phi_left, n, m)
        mode_right = zernike(rho_right, phi_right, n, m)
        # mode /= np.std(mode[pupil_bool_np])
        modes_left.append(mode_left)
        modes_right.append(mode_right)
        ns.append(n)
        ms.append(m)
        if xp.sum(xp.isnan(mode_left)) != 0:
            print(n, m)
            
modes_left = xp.array(modes_left)
modes_right = xp.array(modes_right)
ns = xp.array(ns)
ms = xp.array(ms)
is_rad = 1.*(ms==0)

def get_zernike_phase_screen(power):
    amps_left = (0.5-xp.random.random(len(ns)))*ns**power
    amps_right = (0.5-xp.random.random(len(ns)))*ns**power
    phase_left = xp.sum(amps_left[:, None, None]*modes_left, 0)*mask_left
    phase_right = xp.sum(amps_right[:, None, None]*modes_right, 0)*mask_right
    return phase_left+phase_right
    
def get_zernike_phase_screens(powers):
    amps_left = xp.random.random()*(0.5-xp.random.random((len(powers),len(ns))))*ns[None, :]**powers[:, None]
    amps_left += xp.random.random()*(0.5-xp.random.random((len(powers),len(ns))))*ns[None, :]**powers[:, None]*is_rad[None, :]
    amps_right = xp.random.random()*(0.5-xp.random.random((len(powers),len(ns))))*ns[None, :]**powers[:, None]
    amps_right += xp.random.random()*(0.5-xp.random.random((len(powers),len(ns))))*ns[None, :]**powers[:, None]*is_rad[None, :]
    phase = xp.sum(amps_left[:, :, None, None]*modes_left[None, :, :, :]*mask_left, 1)+xp.sum(amps_right[:, :, None, None]*modes_right[None, :, :, :]*mask_right, 1)
    return phase

#def get_zernike_phase_screens(powers):
#    # amps = (0.5-xp.random.random((len(powers),len(ns))))*ns[None, :]**powers[:, None]
#    amps_st = (0.5-xp.random.random(len(ns)))*ns[None, :]**powers[:, None]
#    amps = xp.random.random((len(powers),len(ns)))*amps_st
#    # amps = xp.random.randn(*(len(powers),len(ns)))*amps_st
#    # amps += xp.random.randn(*(len(powers),len(ns)))*amps_st
#    phase = xp.sum(amps[:, :, None, None]*modes[None, :, :, :], 1)
#    return phase

# Function to calculate the Point Spread Function (PSF) from the phase screen
def get_psf(phase_screen, rms, rms_pist):
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
    ps_pist_scaled = mask_left[None, :, :]*xp.ones(ps_scaled.shape)*rms_pist[:, None, None]/pi.lam*pupil
    psf = abs(xp.fft.fft2(pupil * xp.exp(1j * (ps_scaled+ps_pist_scaled))))**2  # Computing the PSF from the scaled phase screen
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
    power_mean = np.random.random()*pi.power_range[1]-pi.power_range[0]
    
    # Generating the power array with random fluctuations
    power = xp.ones(n_tau) * power_mean
    power = power[:, None] * xp.ones(config.n_frames_in)[None, :]
    power = xp.reshape(power, n_tau * pi.n_frames)
    
    # Generating the power-law screens with the calculated power
    pls = get_zernike_phase_screens(power)[:n_tau * pi.n_frames]
    
    # Generating random RMS values
    rms_mean = xp.random.random() * pi.rms_max
    rms_pist_mean = xp.random.random() * pi.rms_pist_max
    rms_pist_std = xp.random.random() * pi.rms_pist_max/5
    rms_pist_std_2 = xp.random.random() * pi.rms_pist_max
    # Generating the RMS array with random fluctuations
    rms = xp.ones(n_tau) * rms_mean
    rms = rms[:, None] * xp.ones(config.n_frames_in)[None, :]
    rms += xp.random.randn(*rms.shape)*rms_mean/5
    rms = xp.reshape(rms, n_tau * pi.n_frames)
    
    rms_pist = xp.ones(n_tau) * rms_pist_mean + xp.random.randn(n_tau) * rms_pist_std
    rms_pist = rms_pist[:, None] + xp.random.randn(config.n_frames_in)[None, :] * rms_pist_std_2
    rms_pist += xp.random.randn(*rms_pist.shape)*rms_pist_mean/5
    rms_pist = xp.reshape(rms_pist, n_tau * pi.n_frames)
    
    # Generating the PSFs from the power-law screens and RMS values
    psfs = get_psf(pls, rms, rms_pist)
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

def create_lines_error(cube):
#    print(cube.shape)
    mean_cube = 3*xp.random.random()*xp.mean(cube)
    values_1 = mean_cube+xp.random.randn(cube.shape[0])*mean_cube/5
    values_2 = mean_cube+xp.random.randn(cube.shape[0])*mean_cube/5
    # print(xp.mean(cube), values_1, values_2)
    line_num = int(65+20*xp.random.random())
    line_nums = xp.array(line_num+xp.random.randn(cube.shape[2]), dtype='int')
    for k in range(cube.shape[0]):
       cube[k,line_nums[k], :] = values_1[k]
       cube[k,line_nums[k]+1, :] = values_2[k]
    return cube

# Function to normalize and add noise to a data cube
def normalize_and_noise(cube):
    cube = cube/xp.std(cube, (1,2))[:, None, None]
    
    noise_ratio = xp.random.random()
    
    uniform_noise = xp.random.randn(*cube.shape)
    poisson_noise = xp.random.randn(*cube.shape)*cube**0.5
    poisson_noise = poisson_noise/xp.std(poisson_noise, (1,2))[:, None, None]
    
    noise = uniform_noise*noise_ratio+poisson_noise*(1-noise_ratio)
    snr = pi.snr_min+xp.random.random()*(pi.snr_max-pi.snr_min)
    nsr = pi.nsr_min+xp.random.random()*(pi.nsr_max-pi.nsr_min)
    snr = 1/nsr**2
#    print(nsr, snr)
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
    final_cube = normalize_and_noise(create_lines_error(cube_recentered))
    
    return final_cube  # Returning the cube of observed images


def split(a, n):
    k, m = divmod(len(a), n)
    return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))




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


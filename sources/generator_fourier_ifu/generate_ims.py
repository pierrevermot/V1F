import os
import numpy as np
from joblib import Parallel, delayed
import config

if config.source.compute_mode == "CPU":
    import numpy as xp  # type: ignore
    gpu = False
elif config.source.compute_mode == "GPU":
    import cupy as xp  # type: ignore
    gpu = True

# Initializing global variables
x_np = np.arange(config.n_pix)
freqs_np = np.fft.fftfreq(config.n_pix)
mesh_np = np.meshgrid(freqs_np, freqs_np)
r_np = (mesh_np[0]**2 + mesh_np[1]**2)**0.5

# Function to generate exponential phase screen
def get_exp_phase_screen(scale):
    """
    Generates a phase screen using a given scale factor.
    """
    r = xp.array(r_np)
    power_spectrum = xp.exp(-r * scale)  # Calculate the power spectrum
    phase = 2 * xp.pi * xp.random.random(xp.shape(power_spectrum))  # Random phase
    complex_fft = power_spectrum * xp.exp(1j * phase)  # Calculate the FFT
    complex_screen = xp.fft.fft2(complex_fft)  # 2D FFT
    return complex_screen.real, complex_screen.imag  # Return the real and imaginary parts

# Function to get image from screen
def get_im_from_screen(screen, sigma):
    """
    Processes the screen to get an image.
    """
    screen /= xp.std(screen)  # Normalize by the standard deviation
    im = (screen - sigma) * (screen > sigma)  # Threshold the image
    mean = xp.mean(im)  # Calculate mean
    return im / mean if mean > 0 else im  # Normalize by mean if mean > 0

# Function to generate images
def make_im():
    """
    Generates images by summing up objects in the image.
    """
    ims = []
    for k in range(1 + int(xp.random.random() * config.source.max_obj_per_im)):
        screen_a, screen_b = get_exp_phase_screen(xp.random.random() * config.n_pix)
        im_a = get_im_from_screen(screen_a, config.source.max_std * xp.random.randn())
        im_b = get_im_from_screen(screen_b, config.source.max_std * xp.random.randn())
        ims.append(xp.random.random() * im_a)  # Random weighting and append
        ims.append(xp.random.random() * im_b)  # Random weighting and append
        
    im = xp.sum(xp.array(ims), 0)  # Sum up all the images
    std = xp.std(im)  # Calculate standard deviation
    return im / std if std > 0 else im  # Normalize by std if std > 0


def split(a, n):
    k, m = divmod(len(a), n)
    return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

# Function to process and save images
def process(rep_bulk):
    """
    Processes and saves images to a specified directory.
    """
    def process_ind(device_number, n_file):
        if gpu:        
            with xp.cuda.Device(device_number):
                ims = []
                for n_im in range(config.source.N_ims_per_file):
                    ims.append(make_im())  # Append generated images
                ims = xp.array(ims)
                xp.save(f'{rep_bulk}/IMS_{str(n_file)}', ims)  # Save the images as .npy files
        else:
            ims = []
            for n_im in range(config.source.N_ims_per_file):
                ims.append(make_im())  # Append generated images
            ims = xp.array(ims)
            xp.save(f'{rep_bulk}/IMS_{str(n_file)}', ims)  # Save the images as .npy files   
        
    n_files = xp.arange(config.source.N_files)
    if config.source.compute_mode == 'CPU':
        n_cores = os.cpu_count()
    elif config.source.compute_mode == 'GPU':
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

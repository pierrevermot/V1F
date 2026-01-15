import os
import glob
import math
import sys
import numpy as np
from joblib import Parallel, delayed
import tensorflow as tf
import config
from instruments.elt.make_pupil import generate_elt_pupil

pi = config.instrument  # alias for readability

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



def generate_elt_pupil(
    n_pix: int,
    *,
    diameter_m: float = 40.0,
    spider_width_m: float = 0.51,
    gap_m: float = 4e-3,
    rotation_deg: float = 0.0,
    central_obscuration_ratio: float = 0.0,
    reflectivity_std: float | None = None,
    missing_segments: int = 0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Return a float32 array containing the ELT pupil mask.

    Parameters mirror the COMPASS ParamTel settings:
      * n_pix: number of pixels across the pupil support (use p_geom.pupdiam).
      * diameter_m: primary mirror diameter in metres.
      * spider_width_m: width of the three spiders, metres.
      * gap_m: inter-segment gap, metres.
      * rotation_deg: pupil rotation angle, degrees.
      * central_obscuration_ratio: central obscuration diameter ratio (0–1).
      * reflectivity_std: optional per-segment reflectivity std dev (metres).
      * missing_segments: number of random missing segments (set to 0 for full pupil).
      * rng: optional numpy RNG or seed for reproducibility (defaults to 42).
    """
    if n_pix <= 0:
        raise ValueError("n_pix must be positive")

    if rng is None:
        rng = np.random.default_rng(42)
    elif not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    def _dist(N, xc, yc):
        x = np.arange(N, dtype=np.float64) - xc
        y = np.arange(N, dtype=np.float64) - yc
        X, Y = np.meshgrid(x, y, indexing="ij")
        return np.sqrt(X**2 + Y**2)

    def _fill_polygon(x, y, i0, j0, scale, gap, N, index=0):
        X = (np.arange(N, dtype=np.float64) - i0) * scale
        Y = (np.arange(N, dtype=np.float64) - j0) * scale
        X, Y = np.meshgrid(X, Y, indexing="ij")
        x0 = np.mean(x)
        y0 = np.mean(y)
        T = (np.arctan2(Y - y0, X - x0) + 2 * np.pi) % (2 * np.pi)
        t = (np.arctan2(y - y0, x - x0) + 2 * np.pi) % (2 * np.pi)
        sens = np.median(np.diff(np.unwrap(t)))
        if sens < 0:
            x = x[::-1]
            y = y[::-1]
            t = t[::-1]
        imin = t.argmin()
        if imin != 0:
            x = np.roll(x, -imin)
            y = np.roll(y, -imin)
            t = np.roll(t, -imin)
        n = x.shape[0]
        indx = np.array([], dtype=np.int64)
        indy = np.array([], dtype=np.int64)
        distedge = np.array([], dtype=np.float64)
        for i in range(n):
            j = (i + 1) % n
            if j == 0:
                sub = np.where((T >= t[-1]) | (T <= t[0]))
            else:
                sub = np.where((T >= t[i]) & (T <= t[j]))
            dy = y[j] - y[i]
            dx = x[j] - x[i]
            vnorm = np.hypot(dx, dy)
            if vnorm == 0:
                continue
            dx /= vnorm
            dy /= vnorm
            crossprod = dx * (Y[sub] - y[i]) - dy * (X[sub] - x[i])
            tmp = crossprod > gap
            indx = np.append(indx, sub[0][tmp])
            indy = np.append(indy, sub[1][tmp])
            distedge = np.append(distedge, crossprod[tmp])
        if index == 1:
            return indx.astype(np.int64), indy.astype(np.int64), distedge
        pol = np.zeros((N, N), dtype=bool)
        pol[indx, indy] = True
        return pol

    def _fill_spider(N, nspider, dspider, i0, j0, scale, rot):
        mask = np.ones((N, N), dtype=bool)
        x = (np.arange(N, dtype=np.float64) - i0) * scale
        y = (np.arange(N, dtype=np.float64) - j0) * scale
        X, Y = np.meshgrid(x, y, indexing="ij")
        w = 2 * np.pi / nspider
        for i in range(nspider):
            nn = np.abs(X * np.cos(i * w - rot) + Y * np.sin(i * w - rot)) < dspider / 2.0
            mask[nn] = False
        return mask

    def _create_hexa_pattern(pitch, support_size):
        v3 = np.sqrt(3.0)
        nx = int(np.ceil((support_size / 2.0) / pitch) + 1)
        x = pitch * (np.arange(2 * nx + 1) - nx)
        ny = int(np.ceil((support_size / 2.0) / pitch / v3) + 1)
        y = (v3 * pitch) * (np.arange(2 * ny + 1) - ny)
        x, y = np.meshgrid(x, y, indexing="ij")
        x = x.flatten()
        y = y.flatten()
        peak_axis = np.append(x, x + pitch / 2.0)
        flat_axis = np.append(y, y + pitch * v3 / 2.0)
        return flat_axis, peak_axis

    def _reorganize_segments_order_eso(x, y):
        pi_3 = np.pi / 3
        pi_6 = np.pi / 6
        two_pi = 2 * np.pi
        t = (np.arctan2(y, x) + pi_6 - 1e-3) % two_pi
        X = np.array([])
        Y = np.array([])
        A = 100.0
        for k in range(6):
            sector = (t > k * pi_3) & (t < (k + 1) * pi_3)
            u = k * pi_3
            distance = (A * np.cos(u) - np.sin(u)) * x[sector] + (np.cos(u) + A * np.sin(u)) * y[sector]
            indsort = np.argsort(distance)
            X = np.append(X, x[sector][indsort])
            Y = np.append(Y, y[sector][indsort])
        return X, Y

    def _generate_coord_segments(
        D,
        rot,
        pitch=1.244683637214,
        nseg=33,
        inner_rad=4.1,
        outer_rad=15.4,
        R=95.7853,
        nominalD=40,
    ):
        v3 = np.sqrt(3.0)
        lx, ly = _create_hexa_pattern(pitch, (nseg + 2) * pitch)
        ll = np.sqrt(lx**2 + ly**2)
        valid = (ll > inner_rad * pitch) & (ll < outer_rad * pitch)
        lx = lx[valid]
        ly = ly[valid]
        lx, ly = _reorganize_segments_order_eso(lx, ly)
        th = np.linspace(0, 2 * np.pi, 7)[:-1]
        hx = np.cos(th) * pitch / v3
        hy = np.sin(th) * pitch / v3
        x = lx[None, :] + hx[:, None]
        y = ly[None, :] + hy[:, None]
        r = np.sqrt(x**2 + y**2)
        rrc = R / r * np.arctan(r / R)
        x *= rrc
        y *= rrc
        if D != nominalD:
            scale = D / nominalD
            x *= scale
            y *= scale
        mrot = np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])
        xyrot = np.dot(mrot, np.transpose(np.array([x, y]), (1, 0, 2)))
        return xyrot[0], xyrot[1]

    def _generate_segment_properties(
        attribute,
        hx,
        hy,
        i0,
        j0,
        scale,
        gap,
        N,
        D,
        nominalD=40,
        pitch=1.244683637214,
        half_seg=0.75,
    ):
        nseg = hx.shape[-1]
        attr = np.asarray(attribute, dtype=np.float64)
        if attr.ndim == 0:
            attr = np.full(nseg, attr, dtype=np.float64)
        if attr.size != nseg:
            raise ValueError(f"attribute must have length {nseg}, got {attr.size}")
        pupil = np.zeros((N, N), dtype=np.float64)
        x0 = np.mean(hx, axis=0)
        y0 = np.mean(hy, axis=0)
        x0 = x0 / scale + i0
        y0 = y0 / scale + j0
        hexrad = half_seg * D / nominalD / scale
        ix0 = np.floor(x0 - hexrad).astype(int) - 1
        iy0 = np.floor(y0 - hexrad).astype(int) - 1
        segdiam = int(np.ceil(hexrad * 2 + 1)) + 1
        for idx in range(nseg):
            subx, suby, _ = _fill_polygon(
                hx[:, idx],
                hy[:, idx],
                i0 - ix0[idx],
                j0 - iy0[idx],
                scale,
                gap,
                segdiam,
                index=1,
            )
            sx = subx + ix0[idx]
            sy = suby + iy0[idx]
            valid = (sx >= 0) & (sx < N) & (sy >= 0) & (sy < N)
            pupil[sx[valid], sy[valid]] = attr[idx]
        return pupil

    pixscale = diameter_m / n_pix
    centre = n_pix / 2.0 - 0.5
    hx, hy = _generate_coord_segments(
        diameter_m,
        np.deg2rad(rotation_deg),
        pitch=1.244683637214,
        nseg=33,
        inner_rad=4.1,
        outer_rad=15.4,
        R=95.7853,
        nominalD=40,
    )
    nseg = hx.shape[-1]
    attribute = np.ones(nseg, dtype=np.float64)
    if reflectivity_std is not None and reflectivity_std > 0:
        attribute -= rng.normal(scale=reflectivity_std, size=nseg)
        attribute = np.clip(attribute, 0.0, None)
    if missing_segments:
        missing_segments = int(min(max(missing_segments, 0), nseg))
        if missing_segments > 0:
            idx = rng.choice(nseg, size=missing_segments, replace=False)
            attribute[idx] = 0.0
    pupil = _generate_segment_properties(
        attribute,
        hx,
        hy,
        centre,
        centre,
        pixscale,
        gap_m,
        n_pix,
        diameter_m,
        nominalD=40,
        pitch=1.244683637214,
        half_seg=0.75,
    )
    if spider_width_m > 0:
        spider_mask = _fill_spider(n_pix, 3, spider_width_m, centre, centre, pixscale, np.deg2rad(rotation_deg))
        pupil *= spider_mask
    if central_obscuration_ratio > 0:
        radius_mask = _dist(n_pix, centre, centre) >= (n_pix * central_obscuration_ratio + 1.0) * 0.5
        pupil *= radius_mask
    return pupil.astype(np.float32)


# Definitions related to the pupil - ELT specific
# Generate ELT pupil at high resolution and rebin to target size
pupil_large_np = generate_elt_pupil(n_pix=pi.elt_pupil_size)
if gpu:
    pupil_large_np = xp.asarray(pupil_large_np)
else:
    pupil_large_np = xp.array(pupil_large_np)
pupil_rebin_np = rebin(pupil_large_np, (config.n_pix, config.n_pix))  # Rebinning the pupil
pupil_np = pupil_rebin_np  # Setting pupil
pupil_bool_np = xp.array(pupil_np, dtype=bool)  # Boolean array of the pupil

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
x = xp.linspace(-rad, rad, config.n_pix)
mesh = xp.meshgrid(x, x)
rho = (mesh[0]**2+mesh[1]**2)**0.5/(pi.D/2)
phi = xp.arctan2(mesh[0], mesh[1])

modes = []
ns = []
ms = []
for n in xp.arange(2, 5, 1):
    n = int(n)
    for m in xp.arange(-n, n+1, 2):
        mode = zernike(rho, phi, n, m)
        # mode /= np.std(mode[pupil_bool_np])
        modes.append(mode)
        ns.append(n)
        ms.append(m)
        if xp.sum(xp.isnan(mode)) != 0:
            print(n, m)
            
modes = xp.array(modes)
ns = xp.array(ns)
ms = xp.array(ms)
is_rad = 1.*(ms==0)

def get_zernike_phase_screen(power):
    amps = (0.5-xp.random.random(len(ns)))*ns**power
    phase = xp.sum(amps[:, None, None]*modes, 0)
    return phase
    
def get_zernike_phase_screens(powers):
    amps = xp.random.random()*(0.5-xp.random.random((len(powers),len(ns))))*ns[None, :]**powers[:, None]
    amps += xp.random.random()*(0.5-xp.random.random((len(powers),len(ns))))*ns[None, :]**powers[:, None]*is_rad[None, :]
    phase = xp.sum(amps[:, :, None, None]*modes[None, :, :, :], 1)
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
    power_mean = np.random.random()*pi.power_range[1]-pi.power_range[0]
    
    # Generating the power array with random fluctuations
    power = xp.ones(n_tau) * power_mean
    power = power[:, None] * xp.ones(config.n_frames_in)[None, :]
    power = xp.reshape(power, n_tau * config.n_frames_in)
    
    # Generating the power-law screens with the calculated power
    pls = get_zernike_phase_screens(power)[:n_tau * pi.n_frames]
    
    # Generating random RMS values
    rms_mean = xp.random.random() * pi.rms_max
    
    # Generating the RMS array with random fluctuations
    rms = xp.ones(n_tau) * rms_mean
    rms = rms[:, None] * xp.ones(config.n_frames_in)[None, :]
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


# Function to make observations by convolving recentered PSFs with an image
def make_obs_w_psfs(im):
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
    
    return final_cube, psfs_recentered  # Returning the cube of observed images



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
                xp.save(f'{rep_bulk_obs}/OBS_{str(n_file)}.npy', obss)  # Save the images as .npy files
        else:
            ims = xp.load(f'{rep_bulk_ims}/IMS_{str(n_file)}.npy')
            obss = []
            for n_im in range(len(ims)):
                obss.append(make_obs(ims[n_im]))  # Append generated images
            obss = xp.array(obss)
            xp.save(f'{rep_bulk_obs}/OBS_{str(n_file)}.npy', obss)  # Save the images as .npy files

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

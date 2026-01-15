import sys
sys.path.append("../../")

import params
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

x = np.arange(params.n_pix)
freqs = np.fft.fftfreq(params.n_pix)
mesh = np.meshgrid(freqs, freqs)
r = (mesh[0]**2+mesh[1]**2)**0.5

def get_power_phase_screen(power):
    power_spectrum = r**-power
    phase = 2*np.pi*np.random.random(np.shape(power_spectrum))
    complex_fft = power_spectrum*np.exp(1j*phase)
    complex_screen = np.fft.fft2(complex_fft)
    return complex_screen.real, complex_screen.imag

def get_psf(screen, n_tau):
    
# Image parameters

n_pix = 128
n_frames = 32
lam = 2.18e-6 #4.78e-6 #2.18e-6i
pixel_scale = 0.013/206265#0.01225/206265

# Pupil parameters

upscaling = 32
D=8.2
D2=1.116

# Randomness parameters

max_n_tau = 30

power_range = [0, -2]

rms_max = lam/5

snr_min = 0
snr_max = 50


# Compute mode

compute_mode = "GPU"

# Image parameters

n_pix = 128
n_frames = 32
lam = 4.78e-6#3.80e-6 #4.78e-6 #2.18e-6i
pixel_scale = 0.02719/206265#0.01225/206265#0.01225/206265

# Pupil parameters - ELT specific

elt_pupil_size = 8192  # Size for generating the ELT pupil
D = 40.0  # ELT primary mirror diameter in meters

# Randomness parameters

max_n_tau = 30

power_range = [0, -2]

rms_max = lam/5

snr_min = 0
snr_max = 50


# Compute mode

compute_mode = "GPU"

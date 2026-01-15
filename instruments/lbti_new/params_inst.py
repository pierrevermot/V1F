# Image parameters

n_pix = 100
n_frames = 32
lam = 8.699e-6
pixel_scale = 0.018/206265

# Pupil parameters

upscaling = 32
D = 8.4 #8.2
D2 = 0.9 #1.116

c_to_c = 14.4 #0 


# Randomness parameters

max_n_tau = 30

power_range = [0, -2]

rms_max = lam/6
rms_pist_max = lam*5

snr_min = 0
snr_max = 15

nsr_min = 0
nsr_max = 3


# Compute mode

compute_mode = "GPU"

# Image parameters

n_pix = 128
n_frames = 32
lam = 2.182e-6# 4.78e-6 # 3.8e-6 # 2.18e-6
pixel_scale = 0.01225/206265# 0.027/206265 # 0.013/206265

# Pupil parameters

upscaling = 32
D=8.2
D2=1.116

# Randomness parameters

max_n_tau = 30

mean_power = -5/3
std_max_power = 1/3
std_max_power_interframe = 1/9
std_max_power_intraframe = std_max_power_interframe/3

rms_max = lam/5
std_max_rms_interframe = lam/10
std_max_rms_intraframe = std_max_rms_interframe/3

snr_min = 1
snr_max = 100

# Compute mode

compute_mode = "GPU"

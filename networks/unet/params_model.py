import os
from datetime import datetime
name = "fpm_128_NACO_M"#"decK_UNET"
rep_bulk = os.environ.get('SCRATCH')+'/'+name

# Model parameters 

float_precision = 'float32' # Can be float16, float32 (recommended), float64 or mixed
inner_activation = 'softplus' # Can be linear, relu, LeakyReLU, softplus, etc..
output_activation = 'linear' # Can be linear, relu, LeakyReLU, softplus, etc..

# Input and output chape

n_pix = 128
n_frames_in = 32
n_frames_out = 1

# Training parameters

batch_size = 16   #16
epochs = 2500
#max_steps = 1000000
learning_rate = 5e-4
clipnorm = 5
loss = "mean_absolute_error"
test_dataset_length = 1000

# Output directories

output_dir = './outputs/'
datedir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
logdir = os.path.join(datedir, "logs")
savedir = os.path.join(datedir, "model")
plotdir = os.path.join(datedir, "plots")

# Data plotting

plot_data = True
pathsdata = ["/lustre/fswork/projects/rech/hfk/udl61tt/NEBRAA_V1E/observations/naco_1068_M/data.npy"]  # UPDATED path for NEBRAA_V1E
datanames = ['NACO_M']

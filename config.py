"""Unified configuration for the project.

This single file replaces previous scattered parameter modules:
 - params_glob.py
 - sources/*/params_im_gen.py
 - instruments/*/params_inst.py
 - networks/unet/params_model.py

It also centralizes run-mode control (generation vs training) and
TFRecord regeneration policies.
"""

from __future__ import annotations

import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Literal

# --------------------------------------------------------------------------------------
# High-level run control
# --------------------------------------------------------------------------------------

# run_mode controls which stages execute in main.py
#  - 'full'          : generate images + observations + tfrecords then train
#  - 'generate_only' : generate up to tfrecords, skip training
#  - 'train'         : assume tfrecords already exist, only train
run_mode: Literal['full', 'generate_only', 'train'] = 'full'

# Regeneration policy for each data layer; choices: 'if_missing' | 'always' | 'never'
regen_images: Literal['if_missing', 'always', 'never'] = 'if_missing'
regen_observations: Literal['if_missing', 'always', 'never'] = 'if_missing'
regen_tfrecords: Literal['if_missing', 'always', 'never'] = 'if_missing'

# --------------------------------------------------------------------------------------
# Selection of pipeline components
# --------------------------------------------------------------------------------------

SOURCE = 'generator_fourier_ps'       # directory under sources/
INSTRUMENT = 'ao_zer_radplus'         # directory under instruments/
NETWORK = 'unet'                      # directory under networks/

# --------------------------------------------------------------------------------------
# Core shared geometry / shapes
# --------------------------------------------------------------------------------------

n_pix = 128              # Spatial dimension (assumed square)
n_frames_in = 32         # Number of temporal frames provided to network (instrument output depth)
n_frames_out = 1         # Network output depth

# --------------------------------------------------------------------------------------
# Naming / directories
# --------------------------------------------------------------------------------------

name_ims = 'fpm_128'
name_obs = 'NACO_M'
run_tag = f'{name_ims}_{name_obs}'

SCRATCH = os.environ.get('SCRATCH', '/tmp')
rep_bulk_ims = os.path.join(SCRATCH, name_ims)
rep_bulk_obs = os.path.join(SCRATCH, run_tag)
rep_tfrecords = os.path.join(rep_bulk_obs, 'tfrecords')

output_dir = './outputs'
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
datedir = os.path.join(output_dir, timestamp)
logdir = os.path.join(datedir, 'logs')
savedir = os.path.join(datedir, 'model')
plotdir = os.path.join(datedir, 'plots')

# The training code expects pm.rep_bulk to point to directory containing *.tfrecords
rep_bulk = rep_tfrecords

# --------------------------------------------------------------------------------------
# Source generation parameters (previously in params_im_gen)
# --------------------------------------------------------------------------------------

@dataclass
class SourceConfig:
    N_files: int = 4096
    N_ims_per_file: int = 256
    max_obj_per_im: int = 40
    max_std: float = 5.0
    max_ns: int = 10
    compute_mode: Literal['CPU', 'GPU'] = 'CPU'

source = SourceConfig()

# --------------------------------------------------------------------------------------
# Instrument simulation parameters (previously in params_inst)
# --------------------------------------------------------------------------------------

@dataclass
class InstrumentConfig:
    lam: float = 4.78e-6
    pixel_scale: float = 0.02719 / 206265
    upscaling: int = 32
    D: float = 8.2
    D2: float = 1.116
    max_n_tau: int = 30
    power_range: List[float] = field(default_factory=lambda: [0, -2])
    rms_max: float = 4.78e-6 / 5
    snr_min: float = 0.0
    snr_max: float = 50.0
    # Extended / optional fields for other instruments
    mean_power: float = -5/3
    std_max_power: float = 1/3
    std_max_power_interframe: float = 1/9
    std_max_power_intraframe: float = (1/9)/3
    std_max_rms_interframe: float = 0.0
    std_max_rms_intraframe: float = 0.0
    rms_pist_max: float = 0.0
    c_to_c: float = 0.0
    nsr_min: float = 0.0
    nsr_max: float = 0.0
    compute_mode: Literal['CPU', 'GPU'] = 'GPU'

instrument = InstrumentConfig()

# --------------------------------------------------------------------------------------
# Overrides for specific source / instrument variants (migrated from legacy param files)
# --------------------------------------------------------------------------------------

source_overrides = {
    'generator_fourier': dict(N_files=12, max_obj_per_im=10, compute_mode='GPU'),
    'generator_fourier_ifu': dict(N_files=12, max_obj_per_im=10, compute_mode='GPU'),
}

instrument_overrides = {
    'ao_psd': dict(
        lam=2.182e-6,
        pixel_scale=0.01225/206265,
        rms_max=2.182e-6/5,
        snr_min=1,
        snr_max=100,
        mean_power=-5/3,
        std_max_power=1/3,
        std_max_power_interframe=1/9,
        std_max_power_intraframe=(1/9)/3,
        std_max_rms_interframe=(2.182e-6)/10,
        std_max_rms_intraframe=(2.182e-6)/10/3,
    ),
    'ao_zer': dict(
        lam=2.18e-6,
        pixel_scale=0.013/206265,
        rms_max=2.18e-6/5,
        snr_min=0,
        snr_max=50,
    ),
    'lbti_new': dict(
        n_pix=100,
        lam=8.699e-6,
        pixel_scale=0.018/206265,
        D=8.4,
        D2=0.9,
        c_to_c=14.4,
        power_range=[0, -2],
        rms_max=8.699e-6/6,
        rms_pist_max=8.699e-6*5,
        snr_min=0,
        snr_max=15,
        nsr_min=0,
        nsr_max=3,
    ),
    'ao_zer_radplus_lbti': dict(
        n_pix=100,
        lam=8.699e-6,
        pixel_scale=0.018/206265,
        D=8.4,
        D2=0.9,
        c_to_c=14.4,
        power_range=[0, -2],
        rms_max=8.699e-6/10,
        snr_min=0,
        snr_max=50,
    ),
}

# --------------------------------------------------------------------------------------
# Training parameters (previously in params_model)
# --------------------------------------------------------------------------------------

@dataclass
class TrainConfig:
    float_precision: str = 'float32'   # float16 | float32 | float64 | mixed
    inner_activation: str = 'softplus'
    output_activation: str = 'linear'
    batch_size: int = 16
    epochs: int = 2500
    learning_rate: float = 5e-4
    clipnorm: float = 5.0
    loss: str = 'mean_absolute_error'
    test_dataset_length: int = 1000
    plot_data: bool = True
    pathsdata: List[str] = field(default_factory=lambda: [
    '/lustre/fswork/projects/rech/hfk/udl61tt/NEBRAA_V1E/observations/naco_1068_M/data.npy'
    ])
    datanames: List[str] = field(default_factory=lambda: ['NACO_M'])

train = TrainConfig()

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def ensure_dirs():
    for d in [rep_bulk_ims, rep_bulk_obs, rep_tfrecords, output_dir, datedir, logdir, savedir, plotdir]:
        os.makedirs(d, exist_ok=True)


def need_regenerate(target_dir: str, pattern_ext: str, policy: str) -> bool:
    """Determine if a generation step should run.

    pattern_ext: file extension to search for (e.g. '.npy' or '.tfrecords')
    """
    if policy == 'always':
        return True
    if policy == 'never':
        return False
    # if_missing
    if not os.path.isdir(target_dir):
        return True
    for f in os.listdir(target_dir):
        if f.endswith(pattern_ext):
            return False
    return True


# Backwards compatibility alias expectations in other modules:
# The training script imports variables like pm.float_precision, pm.rep_bulk, etc.
# By assigning pm = config (in model.py) we keep attribute parity.


def apply_overrides():
    # Source overrides
    so = source_overrides.get(SOURCE, {})
    for k, v in so.items():
        setattr(source, k, v)
    # Instrument overrides (may adjust global geometry)
    io = instrument_overrides.get(INSTRUMENT, {})
    global n_pix, n_frames_in
    if 'n_pix' in io:
        n_pix = io['n_pix']
    if 'n_frames' in io:
        n_frames_in = io['n_frames']
    for k, v in io.items():
        if k in ('n_pix', 'n_frames'):  # already handled
            continue
        setattr(instrument, k, v)


apply_overrides()

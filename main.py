"""Main orchestrator using unified config.

Stages (depending on config.run_mode):
 1. Ensure directories
 2. Generate images (IMS)      -> rep_bulk_ims
 3. Generate observations (OBS)-> rep_bulk_obs
 4. Build TFRecords            -> rep_tfrecords
 5. Train network              -> outputs
"""

import os
import time
import importlib

import config
from data_pipeline.tfrecords import build_tfrecords


def dynamic_import(path: str):
    return importlib.import_module(path)


def get_modules():
    src_mod = dynamic_import(f'sources.{config.SOURCE}.generate_ims')
    inst_mod = dynamic_import(f'instruments.{config.INSTRUMENT}.generate_obs')
    net_mod = dynamic_import(f'networks.{config.NETWORK}.model')
    return src_mod, inst_mod, net_mod


def maybe_generate_images(src_mod):
    if config.need_regenerate(config.rep_bulk_ims, '.npy', config.regen_images):
        print('[IMS] Generating images...')
        t0 = time.time()
        src_mod.process(config.rep_bulk_ims)
        print(f'[IMS] Done in {time.time()-t0:.2f}s')
    else:
        print('[IMS] Skipped (existing)')


def maybe_generate_observations(inst_mod):
    if config.need_regenerate(config.rep_bulk_obs, '.npy', config.regen_observations):
        print('[OBS] Generating observations...')
        t0 = time.time()
        inst_mod.process(config.rep_bulk_ims, config.rep_bulk_obs)
        print(f'[OBS] Done in {time.time()-t0:.2f}s')
    else:
        print('[OBS] Skipped (existing)')


def maybe_build_tfrecords():
    force = (config.regen_tfrecords == 'always')
    if config.regen_tfrecords == 'never':
        print('[TFRecords] Skipped by policy (never)')
        return
    status = build_tfrecords(config.rep_bulk_ims, config.rep_bulk_obs, config.rep_tfrecords, force=force)
    print(f'[TFRecords] Status: {status}')


def maybe_train(net_mod):
    print('[TRAIN] Starting training')
    net_mod.train()
    print('[TRAIN] Finished')


def main():
    config.ensure_dirs()
    src_mod, inst_mod, net_mod = get_modules()

    if config.run_mode in ('full', 'generate_only'):
        maybe_generate_images(src_mod)
        maybe_generate_observations(inst_mod)
        maybe_build_tfrecords()

    if config.run_mode in ('full', 'train'):
        maybe_train(net_mod)
    else:
        print('[MAIN] Training skipped (generate_only mode)')


if __name__ == '__main__':
    main()


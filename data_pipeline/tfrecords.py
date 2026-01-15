"""TFRecord builder integrating previous standalone script logic.

Writes TFRecords pairing observations (X) and images (y).

File naming convention: DATA_<index>.tfrecords where <index> matches the IMS/OBS number.
"""
from __future__ import annotations

import os
import glob
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_array(array):
    return tf.io.serialize_tensor(array)


def _write_example(writer: tf.io.TFRecordWriter, X, y):
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    X_ser = serialize_array(X)
    y_ser = serialize_array(y)
    feat = {'X': _bytes_feature(X_ser), 'y': _bytes_feature(y_ser)}
    example = tf.train.Example(features=tf.train.Features(feature=feat))
    writer.write(example.SerializeToString())


def _process_file(idx: int, ims_path: str, rep_bulk_obs: str, rep_tfrecords: str):
    obs_path = os.path.join(rep_bulk_obs, f'OBS_{idx}.npy')
    if not os.path.isfile(obs_path):
        raise FileNotFoundError(f'Missing observation file {obs_path} for image file {ims_path}')
    ims = np.load(ims_path)              # Shape: (N_ims_per_file, n_pix, n_pix)
    obss = np.load(obs_path)             # Shape: (N_ims_per_file, n_frames, n_pix, n_pix)
    # Move frames axis to last to get (n_pix, n_pix, n_frames)
    obss = np.moveaxis(obss, 1, 3)
    out_path = os.path.join(rep_tfrecords, f'DATA_{idx}.tfrecords')
    with tf.io.TFRecordWriter(out_path) as w:
        for X, y in zip(obss, ims):
            _write_example(w, X, y)
    return out_path


def build_tfrecords(rep_bulk_ims: str, rep_bulk_obs: str, rep_tfrecords: str, force: bool = False, n_jobs: int | None = None):
    os.makedirs(rep_tfrecords, exist_ok=True)
    ims_files = sorted(glob.glob(os.path.join(rep_bulk_ims, 'IMS_*.npy')))
    if not ims_files:
        raise RuntimeError(f'No IMS_*.npy files found in {rep_bulk_ims}')
    # If not forcing and counts already match, skip
    if not force:
        existing = glob.glob(os.path.join(rep_tfrecords, 'DATA_*.tfrecords'))
        if existing and len(existing) == len(ims_files):
            return 'skipped'
    # Derive indices
    indices = []
    for f in ims_files:
        try:
            idx = int(os.path.basename(f).split('IMS_')[-1].split('.')[0])
            indices.append(idx)
        except ValueError:
            continue
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1

    def task(fpath):
        idx_local = int(os.path.basename(fpath).split('IMS_')[-1].split('.')[0])
        return _process_file(idx_local, fpath, rep_bulk_obs, rep_tfrecords)

    Parallel(n_jobs=n_jobs, backend='threading')(delayed(task)(f) for f in ims_files)
    return 'built'

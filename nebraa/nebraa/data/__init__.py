"""
Data I/O and TFRecord handling.

Provides utilities for:
- Building TFRecords from numpy arrays
- Loading TFRecord datasets for training
- Data augmentation
"""

from __future__ import annotations

import os
import glob
from typing import Optional, Tuple, Callable

import numpy as np

from ..utils.logging import get_logger, Timer


# =============================================================================
# TFRecord Building
# =============================================================================

def _bytes_feature(value):
    """Create bytes feature for TFRecord."""
    import tensorflow as tf
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _serialize_example(X, y):
    """Serialize a single example to TFRecord format."""
    import tensorflow as tf
    
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    
    X_ser = tf.io.serialize_tensor(X)
    y_ser = tf.io.serialize_tensor(y)
    
    feature = {
        'X': _bytes_feature(X_ser),
        'y': _bytes_feature(y_ser),
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def build_tfrecord_file(
    images_path: str,
    observations_path: str,
    output_path: str,
):
    """
    Build a single TFRecord file from numpy files.
    
    Args:
        images_path: Path to images .npy file (ground truth)
        observations_path: Path to observations .npy file (input)
        output_path: Path for output .tfrecords file
    """
    import tensorflow as tf
    
    # Load data
    images = np.load(images_path)  # (N, H, W)
    observations = np.load(observations_path)  # (N, n_frames, H, W)
    
    # Move frames axis to last: (N, H, W, n_frames)
    observations = np.moveaxis(observations, 1, 3)
    
    # Write TFRecord
    with tf.io.TFRecordWriter(output_path) as writer:
        for X, y in zip(observations, images):
            writer.write(_serialize_example(X.astype(np.float32), y.astype(np.float32)))


def build_tfrecords(
    images_dir: str,
    observations_dir: str,
    output_dir: str,
    force: bool = False,
    n_jobs: Optional[int] = None,
) -> str:
    """
    Build TFRecords from image and observation directories.
    
    Expects matching IMS_*.npy and OBS_*.npy files.
    
    Args:
        images_dir: Directory containing IMS_*.npy files
        observations_dir: Directory containing OBS_*.npy files
        output_dir: Output directory for .tfrecords files
        force: If True, rebuild even if files exist
        n_jobs: Number of parallel jobs
    
    Returns:
        Status string: 'built', 'skipped', or 'error'
    """
    from joblib import Parallel, delayed
    
    logger = get_logger()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find image files
    image_files = sorted(glob.glob(os.path.join(images_dir, 'IMS_*.npy')))
    if not image_files:
        logger.error(f"No IMS_*.npy files found in {images_dir}")
        return 'error'
    
    # Check if already built
    if not force:
        existing = glob.glob(os.path.join(output_dir, 'DATA_*.tfrecords'))
        if len(existing) >= len(image_files):
            logger.info(f"TFRecords already exist ({len(existing)} files)")
            return 'skipped'
    
    # Extract indices
    def get_index(path):
        base = os.path.basename(path)
        return int(base.split('IMS_')[-1].split('.')[0])
    
    indices = [get_index(f) for f in image_files]
    
    # Build function
    def process_one(idx):
        img_path = os.path.join(images_dir, f'IMS_{idx}.npy')
        obs_path = os.path.join(observations_dir, f'OBS_{idx}.npy')
        out_path = os.path.join(output_dir, f'DATA_{idx}.tfrecords')
        
        if not os.path.exists(obs_path):
            return f"Missing {obs_path}"
        
        build_tfrecord_file(img_path, obs_path, out_path)
        return None
    
    n_jobs = n_jobs or os.cpu_count() or 1
    
    with Timer("Building TFRecords", logger):
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(process_one)(idx) for idx in indices
        )
    
    # Check for errors
    errors = [r for r in results if r is not None]
    if errors:
        for err in errors[:5]:
            logger.warning(err)
        return 'error'
    
    logger.info(f"Built {len(indices)} TFRecord files")
    return 'built'


# =============================================================================
# TFRecord Loading
# =============================================================================

def parse_tfrecord_fn(example):
    """Parse function for TFRecord dataset."""
    import tensorflow as tf
    
    features = {
        'X': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed = tf.io.parse_single_example(example, features)
    
    X = tf.io.parse_tensor(parsed['X'], out_type=tf.float32)
    y = tf.io.parse_tensor(parsed['y'], out_type=tf.float32)
    
    return X, y


def load_tfrecord_dataset(
    tfrecords_dir: str,
    batch_size: int = 16,
    shuffle_buffer: int = 1000,
    augment: bool = True,
    num_parallel_calls: Optional[int] = None,
) -> Tuple:
    """
    Load TFRecord dataset for training.
    
    Args:
        tfrecords_dir: Directory containing .tfrecords files
        batch_size: Training batch size
        shuffle_buffer: Shuffle buffer size
        augment: Apply data augmentation
        num_parallel_calls: Parallelization level
    
    Returns:
        (train_dataset, test_dataset) tuple
    """
    import tensorflow as tf
    
    if num_parallel_calls is None:
        num_parallel_calls = tf.data.AUTOTUNE
    
    # List files
    files = tf.data.Dataset.list_files(
        os.path.join(tfrecords_dir, '*.tfrecords'),
        shuffle=True,
    )
    
    # Interleave and parse
    dataset = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=num_parallel_calls,
        block_length=1,
    )
    
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=num_parallel_calls)
    
    # Augmentation
    if augment:
        def augment_fn(X, y):
            # Random scaling
            scale = 2.0 * tf.random.uniform(shape=[])
            return scale * X, scale * y
        
        dataset = dataset.map(augment_fn, num_parallel_calls=num_parallel_calls)
    
    # Shuffle, batch, prefetch
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_train_test_split(
    tfrecords_dir: str,
    test_size: int = 1000,
    batch_size: int = 16,
    augment: bool = True,
) -> Tuple:
    """
    Create train/test split from TFRecords.
    
    Args:
        tfrecords_dir: Directory containing .tfrecords files
        test_size: Number of test samples
        batch_size: Batch size
        augment: Apply augmentation to training data
    
    Returns:
        (train_dataset, test_dataset)
    """
    import tensorflow as tf
    
    num_parallel_calls = tf.data.AUTOTUNE
    
    # Load full dataset
    files = tf.data.Dataset.list_files(
        os.path.join(tfrecords_dir, '*.tfrecords'),
        shuffle=True,
    )
    
    dataset = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=num_parallel_calls,
        block_length=1,
    )
    
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=num_parallel_calls)
    
    # Split
    test_dataset = dataset.take(test_size)
    train_dataset = dataset.skip(test_size)
    
    # Augment training only
    if augment:
        def augment_fn(X, y):
            scale = 2.0 * tf.random.uniform(shape=[])
            return scale * X, scale * y
        
        train_dataset = train_dataset.map(augment_fn, num_parallel_calls=num_parallel_calls)
    
    # Batch and prefetch
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(1000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset


__all__ = [
    'build_tfrecords',
    'build_tfrecord_file',
    'load_tfrecord_dataset',
    'create_train_test_split',
    'parse_tfrecord_fn',
]

#!/usr/bin/env python3
"""
Command-line interface for NEBRAA.

Usage:
    nebraa generate [options]     Generate observations dataset
    nebraa build-tfrecords        Build TFRecords from numpy files
    nebraa train [options]        Train reconstruction model
    nebraa predict [options]      Run inference on observations
    nebraa info                   Show configuration info
"""

import argparse
import sys
import os
from pathlib import Path


def cmd_generate(args):
    """Generate observations dataset."""
    from .config import load_config, Config, PipelineConfig, SourceConfig, InstrumentConfig
    from .utils.compute import init_backend, get_worker_items, Backend
    from .utils.logging import get_logger, Timer
    from .instruments import get_instrument
    from .sources import SourceGenerator
    
    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = Config(
            pipeline=PipelineConfig(
                n_samples=args.n_samples,
                n_frames=args.n_frames,
                output_dir=args.output_dir or './data/generated',
            ),
            source=SourceConfig(
                mode=args.source_mode,
                image_size=args.image_size,
            ),
            instrument=InstrumentConfig(
                name=args.instrument,
            ),
        )
    
    config.ensure_dirs()
    
    # Initialize backend
    backend = init_backend(gpu=not args.cpu)
    logger = get_logger()
    
    logger.info(f"Generating observations with {config.instrument.name}")
    logger.info(f"Backend: {backend.name}, Device: {backend.device_id}")
    logger.info(f"Workers: {backend.world_size}, Rank: {backend.rank}")
    
    # Get instrument and source generator
    instrument = get_instrument(config.instrument.name, config.instrument)
    source_gen = SourceGenerator(config.source)
    
    xp = backend.xp
    
    # Determine this worker's samples
    total_samples = config.pipeline.n_samples
    n_per_file = args.samples_per_file or 100
    n_files = (total_samples + n_per_file - 1) // n_per_file
    
    file_indices = get_worker_items(list(range(n_files)))
    logger.info(f"Worker {backend.rank} handling files: {file_indices}")
    
    images_dir = os.path.join(config.pipeline.output_dir, 'images')
    obs_dir = os.path.join(config.pipeline.output_dir, 'observations')
    
    for subdir in [images_dir, obs_dir]:
        os.makedirs(subdir, exist_ok=True)
    
    for file_idx in file_indices:
        n_this_file = min(n_per_file, total_samples - file_idx * n_per_file)
        
        images_list = []
        obs_list = []
        
        with Timer(f"File {file_idx} ({n_this_file} samples)", logger):
            for _ in range(n_this_file):
                # Generate source image
                image = source_gen.generate()
                
                # Generate multi-frame observation
                frames = []
                for _ in range(config.pipeline.n_frames):
                    obs_frame = instrument.simulate_observation(image)
                    frames.append(backend.to_numpy(obs_frame))
                
                images_list.append(backend.to_numpy(image))
                obs_list.append(np.stack(frames, axis=0))
        
        # Save
        import numpy as np
        images_arr = np.stack(images_list, axis=0)
        obs_arr = np.stack(obs_list, axis=0)
        
        np.save(os.path.join(images_dir, f'IMS_{file_idx}.npy'), images_arr)
        np.save(os.path.join(obs_dir, f'OBS_{file_idx}.npy'), obs_arr)
        
        logger.info(f"Saved file {file_idx}: images {images_arr.shape}, obs {obs_arr.shape}")
    
    logger.info("Generation complete")


def cmd_build_tfrecords(args):
    """Build TFRecords from numpy files."""
    from .data import build_tfrecords
    from .utils.logging import get_logger
    
    logger = get_logger()
    
    images_dir = args.images_dir or os.path.join(args.data_dir, 'images')
    obs_dir = args.observations_dir or os.path.join(args.data_dir, 'observations')
    output_dir = args.output_dir or os.path.join(args.data_dir, 'tfrecords')
    
    result = build_tfrecords(
        images_dir=images_dir,
        observations_dir=obs_dir,
        output_dir=output_dir,
        force=args.force,
        n_jobs=args.n_jobs,
    )
    
    logger.info(f"TFRecords status: {result}")


def cmd_train(args):
    """Train reconstruction model."""
    from .config import load_config, Config, TrainConfig
    from .utils.logging import get_logger
    from .data import create_train_test_split
    from .models.unet import UNet
    
    logger = get_logger()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = Config(
            train=TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
            ),
        )
    
    # Setup TensorFlow
    import tensorflow as tf
    
    strategy = None
    if args.distributed:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Distributed training on {strategy.num_replicas_in_sync} devices")
    
    # Load data
    tfrecords_dir = args.tfrecords_dir or './data/tfrecords'
    train_ds, test_ds = create_train_test_split(
        tfrecords_dir,
        test_size=config.train.test_size,
        batch_size=config.train.batch_size,
        augment=True,
    )
    
    # Build model
    def build_and_compile():
        model = UNet(
            input_channels=args.n_frames or 10,
            output_channels=1,
            depth=config.train.model_depth,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.train.learning_rate),
            loss='mse',
            metrics=['mae'],
        )
        return model
    
    if strategy:
        with strategy.scope():
            model = build_and_compile()
    else:
        model = build_and_compile()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.checkpoint_dir or './checkpoints/model.keras',
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=args.log_dir or './logs',
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
        ),
    ]
    
    # Train
    logger.info("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=config.train.epochs,
        callbacks=callbacks,
    )
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir or './checkpoints', 'final_model.keras')
    model.save(final_path)
    logger.info(f"Model saved to {final_path}")


def cmd_predict(args):
    """Run inference on observations."""
    import numpy as np
    import tensorflow as tf
    from .utils.logging import get_logger
    
    logger = get_logger()
    
    # Load model
    model = tf.keras.models.load_model(args.model_path)
    logger.info(f"Loaded model from {args.model_path}")
    
    # Load observations
    observations = np.load(args.input)
    logger.info(f"Loaded observations: {observations.shape}")
    
    # Prepare input (move frames to last axis if needed)
    if observations.ndim == 3:
        observations = observations[None, ...]
    if observations.shape[1] < observations.shape[-1]:
        # Likely (N, frames, H, W) -> (N, H, W, frames)
        observations = np.moveaxis(observations, 1, -1)
    
    # Run inference
    predictions = model.predict(observations)
    
    # Save
    np.save(args.output, predictions)
    logger.info(f"Saved predictions to {args.output}")


def cmd_info(args):
    """Show configuration info."""
    from .utils.compute import init_backend
    
    backend = init_backend(gpu=True)
    
    print("NEBRAA Configuration Info")
    print("=" * 50)
    print(f"Backend: {backend.name}")
    print(f"Device: {backend.device_id}")
    print(f"World size: {backend.world_size}")
    print(f"Rank: {backend.rank}")
    print(f"Is chief: {backend.is_chief}")
    
    # Show available instruments
    from .instruments import get_instrument
    print("\nAvailable instruments:")
    print("  - vlt")
    print("  - lbti (coming soon)")
    
    # Show model info
    print("\nAvailable models:")
    print("  - unet")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='NEBRAA: Neural Bayesian Reconstruction of Astronomical Atmospherics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate observations dataset')
    gen_parser.add_argument('--config', '-c', help='Configuration file (YAML or JSON)')
    gen_parser.add_argument('--n-samples', '-n', type=int, default=1000, help='Number of samples')
    gen_parser.add_argument('--n-frames', type=int, default=10, help='Frames per observation')
    gen_parser.add_argument('--output-dir', '-o', help='Output directory')
    gen_parser.add_argument('--instrument', '-i', default='vlt', help='Instrument name')
    gen_parser.add_argument('--source-mode', default='extended', 
                           choices=['extended', 'point_source', 'ifu', 'mixed'])
    gen_parser.add_argument('--image-size', type=int, default=128, help='Image size')
    gen_parser.add_argument('--samples-per-file', type=int, default=100)
    gen_parser.add_argument('--cpu', action='store_true', help='Force CPU backend')
    
    # Build TFRecords command
    tf_parser = subparsers.add_parser('build-tfrecords', help='Build TFRecords')
    tf_parser.add_argument('--data-dir', '-d', default='./data', help='Base data directory')
    tf_parser.add_argument('--images-dir', help='Images directory (default: data_dir/images)')
    tf_parser.add_argument('--observations-dir', help='Observations directory')
    tf_parser.add_argument('--output-dir', '-o', help='Output directory for TFRecords')
    tf_parser.add_argument('--force', '-f', action='store_true', help='Force rebuild')
    tf_parser.add_argument('--n-jobs', '-j', type=int, help='Number of parallel jobs')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train reconstruction model')
    train_parser.add_argument('--config', '-c', help='Configuration file')
    train_parser.add_argument('--tfrecords-dir', help='TFRecords directory')
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--batch-size', type=int, default=16)
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--n-frames', type=int, default=10)
    train_parser.add_argument('--checkpoint-dir', help='Checkpoint directory')
    train_parser.add_argument('--log-dir', help='TensorBoard log directory')
    train_parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    
    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Run inference')
    pred_parser.add_argument('model_path', help='Path to trained model')
    pred_parser.add_argument('input', help='Input observations file (.npy)')
    pred_parser.add_argument('output', help='Output predictions file (.npy)')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show configuration info')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch
    commands = {
        'generate': cmd_generate,
        'build-tfrecords': cmd_build_tfrecords,
        'train': cmd_train,
        'predict': cmd_predict,
        'info': cmd_info,
    }
    
    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if os.environ.get('NEBRAA_DEBUG'):
            raise
        sys.exit(1)


if __name__ == '__main__':
    main()

"""
U-Net architecture for image reconstruction.

Implements a U-Net with configurable depth, activations, and
support for distributed training via TensorFlow.
"""

from __future__ import annotations

import os
from typing import Optional, List
from dataclasses import dataclass

import numpy as np

from ..config import TrainingConfig
from ..utils.logging import get_logger
from . import register_model


@dataclass
class UNetConfig:
    """U-Net architecture configuration."""
    
    n_frames_in: int = 32
    n_frames_out: int = 1
    base_filters: int = 32
    depth: int = 3
    inner_activation: str = 'softplus'
    output_activation: str = 'linear'


@register_model('unet')
class UNet:
    """
    U-Net model for image deconvolution.
    
    Architecture:
    - Encoder: Conv -> Act -> Conv -> Act -> MaxPool (repeated)
    - Bridge: Conv -> Act -> Conv -> Act
    - Decoder: Upsample + Concat -> Conv -> Act -> Conv -> Act (repeated)
    - Output: 1x1 Conv
    """
    
    def __init__(
        self,
        n_frames_in: int = 32,
        n_frames_out: int = 1,
        base_filters: int = 32,
        depth: int = 3,
        inner_activation: str = 'softplus',
        output_activation: str = 'linear',
    ):
        self.config = UNetConfig(
            n_frames_in=n_frames_in,
            n_frames_out=n_frames_out,
            base_filters=base_filters,
            depth=depth,
            inner_activation=inner_activation,
            output_activation=output_activation,
        )
        self.model = None
        self.logger = get_logger()
    
    def build(self, strategy=None):
        """
        Build the Keras model.
        
        Args:
            strategy: Optional TensorFlow distribution strategy
        
        Returns:
            Keras Model instance
        """
        import tensorflow as tf
        from tensorflow.keras.layers import (
            Input, Conv2D, MaxPooling2D, UpSampling2D, 
            concatenate, Activation, LeakyReLU
        )
        from tensorflow.keras.models import Model
        
        def get_activation(name):
            if name == 'LeakyReLU':
                return LeakyReLU()
            return Activation(name)
        
        cfg = self.config
        
        def build_model():
            # Input: flexible spatial dimensions
            inputs = Input(shape=(None, None, cfg.n_frames_in))
            
            # Encoder path
            encoders = []
            x = inputs
            filters = cfg.base_filters
            
            for i in range(cfg.depth):
                # Double convolution
                x = Conv2D(filters, (3, 3), padding='same')(x)
                x = get_activation(cfg.inner_activation)(x)
                x = Conv2D(filters, (3, 3), padding='same')(x)
                x = get_activation(cfg.inner_activation)(x)
                
                encoders.append(x)
                x = MaxPooling2D(pool_size=(2, 2))(x)
                filters *= 2
            
            # Bridge
            x = Conv2D(filters, (3, 3), padding='same')(x)
            x = get_activation(cfg.inner_activation)(x)
            x = Conv2D(filters, (3, 3), padding='same')(x)
            x = get_activation(cfg.inner_activation)(x)
            
            # Decoder path
            for i in range(cfg.depth - 1, -1, -1):
                filters //= 2
                x = concatenate([UpSampling2D(size=(2, 2))(x), encoders[i]], axis=3)
                x = Conv2D(filters, (3, 3), padding='same')(x)
                x = get_activation(cfg.inner_activation)(x)
                x = Conv2D(filters, (3, 3), padding='same')(x)
                x = get_activation(cfg.inner_activation)(x)
            
            # Output
            output = Conv2D(cfg.n_frames_out, (1, 1))(x)
            output = get_activation(cfg.output_activation)(output)
            
            return Model(inputs=inputs, outputs=output)
        
        if strategy is not None:
            with strategy.scope():
                self.model = build_model()
        else:
            self.model = build_model()
        
        return self.model
    
    def compile(
        self,
        learning_rate: float = 5e-4,
        clipnorm: float = 5.0,
        loss: str = 'mean_absolute_error',
    ):
        """Compile the model."""
        import tensorflow as tf
        from tensorflow.keras.optimizers import Adam
        
        if self.model is None:
            self.build()
        
        optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
        self.model.compile(loss=loss, optimizer=optimizer)
        
        return self
    
    def summary(self):
        """Print model summary."""
        if self.model is not None:
            self.model.summary()


def create_training_callbacks(
    save_dir: str,
    log_dir: str,
    plot_dir: str,
    validation_data=None,
    observation_paths: Optional[List[str]] = None,
    observation_names: Optional[List[str]] = None,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    reduce_lr_min: float = 5e-6,
):
    """
    Create standard training callbacks.
    
    Args:
        save_dir: Directory for model checkpoints
        log_dir: Directory for TensorBoard logs
        plot_dir: Directory for prediction plots
        validation_data: Validation dataset for plotting
        observation_paths: Paths to real observation files for plotting
        observation_names: Names for observation plots
        reduce_lr_patience: Epochs before reducing learning rate
        reduce_lr_factor: Factor to reduce learning rate
        reduce_lr_min: Minimum learning rate
    
    Returns:
        List of Keras callbacks
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        ModelCheckpoint, TensorBoard, ReduceLROnPlateau, Callback
    )
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    callbacks = []
    
    # Checkpoint callback
    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(save_dir, 'model_epoch_{epoch:02d}.h5'),
        save_freq='epoch',
        save_best_only=False,
        verbose=1,
    )
    callbacks.append(checkpoint_cb)
    
    # TensorBoard callback
    tensorboard_cb = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
    )
    callbacks.append(tensorboard_cb)
    
    # Learning rate reducer
    reduce_lr_cb = ReduceLROnPlateau(
        monitor='val_loss',
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=reduce_lr_min,
        min_delta=1e-4,
    )
    callbacks.append(reduce_lr_cb)
    
    # Validation prediction plotting
    if validation_data is not None:
        class PlotValidationCallback(Callback):
            def __init__(self, data, dirplot):
                super().__init__()
                self.dirplot = dirplot
                data = data.take(1)
                self.data = [(x.numpy(), y.numpy()) for x, y in data][0]
            
            def on_epoch_end(self, epoch, logs=None):
                truths = self.data[1][:9]
                obss = self.data[0][:9]
                preds = self.model.predict(obss, verbose=0)
                
                fig, axes = plt.subplots(3, 9, figsize=(30, 9))
                for i in range(9):
                    axes[0, i].imshow(truths[i], cmap='hot')
                    axes[0, i].axis('off')
                    axes[1, i].imshow(np.mean(obss[i], -1), cmap='hot')
                    axes[1, i].axis('off')
                    axes[2, i].imshow(preds[i], cmap='hot')
                    axes[2, i].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.dirplot, 'predictions.png'))
                plt.close()
        
        callbacks.append(PlotValidationCallback(validation_data, plot_dir))
    
    # Real observation plotting
    if observation_paths:
        class PlotObservationCallback(Callback):
            def __init__(self, paths, names, dirplot):
                super().__init__()
                self.paths = paths
                self.names = names or [f'obs_{i}' for i in range(len(paths))]
                self.dirplot = dirplot
            
            def on_epoch_end(self, epoch, logs=None):
                for path, name in zip(self.paths, self.names):
                    if not os.path.exists(path):
                        continue
                    
                    data = np.load(path)
                    pred = self.model.predict(data[np.newaxis], verbose=0)[0]
                    
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    axes[0].imshow(np.mean(data, -1), cmap='hot')
                    axes[0].set_title('Observation')
                    axes[0].axis('off')
                    axes[1].imshow(pred, cmap='hot')
                    axes[1].set_title('Reconstruction')
                    axes[1].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.dirplot, f'{name}.png'))
                    plt.close()
        
        callbacks.append(PlotObservationCallback(
            observation_paths,
            observation_names,
            plot_dir,
        ))
    
    return callbacks


def setup_distributed_strategy():
    """
    Setup TensorFlow distributed strategy for multi-GPU/multi-node training.
    
    Automatically detects SLURM environment.
    
    Returns:
        TensorFlow distribution strategy
    """
    import tensorflow as tf
    
    try:
        # Try SLURM cluster resolver
        cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(
            port_base=15000
        )
        
        implementation = tf.distribute.experimental.CommunicationImplementation.NCCL
        communication_options = tf.distribute.experimental.CommunicationOptions(
            implementation=implementation
        )
        
        strategy = tf.distribute.MultiWorkerMirroredStrategy(
            cluster_resolver=cluster_resolver,
            communication_options=communication_options,
        )
        
        return strategy
        
    except Exception:
        # Fall back to mirrored strategy for single-node multi-GPU
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 1:
            return tf.distribute.MirroredStrategy()
        
        # Single GPU or CPU
        return tf.distribute.get_strategy()


__all__ = [
    'UNet',
    'UNetConfig',
    'create_training_callbacks',
    'setup_distributed_strategy',
]

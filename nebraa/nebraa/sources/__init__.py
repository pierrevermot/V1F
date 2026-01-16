"""
Astronomical source image generators.

Provides unified interface for generating synthetic astronomical source
images with different morphologies (extended, point sources, IFU).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional, List
from enum import Enum

from ..config import SourceConfig
from ..utils.compute import get_backend, split_work, get_device_count, get_cpu_count
from ..utils.logging import get_logger, Timer


class SourceMode(Enum):
    """Source generation modes."""
    FOURIER = 'fourier'
    FOURIER_PS = 'fourier_ps'    # With point sources
    FOURIER_IFU = 'fourier_ifu'  # IFU spectral cube
    SIMPLE_3D = '3d'


class SourceGenerator(ABC):
    """
    Abstract base class for source image generators.
    """
    
    def __init__(self, config: SourceConfig, n_pix: int):
        """
        Initialize generator.
        
        Args:
            config: Source generation configuration
            n_pix: Image size in pixels
        """
        self.config = config
        self.n_pix = n_pix
        self.logger = get_logger()
    
    @abstractmethod
    def generate_single(self):
        """Generate a single source image."""
        pass
    
    def generate_batch(self, n: int, seed: Optional[int] = None):
        """
        Generate a batch of source images.
        
        Args:
            n: Number of images
            seed: Random seed
        
        Returns:
            Array of shape (n, n_pix, n_pix)
        """
        backend = get_backend()
        xp = backend.xp
        
        if seed is not None:
            xp.random.seed(seed)
        
        images = []
        for _ in range(n):
            images.append(self.generate_single())
        
        return xp.stack(images, axis=0)


class FourierGenerator(SourceGenerator):
    """
    Fourier-domain source generator.
    
    Generates extended source images using filtered random noise
    in the Fourier domain, creating smooth morphologies.
    """
    
    def __init__(self, config: SourceConfig, n_pix: int):
        super().__init__(config, n_pix)
        
        backend = get_backend()
        xp = backend.xp
        
        # Precompute frequency grid
        freqs = xp.fft.fftfreq(n_pix)
        fx, fy = xp.meshgrid(freqs, freqs)
        self.r_freq = xp.sqrt(fx**2 + fy**2).astype(xp.float32)
    
    def _exponential_phase_screen(self, scale: float):
        """Generate phase screen with exponential power spectrum."""
        backend = get_backend()
        xp = backend.xp
        
        power_spectrum = xp.exp(-self.r_freq * scale)
        phase = 2 * xp.pi * xp.random.random(power_spectrum.shape).astype(xp.float32)
        complex_fft = power_spectrum * xp.exp(1j * phase)
        screen = xp.fft.fft2(complex_fft)
        
        return screen.real, screen.imag
    
    def _threshold_image(self, screen, sigma: float):
        """Apply threshold and normalize."""
        backend = get_backend()
        xp = backend.xp
        
        std = xp.std(screen)
        if std > 0:
            screen = screen / std
        
        im = (screen - sigma) * (screen > sigma)
        mean = xp.mean(im)
        
        return im / mean if mean > 0 else im
    
    def generate_single(self):
        """Generate a single extended source image."""
        backend = get_backend()
        xp = backend.xp
        
        cfg = self.config
        
        images = []
        n_objects = 1 + int(xp.random.random() * cfg.max_objects_per_image)
        
        for _ in range(n_objects):
            scale = float(xp.random.random() * self.n_pix)
            screen_a, screen_b = self._exponential_phase_screen(scale)
            
            sigma_a = cfg.max_std_threshold * float(xp.random.randn())
            sigma_b = cfg.max_std_threshold * float(xp.random.randn())
            
            im_a = self._threshold_image(screen_a, sigma_a)
            im_b = self._threshold_image(screen_b, sigma_b)
            
            weight_a = float(xp.random.random())
            weight_b = float(xp.random.random())
            
            images.append(weight_a * im_a)
            images.append(weight_b * im_b)
        
        combined = xp.sum(xp.stack(images), axis=0)
        std = xp.std(combined)
        
        return (combined / std if std > 0 else combined).astype(xp.float32)


class FourierPointSourceGenerator(FourierGenerator):
    """
    Fourier generator with added point sources.
    
    Combines extended emission with unresolved point sources.
    """
    
    def generate_single(self):
        """Generate extended + point source image."""
        backend = get_backend()
        xp = backend.xp
        
        cfg = self.config
        
        # Generate extended component
        extended = super().generate_single()
        
        # Add point sources
        n_ps = int(xp.random.randint(0, cfg.max_point_sources + 1))
        
        if n_ps > 0:
            ps_image = xp.zeros((self.n_pix, self.n_pix), dtype=xp.float32)
            
            rows = xp.random.randint(0, self.n_pix, n_ps)
            cols = xp.random.randint(0, self.n_pix, n_ps)
            fluxes = xp.random.random(n_ps).astype(xp.float32) * float(xp.sum(extended))
            
            # Use advanced indexing
            for i in range(n_ps):
                ps_image[int(rows[i]), int(cols[i])] += float(fluxes[i])
            
            # Combine with random weights
            w_ext = float(xp.random.random())
            w_ps = float(xp.random.random())
            combined = w_ext * extended + w_ps * ps_image
        else:
            combined = extended
        
        # Normalize by mean
        mean = xp.mean(combined)
        return (combined / mean if mean > 0 else combined).astype(xp.float32)


# =============================================================================
# Factory and Processing
# =============================================================================

def get_generator(mode: str, config: SourceConfig, n_pix: int) -> SourceGenerator:
    """
    Get generator instance by mode.
    
    Args:
        mode: Generator mode ('fourier', 'fourier_ps', etc.)
        config: Source configuration
        n_pix: Image size
    
    Returns:
        Generator instance
    """
    mode = mode.lower()
    
    generators = {
        'fourier': FourierGenerator,
        'fourier_ps': FourierPointSourceGenerator,
        'fourier_ifu': FourierGenerator,  # Same as fourier for now
    }
    
    if mode not in generators:
        raise ValueError(f"Unknown generator mode: {mode}. Available: {list(generators.keys())}")
    
    return generators[mode](config, n_pix)


def process_sources(
    config: SourceConfig,
    n_pix: int,
    output_dir: str,
    n_files: Optional[int] = None,
    n_per_file: Optional[int] = None,
):
    """
    Generate source images and save to files.
    
    Uses parallel execution on CPU or GPU.
    
    Args:
        config: Source configuration
        n_pix: Image size
        output_dir: Output directory
        n_files: Number of output files (default from config)
        n_per_file: Images per file (default from config)
    """
    from joblib import Parallel, delayed
    
    backend = get_backend()
    xp = backend.xp
    np = backend.np
    
    logger = get_logger()
    
    n_files = n_files or config.n_files
    n_per_file = n_per_file or config.n_images_per_file
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine parallelization
    if config.compute_mode == 'GPU':
        n_workers = get_device_count() or 1
    else:
        n_workers = get_cpu_count()
    
    logger.info(f"Generating {n_files} files with {n_per_file} images each using {n_workers} workers")
    
    def process_file(file_idx: int, device_id: int):
        """Process a single file."""
        # Set device for GPU
        if config.compute_mode == 'GPU' and backend.gpu:
            import cupy as cp
            cp.cuda.Device(device_id).use()
        
        generator = get_generator(config.mode, config, n_pix)
        images = generator.generate_batch(n_per_file)
        
        # Convert to numpy for saving
        if hasattr(images, 'get'):
            images = images.get()
        
        np.save(os.path.join(output_dir, f'IMS_{file_idx}.npy'), images)
    
    # Split files across workers
    file_indices = list(range(n_files))
    work_splits = split_work(file_indices, n_workers)
    
    with Timer("Source generation", logger):
        for batch_idx in range(max(len(split) for split in work_splits)):
            tasks = []
            for worker_id, split in enumerate(work_splits):
                if batch_idx < len(split):
                    tasks.append((split[batch_idx], worker_id))
            
            Parallel(n_jobs=len(tasks), backend='threading')(
                delayed(process_file)(fidx, did) for fidx, did in tasks
            )
    
    logger.info(f"Generated {n_files} source files to {output_dir}")


__all__ = [
    'SourceGenerator',
    'FourierGenerator',
    'FourierPointSourceGenerator',
    'get_generator',
    'process_sources',
]

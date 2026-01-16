"""
Logging utilities with distributed execution support.

Provides rank-aware logging that avoids duplicate messages in
multi-process environments.
"""

from __future__ import annotations

import logging
import sys
import os
from typing import Optional
from datetime import datetime


# =============================================================================
# Distributed State
# =============================================================================

def _get_rank() -> int:
    """Get current process rank."""
    for key in ["RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK"]:
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


def _get_world_size() -> int:
    """Get total number of processes."""
    for key in ["WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"]:
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return 1


# =============================================================================
# Custom Formatter
# =============================================================================

class RankAwareFormatter(logging.Formatter):
    """
    Formatter that includes rank information for distributed jobs.
    """
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt)
        self.rank = _get_rank()
        self.world_size = _get_world_size()
    
    def format(self, record: logging.LogRecord) -> str:
        # Add rank info to record
        record.rank = self.rank
        record.world_size = self.world_size
        
        # Only add rank prefix if distributed
        if self.world_size > 1:
            record.rank_prefix = f"[R{self.rank}/{self.world_size}] "
        else:
            record.rank_prefix = ""
        
        return super().format(record)


# =============================================================================
# Logger Setup
# =============================================================================

_loggers: dict = {}


def get_logger(
    name: str = "nebraa",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    rank_filter: bool = True,
) -> logging.Logger:
    """
    Get a configured logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        rank_filter: If True, only rank 0 logs to console in distributed settings
    
    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear any existing handlers
    
    rank = _get_rank()
    world_size = _get_world_size()
    
    # Console handler (rank-filtered)
    if not rank_filter or rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if world_size > 1:
            console_fmt = "%(rank_prefix)s%(asctime)s [%(levelname)s] %(message)s"
        else:
            console_fmt = "%(asctime)s [%(levelname)s] %(message)s"
        
        console_handler.setFormatter(RankAwareFormatter(
            fmt=console_fmt,
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(console_handler)
    
    # File handler (all ranks)
    if log_file:
        # Include rank in filename if distributed
        if world_size > 1:
            base, ext = os.path.splitext(log_file)
            log_file = f"{base}_rank{rank}{ext}"
        
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(RankAwareFormatter(
            fmt="%(asctime)s [%(levelname)s] [R%(rank)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger


def log_once(logger: logging.Logger, level: int, msg: str, *args, **kwargs):
    """
    Log a message only from rank 0.
    
    Useful for one-time messages in distributed settings.
    """
    if _get_rank() == 0:
        logger.log(level, msg, *args, **kwargs)


class Timer:
    """
    Context manager for timing code blocks with logging.
    """
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or get_logger()
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        log_once(self.logger, logging.INFO, f"[{self.name}] Starting...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        log_once(self.logger, logging.INFO, f"[{self.name}] Completed in {elapsed:.2f}s")
        return False


# =============================================================================
# Progress Tracking
# =============================================================================

class ProgressTracker:
    """
    Simple progress tracker with optional file-based checkpointing.
    """
    
    def __init__(self, total: int, name: str = "Progress", checkpoint_file: Optional[str] = None):
        self.total = total
        self.name = name
        self.current = 0
        self.checkpoint_file = checkpoint_file
        self.logger = get_logger()
        
        # Load checkpoint if exists
        if checkpoint_file and os.path.exists(checkpoint_file):
            self._load_checkpoint()
    
    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        
        # Log progress at intervals
        if self.current % max(1, self.total // 10) == 0:
            pct = 100 * self.current / self.total
            log_once(self.logger, logging.INFO, f"[{self.name}] {pct:.0f}% ({self.current}/{self.total})")
    
    def save_checkpoint(self):
        """Save current progress to file."""
        if self.checkpoint_file:
            with open(self.checkpoint_file, 'w') as f:
                f.write(str(self.current))
    
    def _load_checkpoint(self):
        """Load progress from checkpoint file."""
        try:
            with open(self.checkpoint_file, 'r') as f:
                self.current = int(f.read().strip())
            log_once(self.logger, logging.INFO, f"[{self.name}] Resumed from {self.current}")
        except Exception:
            pass
    
    @property
    def remaining(self) -> int:
        """Items remaining to process."""
        return max(0, self.total - self.current)
    
    @property
    def is_complete(self) -> bool:
        """Check if all items processed."""
        return self.current >= self.total

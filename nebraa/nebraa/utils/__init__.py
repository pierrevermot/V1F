"""
Utility modules.
"""

from .compute import (
    Backend,
    init_backend,
    get_backend,
    get_xp,
    is_gpu,
    split_work,
    get_worker_items,
    get_device_count,
    get_cpu_count,
    WORLD_SIZE,
    RANK,
    LOCAL_RANK,
    IS_CHIEF,
)

from .logging import (
    get_logger,
    log_once,
    Timer,
    ProgressTracker,
)

__all__ = [
    'Backend',
    'init_backend',
    'get_backend',
    'get_xp',
    'is_gpu',
    'split_work',
    'get_worker_items',
    'get_device_count',
    'get_cpu_count',
    'WORLD_SIZE',
    'RANK',
    'LOCAL_RANK',
    'IS_CHIEF',
    'get_logger',
    'log_once',
    'Timer',
    'ProgressTracker',
]

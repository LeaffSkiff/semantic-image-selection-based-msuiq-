"""Distributed training utilities.

This module provides utility functions for distributed training.
For single GPU/CPU usage, these functions return default values.
"""

import functools

import torch
import torch.distributed as dist


def get_dist_info():
    """Get distributed information.

    Returns:
        tuple: (rank, world_size)
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):
    """Decorator to make a function only execute on master process.

    Args:
        func: The function to decorate.

    Returns:
        The decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)
    return wrapper

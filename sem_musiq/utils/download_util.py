"""
Download utilities for Sem-MUSIQ.

精简版：仅保留需要的下载函数。
"""

import os
import math
from torch.hub import download_url_to_file, get_dir
from tqdm import tqdm
from urllib.parse import urlparse


DEFAULT_CACHE_DIR = os.path.join(get_dir(), 'pyiqa')


def sizeof_fmt(size, suffix='B'):
    """
    Return size in human-readable format.

    Example:
        >>> sizeof_fmt(123)
        '123.0B'
        >>> sizeof_fmt(12345)
        '12.1KB'
        >>> sizeof_fmt(1234567)
        '1.2MB'
    """
    multiplier = 1024.0
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < multiplier:
            return f'{size:.1f}{unit}{suffix}'
        size /= multiplier
    return f'{size:.1f}Y{suffix}'


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """
    Load file from http url, download if necessary.

    Args:
        url (str): URL to be downloaded.
        model_dir (str, optional): Path to save the downloaded model.
            If None, use pytorch hub_dir. Default: None.
        progress (bool, optional): Whether to show download progress. Default: True.
        file_name (str, optional): The downloaded file name. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    model_dir = model_dir or DEFAULT_CACHE_DIR

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file

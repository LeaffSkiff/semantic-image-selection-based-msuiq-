"""
Miscellaneous utilities for Sem-MUSIQ.

精简版：仅保留需要的函数。
"""


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

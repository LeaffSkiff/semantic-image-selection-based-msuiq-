"""
Padding utilities for exact same-padding as TensorFlow/MATLAB.
"""

import math
import collections.abc
from itertools import repeat
from typing import Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F


def _ntuple(n):
    """Convert input to tuple of length n."""
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


def symm_pad(im: torch.Tensor, padding: Tuple[int, int, int, int]):
    """
    Symmetric padding same as TensorFlow.

    Ref: https://discuss.pytorch.org/t/symmetric-padding/19866/3
    """
    h, w = im.shape[-2:]
    left, right, top, bottom = padding

    x_idx = torch.arange(-left, w + right)
    y_idx = torch.arange(-top, h + bottom)

    def reflect(x, minx, maxx):
        """Reflect an array around two points."""
        rng = maxx - minx
        double_rng = 2 * rng
        mod = torch.fmod(x - minx, double_rng)
        normed_mod = torch.where(mod < 0, mod + double_rng, mod)
        out = torch.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        return out

    x_pad = reflect(x_idx.to(torch.float32), -0.5, w - 0.5).to(torch.int64)
    y_pad = reflect(y_idx.to(torch.float32), -0.5, h - 0.5).to(torch.int64)
    xx, yy = torch.meshgrid(x_pad, y_pad, indexing='ij')
    return im[..., yy, xx]


def exact_padding_2d(x, kernel, stride=1, dilation=1, mode='same'):
    """
    Calculate exact padding values for 4D tensor inputs.

    Args:
        x: Input tensor of shape (B, C, H, W).
        kernel: Kernel size (int or tuple).
        stride: Stride size (int or tuple). Default: 1.
        dilation: Dilation size (int or tuple). Default: 1.
        mode: Padding mode ('same', 'symmetric', 'replicate', 'circular').

    Returns:
        Padded tensor.
    """
    assert len(x.shape) == 4, f'Only support 4D tensor input, but got {x.shape}'
    kernel = to_2tuple(kernel)
    stride = to_2tuple(stride)
    dilation = to_2tuple(dilation)
    b, c, h, w = x.shape

    h2 = math.ceil(h / stride[0])
    w2 = math.ceil(w / stride[1])
    pad_row = (h2 - 1) * stride[0] + (kernel[0] - 1) * dilation[0] + 1 - h
    pad_col = (w2 - 1) * stride[1] + (kernel[1] - 1) * dilation[1] + 1 - w
    pad_l, pad_r, pad_t, pad_b = (
        pad_col // 2, pad_col - pad_col // 2,
        pad_row // 2, pad_row - pad_row // 2,
    )

    mode = mode if mode != 'same' else 'constant'
    if mode != 'symmetric':
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    elif mode == 'symmetric':
        x = symm_pad(x, (pad_l, pad_r, pad_t, pad_b))

    return x


class ExactPadding2d(nn.Module):
    r"""
    Calculate exact padding for 4D tensor inputs.

    Args:
        kernel: Kernel size (int or tuple).
        stride: Stride size (int or tuple). Default: 1.
        dilation: Dilation size (int or tuple). Default: 1.
        mode: Padding mode ('same', 'symmetric', 'replicate', 'circular').
    """

    def __init__(self, kernel, stride=1, dilation=1, mode='same'):
        super().__init__()
        self.kernel = to_2tuple(kernel)
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)
        self.mode = mode

    def forward(self, x):
        if self.mode is None:
            return x
        return exact_padding_2d(x, self.kernel, self.stride, self.dilation, self.mode)

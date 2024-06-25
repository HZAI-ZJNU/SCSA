import os
import tarfile
import typing as t
import torch

__all__ = ['auto_pad']


def auto_pad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [k // 2 for x in k]  # auto-pad
    return p

"""
NOTE: This module is a modified version of the `spatial` module in lucid so that it works with pytorch tensors.
"""

import numpy as np
import torch


def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies (these are the k-values in DFT)"""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


def fourier_parameterized_image(shape, sd=0.01):
    batch, h, w, ch = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (2, batch, ch) + freqs.shape
    init_val = torch.empty(init_val_size).normal_(std=sd)

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h))
    scale = torch.from_numpy(scale).float()
    init_val *= scale

    init_val = init_val.permute(1, 2, 3, 4, 0)
    image = torch.irfft(init_val, signal_ndim=2, normalized=True, onesided=True, signal_sizes=(h, w))
    image = image / 4.0  # the four here is a magic number defined in the lucid implementation

    return image

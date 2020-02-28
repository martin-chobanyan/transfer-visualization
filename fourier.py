"""
This module is based off of the `spatial` module in lucid.
"""

import numpy as np
import torch


def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies (these are the k-values in DFT)"""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


def fft_image(shape):
    sd = 0.01
    batch, h, w, ch = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (2, batch, ch) + freqs.shape
    init_val_size = (batch, ch) + freqs.shape + (2,)
    
    init_val = np.random.normal(size=init_val_size, scale=sd).astype(np.float32)
    init_val = torch.from_numpy(init_val)
    
    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h))
#     scale *= np.sqrt(w * h)
    scale = torch.from_numpy(scale).float()
    init_val = scale * init_val
    
    init_val = init_val.permute(1, 2, 3, 4, 0)
    
    image_t = torch.irfft(init_val, signal_ndim=2, normalized=True)
    image_t = image_t[:batch, :ch, :h, :w]
    image_t = image_t / 4.0
    
    return image_t

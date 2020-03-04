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


def fourier_parameterized_image(shape=(1, 224, 224, 3), std_dev=0.01):
    """Create the initial image using the Fourier Transform

    Initializing the image with a Gaussian distribution in the pixel space will result in a lot of high frequency
    artifacts. Instead, this function initializes the Gaussian in the Fourier space and then uses the inverse Fourier
    Transform to bring the image back to the pixel space.

    Parameters
    ----------
    shape: tuple[int], optional
        The desired image shape. The format is as follows: (batch_size, height, width, num_channels).
        The default shape is (1, 224, 224, 3), resulting in a single 3-channel image.
    std_dev: float, optional
        The standard deviation used by the normal distribution when initializing the Fourier spectrum.
        Lower values will result in lower frequencies (the image will look more gray).

    Returns
    -------
    torch.FloatTensor
        A pytorch tensor containing the Fourier parameterized images.
    """
    batch, h, w, ch = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (2, batch, ch) + freqs.shape
    init_val = torch.empty(init_val_size).normal_(std=std_dev)

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h))
    scale = torch.from_numpy(scale).float()
    init_val *= scale

    init_val = init_val.permute(1, 2, 3, 4, 0)
    image = torch.irfft(init_val, signal_ndim=2, normalized=True, onesided=True, signal_sizes=(h, w))
    image = image / 4.0  # the four here is a magic number defined in the lucid implementation

    return image

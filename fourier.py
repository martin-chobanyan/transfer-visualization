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


class FourierParam:
    """Initialize an image in the frequency space and define the mapping to the pixel space

    Initializing the image with a Gaussian distribution in the pixel space will result in a lot of high frequency
    artifacts. Instead, this class initializes the Gaussian in the Fourier space. Optimizing this
    Fourier parameterization will have a regularizing effect on the feature visualization.

    Parameters
    ----------
    shape: tuple[int], optional
        The desired image shape. The format is as follows: (batch_size, height, width, num_channels).
        The default shape is (1, 224, 224, 3), resulting in a single 3-channel image.
    std_dev: float, optional
        The standard deviation used by the normal distribution when initializing the Fourier spectrum.
        Lower values will result in lower frequencies (the image will look more gray).
    """
    def __init__(self, shape, std_dev=0.01, decay_power=1.0, device='cpu'):
        self.batch_size, self.height, self.width, self.channels = shape
        self.std_dev = std_dev
        self.decay_power = decay_power
        self.device = device

        # define the sampled frequencies to use in the Fourier spectrum
        # (this will only be used to scale the random initializations)
        self.freqs = rfft2d_freqs(self.height, self.width)
        self.init_val_size = (2, self.batch_size, self.channels) + self.freqs.shape

        # define the scaling
        self.scale = 1.0 / np.maximum(self.freqs, 1.0 / max(self.height, self.width)) ** self.decay_power
        self.scale = torch.from_numpy(self.scale).float().to(self.device)

    def init_spectrum(self):
        """Initialize the Fourier spectrum using a normal distribution

        Returns
        -------
        torch.FloatTensor
        """
        return torch.empty(self.init_val_size).normal_(std=self.std_dev)

    def map_to_pixel_space(self, init_val):
        """Map the initialized frequency space values to the pixel space

        Parameters
        ----------
        init_val: torch.FloatTensor
            The output of the `init_spectrum` method

        Returns
        -------
        torch.FloatTensor
            The (scaled) pixel space representation of the frequency space initialization
        """
        init_val = init_val * self.scale
        init_val = init_val.permute(1, 2, 3, 4, 0)

        image_shape = (self.height, self.width)
        image = torch.irfft(init_val, signal_ndim=2, normalized=True, onesided=True, signal_sizes=image_shape)
        # image = torch.irfft(init_val, signal_ndim=2, normalized=True, signal_sizes=image_shape)
        image /= 4.0  # the four here is a magic number defined in the lucid implementation
        return image

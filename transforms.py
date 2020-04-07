"""
This module provides helper objects for transforming image tensors using the kornia library.
"""

import random
import torch
from kornia.geometry import rotate, scale, translate


class CropTensorPadding:
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, x):
        p = self.padding
        x = x[..., p:-p, p:-p]
        return x


class RotateTensor:
    """Rotate a pytorch tensor

    Rotates a pytorch tensor representing image along its center.
    When calling this object, the input must be a tensor with shape (batch, channel, height, width).

    Parameters
    ----------
    angles: list[float]
        A list of rotation angles to sample from
    """

    def __init__(self, angles, device='cpu'):
        self.angles = angles
        self.device = device

    def __call__(self, x):
        batch_size = x.size(0)
        angles = random.choices(self.angles, k=batch_size)
        angles = torch.tensor(angles).to(self.device)
        return rotate(x, angles)


class ScaleTensor:
    """Scale a pytorch tensor

    When calling this object, the input must be a tensor with shape (batch, channel, height, width).

    Parameters
    ----------
    scale_factors: list[float]
        A list of scale factors to choose from for each image
    """
    def __init__(self, scale_factors, device='cpu'):
        self.scale_factors = scale_factors
        self.device = device

    def __call__(self, x):
        batch_size = x.size(0)
        scale_factors = random.choices(self.scale_factors, k=batch_size)
        scale_factors = torch.tensor(scale_factors).to(self.device)
        return scale(x, scale_factors)


class TranslateTensor:
    """Translate a pytorch tensor

    Performs a translation on a pytorch tensor.
    When calling this object, the input must be a tensor with shape (batch, channel, height, width).

    Parameters
    ----------
    shift: float
        The amount to shift the tensor along its height and width dimensions.
        The direction of the shift is randomly chosen for both dimensions.
    """

    def __init__(self, shift, device='cpu'):
        self.shift = float(shift)
        self.device = device

    def __call__(self, x):
        batch_size = x.size(0)
        options = torch.tensor([self.shift, -self.shift])
        shifts = options[torch.randint(0, 2, (batch_size, 2))].to(self.device)
        return translate(x, shifts)

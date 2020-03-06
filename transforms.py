"""
This module provides helper objects for transforming image tensors using the kornia library.
"""

import random
import torch
from kornia.geometry import rotate, translate


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

    def __init__(self, shift):
        self.shift = float(shift)

    def __call__(self, x):
        # select the amount to shift in both directions
        h = random.choice([self.shift, -self.shift])
        v = random.choice([self.shift, -self.shift])
        return translate(x, torch.tensor([[h, v]]))


class RotateTensor:
    """Rotate a pytorch tensor

    Rotates a pytorch tensor representing image along its center.
    When calling this object, the input must be a tensor with shape (batch, channel, height, width).

    Parameters
    ----------
    angles: list[float]
        A list of rotation angles to sample from
    """

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return rotate(x, torch.tensor(angle))

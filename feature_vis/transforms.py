"""
This module provides helper objects for transforming image tensors using the kornia library.
"""

import random
import torch
from torch.nn import ReflectionPad2d
from torchvision.transforms import Compose, Normalize
from kornia.geometry import rotate, scale, translate
from .utils import torch_uniform

IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDEVS = [0.229, 0.224, 0.225]


class ImagenetNorm(Normalize):
    def __init__(self):
        super().__init__(mean=IMAGENET_MEANS, std=IMAGENET_STDEVS)


# the default set of transformations matches those as in the "Feature Visualization" distill article
# https://distill.pub/2017/feature-visualization/
def get_default_transforms(device):
    return Compose([
        ImagenetNorm(),
        AddBatchDim(),
        ReflectionPad2d(16),
        TranslateTensor(max_shift=16, device=device),
        ScaleTensor(scale_factors=[1, 0.975, 1.025, 0.95, 1.05], device=device),
        RotateTensor(angles=list(range(-5, 6)), device=device),
        TranslateTensor(max_shift=8, device=device),
        CropTensorPadding(16)
    ])


class AddBatchDim:
    def __call__(self, t):
        return t.unsqueeze(0)


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
    max_shift: int
        The maximum amount to shift the tensor along its height and width dimensions.
        The direction of the shift is randomly chosen for both dimensions.
    """

    def __init__(self, max_shift, device='cpu'):
        self.shift = max_shift
        self.device = device

    def __call__(self, x):
        batch_size = x.size(0)
        shifts = torch_uniform(-self.shift, self.shift, (batch_size, 2)).to(self.device)
        return translate(x, shifts)

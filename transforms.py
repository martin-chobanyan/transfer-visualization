import random
import torch
from kornia.geometry import rotate


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


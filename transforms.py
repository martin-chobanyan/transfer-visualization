import random
import torch
from kornia.geometry import rotate


class RotateTensor:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return rotate(x, torch.tensor(angle))

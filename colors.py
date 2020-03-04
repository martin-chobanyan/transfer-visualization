"""
NOTE: This module is a modified version of the `color` module in lucid so that it works with pytorch tensors.
"""

import numpy as np
import torch

COLOR_CORRELATION_SVD_SQRT = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")
MAX_NORM_SVD_SQRT = np.max(np.linalg.norm(COLOR_CORRELATION_SVD_SQRT, axis=0))
COLOR_CORRELATION_NORMALIZED = torch.from_numpy(COLOR_CORRELATION_SVD_SQRT / MAX_NORM_SVD_SQRT)


def _linear_decorelate_color(t):
    if t.numel() % 3 != 0:
        raise ValueError('Input tensor must have three channels!')
    t_flat = t.view(-1, 3)
    t_flat = torch.matmul(t_flat, COLOR_CORRELATION_NORMALIZED.T)
    t = t_flat.view(t.shape)
    return t

def to_valid_rgb(t):
    t = _linear_decorelate_color(t)
    return torch.sigmoid(t)

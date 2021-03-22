"""
NOTE: This module is a modified version of the `color` module in lucid so that it works with pytorch tensors.
"""

import numpy as np
import torch

COLOR_CORRELATION_SVD_SQRT = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")


class DecorrelateColors:
    """Decorrelate the colors in an RGB image

    Parameters
    ----------
    color_corr_svd_sqrt: np.ndarray, optional
        The color correlation matrix (defualt is to use the pre-calculated matrix for ImageNet)
    device: str or torch.device, optional
        The pytorch device (default is cpu)
    """
    def __init__(self, color_corr_svd_sqrt=COLOR_CORRELATION_SVD_SQRT, device='cpu'):
        self.device = device
        self.color_corr_svd_sqrt = color_corr_svd_sqrt
        self.max_norm_svd_sqrt = np.max(np.linalg.norm(self.color_corr_svd_sqrt, axis=0))
        self.normalized_color_corr = torch.from_numpy(self.color_corr_svd_sqrt / self.max_norm_svd_sqrt).to(self.device)

    def __linear_decorrelate_colors(self, t):
        c, h, w = t.shape
        # assert c == 3, 'Input tensor must have three channels!'
        t_flat = t.permute(1, 2, 0)
        t_flat = t_flat.view(-1, c)
        t_flat = torch.matmul(t_flat, self.normalized_color_corr.T)
        t = t_flat.view((h, w, c)).permute(2, 0, 1)
        return t

    def __call__(self, t):
        """Decorrelate colors

        Parameters
        ----------
        t: torch.FloatTensor
            The input image as a pytorch tensor with shape (channels, height, width)

        Returns
        -------
        torch.FloatTensor
            The image as a pytorch tensor with shape (height, width, channels) and values in range (0, 1)
        """
        t = self.__linear_decorrelate_colors(t)
        return torch.sigmoid(t)

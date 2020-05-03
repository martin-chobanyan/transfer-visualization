import torch
from torch.nn import Module, Sequential


def calc_gram_matrix(features):
    """Calculate the gram matrix for a batch of images

    Parameters
    ----------
    features: torch.Tensor
        The batch of feature maps with shape (batch_size, num_channels, height, width)

    Returns
    -------
    torch.Tensor
    """
    b, c, h, w = features.size()
    flat_features = features.view(b, c, h * w)
    gram = torch.bmm(flat_features, flat_features.transpose(1, 2))
    return gram


class GramMatrixDistance(Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img1, img2):
        if img1.size() != img2.size():
            raise ValueError('Input images to GramMatrixLoss must have the same shape!')

        features1 = self.model(img1)
        features2 = self.model(img2)
        b, c, h, w = features1.size()

        gram1 = calc_gram_matrix(features1)
        gram2 = calc_gram_matrix(features2)

        scale = 1 / (4 * (c ** 2) * ((h * w) ** 2))
        gram_diff = (gram1 - gram2) ** 2
        gram_diff = gram_diff.view(b, -1)
        return scale * torch.sum(gram_diff, dim=1)

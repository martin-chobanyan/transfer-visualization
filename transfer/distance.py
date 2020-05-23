import torch
from torch.nn import Module, ModuleList, Sequential


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


class GramMatrixLoss(Module):
    def forward(self, features1, features2):
        # calculate the gram matrices
        gram1 = calc_gram_matrix(features1)
        gram2 = calc_gram_matrix(features2)

        # calculate the denominator of the scaling factor
        b, c, h, w = features1.size()
        n_l, m_l = c, h * w  # as defined in the paper
        scale = 4 * (n_l ** 2) * (m_l ** 2)

        # calculate and scale the sum of sq errors
        gram_err = (gram1 - gram2) ** 2
        gram_err = gram_err.view(b, -1)
        return torch.sum(gram_err, dim=1) / scale


class GramDistanceResnet50(Module):
    def __init__(self, resnet_model, target_layers=None):
        super().__init__()
        self.target_layers = target_layers
        if self.target_layers is None:
            self.target_layers = ['layer1', 'layer2', 'layer3', 'layer4']
        self.layers = ModuleList(self.group_model_layers(resnet_model))
        self.layers.eval()

    def group_model_layers(self, full_model):
        final_layers = []
        layer_bank = []
        for name, child in full_model.named_children():
            layer_bank.append(child)
            if name in self.target_layers:
                final_layers.append(Sequential(*layer_bank))
                final_layers.append(GramMatrixLoss())
                layer_bank.clear()
        return final_layers

    def forward(self, img1, img2):
        if img1.size() != img2.size():
            raise ValueError('Input images to GramMatrixLoss must have the same shape!')
        features1 = img1
        features2 = img2

        losses = []
        for layer in self.layers:
            if isinstance(layer, GramMatrixLoss):
                loss = layer(features1, features2)
                losses.append(loss)
            else:
                features1 = layer(features1)
                features2 = layer(features2)
        losses = torch.stack(losses)
        return losses

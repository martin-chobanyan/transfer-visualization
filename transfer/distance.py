import torch
from torch.nn import Conv2d, Module, ModuleList, Sequential
from torchvision.models import resnet50, vgg19


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


class GramDistanceModel(Module):
    def __init__(self):
        super().__init__()
        self.layers = ModuleList(self.group_model_layers())
        self.layers.eval()
        self.num_gram_layers = sum(1 for layer in self.layers if isinstance(layer, GramMatrixLoss))

    def group_model_layers(self):
        raise NotImplementedError

    def forward(self, img1, img2):
        """Calculate the Gram matrix distances between pairs of images

        Parameters
        ----------
        img1: torch.Tensor
            The first batch of images as a tensor with shape (batch, channels, height, width)
        img2: torch.Tensor
            The second batch of images as a tensor with shape (batch, channels, height, width)

        Returns
        -------
        torch.Tensor
            A tensor containing the Gram matrix distances for each example in the batch with shape
            (batch, layers), where layers corresponds to the number of GramMatrixLoss layers.
        """
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


class GramDistanceVGG19(GramDistanceModel):
    def group_model_layers(self):
        final_layers = []
        layer_bank = []
        model = vgg19(pretrained=True).features
        for child in model.children():
            layer_bank.append(child)
            if isinstance(child, Conv2d):
                final_layers.append(Sequential(*layer_bank))
                final_layers.append(GramMatrixLoss())
                layer_bank.clear()
        return final_layers


class GramDistanceResnet50(GramDistanceModel):
    def group_model_layers(self):
        final_layers = []
        layer_bank = []
        model = resnet50(pretrained=True)
        for name, child in model.named_children():
            layer_bank.append(child)
            if name in ['relu', 'layer1', 'layer2', 'layer3', 'layer4']:
                final_layers.append(Sequential(*layer_bank))
                final_layers.append(GramMatrixLoss())
                layer_bank.clear()
        return final_layers

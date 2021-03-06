"""This file contains callable objects which calculate the distance or similarity between two input images"""
import torch
from torch.nn import Module, ModuleList, Sequential
from torch.nn.functional import cosine_similarity
from torchvision.models import resnet50


# ----------------------------------------------------------------------------------------------------------------------
# Cosine Similarity Metric
# ----------------------------------------------------------------------------------------------------------------------


class CosineSimResnet50(Module):
    """Calculate the cosine similarity of two images using their ResNet50 embeddings before the fully connected layer"""
    def __init__(self):
        super().__init__()
        self.resnet_cnn = Sequential(*list(resnet50(pretrained=True).children())[:-1])
        self.output_dim = 2048

    def forward(self, img1, img2):
        """Calculate the cosine similarity of the ResNet-50 embeddings of two batches of images

        Parameters
        ----------
        img1: torch.Tensor
            The first batch of images as a tensor with shape (batch, channels, height, width)
        img2: torch.Tensor
            The second batch of images as a tensor with shape (batch, channels, height, width)

        Returns
        -------
        torch.Tensor
            A tensor containing the cosine similarity for each example in the batch
        """
        if img1.size() != img2.size():
            raise ValueError('Input images to GramMatrixLoss must have the same shape!')
        batch_size, *_ = img1.size()  # extract the batch size

        self.resnet_cnn.eval()
        with torch.no_grad():
            emb1 = self.resnet_cnn(img1).view(batch_size, self.output_dim)
            emb2 = self.resnet_cnn(img2).view(batch_size, self.output_dim)
        return cosine_similarity(emb1, emb2)


# ----------------------------------------------------------------------------------------------------------------------
# Gram Matrix Distance Metric
# ----------------------------------------------------------------------------------------------------------------------


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
    """Calculate the difference in the Gram Matrix representation of two input feature volumes"""
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
    """A module which calculates the GramMatrixLoss of two images after every bottleneck layer in ResNet50"""
    def __init__(self):
        super().__init__()
        self.layers = ModuleList(self.prepare_layers())
        self.layers.eval()
        self.num_gram_layers = sum(1 for layer in self.layers if isinstance(layer, GramMatrixLoss))

    def prepare_layers(self):
        """Insert GramMatrixLoss layers after every bottleneck layer in the Resnet50 architecture"""
        base_model = resnet50(pretrained=True)
        layers = [
            base_model.conv1,
            GramMatrixLoss(),
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1[0],
            GramMatrixLoss(),
            base_model.layer1[1],
            GramMatrixLoss(),
            base_model.layer1[2],
            GramMatrixLoss(),
            base_model.layer2[0],
            GramMatrixLoss(),
            base_model.layer2[1],
            GramMatrixLoss(),
            base_model.layer2[2],
            GramMatrixLoss(),
            base_model.layer2[3],
            GramMatrixLoss(),
            base_model.layer3[0],
            GramMatrixLoss(),
            base_model.layer3[1],
            GramMatrixLoss(),
            base_model.layer3[2],
            GramMatrixLoss(),
            base_model.layer3[3],
            GramMatrixLoss(),
            base_model.layer3[4],
            GramMatrixLoss(),
            base_model.layer3[5],
            GramMatrixLoss(),
            base_model.layer4[0],
            GramMatrixLoss(),
            base_model.layer4[1],
            GramMatrixLoss(),
            base_model.layer4[2],
            GramMatrixLoss()
        ]
        return layers

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

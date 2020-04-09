import numpy as np
import torch
from visdom import Visdom

vis = Visdom()


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def create_white_noise(img_shape=(224, 224), num_channels=3):
    return torch.rand((num_channels, *img_shape))


def slice_model(model, idx):
    return torch.nn.Sequential(*list(model.children())[:idx])


def torch_uniform(low, high, size):
    return (high - low) * torch.rand(size) + low


def visdom_show(img, window='test'):
    vis.image(np.array(img).transpose((2, 0, 1)), window)

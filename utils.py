import torch


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def create_white_noise(img_shape=(224, 224), num_channels=3):
    return torch.rand((num_channels, *img_shape))


def slice_model(model, idx):
    return torch.nn.Sequential(*list(model.children())[:idx])

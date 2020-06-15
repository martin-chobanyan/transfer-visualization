import torch


def _set_param_training_status(model, status):
    for param in model.parameters():
        param.requires_grad = status
    return model


def freeze_parameters(model):
    return _set_param_training_status(model, status=False)


def unfreeze_parameters(model):
    return _set_param_training_status(model, status=True)


def slice_model(model, idx):
    return torch.nn.Sequential(*list(model.children())[:idx])


def torch_uniform(low, high, size):
    return (high - low) * torch.rand(size) + low

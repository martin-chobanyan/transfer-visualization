"""Utility functions for feature visualization"""
import torch


def _set_param_training_status(model, status):
    for param in model.parameters():
        param.requires_grad = status
    return model


def freeze_parameters(model):
    """Freeze all parameters in a network (set requires_grad to False)

    Parameters
    ----------
    model: torch.nn.Module

    Returns
    -------
    torch.nn.Module
    """
    return _set_param_training_status(model, status=False)


def unfreeze_parameters(model):
    """Unfreeze all parameters in a network (set requires_grad to True)

    Parameters
    ----------
    model: torch.nn.Module

    Returns
    -------
    torch.nn.Module
    """
    return _set_param_training_status(model, status=True)


def slice_model(model, idx):
    """Slice a model at the given index

    This function creates a subset of a pytorch module by creating a Sequential module with the first `idx` number of
    children modules in the original model.

    This function is useful for creating subsets of neural networks in order to visualize channels in their hidden layers.

    Parameters
    ----------
    model: torch.nn.Module
        The original, complete model
    idx: int
        The slicing index. The sliced model will contain as many children as this argument.

    Returns
    -------
    torch.nn.Module
    """
    return torch.nn.Sequential(*list(model.children())[:idx])


def torch_uniform(low, high, size):
    """Sample a pytorch tensor from a uniform distribution

    Parameters
    ----------
    low: float
        The lower bound for the uniform distribution
    high: float
        The upper bound for the uniform distribution
    size: tuple[int]
        The shape for the resulting tensor

    Returns
    -------
    torch.FloatTensor
    """
    return (high - low) * torch.rand(size) + low


def get_device():
    """Get the cuda device if it is available

    Returns
    -------
    torch.device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

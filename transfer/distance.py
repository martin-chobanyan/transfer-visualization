import torch
from torch.nn import Module, Sequential

def calc_gram_matrices(imgs):
    b, c, h, w = imgs.size()
    features = imgs.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    gram /= imgs.numel()  # normalize the gram matrix by the total number of elements
    return gram

# class GramMatrixLoss(Module):
#     def __init__(self, )
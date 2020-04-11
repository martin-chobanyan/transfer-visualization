# public imports
from torch import mean
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import ToPILImage
from tqdm import tqdm

# local imports
from .colors import DecorrelateColors
from .fourier import FourierParam
from .transforms import get_default_transforms


class FeatureVisualizer:
    def __init__(self,
                 model,
                 img_shape=(400, 400),
                 fourier_param=None,
                 color_decorr=None,
                 transforms=None,
                 device='cpu'):
        # initialize the attributes
        self.model = model
        self.img_shape = img_shape
        self.fourier_param = fourier_param
        self.color_decorr = color_decorr
        self.transforms = transforms
        self.device = device

        # set the default fourier parameterization and color decorrelation objects
        if self.fourier_param is None:
            self.fourier_param = FourierParam(shape=(*img_shape, 3),
                                              device=self.device)
        if self.color_decorr is None:
            self.color_decorr = DecorrelateColors(device=self.device)

        if self.transforms is None:
            self.transforms = get_default_transforms(self.device)

    def __retrieve_image(self, x):
        img = self.fourier_param.map_to_pixel_space(x)
        img = self.color_decorr(img)
        return ToPILImage()(img.squeeze().cpu().detach())

    def __call__(self,
                 act_idx,
                 epochs=200,
                 lr=0.05,
                 optim=None,
                 maximize=True,
                 progress=True,
                 hook_output_fn=None,
                 *args,
                 **kwargs):
        # initialize the image in fourier space
        x = self.fourier_param.init_spectrum()
        x = x.to(self.device)
        x = nn.Parameter(x, requires_grad=True)

        # set up the optimizer
        if optim is None:
            optimizer = Adam([x], lr=lr)
        else:
            optimizer = optim([x], lr=lr, *args, **kwargs)

        epoch_range = tqdm(range(epochs)) if progress else range(epochs)
        for _ in epoch_range:
            optimizer.zero_grad()

            img = self.fourier_param.map_to_pixel_space(x)
            img = self.color_decorr(img)
            img = self.transforms(img)

            out = self.model(img)[0]
            if hook_output_fn is not None:
                out = hook_output_fn(self.model)

            act = mean(out[act_idx])
            if maximize:
                act *= -1
            act.backward()
            optimizer.step()

        return self.__retrieve_image(x)

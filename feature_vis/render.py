"""The render module defines an object which can be used to create feature visualizations"""
from torch import mean
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from .colors import DecorrelateColors
from .fourier import FourierParam
from .transforms import get_default_transforms
from .utils import get_device


class FeatureVisualizer:
    """This class creates feature visualizations from a model's output

    The feature visualization is created using an index in the model's output activation.
    In order to visualize a hidden component in the network, a subset of the model must be created such that its output
    activation includes the target component.

    Alternatively, you can set up pytorch hooks in the model before passing it in as input. This then requires the
    user to define a custom function which collects the activation from the hidden layer using the pytorch hook.
    See the __call__ method of this class for more details.

    Parameters
    ----------
    model: torch.nn.Module
        The pytorch network to visualize

    img_shape: tuple[int], optional
        The desired shape of the feature visualization (default is (400, 400))

    fourier_param: FourierParam, optional
        The fourier parametrization object

    color_decorr: DecorrelateColors, optional
        The color decorrelation object

    transforms: callable, optional
        The image transformations to perform during the optimization
        (see get_default_transforms in the transforms module)

    device: str or torch.device
        The pytorch device to use
    """

    def __init__(self,
                 model,
                 img_shape=(400, 400),
                 fourier_param=None,
                 color_decorr=None,
                 transforms=None,
                 device=None):
        # initialize the attributes
        self.model = model
        self.img_shape = img_shape
        self.fourier_param = fourier_param
        self.color_decorr = color_decorr
        self.transforms = transforms
        self.device = get_device() if device is None else device

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
                 epochs=256,
                 lr=0.05,
                 optim=None,
                 maximize=True,
                 progress=True,
                 hook_output_fn=None,
                 *args,
                 **kwargs):
        """Create the feature visualization given the activation index in the model's output

        Parameters
        ----------
        act_idx: int
            The activation index in the model's output to use for the feature visualization

        epochs: int, optional
            The number of epochs/iterations to run the feature visualization (default=256)

        lr: float, optional
            The learning rate for the optimization (default=0.05)

        optim: pytorch optimizer, optional (default=Adam)

        maximize: bool, optional
            If true, then the feature visualization will maximize the target activation (default).
            If false, then the feature visualization will minimize the target activation.

        progress: bool, optional
            If true, then a tqdm progress bar of the optimization process is included (default=True)

        hook_output_fn: callable, optional
            As an alternative to using a model's output, a hook function can be specified which collects the output
            from the target layer (using pytorch hooks) given the model as input.

        If optim is not None, then all other arguments to this method are fowarded to the custom optimizer as
        *args and **kwargs

        Returns
        -------
        PIL.Image
        """
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

"""This script uses the local `feature_vis` package to generate feature visualizations for target layers"""
import os

import torch
from torchvision.models import resnet50
from tqdm import tqdm

from feature_vis.render import FeatureVisualizer
from feature_vis.utils import slice_model
from finetune_dog_classifier import load_resnet50_layer3_bottleneck5, IMAGE_SHAPE
from train_utils import create_folder, get_device

# define the directories as constants
ROOT_DIR = '/home/mchobanyan/data/research/transfer/vis/'
IMAGENET_DIR = os.path.join(ROOT_DIR, 'pretrained-resnet50')
CAR_DIR = os.path.join(ROOT_DIR, f'finetune-car-resnet50')
DOG_DIR = os.path.join(ROOT_DIR, f'finetune-dog-resnet50')


def visualize(model, channels, device, img_shape=IMAGE_SHAPE, output_dir=None):
    """Create feature visualizations for the target channels in the model's output volume

    The input channel indices must align with the output tensor of the model.

    If the target channel for the feature visualization is in a hidden layer,
    then the input model must be sliced such that its output volume contains the target channel.

    Parameters
    ----------
    model: torch.nn.Module
        The pytorch module to visualize
    channels: list[int]
        A list of integer indices for the target channels in the model's output
    device: torch.device
    img_shape: tuple[int], optional
        The desired (height, width) shape for the feature visualizations (default is (400, 400))
    output_dir: str, optional
        The path to the output directory where the feature visualization of each channel will be saved as
        "channel{index}.png". If None, then the images will not be saved (default is None).

    Returns
    -------
    dict[int, PIL.Image]
        A dictionary mapping the index of a channel in the model's output to its feature visualization as a PIL Image.
    """
    features = dict()
    save_images = (output_dir is not None)
    visualize_features = FeatureVisualizer(model, img_shape, device=device)
    for channel in tqdm(channels):
        img = visualize_features(act_idx=channel, progress=False)
        features[channel] = img
        if save_images:
            img.save(os.path.join(output_dir, f'channel{channel}.png'))
    return features


def visualize_resnet50_layer3_bottleneck5(resnet50_model, root_dir, device):
    """Create feature visualizations for all channels in layer3-bottleneck5 of a ResNet-50 network

    A subset of the model is created by keeping all layers up to and including layer3-bottleneck5.

    Parameters
    ----------
    resnet50_model: torchvision.models.ResNet
        A trained pytorch ResNet-50 network
    root_dir: str
        The root directory to use for saving the resulting images.
        The images are saved under <root_dir>/features/layer3-bottleneck5/
    device: torch.device
    """
    output_dir = os.path.join(root_dir, 'features', 'layer3-bottleneck5')
    create_folder(output_dir)

    # this gets us the output of layer3-bottleneck5
    model_subset = slice_model(resnet50_model, 7)
    model_subset = model_subset.eval()

    num_channels = 1024
    return visualize(model_subset, range(num_channels), device, output_dir=output_dir)


def visualize_resnet50_layer4_bottleneck0(resnet50_model, root_dir, device):
    """Create feature visualizations for all channels in layer4-bottleneck0 of a ResNet-50 network

    A subset of the model is created by keeping all layers up to and including layer4-bottleneck0.

    Parameters
    ----------
    resnet50_model: torchvision.models.ResNet
        A trained pytorch ResNet-50 network
    root_dir: str
        The root directory to use for saving the resulting images.
        The images are saved under <root_dir>/features/layer4-bottleneck0/
    device: torch.device
    """
    output_dir = os.path.join(root_dir, 'features', 'layer4-bottleneck0')
    create_folder(output_dir)

    # this gets us the output of layer4-bottleneck0
    model_subset = slice_model(resnet50_model, 7)
    model_subset.add_module('layer4_bottleneck0', resnet50_model.layer4[0])
    model_subset = model_subset.eval()

    num_channels = 2048
    return visualize(model_subset, range(num_channels), device, output_dir=output_dir)


def visualize_resnet50_layer4_bottleneck1(resnet50_model, root_dir, device):
    """Create feature visualizations for all channels in layer4-bottleneck1 of a ResNet-50 network

    A subset of the model is created by keeping all layers up to and including layer4-bottleneck1.

    Parameters
    ----------
    resnet50_model: torchvision.models.ResNet
        A trained pytorch ResNet-50 network
    root_dir: str
        The root directory to use for saving the resulting images.
        The images are saved under <root_dir>/features/layer4-bottleneck1/
    device: torch.device
    """
    output_dir = os.path.join(root_dir, 'features', 'layer4-bottleneck1')
    create_folder(output_dir)

    # this gets us the output of layer4-bottleneck1
    model_subset = slice_model(resnet50_model, 7)
    model_subset.add_module('layer4_bottleneck0', resnet50_model.layer4[0])
    model_subset.add_module('layer4_bottleneck1', resnet50_model.layer4[1])
    model_subset = model_subset.eval()

    num_channels = 2048
    return visualize(model_subset, range(num_channels), device, output_dir=output_dir)


def visualize_resnet50_layer4_bottleneck2(resnet50_model, root_dir, device):
    """Create feature visualizations for all channels in layer4-bottleneck2 of a ResNet-50 network

    A subset of the model is created by keeping all layers up to and including layer4-bottleneck2.

    Parameters
    ----------
    resnet50_model: torchvision.models.ResNet
        A trained pytorch ResNet-50 network
    root_dir: str
        The root directory to use for saving the resulting images.
        The images are saved under <root_dir>/features/layer4-bottleneck2/
    device: torch.device
    """
    output_dir = os.path.join(root_dir, 'features', 'layer4-bottleneck2')
    create_folder(output_dir)

    # this gets us the output of layer4-bottleneck2
    model_subset = slice_model(resnet50_model, 8)
    model_subset = model_subset.eval()

    num_channels = 2048
    return visualize(model_subset, range(num_channels), device, output_dir=output_dir)


def visualize_target_layers(resnet50_model, target_dir, device=None):
    """Create and save feature visualizations for the channels in the final four bottleneck layers of a ResNet-50 model

    Target layers are:
    1. layer3-bottleneck5
    2. layer4-bottleneck0
    3. layer4-bottleneck1
    4. layer4-bottleneck2

    Parameters
    ----------
    resnet50_model: torchvision.models.ResNet
        The full ResNet-50 model
    target_dir: str
        The target directory where the visualizations will be saved (in nested subdirectories)
    device: torch.device, optional
        By default this will attempt to find and use a cuda device
    """
    if device is None:
        device = get_device()
    resnet50_model = resnet50_model.to(device)
    print('Visualizing layer3-bottleneck5:')
    visualize_resnet50_layer3_bottleneck5(resnet50_model, target_dir, device)
    print('\nVisualizing layer4-bottleneck0:')
    visualize_resnet50_layer4_bottleneck0(resnet50_model, target_dir, device)
    print('\nVisualizing layer4-bottleneck1:')
    visualize_resnet50_layer4_bottleneck1(resnet50_model, target_dir, device)
    print('\nVisualizing layer4-bottleneck2:')
    visualize_resnet50_layer4_bottleneck2(resnet50_model, target_dir, device)


def load_finetuned_renet50_model(filepath, num_classes):
    """Initialize a ResNet-50 network and load the state saved from the fine-tuning stage

    Parameters
    ----------
    filepath: str
        The filepath of the saved state
    num_classes: int
        The number of classes for the fine-tuned task (this is needed to initialize the network)

    Returns
    -------
    torchvision.models.ResNet
    """
    model = load_resnet50_layer3_bottleneck5(num_classes)
    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict)
    return model


def visualize_imagenet_classifier():
    """Create the visualizations for the base ResNet-50 network pre-trained on ImageNet"""
    print('Creating feature visualizations for ResNet-50 pre-trained on ImageNet')
    model = resnet50(pretrained=True)
    visualize_target_layers(model, IMAGENET_DIR)


def visualize_dog_classifier(epoch_idx):
    """Create the visualizations for the ResNet-50 network fine-tuned on the dog dataset"""
    print('Creating feature visualizations for ResNet-50 fine-tuned on the dog dataset')
    model_path = os.path.join(DOG_DIR, 'models', f'model_epoch{epoch_idx}.pt')
    model = load_finetuned_renet50_model(model_path, num_classes=120)
    visualize_target_layers(model, DOG_DIR)


def visualize_car_classifier(epoch_idx):
    """Create the visualizations for the ResNet-50 network fine-tuned on the car dataset"""
    print('Creating feature visualizations for ResNet-50 fine-tuned on the car dataset')
    model_path = os.path.join(CAR_DIR, 'models', f'model_epoch{epoch_idx}.pt')
    model = load_finetuned_renet50_model(model_path, num_classes=196)
    visualize_target_layers(model, CAR_DIR)


if __name__ == '__main__':
    visualize_imagenet_classifier()
    visualize_dog_classifier(epoch_idx=59)  # epoch 59 produced the best results for the dog dataset
    visualize_car_classifier(epoch_idx=29)  # epoch 29 produced the best results for the car dataset

import os

import torch
from torchvision.models import resnet50
from tqdm import tqdm

from feature_vis.render import FeatureVisualizer
from feature_vis.utils import slice_model
from finetune_dog_classifier import load_resnet50_layer3_bottleneck5, IMAGE_SHAPE
from train_utils import create_folder, get_device


def visualize_model(model, channels, device, img_shape=IMAGE_SHAPE, output_dir=None):
    features = dict()
    save_images = (output_dir is not None)
    visualize_features = FeatureVisualizer(model, img_shape, device=device)
    for channel in tqdm(channels):
        img = visualize_features(act_idx=channel, progress=False)
        features[channel] = img
        if save_images:
            img.save(os.path.join(output_dir, f'channel{channel}.png'))
    return features


def visualize_resnet50_layer3_bottleneck5(resnet_model, root_dir, device):
    """Visualize the channel features for the model at layer3-bottleneck5"""
    output_dir = os.path.join(root_dir, 'features', 'layer3-bottleneck5')
    create_folder(output_dir)

    model_subset = slice_model(resnet_model, 7)  # this gets us the output of layer3-bottleneck5
    model_subset = model_subset.eval()

    num_channels = 1024
    return visualize_model(model_subset, range(num_channels), device, output_dir=output_dir)


def visualize_resnet50_layer4_bottleneck0(resnet_model, root_dir, device):
    """Visualize the channel features for the model at layer3-bottleneck5"""
    output_dir = os.path.join(root_dir, 'features', 'layer4-bottleneck0')
    create_folder(output_dir)

    model_subset = slice_model(resnet_model, 7)  # this gets us the output of layer3-bottleneck5
    model_subset.add_module('layer4_bottleneck0', resnet_model.layer4[0])
    model_subset = model_subset.eval()

    num_channels = 2048
    return visualize_model(model_subset, range(num_channels), device, output_dir=output_dir)


def visualize_resnet50_layer4_bottleneck1(resnet_model, root_dir, device):
    """Visualize the channel features for the model at layer3-bottleneck5"""
    output_dir = os.path.join(root_dir, 'features', 'layer4-bottleneck1')
    create_folder(output_dir)

    model_subset = slice_model(resnet_model, 7)  # this gets us the output of layer3-bottleneck5
    model_subset.add_module('layer4_bottleneck0', resnet_model.layer4[0])
    model_subset.add_module('layer4_bottleneck1', resnet_model.layer4[1])
    model_subset = model_subset.eval()

    num_channels = 2048
    return visualize_model(model_subset, range(num_channels), device, output_dir=output_dir)


def visualize_resnet50_layer4_bottleneck2(resnet_model, root_dir, device):
    """Visualize the channel features for the model at layer3-bottleneck5"""
    output_dir = os.path.join(root_dir, 'features', 'layer4-bottleneck2')
    create_folder(output_dir)

    model_subset = slice_model(resnet_model, 8)  # this gets us the output of layer4-bottleneck2
    model_subset = model_subset.eval()

    num_channels = 2048
    return visualize_model(model_subset, range(num_channels), device, output_dir=output_dir)


def visualize_imagenet_classifier():
    print('Visualizing resnet50 features pre-trained on ImageNet')
    DEVICE = get_device()
    model = resnet50(pretrained=True)
    model = model.to(DEVICE)

    ROOT_DIR = '/home/mchobanyan/data/research/transfer/vis/pretrained-resnet50'

    print('Visualizing layer3-bottleneck5:')
    visualize_resnet50_layer3_bottleneck5(model, ROOT_DIR, DEVICE)

    print('\nVisualizing layer4-bottleneck0:')
    visualize_resnet50_layer4_bottleneck0(model, ROOT_DIR, DEVICE)

    print('\nVisualizing layer4-bottleneck1:')
    visualize_resnet50_layer4_bottleneck1(model, ROOT_DIR, DEVICE)

    print('\nVisualizing layer4-bottleneck2:')
    visualize_resnet50_layer4_bottleneck2(model, ROOT_DIR, DEVICE)


def visualize_dog_classifier():
    print('Visualizing resnet50 features fine-tuned on the dog breed dataset')
    DEVICE = get_device()
    model = load_resnet50_layer3_bottleneck5(num_classes=120)
    model = model.to(DEVICE)

    ROOT_DIR = '/home/mchobanyan/data/research/transfer/vis/finetune-dog-resnet50/'
    model_dir = os.path.join(ROOT_DIR, 'models')

    # load the state of the model at the 60th epoch
    epoch_idx = 59
    state_dict = torch.load(os.path.join(model_dir, f'model_epoch{epoch_idx}.pt'))
    model.load_state_dict(state_dict)

    print('Visualizing layer3-bottleneck5:')
    visualize_resnet50_layer3_bottleneck5(model, ROOT_DIR, DEVICE)

    print('\nVisualizing layer4-bottleneck0:')
    visualize_resnet50_layer4_bottleneck0(model, ROOT_DIR, DEVICE)

    print('\nVisualizing layer4-bottleneck1:')
    visualize_resnet50_layer4_bottleneck1(model, ROOT_DIR, DEVICE)

    print('\nVisualizing layer4-bottleneck2:')
    visualize_resnet50_layer4_bottleneck2(model, ROOT_DIR, DEVICE)


def visualize_car_classifier():
    print('Visualizing resnet50 features fine-tuned on the car model dataset')
    DEVICE = get_device()
    model = load_resnet50_layer3_bottleneck5(num_classes=196)
    model = model.to(DEVICE)

    ROOT_DIR = '/home/mchobanyan/data/research/transfer/vis/finetune-car-resnet50/'
    model_dir = os.path.join(ROOT_DIR, 'models')

    # load the state of the model at the 30th epoch
    epoch_idx = 29
    state_dict = torch.load(os.path.join(model_dir, f'model_epoch{epoch_idx}.pt'))
    model.load_state_dict(state_dict)

    print('Visualizing layer3-bottleneck5:')
    visualize_resnet50_layer3_bottleneck5(model, ROOT_DIR, DEVICE)

    print('\nVisualizing layer4-bottleneck0:')
    visualize_resnet50_layer4_bottleneck0(model, ROOT_DIR, DEVICE)
    #
    print('\nVisualizing layer4-bottleneck1:')
    visualize_resnet50_layer4_bottleneck1(model, ROOT_DIR, DEVICE)

    print('\nVisualizing layer4-bottleneck2:')
    visualize_resnet50_layer4_bottleneck2(model, ROOT_DIR, DEVICE)


if __name__ == '__main__':
    visualize_imagenet_classifier()
    visualize_dog_classifier()
    visualize_car_classifier()

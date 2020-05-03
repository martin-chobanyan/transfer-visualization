import os

import torch
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


if __name__ == '__main__':
    device = get_device()
    model = load_resnet50_layer3_bottleneck5(num_classes=120)
    model = model.to(device)

    root_dir = '/home/mchobanyan/data/research/transfer/vis/finetune-dog-resnet50/'
    model_dir = os.path.join(root_dir, 'models')
    output_dir = os.path.join(root_dir, 'features', 'layer3-bottleneck5')
    create_folder(output_dir)

    # load the state of the model at the 60th epoch
    epoch_idx = 59
    state_dict = torch.load(os.path.join(model_dir, f'model_epoch{epoch_idx}.pt'))
    model.load_state_dict(state_dict)
    model_subset = slice_model(model, 7)  # this gets us the output of layer3-bottleneck5
    model_subset = model_subset.eval()

    # visualize the channel features for the model at layer3-bottleneck5
    num_channels = 1024
    images = visualize_model(model_subset, range(num_channels), device, output_dir=output_dir)

    # channels = list(range(500, 600))
    # for channel_idx in channels:
    #     create_folder(os.path.join(output_dir, f'channel{channel_idx}'))
    # n_models = len(os.listdir(model_dir))
    # # for i in tqdm(range(n_models)):
    # for i in [80]:
    #     filename = f'model_epoch{i}.pt'
    #     state_dict = torch.load(os.path.join(model_dir, filename))
    #     model.load_state_dict(state_dict)
    #     model_subset = slice_model(model, 7)  # layer 3 bottleneck 5
    #     # model_subset = slice_model(model, 8)  # layer 4 bottleneck 2
    #     model_subset = model_subset.eval()
    #
    #     result = visualize_model(model_subset, channels, device)
    #     for channel_idx, img in result.items():
    #         img.save(os.path.join(output_dir, f'channel{channel_idx}', f'channel{channel_idx}_epoch{i}.png'))

import os

import torch
from tqdm import tqdm

from feature_vis.render import FeatureVisualizer
from feature_vis.utils import slice_model
from finetune_dog_classifier import load_resnet50_layer3_bottleneck5, IMAGE_SHAPE
from train_utils import create_folder


def visualize_model(model, channels, device, img_shape=IMAGE_SHAPE):
    features = dict()
    visualize_features = FeatureVisualizer(model, img_shape, device=device)
    for channel in tqdm(channels):
        img = visualize_features(act_idx=channel, progress=False, epochs=256)
        features[channel] = img
    return features


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_resnet50_layer3_bottleneck5(num_classes=120)
    model = model.to(device)

    root_dir = '/home/mchobanyan/data/research/transfer/vis/finetune-resnet50-layer3-bottleneck5/'
    feature_dir = os.path.join(root_dir, 'features')
    output_dir = os.path.join(feature_dir, 'finetune-resnet50-layer3-bottleneck5')
    model_dir = os.path.join(root_dir, 'models')

    channels = list(range(500, 600))
    for channel_idx in channels:
        create_folder(os.path.join(output_dir, f'channel{channel_idx}'))

    n_models = len(os.listdir(model_dir))
    # for i in tqdm(range(n_models)):
    for i in [80]:
        filename = f'model_epoch{i}.pt'
        state_dict = torch.load(os.path.join(model_dir, filename))
        model.load_state_dict(state_dict)
        model_subset = slice_model(model, 7)  # layer 3 bottleneck 5
        # model_subset = slice_model(model, 8)  # layer 4 bottleneck 2
        model_subset = model_subset.eval()

        result = visualize_model(model_subset, channels, device)
        for channel_idx, img in result.items():
            img.save(os.path.join(output_dir, f'channel{channel_idx}', f'channel{channel_idx}_epoch{i}.png'))

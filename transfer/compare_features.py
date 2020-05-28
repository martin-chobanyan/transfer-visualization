"""
Prerequisite: This script assumes that the `find_bad_channels.py` script has already been executed.
"""

import os
from statistics import mean

from numpy import log
from pandas import read_csv, concat
from PIL import Image
from torch import no_grad
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from dataset import ImagePairs
from distance import GramDistanceResnet50
from train_utils import create_folder, get_device, Logger
from feature_vis.transforms import ImagenetNorm

# define the constants
ROOT_DIR = '/home/mchobanyan/data/research/transfer/vis/'
IMAGENET_DIR = os.path.join(ROOT_DIR, 'pretrained-resnet50')
DOG_DIR = os.path.join(ROOT_DIR, 'finetune-dog-resnet50')
CAR_DIR = os.path.join(ROOT_DIR, 'finetune-car-resnet50')
COMPARISON_DIR = os.path.join(ROOT_DIR, 'comparisons')
MOST_DIFFERENT_DIR = os.path.join(ROOT_DIR, 'most_different')
MOST_SIMILAR_DIR = os.path.join(ROOT_DIR, 'most_similar')
GRAY_CHANNELS_PATH = '/home/mchobanyan/data/research/transfer/vis/gray-channels.csv'

LAYERS = ['layer3-bottleneck5', 'layer4-bottleneck0', 'layer4-bottleneck1', 'layer4-bottleneck2']
DEVICE = get_device()


class FindGrayChannels:
    """Find the channels with 'grayed-out' feature visualizations"""

    def __init__(self, gray_channel_df):
        self.gray_channel_df = gray_channel_df

    def __call__(self, domain, layer):
        model_name = f'finetune-{domain}-resnet50'
        mask = (self.gray_channel_df['model'] == model_name) & (self.gray_channel_df['layer'] == layer)
        bad_channels = self.gray_channel_df.loc[mask, 'channel'].values
        bad_channels = [int(c.split('channel')[-1]) for c in bad_channels]
        return bad_channels


def concat_horizontally(img1, img2):
    canvas = Image.new('RGB', (img1.width + img2.width, img1.height))
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (img1.width, 0))
    return canvas


def compare_images_with_gram(image_pairs, gram_fn, output_path, device, transforms, title=''):
    # set up the logger
    header = ['channel', 'gram_avg'] + [f'gram_layer{i}' for i in range(1, gram_fn.num_gram_layers + 1)]
    logger = Logger(output_path, header=header)

    for channel_idx in tqdm(range(len(image_pairs)), desc=title):
        img1, img2 = image_pairs[channel_idx]
        img1 = transforms(img1).unsqueeze(0).to(device)
        img2 = transforms(img2).unsqueeze(0).to(device)

        # calculate the gram distances
        with no_grad():
            gram_distances = gram_fn(img1, img2).squeeze().tolist()
        gram_avg = mean(gram_distances)

        logger.add_entry(*([channel_idx, gram_avg] + gram_distances))


def calculate_gram_distances(gram_fn, imagenet_dir, target_dir, output_dir):
    create_folder(output_dir)
    image_transforms = Compose([ToTensor(), ImagenetNorm()])

    # compare model pre-trained on ImageNet vs fine-tuned on the dog breed dataset
    for layer in LAYERS:
        img_dir1 = os.path.join(imagenet_dir, 'features', layer)
        img_dir2 = os.path.join(target_dir, 'features', layer)
        image_pairs = ImagePairs(img_dir1, img_dir2)

        output_path = os.path.join(output_dir, f'imagenet-vs-dogs-{layer}.csv')
        compare_images_with_gram(image_pairs, gram_fn, output_path, DEVICE, image_transforms, layer)


def run_gram_distance_calculations():
    gram_distance_fn = GramDistanceResnet50().to(DEVICE)

    print('Comparing ImageNet vs Dog Breeds feature visualizations')
    calculate_gram_distances(gram_distance_fn, IMAGENET_DIR, DOG_DIR, os.path.join(COMPARISON_DIR, 'dog'))
    print()

    print('Comparing ImageNet vs Car Models feature visualizations')
    calculate_gram_distances(gram_distance_fn, IMAGENET_DIR, CAR_DIR, os.path.join(COMPARISON_DIR, 'car'))


if __name__ == '__main__':
    # Uncomment the following line to create the gram distance CSVs
    # run_gram_distance_calculations()

    # instantiate the helper object to find the "gray channels" given the model and layer
    find_gray_channels = FindGrayChannels(read_csv(GRAY_CHANNELS_PATH))

    domain = 'dog'
    comparison_dir = os.path.join(COMPARISON_DIR, domain)
    for filename in os.listdir(comparison_dir):
        # --- read and prep the data ---
        # read the gram distance comparison CSV
        basename, _ = os.path.splitext(filename)
        layer = '-'.join(basename.split('-')[-2:])
        df = read_csv(os.path.join(comparison_dir, filename))

        # log transform all gram distances
        for col in df.columns:
            if 'gram_layer' in col:
                df[col] = log(df[col])

        # drop the old gram average column
        df = df.drop(labels='gram_avg', axis='columns')

        # --- prep the directories ---
        imagenet_dir = os.path.join(IMAGENET_DIR, 'features', layer)
        target_dir = os.path.join(DOG_DIR, 'features', layer)

        most_diff_dir = os.path.join(MOST_DIFFERENT_DIR, domain, layer)
        most_sim_dir = os.path.join(MOST_SIMILAR_DIR, domain, layer)

        create_folder(most_diff_dir)
        create_folder(most_sim_dir)

        # target layers include all layers in the resnet50 Sequential module
        # except for the first conv layer and the last bottleneck
        target_layers = [f'gram_layer{i}' for i in range(2, 17)]

        # remove all instances of "gray channels" from the comparisons
        bad_channels = find_gray_channels(domain, layer)
        texture_diffs = df.loc[~df['channel'].isin(bad_channels)].copy()

        # calculate the average gram value and drop the original gram columns
        original_cols = [col for col in texture_diffs.columns if 'gram_layer' in col]
        texture_diffs['gram'] = texture_diffs[target_layers].mean(axis=1)
        texture_diffs = texture_diffs.drop(labels=original_cols, axis='columns')

        # sort the channels by their gram distances
        texture_diffs = texture_diffs.sort_values('gram')
        texture_diffs = texture_diffs.reset_index(drop=True)

        # retrieve the top-k and bottom-k channels wrt to gram distance
        k = 20
        n = len(texture_diffs)

        closest_channels = texture_diffs.loc[:k, 'channel'].values
        furthest_channels = texture_diffs.loc[n - k:, 'channel'].values[::-1]

        for rank, channel_idx in enumerate(closest_channels):
            channel_name = f'channel{channel_idx}.png'
            img1 = Image.open(os.path.join(imagenet_dir, channel_name))
            img2 = Image.open(os.path.join(target_dir, channel_name))
            img = concat_horizontally(img1, img2)
            img.save(os.path.join(most_sim_dir, f'rank{rank}_{channel_name}'))

        for rank, channel_idx in enumerate(furthest_channels):
            channel_name = f'channel{channel_idx}.png'
            img1 = Image.open(os.path.join(imagenet_dir, channel_name))
            img2 = Image.open(os.path.join(target_dir, channel_name))
            img = concat_horizontally(img1, img2)
            img.save(os.path.join(most_diff_dir, f'rank{rank}_{channel_name}'))

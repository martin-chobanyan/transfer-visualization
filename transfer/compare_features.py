"""
Prerequisite: This script assumes that the `find_bad_channels.py` script has already been executed.
"""

import os

from pandas import read_csv
from PIL import Image
from torch import no_grad
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from dataset import ImagePairs
from distance import CosineSimResnet50
from train_utils import create_folder, get_device, Logger
from feature_vis.transforms import ImagenetNorm

# define the constants
ROOT_DIR = '/home/mchobanyan/data/research/transfer/vis/'
IMAGENET_DIR = os.path.join(ROOT_DIR, 'pretrained-resnet50')
DOG_DIR = os.path.join(ROOT_DIR, 'finetune-dog-resnet50')
CAR_DIR = os.path.join(ROOT_DIR, 'finetune-car-resnet50')
COMPARISON_DIR = os.path.join(ROOT_DIR, 'comparisons')
MOST_DIFFERENT_DIR = os.path.join(ROOT_DIR, 'most-different')
MOST_SIMILAR_DIR = os.path.join(ROOT_DIR, 'most-similar')
GRAY_CHANNELS_PATH = '/home/mchobanyan/data/research/transfer/vis/gray-channels.csv'

LAYERS = ['layer3-bottleneck5', 'layer4-bottleneck0', 'layer4-bottleneck1', 'layer4-bottleneck2']
DEVICE = get_device()
K = 20


# ----------------------------------------------------------------------------------------------------------------------
# Cosine similarity functions
# ----------------------------------------------------------------------------------------------------------------------


def compare_images_with_cosine_sim(image_pairs, model, output_path, device, transforms, title=''):
    """
    Parameters
    ----------
    image_pairs: ImagePairs
    model: CosineSimResnet50
    output_path: str
    device: torch.device
    transforms: callable
    title: str
    """
    # set up the logger
    header = ['channel'] + [f'cos_sim_layer{i}' for i in range(1, model.num_emb_layers + 1)]
    logger = Logger(output_path, header=header)

    for channel_idx in tqdm(range(len(image_pairs)), desc=title):
        img1, img2 = image_pairs[channel_idx]
        img1 = transforms(img1).unsqueeze(0).to(device)
        img2 = transforms(img2).unsqueeze(0).to(device)

        # calculate the cosine similarities
        with no_grad():
            cosine_sims = model(img1, img2).squeeze().tolist()
        logger.add_entry(channel_idx, *cosine_sims)


def run_cosine_sim(model, imagenet_dir, target_dir, output_dir):
    create_folder(output_dir)
    image_transforms = Compose([ToTensor(), ImagenetNorm()])

    # compare model pre-trained on ImageNet vs fine-tuned on the dog breed dataset
    for layer in LAYERS:
        img_dir1 = os.path.join(imagenet_dir, 'features', layer)
        img_dir2 = os.path.join(target_dir, 'features', layer)
        image_pairs = ImagePairs(img_dir1, img_dir2)

        output_path = os.path.join(output_dir, f'resnet50-cosine-sim-{layer}.csv')
        compare_images_with_cosine_sim(image_pairs, model, output_path, DEVICE, image_transforms, layer)


# ----------------------------------------------------------------------------------------------------------------------
# Gram matrix distance functions
# ----------------------------------------------------------------------------------------------------------------------


def compare_images_with_gram(image_pairs, gram_fn, output_path, device, transforms, title=''):
    header = ['channel'] + [f'gram_layer{i}' for i in range(1, gram_fn.num_gram_layers + 1)]
    logger = Logger(output_path, header=header)
    for channel_idx in tqdm(range(len(image_pairs)), desc=title):
        img1, img2 = image_pairs[channel_idx]
        img1 = transforms(img1).unsqueeze(0).to(device)
        img2 = transforms(img2).unsqueeze(0).to(device)
        with no_grad():
            gram_distances = gram_fn(img1, img2).squeeze().tolist()
        logger.add_entry(*([channel_idx] + gram_distances))


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


# ----------------------------------------------------------------------------------------------------------------------
# Image utilities
# ----------------------------------------------------------------------------------------------------------------------


def concat_horizontally(img1, img2):
    canvas = Image.new('RGB', (img1.width + img2.width, img1.height))
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (img1.width, 0))
    return canvas


class GrayChannelIdx:
    """Find the channels with 'grayed-out' feature visualizations"""

    def __init__(self, gray_channel_df):
        self.gray_channel_df = gray_channel_df

    def __call__(self, domain, layer):
        model_name = f'finetune-{domain}-resnet50'
        mask = (self.gray_channel_df['model'] == model_name) & (self.gray_channel_df['layer'] == layer)
        bad_channels = self.gray_channel_df.loc[mask, 'channel'].values
        bad_channels = [int(c.split('channel')[-1]) for c in bad_channels]
        return bad_channels


# ----------------------------------------------------------------------------------------------------------------------
# Compare the feature visualizations of channels across models trained on different tasks
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # --- Create the cosine similarity CSVs ---
    cos_sim_fn = CosineSimResnet50().to(DEVICE)
    print('Comparing ImageNet vs Dog Breeds feature visualizations with cosine similarity')
    calculate_gram_distances(cos_sim_fn, IMAGENET_DIR, DOG_DIR, os.path.join(COMPARISON_DIR, 'dog'))
    print('\nComparing ImageNet vs Car Models feature visualizations with cosine similarity')
    calculate_gram_distances(cos_sim_fn, IMAGENET_DIR, CAR_DIR, os.path.join(COMPARISON_DIR, 'car'))

    # instantiate the helper object to find the "gray channels" given the model and layer
    find_gray_channels = GrayChannelIdx(read_csv(GRAY_CHANNELS_PATH))

    domain = 'dog'
    for domain in ['car', 'dog']:
        comparison_dir = os.path.join(COMPARISON_DIR, domain)
        for filename in os.listdir(comparison_dir):
            # --- read and prep the data ---
            # read the gram distance comparison CSV
            basename, _ = os.path.splitext(filename)
            layer = '-'.join(basename.split('-')[-2:])
            df = read_csv(os.path.join(comparison_dir, filename))

            # --- prep the directories ---
            imagenet_dir = os.path.join(IMAGENET_DIR, 'features', layer)
            target_dir = os.path.join(DOG_DIR, 'features', layer)

            most_diff_dir = os.path.join(MOST_DIFFERENT_DIR, domain, layer)
            most_sim_dir = os.path.join(MOST_SIMILAR_DIR, domain, layer)

            create_folder(most_diff_dir)
            create_folder(most_sim_dir)

            target_cols = [f'cos_sim_layer{i}' for i in range(1, cos_sim_fn.num_emb_layers)]

            # remove all instances of "gray channels" from the comparisons
            bad_channels = find_gray_channels(domain, layer)
            cosine_sims = df.loc[~df['channel'].isin(bad_channels)].copy()

            # calculate the average cosine similarity and drop the original columns
            cosine_sims['avg_cos_sim'] = cosine_sims[target_cols].mean(axis=1)
            cosine_sims = cosine_sims.drop(labels=target_cols, axis='columns')

            # sort the channels by their cosine similarities
            cosine_sims = cosine_sims.sort_values('gram')
            cosine_sims = cosine_sims.reset_index(drop=True)

            # retrieve the top-k and bottom-k channels with respect to their cosine similarities
            # low cosine similarity --> channels are different
            # high cosine similarity --> channels are similar
            num_channels = len(cosine_sims)
            furthest_channels = cosine_sims.loc[:K, 'channel'].values
            closest_channels = cosine_sims.loc[num_channels - K:, 'channel'].values[::-1]

            for rank, channel_idx in enumerate(furthest_channels):
                channel_name = f'channel{channel_idx}.png'
                img1 = Image.open(os.path.join(imagenet_dir, channel_name))
                img2 = Image.open(os.path.join(target_dir, channel_name))
                img = concat_horizontally(img1, img2)
                img.save(os.path.join(most_diff_dir, f'rank{rank}_{channel_name}'))

            for rank, channel_idx in enumerate(closest_channels):
                channel_name = f'channel{channel_idx}.png'
                img1 = Image.open(os.path.join(imagenet_dir, channel_name))
                img2 = Image.open(os.path.join(target_dir, channel_name))
                img = concat_horizontally(img1, img2)
                img.save(os.path.join(most_sim_dir, f'rank{rank}_{channel_name}'))

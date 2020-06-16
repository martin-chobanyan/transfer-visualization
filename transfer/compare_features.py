"""
Prerequisite: This script assumes that the `find_bad_channels.py` script has already been executed.

This script compares respective feature visualizations across different ResNet-50 models,
finds the top-k most similar and most different images according to the cosine similarity metric,
and compares the cosine similarity distribution between different layers.

For each `domain` in ['car', 'domain'] and `layer` in ['layer3-bottleneck5', 'layer4-bottleneck0', 'layer4-bottleneck1', 'layer4-bottleneck2']
the following will be created:
1. {ROOT_DIR}/comparisons/{domain}/resnet50-cosine-sim-{layer}.csv              (cosine similarity scores per channel)
2. {ROOT_DIR}/most-similar/{domain}/{layer}/channel{channel_index}.png          (top-k most similar channels)
3. {ROOT_DIR>/most-different/{domain}/{layer}/channel{channel_index}.png        (top-k most different channels)
"""
import os

import matplotlib.pyplot as plt
from pandas import concat, read_csv
from PIL import Image
from seaborn import boxplot
from torch import no_grad
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from dataset import ImagePairs
from distance import CosineSimResnet50
from train_utils import create_folder, get_device, Logger
from feature_vis.transforms import ImagenetNorm

# define the constants
DOMAINS = ['car', 'dog']
ROOT_DIR = '/home/mchobanyan/data/research/transfer/vis/'
IMAGENET_DIR = os.path.join(ROOT_DIR, 'pretrained-resnet50')
COMPARISON_DIR = os.path.join(ROOT_DIR, 'comparisons')
CAR_DIR = os.path.join(ROOT_DIR, f'finetune-car-resnet50')
DOG_DIR = os.path.join(ROOT_DIR, f'finetune-dog-resnet50')

MOST_DIFFERENT_DIR = os.path.join(ROOT_DIR, 'most-different')
MOST_SIMILAR_DIR = os.path.join(ROOT_DIR, 'most-similar')
GRAY_CHANNELS_PATH = '/home/mchobanyan/data/research/transfer/vis/gray-channels.csv'

LAYERS = ['layer3-bottleneck5', 'layer4-bottleneck0', 'layer4-bottleneck1', 'layer4-bottleneck2']
DEVICE = get_device()
K = 20


# ----------------------------------------------------------------------------------------------------------------------
# Cosine similarity functions
# ----------------------------------------------------------------------------------------------------------------------


def compare_images_with_cosine_sim(image_pairs, sim_metric, output_path, device, transforms, title=''):
    """Calculate and store the cosine similarity for each pair of feature visualizations in `image_pairs`

    Parameters
    ----------
    image_pairs: ImagePairs
    sim_metric: CosineSimResnet50
    output_path: str
    device: torch.device
    transforms: callable
    title: str
    """
    # set up the logger
    header = ['channel', 'cosine_sim']
    logger = Logger(output_path, header=header)
    for channel_idx in tqdm(range(len(image_pairs)), desc=title):
        img1, img2 = image_pairs[channel_idx]
        img1 = transforms(img1).unsqueeze(0).to(device)
        img2 = transforms(img2).unsqueeze(0).to(device)
        cosine_sim = sim_metric(img1, img2).squeeze().item()
        logger.add_entry(channel_idx, cosine_sim)


def calculate_cosine_similarities(domain_dir, output_dir):
    """Calculate the cosine similarities for each of the target layers in the

    Parameters
    ----------
    domain_dir: str
        The path to the target domain dir (e.g. DOG_DIR for the dog classifier directory)
    output_dir: str
        The path to the output directory (will be created if it does not exist)
    """
    cos_sim_fn = CosineSimResnet50().to(DEVICE)
    create_folder(output_dir)
    image_transforms = Compose([ToTensor(), ImagenetNorm()])
    # compare model pre-trained on ImageNet vs fine-tuned on the target dataset
    for layer in LAYERS:
        img_dir1 = os.path.join(IMAGENET_DIR, 'features', layer)
        img_dir2 = os.path.join(domain_dir, 'features', layer)
        image_pairs = ImagePairs(img_dir1, img_dir2)
        output_path = os.path.join(output_dir, f'resnet50-cosine-sim-{layer}.csv')
        compare_images_with_cosine_sim(image_pairs, cos_sim_fn, output_path, DEVICE, image_transforms, layer)


# ----------------------------------------------------------------------------------------------------------------------
# Gram matrix distance functions (extra code)
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
    """Concat two PIL images horizontally

    Parameters
    ----------
    img1: PIL.Image
    img2: PIL.Image

    Returns
    -------
    PIL.Image
    """
    canvas = Image.new('RGB', (img1.width + img2.width, img1.height))
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (img1.width, 0))
    return canvas


class GrayChannelIdx:
    """Find the channels with 'grayed-out' feature visualizations

    Parameters
    ----------
    gray_channel_df: pandas.DataFrame
        The output of the `find_bad_channels` script as a pandas DataFrame
    """

    def __init__(self, gray_channel_df):
        self.gray_channel_df = gray_channel_df

    def __call__(self, domain, layer):
        """Find the channels with faulty feature visualizations in the given domain's model and layer.

        Parameters
        ----------
        domain: str
            The model domain (e.g. "car", "domain")
        layer: str
            The layer names (see LAYERS)

        Returns
        -------
        list[int]
            A list of the channel indices with faulty optimizations
        """
        model_name = f'finetune-{domain}-resnet50'
        mask = (self.gray_channel_df['model'] == model_name) & (self.gray_channel_df['layer'] == layer)
        bad_channels = self.gray_channel_df.loc[mask, 'channel'].values
        bad_channels = [int(c.split('channel')[-1]) for c in bad_channels]
        return bad_channels


def set_plot_configs():
    """Config matplotlib to produce more visible plots"""
    plt.rcParams.update({
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'font.size': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# ----------------------------------------------------------------------------------------------------------------------
# Compare the feature visualizations of channels across models trained on different tasks
# ----------------------------------------------------------------------------------------------------------------------


def main():
    # Create the cosine similarity CSV files
    print('Comparing ImageNet vs Dog Breeds feature visualizations with cosine similarity')
    calculate_cosine_similarities(DOG_DIR, output_dir=os.path.join(COMPARISON_DIR, 'dog'))
    print('\nComparing ImageNet vs Car Models feature visualizations with cosine similarity')
    calculate_cosine_similarities(CAR_DIR, output_dir=os.path.join(COMPARISON_DIR, 'car'))

    # instantiate the helper object to find the "gray channels" given the model and layer
    find_gray_channels = GrayChannelIdx(read_csv(GRAY_CHANNELS_PATH))

    for domain in DOMAINS:
        print(f'\nFinding the {K} most similar and different channels for {domain}s')
        data_by_layer = []
        comparison_dir = os.path.join(COMPARISON_DIR, domain)
        for filename in os.listdir(comparison_dir):

            # read the cosine similarity comparison CSV
            basename, _ = os.path.splitext(filename)
            layer = '-'.join(basename.split('-')[-2:])
            df = read_csv(os.path.join(comparison_dir, filename))

            # prep the directories
            imagenet_dir = os.path.join(IMAGENET_DIR, 'features', layer)
            target_dir = os.path.join(CAR_DIR if domain == 'car' else DOG_DIR, 'features', layer)

            most_diff_dir = os.path.join(MOST_DIFFERENT_DIR, domain, layer)
            most_sim_dir = os.path.join(MOST_SIMILAR_DIR, domain, layer)
            create_folder(most_diff_dir)
            create_folder(most_sim_dir)

            # remove all instances of "gray channels" from the comparisons
            bad_channels = find_gray_channels(domain, layer)
            cosine_sims = df.loc[~df['channel'].isin(bad_channels)].copy()

            # sort the channels by their cosine similarities
            cosine_sims = cosine_sims.sort_values('cosine_sim')
            cosine_sims = cosine_sims.reset_index(drop=True)

            # append this cosine similarity data to the global list
            cosine_sims['layer'] = layer
            data_by_layer.append(cosine_sims)

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

        # create a boxplot of the average cosine similarity across the four layers
        set_plot_configs()
        fig, ax = plt.subplots(figsize=(14, 7))
        data_by_layer = concat(data_by_layer).sort_values('layer')  # sort by layer so that the boxplots are in order
        boxplot(data=data_by_layer, x='layer', y='cosine_sim', ax=ax)
        plt.title(f'Cosine Similarities Across Layers ({domain}s)')
        plt.xlabel('Resnet-50 Layers')
        plt.ylabel('Cosine Similarity')
        plt.show()
    print('Done!')


if __name__ == '__main__':
    main()

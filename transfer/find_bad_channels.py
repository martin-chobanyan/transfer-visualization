"""Find the channels with faulty / unoptimized feature visualizations characterized by gray images"""
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT_DIR = '/home/mchobanyan/data/research/transfer/vis'  # root directory of the project
MODELS = ['pretrained-resnet50', 'finetune-dog-resnet50', 'finetune-car-resnet50']  # subdirectories for each model
LAYERS = ['layer3-bottleneck5', 'layer4-bottleneck0', 'layer4-bottleneck1',
          'layer4-bottleneck2']  # subdirectories for each layer
SCORE_CHANGE_CUTOFF = 0.1  # the threshold for the spike in the grayscale score (default is 10%)


def grayscale_score(img):
    """Score the input image based on how gray it is

    Convert the input RGB image into grayscale, subtract the channel from the original image, and add up the square
    of the residuals into a scalar score. Images which are "gray-ish" will have a lower grayscale score than images
    which are colorful.

    Parameters
    ----------
    img: PIL.Image

    Returns
    -------
    float
    """
    x_color = np.array(img)
    h, w, _ = x_color.shape
    x_gray = np.array(img.convert('L')).reshape((h, w, 1))
    return np.sqrt(np.sum((x_color - x_gray) ** 2))


def calculate_scores(channel_dir, title=''):
    """Calculate the grayscale scores for all feature visualizations in the given directory

    Parameters
    ----------
    channel_dir: str
    title: str, optional
        Optional title / description for the tqdm progress bar (default is empty)

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with columns for the channel index
        and the grayscale score for the corresponding feature visualization
    """
    scores = []
    channels = [f'channel{i}' for i in range(len(os.listdir(channel_dir)))]
    for channel in tqdm(channels, desc=title):
        img = Image.open(os.path.join(channel_dir, f'{channel}.png'))
        scores.append(grayscale_score(img))
    return pd.DataFrame({'channel': channels, 'score': scores})


def find_changepoint(scores):
    """Find where in the sorted array of grayscale scores the spike occurs

    Note: this assumes that the `SCORE_CHANGE_CUTOFF` is big enough such that smaller increases are ignored.

    Parameters
    ----------
    scores: np.ndarray
        An array of sorted grayscale scores in ascending order

    Returns
    -------
    int
        The index of the last element in the array before the large spike in grayscale scores
    """
    max_score = scores.max()
    changes = np.diff(scores) / max_score
    changepoint, = np.where(changes > SCORE_CHANGE_CUTOFF)
    if len(changepoint) > 0:
        return changepoint[0]
    else:
        return -1


if __name__ == '__main__':
    rows = []
    for model_name in MODELS:
        for layer in LAYERS:
            target_dir = os.path.join(ROOT_DIR, model_name, 'features', layer)
            score_df = calculate_scores(target_dir, title=f'{model_name}_{layer}')
            score_df = score_df.sort_values('score')
            score_df = score_df.reset_index(drop=True)
            idx = find_changepoint(score_df['score'].values)
            for channel in score_df.loc[:idx, 'channel']:
                rows.append({'model': model_name, 'layer': layer, 'channel': channel})

    # join and save the results
    output_path = os.path.join(ROOT_DIR, 'gray-channels.csv')
    gray_channels_df = pd.DataFrame(rows)
    gray_channels_df.to_csv(output_path, index=False)

    # find the number of bad channels per group
    counts = gray_channels_df.groupby(['model', 'layer']).count()
    counts.columns = ['Bad Channel Count']
    print(gray_channels_df)

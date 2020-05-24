import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT_DIR = '/home/mchobanyan/data/research/transfer/vis'
MODELS = ['pretrained-resnet50', 'finetune-dog-resnet50', 'finetune-car-resnet50']
LAYERS = ['layer3-bottleneck5', 'layer4-bottleneck0', 'layer4-bottleneck1', 'layer4-bottleneck2']
SCORE_CHANGE_CUTOFF = 0.1


def grayscale_score(img):
    x_color = np.array(img)
    h, w, _ = x_color.shape
    x_gray = np.array(img.convert('L')).reshape((h, w, 1))
    return np.sqrt(np.sum((x_color - x_gray) ** 2))


def calculate_scores(channel_dir, title=''):
    scores = []
    channels = [f'channel{i}' for i in range(len(os.listdir(channel_dir)))]
    for channel in tqdm(channels, desc=title):
        img = Image.open(os.path.join(channel_dir, f'{channel}.png'))
        scores.append(grayscale_score(img))
    return pd.DataFrame({'channel': channels, 'score': scores})


def find_changepoint(scores):
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

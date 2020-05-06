import os

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotation, ToTensor
from tqdm import tqdm

from feature_vis.transforms import IMAGENET_MEANS, IMAGENET_STDEVS
from dataset import CarModels
from train_utils import *

# define the constants
IMAGE_SHAPE = (400, 400)
P_TRAIN = 0.8
BATCH_SIZE = 100
LEARNING_RATE = 0.00001
NUM_EPOCHS = 60

if __name__ == '__main__':
    # define the dog breed dataset
    root_dir = '/home/mchobanyan/data/stanford-cars/'
    transforms = Compose([
        RandomCrop(IMAGE_SHAPE, pad_if_needed=True),
        RandomHorizontalFlip(),
        RandomRotation(30),
        ToTensor(),
        Normalize(IMAGENET_MEANS, IMAGENET_STDEVS)
    ])

    train_set = CarModels(root_dir, train=True)
    test_set = CarModels(root_dir, train=False)
    print(len(test_set.load_annotations()))

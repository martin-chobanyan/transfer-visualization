import os

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotation, ToTensor
from tqdm import tqdm

from feature_vis.transforms import IMAGENET_MEANS, IMAGENET_STDEVS
from dataset import CarModels
from train_utils import *

# define the constants
IMAGE_SHAPE = (400, 400)
BATCH_SIZE = 100
LEARNING_RATE = 0.00001
NUM_EPOCHS = 100

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

    train_data = CarModels(root_dir, train=True, transforms=transforms)
    test_data = CarModels(root_dir, train=False, transforms=transforms)
    num_models = len(set(train_data.label_map.values()))
    num_models2 = len(set(test_data.label_map.values()))
    # print(num_models, num_models2)
    # print(sorted(list(set(test_data.label_map.values()))))
    print(sorted(list(set(train_data.label_map.values()))))
    print(len(list(set(train_data.label_map.values()))))

    # set up the train and test data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # set up the model
    # device = get_device()
    # model = load_resnet50_layer3_bottleneck5(num_breeds)
    # model = model.to(device)

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
    dataset = CarModels(root_dir, transforms=transforms)
    num_cars = len(dataset)
    num_car_models = len(set(dataset.label_map.values()))

    # randomly split the dataset into train and test
    num_train = int(P_TRAIN * num_cars)
    num_test = num_cars - num_train
    train_data, test_data = random_split(dataset, [num_train, num_test])

    # set up the train and test data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # set up the model
    device = get_device()
    model = load_resnet50_layer3_bottleneck5(num_car_models)
    model = model.to(device)

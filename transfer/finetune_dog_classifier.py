"""This script fine-tunes ResNet-50 pre-trained on ImageNet to the Stanford Dog Dataset"""
import os

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, RandomRotation, ToTensor
from tqdm import tqdm

from feature_vis.transforms import ImagenetNorm
from feature_vis.utils import get_device
from dataset import DogBreeds
from train_utils import *

# define the constants
IMAGE_SHAPE = (400, 400)
P_TRAIN = 0.8
BATCH_SIZE = 100
LEARNING_RATE = 0.00001
NUM_EPOCHS = 60

if __name__ == '__main__':
    # root directory of the dog dataset
    data_dir = '/home/mchobanyan/data/kaggle/dog_breeds/'

    image_transforms = Compose([
        RandomCrop(IMAGE_SHAPE, pad_if_needed=True),
        RandomHorizontalFlip(),
        RandomRotation(30),
        ToTensor(),
        ImagenetNorm()
    ])

    # define the dataset class
    dataset = DogBreeds(data_dir, image_transforms)
    num_dogs = len(dataset)
    num_breeds = len(dataset.breeds)

    # randomly split the dataset into train and test
    num_train = int(P_TRAIN * num_dogs)
    num_test = num_dogs - num_train
    train_data, test_data = random_split(dataset, [num_train, num_test])

    # set up the train and test data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # load ResNet-50 with every layer frozen except for layer3-bottleneck5 and beyond,
    # and a new fully-connected network which outputs a 120-dim vector
    device = get_device()
    model = load_resnet50_layer3_bottleneck5(num_breeds)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # set up the output logger
    output_dir = '/home/mchobanyan/data/research/transfer/vis/finetune-dog-resnet50'
    create_folder(os.path.join(output_dir, 'models'))
    logger = TrainingLogger(filepath=os.path.join(output_dir, 'training-log.csv'))

    for epoch in tqdm(range(NUM_EPOCHS)):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        logger.add_entry(epoch, train_loss, test_loss, train_acc, test_acc)
        checkpoint(model, os.path.join(output_dir, 'models', f'model_epoch{epoch}.pt'))

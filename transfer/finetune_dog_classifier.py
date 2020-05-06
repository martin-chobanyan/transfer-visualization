import os

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotation, ToTensor
from tqdm import tqdm

from feature_vis.transforms import IMAGENET_MEANS, IMAGENET_STDEVS
from feature_vis.utils import freeze_parameters, unfreeze_parameters
from dataset import DogBreedDataset
from train_utils import checkpoint, create_folder, get_device, train_epoch, test_epoch, TrainingLogger

# define the constants
IMAGE_SHAPE = (400, 400)
P_TRAIN = 0.8
BATCH_SIZE = 100
LEARNING_RATE = 0.00001
NUM_EPOCHS = 60


def load_resnet50_layer3_bottleneck5(num_classes):
    """
    Loads a pretrained resnet-50, swaps the fully connected layer,
    and freezes all parameters except for those in layer3.bottleneck5, layer4 and the fully connected layer.
    """
    resnet_model = resnet50(pretrained=True)
    resnet_model = freeze_parameters(resnet_model)

    unfreeze_parameters(resnet_model.layer3[5])
    unfreeze_parameters(resnet_model.layer4)

    fc_input_dim = resnet_model.fc.in_features
    new_fc = nn.Linear(fc_input_dim, num_classes)
    resnet_model.fc = new_fc

    return resnet_model


def load_resnet50_layer4(num_classes):
    """
    Loads a pretrained resnet-50, swaps the fully connected layer,
    and freezes all parameters except for those in layer4 and the fully connected layer.
    """
    resnet_model = resnet50(pretrained=True)
    resnet_model = freeze_parameters(resnet_model)

    unfreeze_parameters(resnet_model.layer4)

    fc_input_dim = resnet_model.fc.in_features
    new_fc = nn.Linear(fc_input_dim, num_classes)
    resnet_model.fc = new_fc

    return resnet_model


if __name__ == '__main__':
    # define the dog breed dataset
    root_dir = '/home/mchobanyan/data/kaggle/dog_breeds/'
    transforms = Compose([
        RandomCrop(IMAGE_SHAPE, pad_if_needed=True),
        RandomHorizontalFlip(),
        RandomRotation(30),
        ToTensor(),
        Normalize(IMAGENET_MEANS, IMAGENET_STDEVS)
    ])
    dataset = DogBreedDataset(root_dir, transforms)
    num_dogs = len(dataset)
    num_breeds = len(dataset.breeds)

    # randomly split the dataset into train and test
    num_train = int(P_TRAIN * num_dogs)
    num_test = num_dogs - num_train
    train_data, test_data = random_split(dataset, [num_train, num_test])

    # set up the train and test data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # set up the model
    device = get_device()
    model = load_resnet50_layer3_bottleneck5(num_breeds)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # set up the output directory
    output_dir = '/home/mchobanyan/data/research/transfer/vis/finetune-dog-resnet50'
    create_folder(os.path.join(output_dir, 'models'))
    logger = TrainingLogger(filepath=os.path.join(output_dir, 'training-log.csv'))

    for epoch in tqdm(range(NUM_EPOCHS)):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        logger.add_entry(epoch, train_loss, test_loss, train_acc, test_acc)
        checkpoint(model, os.path.join(output_dir, 'models', f'model_epoch{epoch}.pt'))

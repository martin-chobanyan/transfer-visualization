import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotation, ToTensor

from feature_vis.transforms import IMAGENET_MEANS, IMAGENET_STDEVS
from feature_vis.utils import freeze_parameters, unfreeze_parameters
from dataset import DogBreedDataset


def load_resnet50_layer4(num_classes):
    # load a pretrained resnet-50 and freeze all of its parameters
    model = resnet50(pretrained=True)
    model = freeze_parameters(model)

    # make all modules in layer 4 trainable
    unfreeze_parameters(model.layer4)

    # swap the fully connected layer
    fc_input_dim = model.fc.in_features
    new_fc = nn.Linear(fc_input_dim, num_classes)
    model.fc = new_fc

    return model


if __name__ == '__main__':
    # define the constants
    IMAGE_SHAPE = (400, 400)
    P_TRAIN = 0.8
    BATCH_SIZE = 32

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
    n_dogs = len(dataset)
    n_breeds = len(dataset.breeds)

    # randomly split the dataset into train and test
    n_train = int(P_TRAIN * n_dogs)
    n_test = n_dogs - n_train
    train_data, test_data = random_split(dataset, [n_train, n_test])

    # set up the train and test data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    # set up the model
    model = load_resnet50_layer4(n_breeds)

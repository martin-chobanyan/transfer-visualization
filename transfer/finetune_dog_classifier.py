import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotation, ToTensor
from tqdm import tqdm

from feature_vis.transforms import IMAGENET_MEANS, IMAGENET_STDEVS
from feature_vis.utils import freeze_parameters, unfreeze_parameters
from dataset import DogBreedDataset
from train_utils import *

# define the constants
IMAGE_SHAPE = (400, 400)
P_TRAIN = 0.8
BATCH_SIZE = 100


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_resnet50_layer3_bottleneck5(n_breeds)
    model = model.to(device)

    learning_rate = 0.00001
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # set up the output directory
    output_dir = '/home/mchobanyan/data/research/transfer/vis/finetune-resnet50-layer3-bottleneck5'
    create_folder(os.path.join(output_dir, 'models'))
    logger = TrainingLogger(filepath=os.path.join(output_dir, 'training-log.csv'))

    num_epochs = 200
    for epoch in tqdm(range(num_epochs)):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = train_epoch(model, test_loader, criterion, optimizer, device)
        logger.add_entry(epoch, train_loss, test_loss, train_acc, test_acc)
        checkpoint(model, os.path.join(output_dir, 'models', f'model_epoch{epoch}.pt'))

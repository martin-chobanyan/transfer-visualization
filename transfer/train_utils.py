"""Utility classes and functions for training a neural network"""
import os
from csv import writer as csv_writer

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torchvision.models import resnet50

from feature_vis.utils import freeze_parameters, unfreeze_parameters


def create_folder(path):
    """Create a folder if it does not already exist"""
    if not (os.path.exists(path) or os.path.isdir(path)):
        os.makedirs(path, exist_ok=True)


def get_device():
    """Get the cuda device if it is available

    Returns
    -------
    torch.device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_resnet50_layer3_bottleneck5(num_classes):
    """Prepare the base ResNet-50 network

    Loads a pre-trained ResNet-50,
    swaps the fully connected layer with one which outputs `num_classes` dimensions,
    and freezes all parameters except for those in layer3.bottleneck5, layer4 and the fully connected layer.

    Parameters
    ----------
    num_classes: int
        The dimension for the output vector

    Returns
    -------
    torchvision.models.ResNet
    """
    resnet_model = resnet50(pretrained=True)
    resnet_model = freeze_parameters(resnet_model)

    unfreeze_parameters(resnet_model.layer3[5])
    unfreeze_parameters(resnet_model.layer4)

    fc_input_dim = resnet_model.fc.in_features
    new_fc = Linear(fc_input_dim, num_classes)
    resnet_model.fc = new_fc

    return resnet_model


class Logger:
    """A CSV logger

    Parameters
    ----------
    filepath: str
        The filepath where the logger will be created.
    header: list[str]
        The columns for the CSV file as a list of strings
    """
    def __init__(self, filepath, header):
        self.filepath = filepath
        self.header = header
        with open(filepath, 'w') as file:
            writer = csv_writer(file)
            writer.writerow(header)

    def add_entry(self, *args):
        """Append a row to the CSV file
        The arguments for this function must match the length and order of the initialized headers.
        """
        if len(args) != len(self.header):
            raise ValueError('Entry length must match the header length!')
        with open(self.filepath, 'a') as file:
            writer = csv_writer(file)
            writer.writerow(args)


class TrainingLogger(Logger):
    """An extension of the Logger class for training a classifier

    The headers are fixed to include: 'Epoch', 'Train Loss', 'Test Loss', 'Train Accuracy', and 'Test Accuracy'

    Parameters
    ----------
    filepath: str
        The filepath where the logger will be created
    """
    def __init__(self, filepath):
        header = ['Epoch', 'Train Loss', 'Test Loss', 'Train Accuracy', 'Test Accuracy']
        super().__init__(filepath, header)

    def add_entry(self, epoch, train_loss, test_loss, train_acc, test_acc):
        """Append a row to the CSV file"""
        super().add_entry(epoch, train_loss, test_loss, train_acc, test_acc)


def accuracy(model_out, true_labels):
    """Calculate the accuracy of a batch of predictions

    Parameters
    ----------
    model_out: torch.FloatTensor
        The output of the classifier with shape (batch size, number of classes)
    true_labels: torch.LongTensor
        The true labels aligned with the model's output with shape (batch size,)

    Returns
    -------
    float
    """
    pred = torch.argmax(F.softmax(model_out, dim=1), dim=1)
    acc = (pred == true_labels.squeeze())
    return float(acc.sum()) / acc.size(0)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train a neural network for a single epoch

    Parameters
    ----------
    model: nn.Module
    dataloader: DataLoader
    criterion: callable loss function
    optimizer: pytorch optimizer
    device: torch.device

    Returns
    -------
    tuple[float]
        A tuple containing the average training loss and accuracy for this epoch
    """
    avg_loss = []
    avg_acc = []
    model.train()
    for batch_image, batch_label in dataloader:
        batch_image = batch_image.to(device)
        batch_label = batch_label.to(device)
        optimizer.zero_grad()
        output = model(batch_image)
        loss = criterion(output, batch_label)
        loss.backward()
        optimizer.step()
        acc = accuracy(output, batch_label)
        avg_loss.append(loss.item())
        avg_acc.append(acc)
    return sum(avg_loss) / len(avg_loss), sum(avg_acc) / len(avg_acc)


def test_epoch(model, dataloader, criterion, device):
    """Test a neural network for a single epoch

    Parameters
    ----------
    model: nn.Module
    dataloader: DataLoader
    criterion: callable loss function
    device: torch.device

    Returns
    -------
    tuple[float]
        A tuple containing the average testing loss and accuracy for this epoch
    """
    avg_loss = []
    avg_acc = []
    model.eval()
    with torch.no_grad():
        for batch_image, batch_label in dataloader:
            batch_image = batch_image.to(device)
            batch_label = batch_label.to(device)
            output = model(batch_image)
            loss = criterion(output, batch_label)
            acc = accuracy(output, batch_label)
            avg_loss.append(loss.item())
            avg_acc.append(acc)
    return sum(avg_loss) / len(avg_loss), sum(avg_acc) / len(avg_acc)


def checkpoint(model, filepath):
    """Save the state of the model

    To restore the model do the following:
    >> the_model = TheModelClass(*args, **kwargs)
    >> the_model.load_state_dict(torch.load(PATH))

    Parameters
    ----------
    model: nn.Module
        The pytorch model to be saved
    filepath: str
        The filepath of the pickle
    """
    torch.save(model.state_dict(), filepath)

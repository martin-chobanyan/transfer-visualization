import os
from csv import writer as csv_writer

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torchvision.models import resnet50

from feature_vis.utils import freeze_parameters, unfreeze_parameters


def create_folder(path):
    if not (os.path.exists(path) or os.path.isdir(path)):
        os.makedirs(path, exist_ok=True)


def get_device():
    """Get the cuda device if it is available

    Note: this assumes that there is only one GPU device

    Returns
    -------
    torch.device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    new_fc = Linear(fc_input_dim, num_classes)
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
    new_fc = Linear(fc_input_dim, num_classes)
    resnet_model.fc = new_fc

    return resnet_model


class Logger:
    def __init__(self, filepath, header):
        self.filepath = filepath
        self.header = header
        with open(filepath, 'w') as file:
            writer = csv_writer(file)
            writer.writerow(header)

    def add_entry(self, *args):
        if len(args) != len(self.header):
            raise ValueError('Entry length must match the header length!')
        with open(self.filepath, 'a') as file:
            writer = csv_writer(file)
            writer.writerow(args)


class TrainingLogger(Logger):
    def __init__(self, filepath):
        header = ['Epoch', 'Train Loss', 'Test Loss', 'Train Accuracy', 'Test Accuracy']
        super().__init__(filepath, header)

    def add_entry(self, epoch, train_loss, test_loss, train_acc, test_acc):
        super().add_entry(epoch, train_loss, test_loss, train_acc, test_acc)


def accuracy(model_out, true_labels):
    """Calculate the accuracy of a batch of predictions

    Parameters
    ----------
    model_out: torch.FloatTensor
        The output of the emotion classifier with shape [batch size, num emotions]
    true_labels: torch.LongTensor
        The true encoded emotion labels aligned with the model's output

    Returns
    -------
    float
    """
    pred = torch.argmax(F.softmax(model_out, dim=1), dim=1)
    acc = (pred == true_labels.squeeze())
    return float(acc.sum()) / acc.size(0)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for an epoch

    Parameters
    ----------
    model: nn.Module
    dataloader: DataLoader
    criterion: callable loss function
    optimizer: pytorch optimizer
    device: torch.device

    Returns
    -------
    float
        The average loss
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
    """Run the model for a validation epoch

    Parameters
    ----------
    model: nn.Module
    dataloader: DataLoader
    criterion: callable loss function
    device: torch.device

    Returns
    -------
    float, float
        The average loss and the average accuracy
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

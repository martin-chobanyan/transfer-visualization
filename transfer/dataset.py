import os

from pandas import read_csv
from PIL import Image
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class DogBreeds(Dataset):
    def __init__(self, root_dir, transforms=None):
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        self.img_dir = os.path.join(root_dir, 'train')

        dog_df = read_csv(os.path.join(root_dir, 'labels.csv'))
        self.dog_ids = dog_df['id'].values

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(dog_df['breed'].values)

    @property
    def breeds(self):
        return self.label_encoder.classes_

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.img_dir, f'{self.dog_ids[item]}.jpg'))
        label = self.labels[item]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.dog_ids)


class CarModels(Dataset):
    def __init__(self, root_dir, train, transforms=None):
        super().__init__()
        self.root_dir = root_dir
        self.train = train
        self.transforms = transforms
        self.data_dir = os.path.join(root_dir, 'cars_train' if train else 'cars_test')
        self.filenames = os.listdir(self.data_dir)
        self.label_map = self.load_annotations()

    def load_annotations(self):
        mode = 'train' if self.train else 'test'
        mat_data = loadmat(os.path.join(self.root_dir, 'devkit', f'cars_{mode}_annos.mat'))
        mat_data = mat_data['annotations'].squeeze()

        label_map = dict()
        for (*_, label, filename) in mat_data:
            label_map[filename.item()] = label.item()
        return label_map

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        car_model = self.label_map[filename]
        img = Image.open(os.path.join(self.data_dir, filename))
        if self.transforms is not None:
            img = self.transforms(img)
        return img, car_model

    def __len__(self):
        return len(self.filenames)

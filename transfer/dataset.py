import os

import numpy as np
from pandas import read_csv
from PIL import Image
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


def is_grayscale(img):
    return (img.mode == 'L')


def grayscale_to_rgb(img):
    arr = np.array(img)
    arr = np.expand_dims(arr, -1)
    arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr)


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
    def __init__(self, root_dir, transforms=None):
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        self.img_dir = os.path.join(root_dir, 'cars_train')
        self.filenames = os.listdir(self.img_dir)
        self.label_map = self.load_annotations()

    def load_annotations(self):
        mat_data = loadmat(os.path.join(self.root_dir, 'devkit', f'cars_train_annos.mat'))
        mat_data = mat_data['annotations'].squeeze()

        label_map = dict()
        for (*_, label, filename) in mat_data:
            label_map[filename.item()] = label.item() - 1  # we subtract since the labels start from 1 instead of 0
        return label_map

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        car_model = self.label_map[filename]
        img = Image.open(os.path.join(self.img_dir, filename))
        if is_grayscale(img):
            img = grayscale_to_rgb(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, car_model

    def __len__(self):
        return len(self.filenames)


class ImagePairs(Dataset):
    def __init__(self, img_dir1, img_dir2, transforms=None):
        super().__init__()
        self.img_dir1 = img_dir1
        self.img_dir2 = img_dir2
        self.transforms = transforms

        num_imgs1 = len(os.listdir(img_dir1))
        num_imgs2 = len(os.listdir(img_dir2))
        self.num_pairs = num_imgs1
        if num_imgs1 != num_imgs2:
            raise ValueError('The number of images in each directory should match!')

    def __getitem__(self, idx):
        path1 = os.path.join(self.img_dir1, f'channel{idx}.png')
        path2 = os.path.join(self.img_dir2, f'channel{idx}.png')

        img1 = Image.open(path1)
        img2 = Image.open(path2)

        if self.transforms is not None:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        return img1, img2

    def __len__(self):
        return self.num_pairs

import os

from pandas import read_csv
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

class DogBreedDataset(Dataset):
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

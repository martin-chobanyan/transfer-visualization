from torch.utils.data import random_split
from dataset import DogBreedDataset

if __name__ == '__main__':
    # define the dog breed dataset
    dataset = DogBreedDataset(root_dir='/home/mchobanyan/data/kaggle/dog_breeds/')
    n_dogs = len(dataset)

    # randomly split the dataset into train and test
    n_train = int(0.8 * n_dogs)
    n_test = n_dogs - n_train
    train_data, test_data = random_split(dataset, [n_train, n_test])

    print(len(train_data), len(test_data))

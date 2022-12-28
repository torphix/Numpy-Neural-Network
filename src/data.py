import random
import numpy as np
from sklearn.datasets import load_digits


def load_data(train_test_val_split: list, n_samples: int = -1):
    train_size, val_size, test_size = train_test_val_split
    X, Y = load_digits(n_class=10, return_X_y=True, as_frame=False)

    if n_samples == -1:
        n_samples = len(X)

    X /= 16  # Normalize
    X = X[:n_samples, :]
    Y = Y[:n_samples]

    n_samples = X.shape[0]
    train_slice = int(n_samples*train_size)
    val_slice = int(n_samples*val_size) + train_slice
    train = X[:train_slice, :], Y[:train_slice]
    val = X[train_slice+1:val_slice, :], Y[train_slice+1:val_slice]
    test = X[val_slice+1:, :], Y[train_slice+1:]
    return train, val, test


class MNISTDataloader:
    '''
    Generator that holds each dataset
    train_val_test_split: 
        - splits the data pass list of each split size
        - exclude a size in order to not have that dataset
    n_samples: number of datapoints to use, pass -1 for all
    '''

    def __init__(self, train_val_test_split: list, shuffle: bool, n_samples: int):

        self.should_shuffle = shuffle
        self.train_data, self.val_data, self.test_data = load_data(
            train_val_test_split, n_samples)

        if n_samples == -1:
            self.n_samples = len(self.train_data[0])

    @property
    def train_len(self):
        return len(self.train_data[0])

    @property
    def val_len(self):
        return len(self.val_data[0])

    @property
    def test_len(self):
        return len(self.test_data[0])

    def shuffle_data(self, data):
        random.shuffle(list(zip(data)))
        return zip(*data)

    def train_dataloader(self):
        '''
        n_samples: number of datapoints to use pass -1 for all
        '''
        X, Y = self.train_data
        for i in range(len(X)):
            yield np.array([X[i]]), np.array([Y[i]])

        self.shuffle_data(self.train_data)

    def val_dataloader(self):
        self.shuffle_data(self.val_data)
        X, Y = self.val_data
        for i in range(len(X)):
            yield np.array([X[i]]), np.array([Y[i]])

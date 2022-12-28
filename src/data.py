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
    test = X[val_slice+1:, :], Y[val_slice+1:]
    return train, val, test


class Dataloader:
    def __init__(self, data:tuple, shuffle: bool):
        self.data = data 
        self.should_shuffle = shuffle

    def __len__(self):
        return len(self.data[0])

    def shuffle_data(self):
        if self.should_shuffle:
            data = list(zip(self.data[0], self.data[1]))
            random.shuffle(data)
            x,y = zip(*data)
            self.data = x,y

    def __iter__(self):
        self.curr_idx = 0
        self.shuffle_data()            
        return self

    def __next__(self):
        if self.curr_idx == len(self.data[0]):
            self.curr_idx = 0
            self.shuffle_data()
            raise StopIteration
        else:
            x, y = self.data[0][self.curr_idx], self.data[1][self.curr_idx]
            self.curr_idx += 1
            return (np.expand_dims(np.array(x), axis=0), 
                    np.expand_dims(np.array(y), axis=0))

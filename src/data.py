import random
import numpy as np
from utils import load_data

class MNISTDataloader:
    '''
    Generator that holds each dataset
    train_val_test_split: 
        - splits the data pass list of each split size
        - exclude a size in order to not have that dataset
    n_samples: number of datapoints to use, pass -1 for all
    '''
    def __init__(self, train_val_test_split:list, shuffle:bool, n_samples:int):

        self.should_shuffle = shuffle
        self.train_data, self.val_data = load_data(train_val_test_split, n_samples)

    @property
    def train_len(self):
        return len(self.train_data)
        
    @property
    def val_len(self):
        return len(self.val_data)

    def shuffle_data(self, data):
        random.shuffle(list(zip(data)))
        return zip(*data)

    def train_dataloader(self):
        '''
        n_samples: number of datapoints to use pass -1 for all
        '''
        X,Y = self.train_data
        for i in range(len(self.train_data)):
            yield np.array([X[i]]), np.array([Y[i]])

        if self.should_shuffle:
            self.shuffle_data(self.train_data)

    def val_dataloader(self):
        X,Y = self.val_data
        for i in range(len(self.val_data)):
            yield np.array([X[i]]), np.array([Y[i]])
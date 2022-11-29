import yaml
import numpy as np
from sklearn.datasets import load_digits


def open_config():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config

def one_hot_encode(class_label, n_classes=10):
  one_hot = np.zeros((n_classes))
  one_hot[class_label] = 1
  return one_hot

def load_data(train_test_val_split:list, n_samples:int=-1):
  train_size, val_size, test_size = train_test_val_split
  X, Y = load_digits(n_class=10, return_X_y=True, as_frame=False)

  if n_samples == -1:
    n_samples = len(X)

  X /= 16
  train_size = int(len(X) * train_size)
  return (X[:train_size,:], Y[:train_size]), (X[train_size+1:,:], Y[train_size+1:])
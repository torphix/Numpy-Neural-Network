import yaml
import numpy as np


def open_config():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config


def one_hot_encode(class_label, n_classes=10):
    one_hot = np.zeros((n_classes))
    one_hot[class_label] = 1
    return one_hot

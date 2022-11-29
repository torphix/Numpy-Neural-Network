import sys
from src.train import Trainer
from utils import open_config


if __name__ == '__main__':
    command = sys.argv[1]

    if command == 'train':
        config = open_config()
        trainer = Trainer(config)
        trainer()
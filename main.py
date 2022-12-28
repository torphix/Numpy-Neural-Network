import sys
import unittest
from src.train import Trainer
from utils import open_config
from tests import *

if __name__ == '__main__':
    command = sys.argv[1]

    if command == 'train':
        config = open_config()
        trainer = Trainer(config)
        trainer()

    elif command == 'test':
        unittest.main(argv=['first-arg-is-ignored'], exit=False)

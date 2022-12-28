import unittest
import numpy as np
from src.train import Trainer
from utils import open_config
from src.model.layers import LinearLayer


class LayerTests(unittest.TestCase):

    def test_dims(self):

        self.linear_relu = LinearLayer(10, 20, 'relu')

        test_input = np.random.randn(1, 10)
        out_relu = self.linear_relu(test_input)
        self.assertEqual(out_relu.shape, (1, 20))


class NeuralNetworkTests(unittest.TestCase):

    def test_convergance_basic(self):
        # Tests if network can overfit on small subset of data
        # Intuition is if it can't overfit on small dataset then there is a problem
        config = open_config()
        config['use_n_datasamples'] = 20
        config['train_val_test_split'] = [0.8, 0.1, 0.1]
        config['log_run'] = False
        config['epochs'] = 1500
        config['layers'] = [
            {'in_d':64, 'out_d':100, 'activation': 'relu', 'dropout': 0.5},
            {'in_d':100, 'out_d':20, 'activation': 'relu', 'dropout': 0.0},
            {'in_d':20, 'out_d':10, 'activation': None, 'dropout': 0.0},
        ]
        trainer = Trainer(config)
        final_outputs = trainer()


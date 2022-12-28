import unittest
import numpy as np
from src.model.layers import LinearLayer
from src.train import Trainer
from utils import open_config


class LayerTests(unittest.TestCase):

    def test_dims(self):

        self.linear_relu = LinearLayer(10, 20, 'relu')

        test_input = np.random.randn(1, 10)
        out_relu = self.linear_relu(test_input)
        self.assertEqual(out_relu.shape, (1, 20))


class NeuralNetworkTest(unittest.TestCase):

    def test_convergance(self):
        # Tests if network can overfit on small subset of data
        # Intuition is if it can't then there is problem with neural network
        config = open_config()
        config['use_n_datasamples'] = 100
        config['log_run'] = False
        config['epochs'] = 30
        trainer = Trainer(config)
        final_outputs = trainer()
        print(final_outputs)


def exec_layer_tests():
    unittest.main()

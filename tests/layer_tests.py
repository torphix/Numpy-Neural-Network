import unittest
import numpy as np
from src.train import Trainer
from utils import open_config
from src.model.layers import LinearLayer, Dropout


class LayerTests(unittest.TestCase):

    def test_dims(self):

        self.linear_relu = LinearLayer(10, 20, 'relu')

        test_input = np.random.randn(1, 10)
        out_relu = self.linear_relu(test_input)
        self.assertEqual(out_relu.shape, (1, 20))


class DropoutTest(unittest.TestCase):

    def test_dropout(self):
        dropout = Dropout(0.5)
        total = 0
        for i in range(100):
            x = np.random.randn(10, 10) + 1.
            x = dropout(x)
            x[x != 0] = 1
            total += x.sum()
        self.assertTrue(4750 <= int(total) <= 5250)

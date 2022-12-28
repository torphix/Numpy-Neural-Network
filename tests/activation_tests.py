import unittest
import numpy as np
from src.model.activations import Softmax, ReLU, Sigmoid


class SoftmaxTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.softmax = Softmax()

    def test_forward(self):
        x = np.random.randn(10) + 0.0001
        output = self.softmax.forward(x)
        self.assertAlmostEqual(output.sum(), 1., places=2,
                               msg=f'Softmax output: {output}')

    def test_backward(self):
        test_err = 1.
        test_values = np.random.randn(1, 10)
        diagonals = test_values*(1-test_values)
        test_loss = -1*np.matmul(test_values.T, test_values)
        np.fill_diagonal(test_loss, diagonals)
        test_loss = np.expand_dims(np.sum(test_loss*test_err, axis=1), axis=0)
        self.softmax.output_cache = test_values
        acc_loss = self.softmax.backward(test_err)
        self.assertEqual((test_loss == acc_loss).all(), True)


class ReluTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = ReLU()

    def test_forward(self):
        x = np.array([0.1, 0.0, -0.1, 1])
        output = self.relu.forward(x)
        x[x < 0] = 0
        self.assertEqual((output == x).all(), True)

    def test_backward(self):
        x = np.array([0.1, 0.0, -0.1, 1])
        output = self.relu.backward(x)
        x[x < 0] = 0
        self.assertEqual((output == x).all(), True)


class SigmoidTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmoid = Sigmoid()

    def test_forward(self):
        x = np.array([-50, 0.0, 50])
        output = self.sigmoid.forward(x)
        self.assertAlmostEqual(output[0], 0)
        self.assertAlmostEqual(output[1], 0.5)
        self.assertAlmostEqual(output[2], 1)

    def test_backward(self):
        x = np.array([-50, 0.0, 50])
        output = self.sigmoid.backward(x)
        self.assertAlmostEqual(output[0], 0)
        self.assertAlmostEqual(output[1], 0.25)
        self.assertAlmostEqual(output[2], 0)


def exec_activation_tests():
    unittest.main()

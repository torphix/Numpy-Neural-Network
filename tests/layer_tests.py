import numpy as np
from src.model.activations import Softmax, ReLU

class SoftmaxTests:

    def __init__(self):
        self.softmax = Softmax()

    def test_forward(self):
        x = np.random.randn(10) +0.01
        output = self.softmax.forward(x)
        if round(output.sum(), 2) == 1.0:
            print('Softmax Forward Passed!')
        else:
            print('Softmax Forward Failed!')

    def test_backward(self):
        self.softmax.backward()
        pass

class ReluTests:
    def __init__(self):
        self.relu = ReLU()

    def test_forward(self):
        x = np.array([0.1, 0.0, -0.1, 1])
        output = self.relu.forward(x)
        x[x < 0] = 0
        if (output == x).all() == True:
            print('Relu forward pass Passed!')
        else:
            print('Relu forward pass Failed!')

    def test_backward(self):
        x = np.array([0.1, 0.0, -0.1, 1])
        output = self.relu.backward(x)
        x[x < 0] = 0
        if (output == x).all() == True:
            print('Relu forward pass Passed!')
        else:
            print('Relu forward pass Failed!')


def exec_layer_tests():
    softmax_test = SoftmaxTests()
    softmax_test.test_forward()
    relu_test = ReluTests()
    relu_test.test_forward()
    relu_test.test_backward()
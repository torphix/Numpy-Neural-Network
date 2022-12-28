import time
import numpy as np
from .activations import BlankActivation, ReLU, Sigmoid, Softmax


class Dropout():
    def __init__(self, probability=0.5):
        self.p = int(probability * 100)

    def __call__(self, x):
        # Construct dropout matrix (ie: set some values to 0 with probability p)
        dropout_matrix = np.random.randint(1, 100, size=x.shape)
        dropout_matrix[dropout_matrix <= self.p] = 0
        dropout_matrix[dropout_matrix > self.p] = 1
        self.matrix = dropout_matrix 
        return x * self.matrix

class LayerNorm():
    def __init__(self):
        pass

    def forward(self, x):
        mu = np.mean(x, axis=-1)
        sigma = np.sqrt(np.mean((x-mu)**2, axis=-1))
        return (x-mu)/sigma

    def backward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)

class LinearLayer():
    def __init__(self, in_d, out_d, activation=None, dropout=0.):
        '''A differentiable array'''
        assert activation in ['relu', 'sigmoid', 'softmax', None], \
            f'Activation: {activation} not implemented'
        self.W = (np.random.randn(in_d, out_d)+1) * 0.1
        self.B = np.ones((1, out_d))
        self.input_cache = 0
        self.dropout = Dropout(dropout)

        if activation == 'relu':
            self.activation = ReLU()
        elif activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'softmax':
            self.activation = Softmax()
        else:
            self.activation = BlankActivation()


    def __call__(self, x):
        self.input_cache = x
        return self.forward(x)

    def forward(self, x):
        x = (x @ self.dropout(self.W)) + self.B
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

    def backward(self, global_err):
        self.W_grad = self.input_cache.swapaxes(0, 1) @ global_err
        self.W_grad *= self.dropout.matrix 
        self.B_grad = np.mean(global_err, axis=-1, keepdims=True)
        global_err = (global_err @ self.W.swapaxes(0, 1)) * \
            self.activation.backward(self.input_cache)
        return global_err

    def update(self, lr):
        self.W -= self.W_grad * lr
        self.B -= self.B_grad * lr

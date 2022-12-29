import numpy as np
from .layers import LinearLayer


class NeuralNetwork():
    def __init__(self, layers_config):
        self.layers = []
        for i in range(len(layers_config)):
            self.layers.append(LinearLayer(**layers_config[i]))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def set_eval_mode(self):
        for layer in self.layers:
            layer.set_eval_mode()
            
    def set_train_mode(self):
        for layer in self.layers:
            layer.set_train_mode()

class SGDOptimizer():
    def __init__(self, nn, lr):
        self.lr = lr
        self.nn = nn

    def backward(self, loss):
        '''Calculate the grads & new error'''
        # Calculate global loss (last layers )
        for layer in reversed(self.nn.layers):
            loss = layer.backward(loss)

    def update(self):
        for layer in self.nn.layers:
            W_update = layer.W_grad * self.lr
            B_update = layer.B_grad * self.lr
            layer.update(W_update, B_update)


class AdamOptimizer():
    def __init__(self, nn, lr, b1, b2):
        self.nn = nn
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.mt_W = [np.zeros_like(layer.W) for layer in self.nn.layers]
        self.mt_B = [np.zeros_like(layer.B) for layer in self.nn.layers]
        self.vt_W = [np.zeros_like(layer.W) for layer in self.nn.layers]
        self.vt_B = [np.zeros_like(layer.B) for layer in self.nn.layers]

    def backward(self, loss):
        '''Calculate the grads & new error'''
        # Calculate global loss (last layers )
        for layer in reversed(self.nn.layers):
            loss = layer.backward(loss)

    def update(self):
        for i, layer in enumerate(self.nn.layers):
            # Calculate scalars
            self.mt_W[i] = self.b1 * self.mt_W[i] + (1 - self.b1) * layer.W_grad
            self.mt_B[i] = self.b1 * self.mt_B[i] + (1 - self.b1) * layer.B_grad
            mt_W_hat = self.mt_W[i] / (1-self.b1)
            mt_B_hat = self.mt_B[i] / (1-self.b1)
            self.vt_W[i] = self.b2 * self.vt_W[i] + (1 - self.b2) * layer.W_grad**2
            self.vt_B[i] = self.b2 * self.vt_B[i] + (1 - self.b2) * layer.B_grad**2
            vt_W_hat = self.vt_W[i] / (1-self.b2)
            vt_B_hat = self.vt_B[i] / (1-self.b2)
            # Perform Update
            W_update =  (self.lr * mt_W_hat) / (np.sqrt(vt_W_hat + 1.0e-10))
            B_update = (self.lr * mt_B_hat) / (np.sqrt(vt_B_hat + 1.0e-10))
            layer.update(W_update, B_update)

            
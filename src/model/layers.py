import time
import numpy as np
from .activations import BlankActivation, ReLU, Sigmoid, Softmax


class LinearLayer():
  def __init__(self, in_d, out_d, activation=None):
    '''A differentiable array'''
    assert activation in ['relu','sigmoid','softmax', None], \
      f'Activation: {activation} not implemented'
    self.W = np.random.randn(in_d, out_d)*0.5
    self.B = np.zeros((1,out_d))
    self.input_cache = 0

    if activation == 'relu':
      self.activation = ReLU()
    elif activation == 'sigmoid':
      self.activation = Sigmoid()
    elif activation == 'softmax':
      self.activation = Softmax()
    else:
      self.activation = BlankActivation()

  def __call__(self, x):
    # assert len(x.shape) == 3, \
      # 'Input shape must be 3D'
    self.input_cache = x
    return self.forward(x)

  def forward(self, x):
    x = (x @ self.W) + self.B
    if self.activation is not None:
      return self.activation(x)
    else:
      return x

  def backward(self, global_err):
    self.W_grad = self.input_cache.swapaxes(0,1) @ global_err
    self.B_grad = np.sum(global_err, axis=-1, keepdims=True)
    global_err = (global_err @ self.W.swapaxes(0,1)) * self.activation.backward(self.input_cache)
    return global_err

  def update(self, lr):
    self.W -= self.W_grad * lr
    self.B -= self.B_grad * lr
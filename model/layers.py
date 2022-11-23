import time
import numpy as np
from abc import ABC, abstractmethod
from .activations import ReLU, Sigmoid, Softmax


class BaseLayer(ABC):
  '''
    Caches input values on forward call
  '''
  def __init__(self):
    self.input_cache = 0

  def __call__(self, *args, **kwargs):
    self.input_cache = args, *list(kwargs.values())
    outputs = self.forward(*args, **kwargs)
    return outputs
  
  @abstractmethod
  def forward(self, *args, **kwargs):
    '''Perform the desired computation'''
    raise NotImplementedError()

  @abstractmethod
  def backward(self):
    '''Compute the gradient'''
    raise NotImplementedError()

  @abstractmethod
  def update(self):
    '''Update the weights, Must overwrite even if blank'''
    raise NotImplementedError()


class LinearLayer(BaseLayer):
  def __init__(self, in_d, out_d, activation=None):
    super().__init__()
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
      self.activation = None

  def forward(self, x):
    x = (x @ self.W) + self.B
    if self.activation is not None:
      return self.activation(x)
    else:
      return x

  def backward(self, global_err):
    self.W_grad = self.input_cache.T @ global_err
    self.B_grad = np.sum(global_err, axis=-1, keepdims=True)
    global_err = (global_err @ self.W.T) * self.activation.backward(self.input_cache)
    return global_err

  def update(self, lr):
    self.W -= self.W_grad * lr
    self.B -= self.B_grad * lr
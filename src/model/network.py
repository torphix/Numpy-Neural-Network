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
      layer.update(self.lr)
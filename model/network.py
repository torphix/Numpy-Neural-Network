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


class SGDOptim():
  def __init__(self, nn, lr):
    self.lr = lr
    self.nn = nn

  def backward(self, loss, predictions):
    '''Calculate the grads & new error'''
    # Calculate global loss (last layers )
    loss = loss * self.nn.layers[-1].activation.backward(predictions)
    for layer in reversed(self.nn.layers):
      loss = layer.backward(loss)

  def update(self, lr):
    for layer in self.nn.layers:
      layer.update(lr)
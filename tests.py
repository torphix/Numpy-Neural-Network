import numpy as np
from .model.layers import LinearLayer
from .model.network import NeuralNetwork, SGDOptim


def test_neural_net():
  model_config = {
      'layers': [
          {'in_d':10, 'out_d':20, 'activation': 'relu'},
          {'in_d':20, 'out_d':20, 'activation': 'relu'},
          {'in_d':20, 'out_d':20, 'activation': 'relu'},
          {'in_d':20, 'out_d':9, 'activation': 'softmax'},
        ]
  } 
  LR = 0.01
  nn = NeuralNetwork(model_config['layers'])
  optim = SGDOptim(nn, LR)
  INPUT = np.array([[0,1,0,0,0,1,0,1,0,0]])
  TARGET = np.array([[1,0,1,0,0,1,1,0,1]])
  for i in range(100):
    OUT = nn(INPUT)
    loss = -(TARGET - OUT)
    optim.backward(loss, OUT)
    optim.update(LR)
  assert OUT.round().all() == TARGET.all(), f'Output: {OUT} not equal {TARGET}'
  print(f'Neural Network Working! Last output: {OUT} Target: {TARGET}')


def linear_layer_test(batch_sz=2,length=5, in_d=3, out_d=8):
  input = np.random.randn(batch_sz, length, in_d)
  layer = LinearLayer(in_d, out_d)
  output = layer(input)
  assert output.shape == (batch_sz, length, out_d)
  print('Passed')

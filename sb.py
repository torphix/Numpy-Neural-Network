from src.model.activations import Softmax
import numpy as np


x = np.array([[1,1,1,1,1,1,1,1,1,1]])
sf = Softmax()
x = sf.forward(x)
print(sf.output_cache)
out=  sf.backward(1)
print(out)
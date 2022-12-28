import abc
import numpy as np


# Activation Functions
class BaseActivation(abc.ABC):
    '''
    Base class for node in computational graph
    Must implement the following methods for new nodes
    '''

    def __call__(self, *args, **kwargs):
        outputs = self.forward(*args, **kwargs)
        return outputs

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        '''Perform the desired computation'''
        raise NotImplementedError()

    @abc.abstractmethod
    def backward(self):
        '''Compute the gradient'''
        raise NotImplementedError()


class Sigmoid(BaseActivation):

    def forward(self, x):
        return 1 / (1+(np.e**-x))

    def backward(self, x):
        '''
        Derivative of sigmoid function
        sig(x) * (1-sig(x))
        '''
        return np.multiply(self.forward(x), (1 - self.forward(x)))


class ReLU(BaseActivation):
    def forward(self, x):
        x[x < 0] = 0
        return x

    def backward(self, x):
        x[x <= 0] = 0
        return x


class BlankActivation(BaseActivation):

    def forward(self, x):
        return x

    def backward(self, x):
        return x


class Softmax(BaseActivation):

    def forward(self, x):
        '''
        Calculates the softmax, data should be in form: [BS, L, N]
        '''
        x += 1.0e-25 # Add tiny number to prevent error in np.exp
        x = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
        self.output_cache = x
        return x

    def backward(self, global_loss):
        diagonals = self.output_cache*(1-self.output_cache)
        loss = -1*np.matmul(self.output_cache.T, self.output_cache)
        np.fill_diagonal(loss, diagonals)
        loss = np.expand_dims(np.sum(loss*global_loss, axis=1), axis=0)
        return loss


class CrossEntropy(BaseActivation):
    def forward(self, outputs, targets):
        '''
        Outputs: predicted probability distribution (softmax output)
        Targets: one hot encoded class labels
        '''
        # Add a tiny number to prevent taking log of 0 (nan)
        return -np.log(outputs[:, targets.argmax()] + 1.0e-10)

    def backward(self, outputs, targets):
        '''
        derivative of a log: 1/x (natural)
        '''
        loss = -targets * (1/np.maximum(1.0e-25, outputs))
        return loss

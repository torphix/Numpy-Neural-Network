import numpy as np
from .data import MNISTDataloader
from utils import load_data, one_hot_encode
from .model.activations import CrossEntropy, Softmax
from .model.network import NeuralNetwork, SGDOptimizer


class Trainer:
    def __init__(self, config):
        # Init hyperparams
        self.epochs = config['epochs']
        # Init Layers
        self.model = NeuralNetwork(config['layers'])
        self.optim = SGDOptimizer(self.model, config['learning_rate'])
        self.loss = CrossEntropy()
        self.softmax = Softmax()
        # Init Data
        self.dataloader = MNISTDataloader(config['train_val_test_split'], shuffle=True, n_samples=-1) 

    def __call__(self):
        train_metrics, val_metrics = {}, {}
        prev_val_acc = 0
        for e in range(self.epochs):
            train_loss, train_acc = self.train_iter(self.dataloader.train_dataloader())
            val_loss, val_acc = self.eval_iter(self.dataloader.val_dataloader())
            if val_acc > prev_val_acc:
                print(val_acc)
            prev_val_acc = val_acc
        pass

    def train_iter(self, dataloader):
        running_loss, running_acc = 0, 0
        for x, y in dataloader:
            outputs = self.model(x)
            outputs = self.softmax(outputs)
            target = one_hot_encode(y)
            running_loss += self.loss.forward(outputs, target)
            loss = self.loss.backward()
            loss = self.softmax.backward(loss)
            self.optim.backward(loss)
            self.optim.update()
            running_acc += int(outputs.argmax() == target.argmax())
        running_acc /= self.dataloader.train_len
        running_loss /= self.dataloader.train_len
        return running_loss, running_acc

    def eval_iter(self, dataloader):
        running_loss, running_acc = 0, 0
        for x, y in dataloader:
            outputs = self.model(x)
            outputs = self.softmax(outputs)
            target = one_hot_encode(y)
            running_loss += self.loss.forward(outputs, target)
            running_acc += int(outputs.argmax() == target.argmax())
        running_acc /= self.dataloader.train_len
        running_loss /= self.dataloader.train_len
        return running_loss, running_acc
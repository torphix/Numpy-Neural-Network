import numpy as np
from utils import one_hot_encode
from .data import MNISTDataloader
from prettytable import PrettyTable
from .model.activations import CrossEntropy, Softmax
from .model.network import NeuralNetwork, SGDOptimizer


class Trainer:
    def __init__(self, config):
        # Init hyperparams
        self.epochs = config['epochs']
        self.log_steps = config['log_n_steps']
        self.log_run = config['log_run']
        # Init Layers
        self.model = NeuralNetwork(config['layers'])
        self.optim = SGDOptimizer(self.model, config['learning_rate'])
        self.loss = CrossEntropy()
        self.softmax = Softmax()
        # Init Data
        self.dataloader = MNISTDataloader(config['train_val_test_split'], shuffle=True, n_samples=config['use_n_datasamples']) 
        self.metrics_table = PrettyTable(['Epoch', 'Train Acc', 'Val Acc', 'LR'])
        self.log_metrics(0,0,0,0)

        if self.log_run:
            print(self.metrics_table)

    def __call__(self):

        for e in range(self.epochs):
            train_loss, train_acc = self.train_iter(self.dataloader.train_dataloader())
            val_loss, val_acc = self.eval_iter(self.dataloader.val_dataloader())

            if e % self.log_steps == 0:
                self.log_metrics(e, train_acc, val_acc, self.optim.lr)                
        final_outputs = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        return final_outputs

    def train_iter(self, dataloader):
        running_loss, running_acc = 0, 0
        for x, y in dataloader:
            y = np.array([y])
            outputs = self.model(x)
            outputs = self.softmax(outputs)
            targets = one_hot_encode(y)
            running_loss += self.loss.forward(outputs, targets)
            loss = self.loss.backward(outputs, targets)
            loss = self.softmax.backward(loss)
            self.optim.backward(loss)
            self.optim.update()
            running_acc += int(outputs.argmax() == targets.argmax())
        running_acc /= self.dataloader.train_len
        # running_loss /= self.dataloader.train_len
        return running_loss.sum(), running_acc

    def eval_iter(self, dataloader):
        running_loss, running_acc = 0, 0
        for x, y in dataloader:
            outputs = self.model(x)
            outputs = self.softmax(outputs)
            targets = one_hot_encode(y)
            running_loss += outputs-targets
            running_acc += int(outputs.argmax() == targets.argmax())
        running_acc /= self.dataloader.val_len
        running_loss /= self.dataloader.val_len
        return running_loss, running_acc

    def log_metrics(self, epoch, train_acc, val_acc, lr):
        self.metrics_table.add_row([
            epoch,
            f'{round(train_acc, 2)}%', 
            f'{round(val_acc, 2)}%', 
            round(lr, 8)])
        if self.log_run:
            print( "\n".join(self.metrics_table.get_string().splitlines()[-2:])) # Print only new row
import sys
import numpy as np
from utils import one_hot_encode
from prettytable import PrettyTable
from .data import Dataloader, load_data
from .model.activations import CrossEntropy, Softmax
from .model.network import NeuralNetwork, SGDOptimizer


class Trainer:
    def __init__(self, config):
        # Init hyperparams
        self.epochs = config['epochs']
        self.log_steps = config['log_n_steps']
        self.log_run = config['log_run']
        self.use_early_stopping = config['use_early_stopping']
        self.early_stopping_patience = config['early_stopping_patience']
        # Init Layers
        self.model = NeuralNetwork(config['layers'])
        self.optim = SGDOptimizer(self.model, config['learning_rate'])
        self.loss = CrossEntropy()
        self.softmax = Softmax()
        # Init Data
        train_data, val_data, test_data = load_data(config['train_val_test_split'],
                                                    n_samples=config['use_n_datasamples'])

        self.train_dataloader = Dataloader(train_data, shuffle=True)
        self.val_dataloader = Dataloader(val_data, shuffle=True)
        self.test_dataloader = Dataloader(test_data, shuffle=True)
        self.metrics_table = PrettyTable(
            ['Epoch', 'Train Loss', 'Train Acc', 'Val Acc', 'LR'])


    def __call__(self):
        test_loss, test_acc = self.eval_iter(self.test_dataloader)
        self.running_val_acc = []
        if self.log_run:
            print(f'Starting Test Accuracy: {round(test_acc, 2)*100}%')
            self.log_metrics(0, 0, 0, 0, self.optim.lr)
            print(self.metrics_table)

        for e in range(self.epochs):
            train_loss, train_acc = self.train_iter(self.train_dataloader)
            val_loss, val_acc = self.eval_iter(self.val_dataloader)

            if self.use_early_stopping:
                self.check_early_stopping(val_acc)

            if e % self.log_steps == 0:
                self.log_metrics(e, train_loss, train_acc,
                                 val_acc, self.optim.lr)

        test_loss, test_acc = self.eval_iter(self.test_dataloader)
        if self.log_run:
            print(f'End Test Accuracy: {round(test_acc, 2)*100}%')

        final_outputs = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }
        return final_outputs

    def check_early_stopping(self, val_acc):
        if self.use_early_stopping:
            running_accs = self.running_val_acc[-self.early_stopping_patience:]
            val_check = sum(
                [1 if val_acc >= acc else 0 for acc in running_accs])
            if val_check == 0:
                test_loss, test_acc = self.eval_iter(self.train_dataloader)
                print(f'Final Testset Accuracy {test_acc}')
                sys.exit(f'Val Accuracy has not increased for {self.early_stopping_patience} epochs, Stopping')

    def train_iter(self, dataloader):
        running_loss, running_acc = 0, 0
        self.model.set_train_mode()
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
        running_acc /= len(dataloader)
        running_loss /= len(dataloader)
        return running_loss.mean(), running_acc

    def eval_iter(self, dataloader):
        running_loss, running_acc = 0, 0
        self.model.set_eval_mode()
        for x, y in dataloader:
            outputs = self.model(x)
            outputs = self.softmax(outputs)
            targets = one_hot_encode(y)
            running_loss += outputs-targets
            running_acc += int(outputs.argmax() == targets.argmax())
        running_loss /= len(dataloader)
        running_acc /= len(dataloader)
        return running_loss.mean(), running_acc

    def log_metrics(self, epoch, train_loss, train_acc, val_acc, lr):
        self.running_val_acc.append(val_acc)
        self.metrics_table.add_row([
            epoch,
            f'{round(train_loss, 4)}',
            f'{round(train_acc, 2)}%',
            f'{round(val_acc, 2)}%',
            round(lr, 8)])
        if self.log_run:
            # Print only new row
            print("\n".join(self.metrics_table.get_string().splitlines()[-2:]))

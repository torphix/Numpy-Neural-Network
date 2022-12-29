import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import one_hot_encode
from prettytable import PrettyTable
from .data import Dataloader, load_data
from .model.activations import CrossEntropy, Softmax
from .model.network import NeuralNetwork, SGDOptimizer, AdamOptimizer


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
        self.optim = self.init_optim(**config['optim'])
        self.loss = CrossEntropy()
        self.softmax = Softmax()
        # Init Data
        train_data, val_data, test_data = load_data(config['train_val_test_split'],
                                                    n_samples=config['use_n_datasamples'])

        self.train_dataloader = Dataloader(train_data, shuffle=True)
        self.val_dataloader = Dataloader(val_data, shuffle=True)
        self.test_dataloader = Dataloader(test_data, shuffle=True)
        self.metrics_table = PrettyTable(
            ['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'LR'])
        self.running_val_acc = []
        self.running_val_loss = []
        self.running_train_acc = []
        self.running_train_loss = []

    def __call__(self):
        test_loss, test_acc = self.eval_iter(self.test_dataloader)
        if self.log_run:
            print(f'Starting Test Accuracy: {round(test_acc, 2)*100}%')
            self.log_metrics(0, 0, 0, 0, 0, self.optim.lr)
            print(self.metrics_table)

        for e in range(self.epochs):
            train_loss, train_acc = self.train_iter(self.train_dataloader)
            val_loss, val_acc = self.eval_iter(self.val_dataloader)

            if self.use_early_stopping:
                self.check_early_stopping(val_acc)

            if e % self.log_steps == 0:
                self.log_metrics(e,
                                 train_loss,
                                 train_acc,
                                 val_loss,
                                 val_acc,
                                 self.optim.lr)

        test_loss, test_acc = self.eval_iter(self.test_dataloader)
        if self.log_run:
            print(f'End Test Accuracy: {round(test_acc, 2)*100}%')

        final_outputs = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }
        self.vis_results()
        return final_outputs

    def vis_results(self):
        fig, axs = plt.subplots(2)
        fig.suptitle('Training Results')
        # Loss Plot
        epochs = [i for i in range(len(self.running_train_loss))]
        axs[0].plot(epochs, self.running_train_loss)
        axs[0].plot(epochs, self.running_val_loss)
        axs[0].set_xlabel('Epochs')
        axs[0].set_title('Training Loss')
        # Accuracy Plot
        axs[1].plot(epochs, [i*100 for i in self.running_train_acc])
        axs[1].plot(epochs, [i*100 for i in self.running_val_acc])
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Training Accuracy')
        axs[1].set_ylim(0, 100)
        plt.show()

    def check_early_stopping(self, val_acc):
        if self.use_early_stopping:
            running_accs = self.running_val_acc[-self.early_stopping_patience:]
            val_check = sum(
                [1 if val_acc >= acc else 0 for acc in running_accs])
            if val_check == 0:
                test_loss, test_acc = self.eval_iter(self.train_dataloader)
                print(f'Final Testset Accuracy {test_acc}')
                sys.exit(
                    f'Val Accuracy has not increased for {self.early_stopping_patience} epochs, Stopping')

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
            running_loss += self.loss.forward(outputs, targets)
            running_acc += int(outputs.argmax() == targets.argmax())
        running_loss /= len(dataloader)
        running_acc /= len(dataloader)
        return running_loss.mean(), running_acc

    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        self.running_val_acc.append(val_acc)
        self.running_val_loss.append(val_loss)
        self.running_train_acc.append(train_acc)
        self.running_train_loss.append(train_loss)

        self.metrics_table.add_row([
            epoch,
            f'{round(train_loss, 4)}',
            f'{round(train_acc, 2)}%',
            f'{round(val_loss, 4)}',
            f'{round(val_acc, 2)}%',
            round(lr, 8)])
        if self.log_run:
            # Print only new row
            print("\n".join(self.metrics_table.get_string().splitlines()[-2:]))

    def init_optim(self, learning_rate, name, b1=0, b2=0):
        name = name.lower()
        if name == 'adam':
            return AdamOptimizer(self.model, learning_rate, b1, b2)
        elif name == 'sgd':
            return SGDOptimizer(self.model, learning_rate)
        else:
            raise NotImplementedError(
                f'{name} optimizer not implemented please use adam or sgd')

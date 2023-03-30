
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger('MODEL.Net')

class Net(nn.Module):
    def __init__(self, config):
        '''
        We define a recurrent network that predicts the future values of a time-dependent variable based on
        past inputs and covariates.
        '''
        super(Net, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.DATA.NUM_CLASS*config.DATA.OUTPUT_DIM, config.MODEL.EMBEDDING_DIM)

        self.lstm = nn.LSTM(input_size=config.DATA.COV_DIM + 1 + config.MODEL.EMBEDDING_DIM,
                            hidden_size=config.MODEL.LSTM_HIDDEN_DIM,
                            num_layers=config.MODEL.LSTM_LAYERS,
                            bias=True,
                            batch_first=False,
                            dropout=config.MODEL.DROPOUT)

        self.dropout = nn.Dropout(config.MODEL.DROPOUT)
        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(config.MODEL.LSTM_HIDDEN_DIM * config.MODEL.LSTM_LAYERS, config.MODEL.LSTM_HIDDEN_DIM * config.MODEL.LSTM_LAYERS)
        self.fc2 = nn.Linear(config.MODEL.LSTM_HIDDEN_DIM * config.MODEL.LSTM_LAYERS, config.MODEL.LSTM_HIDDEN_DIM * config.MODEL.LSTM_LAYERS)

        self.distribution_mu = nn.Linear(config.MODEL.LSTM_HIDDEN_DIM * config.MODEL.LSTM_LAYERS, config.DATA.PREDICT_STEP)


    def forward(self, x, idx):

        onehot_embed = self.embedding(idx)
        lstm_input = torch.cat((x, onehot_embed.repeat(self.config.DATA.PREDICT_START, 1, 1)), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (self.init_hidden(lstm_input.shape[1]), self.init_cell(lstm_input.shape[1])))
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        x = self.relu(self.dropout(self.fc1(hidden_permute)))
        x = self.relu(self.dropout(self.fc2(x)))
        mu = self.distribution_mu(x)

        return mu.view(-1, self.config.DATA.PREDICT_STEP, 1)


    def init_hidden(self, input_size):
        return torch.zeros(self.config.MODEL.LSTM_LAYERS, input_size, self.config.MODEL.LSTM_HIDDEN_DIM,
                           device=self.config.TRAIN.DEVICE)

    def init_cell(self, input_size):
        return torch.zeros(self.config.MODEL.LSTM_LAYERS, input_size, self.config.MODEL.LSTM_HIDDEN_DIM,
                           device=self.config.TRAIN.DEVICE)

    def test(self, x, v_batch, id_batch, hidden, cell, sampling=False):
        batch_size = x.shape[1]
        pred = self(x.unsqueeze(0), id_batch, hidden, cell)

        return pred


def loss_fn_with_weight(mu: Variable, labels: Variable, trend_labels: Variable):
    labels = labels.permute(1, 0, 2).contiguous()
    mse = torch.pow((mu - labels), 2)
    wight = torch.unsqueeze(torch.linspace(1.5, 0.1, mu.shape[1]), 1).repeat(1, mu.shape[2])
    return torch.mean(mse*wight)


def loss_fn_with_mse_trend_weight(mu: Variable, labels: Variable, labels_trend: Variable):
    labels = labels.permute(1, 0, 2).contiguous()
    labels_trend = labels_trend.permute(1, 0, 2).contiguous()

    trend_mu = mu[:, :-1, :] - mu[:, 1:, ]
    trend_mse = torch.pow((labels_trend[:, 1:, :] - trend_mu), 2)

    mse = torch.pow((mu - labels), 2)
    wight = torch.unsqueeze(torch.linspace(1.5, 0.1, mu.shape[1]), 1).repeat(1, mu.shape[2])

    return torch.mean(mse*wight) + 0.5 * torch.mean(trend_mse)



# if relative is set to True, metrics are not normalized by the scale of labels
def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    if relative:
        diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return [diff, summation]

def accuracy_ND_each_predition( mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)

    diff = torch.sum(torch.abs(mu - labels), axis=[0, 2])
    summation = torch.sum(labels, axis=[0, 2])
    return [diff.numpy(), summation.numpy()]




def NUM_wrong_each_predition(mu: torch.Tensor, x_input: torch.Tensor, labels: torch.Tensor, timestamp, config, test=True):

    wrong_1 = torch.sum((torch.abs(mu - labels) >= 1), axis=[0, 2])
    wrong_2 = torch.sum((torch.abs(mu - labels) >= 2), axis=[0, 2])
    wrong_3 = torch.sum((torch.abs(mu - labels) >= 3), axis=[0, 2])


    # 最差的样本
    X, Y, Z = torch.where((torch.abs(mu - labels) >= 2))
    num_samples = config.DATA.TEST_WINDOWS
    tmp = 1
    if X.any():
        x_0 = X[0]
        for x, y, z in zip(X, Y, Z):
            if tmp >= 20:
                break
            if x != x_0 :

                col = timestamp[x]

                wrong_3_pre = mu[x, :, z].numpy()
                wrong_3_label = x_input[x, :, z + config.DATA.COV_DIM].numpy()
                wrong_3_pre = np.concatenate([wrong_3_label[:config.DATA.PREDICT_START], wrong_3_pre])
                wrong_3_label = np.concatenate([wrong_3_label[:config.DATA.PREDICT_START], labels[x, :, z].numpy()])
                x_range = np.arange(start=1, stop=num_samples + 1)
                f = plt.figure(figsize=(20, 5))
                plt.plot(x_range, wrong_3_pre, color='red', label='pre')
                plt.plot(x_range, wrong_3_label, color='blue', label='label',)
                name = 'test_' if test else 'eval_'
                plot_dir = config.TEST.PLOT_DIR if test else config.TRAIN.PLOT_DIR
                plt.title(name + 'wrong_case_' + str(tmp) + '_' + col )
                x_major_locator = plt.MultipleLocator(10)
                ax = plt.gca()
                ax.xaxis.set_major_locator(x_major_locator)
                plt.legend()
                f.savefig(os.path.join('experiments', config.TRAIN.MODEL_FILE_NAME, plot_dir, name + 'wrong_case_' + str(tmp) + '.png'))
                plt.close()
                tmp += 1
                x_0 = x

    # 最好的样本
    X, Z = np.where(torch.sum((torch.abs(mu - labels) >= 0.2), axis=[1])==0)


    tmp = 1
    if X.any():
        x_0 = X[0]
        for x,  z in zip(X,  Z):
            if tmp >= 20:
                break
            if x != x_0 :
                # lz警告，临时写法
                col = timestamp[x]

                right_3_pre = mu[x, :, z].numpy()
                right_3_label = x_input[x, :, z + config.DATA.COV_DIM].numpy()
                right_3_pre = np.concatenate([right_3_label[:config.DATA.PREDICT_START], right_3_pre])
                right_3_label = np.concatenate([right_3_label[:config.DATA.PREDICT_START], labels[x, :, z].numpy()])
                x_range = np.arange(start=1, stop=num_samples +1)
                f = plt.figure(figsize=(20, 5))
                plt.plot(x_range, right_3_pre, color='red', label='pre')
                plt.plot(x_range, right_3_label, color='blue', label='label')
                name = 'test_' if test else 'eval_'
                plot_dir = config.TEST.PLOT_DIR if test else config.TRAIN.PLOT_DIR
                plt.title(name + 'right_case_' + str(tmp) + '_' + timestamp[x])
                x_major_locator = plt.MultipleLocator(10)
                ax = plt.gca()
                ax.xaxis.set_major_locator(x_major_locator)
                plt.legend()
                f.savefig(os.path.join('experiments', config.TRAIN.MODEL_FILE_NAME, plot_dir, name + 'right_case_' + str(tmp) + '.png'))
                plt.close()
                tmp += 1
                x_0 = x


    all_data = torch.sum(mu!=0, axis=[0, 2])


    return [wrong_1.numpy(), wrong_2.numpy(), all_data.numpy()]



def accuracy_test_loss(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    return [F.mse_loss(mu, labels), 1 ]

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging

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
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.MODEL.DROPOUT)
        self.fc_1 = nn.Linear((config.DATA.COV_DIM + 1) * config.DATA.PREDICT_START + config.MODEL.EMBEDDING_DIM, config.DATA.COV_DIM + 1 + config.MODEL.EMBEDDING_DIM)
        self.fc_2 = nn.Linear(config.DATA.COV_DIM + 1 + config.MODEL.EMBEDDING_DIM, config.DATA.COV_DIM + 1 + config.MODEL.EMBEDDING_DIM)
        self.fc_3 = nn.Linear(config.DATA.COV_DIM + 1 + config.MODEL.EMBEDDING_DIM, 2)
        self.softmax = nn.Softmax()


    def forward(self, x, idx):

        onehot_embed = self.embedding(idx)
        fc_input = torch.cat((x.view(x.shape[0], -1), onehot_embed), dim=1)
        x = self.relu(self.dropout(self.fc_1(fc_input)))
        x = self.relu(self.dropout(self.fc_2(fc_input)))
        x = self.fc_3(x)
        output = self.softmax(x, dim=1)
        return output



def loss_fn(mu: Variable, labels_batch, trend_label_batch, positive: Variable,):

    loss = F.cross_entropy(mu, positive)

    return loss


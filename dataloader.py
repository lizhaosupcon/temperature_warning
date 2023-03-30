from __future__ import division
import numpy as np
import torch
import os
import logging
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger('MODEL.Data')

class TrainDataset(Dataset):
    def __init__(self, data_path, data_name, model_type):
        self.data = np.load(os.path.join(data_path, model_type, f'train_data_{data_name}.npy'))
        self.label = np.load(os.path.join(data_path, model_type, f'train_label_{data_name}.npy'))
        self.trend_label = np.load(os.path.join(data_path, model_type, f'train_trend_label_{data_name}.npy'))
        self.positive_label = np.load(os.path.join(data_path, model_type, f'train_positive_label_{data_name}.npy'))
        self.train_len = self.data.shape[0]
        logger.info(f'train_len: {self.train_len}')
        logger.info(f'building datasets from {data_path}...')

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        # 返回的数据定义为： 5+12位特征＋序列标志+label
        return (self.data[index, :, :-1], int(self.data[index, 0, -1]), self.label[index], self.trend_label[index],
                self.positive_label[index])

class TestDataset(Dataset):
    def __init__(self, data_path, data_name, model_type):
        self.data = np.load(os.path.join(data_path, model_type, f'test_data_{data_name}.npy'))
        self.v = np.load(os.path.join(data_path, model_type, f'test_v_{data_name}.npy'))
        self.label = np.load(os.path.join(data_path, model_type, f'test_label_{data_name}.npy'))
        self.label_no_normal = np.load(os.path.join(data_path, model_type, f'test_label_no_normal_{data_name}.npy'))
        self.x_input_no_normal = np.load(os.path.join(data_path, model_type, f'test_x_input_no_normal_{data_name}.npy'))
        self.trend_label = np.load(os.path.join(data_path, model_type, f'test_trend_label_{data_name}.npy'))
        self.positive_label = np.load(os.path.join(data_path, model_type, f'test_positive_label_{data_name}.npy'))
        self.timestamp = np.load(os.path.join(data_path, model_type, f'test_timestamp_{data_name}.npy'))
        self.test_len = self.data.shape[0]
        logger.info(f'test_len: {self.test_len}')
        logger.info(f'building datasets from {data_path}...')

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return (self.data[index, :, :-1], int(self.data[index, 0, -1]), self.v[index], self.label[index],
                self.x_input_no_normal[index], self.label_no_normal[index], self.trend_label[index],
                self.timestamp[index], self.positive_label[index])


class EvalDataset(Dataset):
    def __init__(self, data_path, data_name, model_type):
        self.data = np.load(os.path.join(data_path, model_type, f'eval_data_{data_name}.npy'))
        self.v = np.load(os.path.join(data_path, model_type, f'eval_v_{data_name}.npy'))
        self.label = np.load(os.path.join(data_path, model_type, f'eval_label_{data_name}.npy'))
        self.label_no_normal = np.load(os.path.join(data_path, model_type, f'eval_label_no_normal_{data_name}.npy'))
        self.x_input_no_normal = np.load(os.path.join(data_path, model_type, f'eval_x_input_no_normal_{data_name}.npy'))
        self.trend_label = np.load(os.path.join(data_path, model_type, f'eval_trend_label_{data_name}.npy'))
        self.positive_label = np.load(os.path.join(data_path, model_type, f'eval_positive_label_{data_name}.npy'))
        self.timestamp = np.load(os.path.join(data_path, model_type, f'eval_timestamp_{data_name}.npy'))
        self.test_len = self.data.shape[0]
        logger.info(f'eval_len: {self.test_len}')
        logger.info(f'building datasets from {data_path}...')

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return (self.data[index, :, :-1], int(self.data[index, 0, -1]), self.v[index], self.label[index],
                self.x_input_no_normal[index], self.label_no_normal[index], self.trend_label[index],
                self.timestamp[index], self.positive_label[index])

class WeightedSampler(Sampler):
    def __init__(self, data_path, data_name, replacement=True):
        v = np.load(os.path.join(data_path, f'train_v_{data_name}.npy'))
        self.weights = torch.as_tensor(np.abs(v[:,0])/np.sum(np.abs(v[:,0])), dtype=torch.double)
        logger.info(f'weights: {self.weights}')
        self.num_samples = self.weights.shape[0]
        logger.info(f'num samples: {self.num_samples}')
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples
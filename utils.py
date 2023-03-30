
import json
import logging
import os
import shutil

import torch
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger('MODEL.Utils')



class RunningAverage:
    '''A simple class that maintains the running average of a quantity
    Example:
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    '''

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    '''Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    '''
    _logger = logging.getLogger('MODEL')
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)
    _logger.addHandler(TqdmHandler(fmt))


def save_dict_to_json(d, json_path):
    '''Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    '''
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4, ensure_ascii=False)


def save_checkpoint(state, is_best, epoch, checkpoint, ins_name=-1):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
        ins_name: (int) instance index
    '''
    if ins_name == -1:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}.pth.tar')
    else:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}_ins_{ins_name}.pth.tar')
    if not os.path.exists(checkpoint):
        logger.info(f'Checkpoint Directory does not exist! Making directory {checkpoint}')
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    logger.info(f'Checkpoint saved to {filepath}')
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
        logger.info('Best checkpoint copied to best.pth.tar')


def load_checkpoint(checkpoint, model, optimizer=None):
    '''Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        gpu: which gpu to use
    '''
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def plot_ND_exchage(variable, save_name, location='./figures/'):
    num_samples = variable.shape[0]
    x = np.arange(start=1, stop=num_samples+1)
    f = plt.figure()
    plt.plot(x, variable[:num_samples]*100)
    plt.xlabel('prediction step')
    plt.ylabel(save_name + '%' )
    f.savefig(os.path.join(location, 'best_' + save_name + '.png'))
    plt.close()




def plot_all_epoch(variable, save_name, location='./figures/'):
    num_samples = variable.shape[0]
    x = np.arange(start=1, stop=num_samples + 1)
    f = plt.figure()
    plt.plot(x, variable[:num_samples])
    f.savefig(os.path.join(location, save_name + '_summary.png'))
    plt.close()


# 分类模型相关的矩阵操纵
def init_metrics_classification_model(cfg):
    metrics = {
        'ND': np.zeros(2),
        'test_loss': np.zeros(2),
    }
    return metrics

def update_metrics_classification_model(raw_metrics,
                                        loss_fn,
                                        pre,
                                        labels,
                                        x_input_no_normal,
                                        labels_no_normal,
                                        trend_labels,
                                        timestamp,
                                        positive_batch,
                                        v_batch,
                                        config):
    import model.MLP as net
    raw_metrics['ND'] = raw_metrics['ND'] + net.accuracy_ND(pre, positive_batch)
    raw_metrics['test_loss'] = raw_metrics['test_loss'] + net.loss_fn(pre, positive_batch)
    return raw_metrics

def final_metrics_classification_model(raw_metrics):
    summary_metric = {}
    summary_metric['ND'] = raw_metrics['ND'][0] / raw_metrics['ND'][1]
    summary_metric['test_loss'] = raw_metrics['test_loss'][0] / raw_metrics['test_loss'][1]
    return summary_metric









# 下面是单独为两个预测模型做的分类
def init_metrics_prediction_model(cfg):
    metrics = {
        'ND': np.zeros(2),
        'ND_no_normal': np.zeros(2),
        'ND_each_prediction': np.zeros((2, cfg.TRAIN.PREDICT_STEP)),
        'test_loss': np.zeros(2),
        'num_each_prediction': np.zeros((3, cfg.TRAIN.PREDICT_STEP)),
    }

    return metrics

def update_metrics_prediction_model(raw_metrics,
                                    loss_fn,
                                    pre,
                                    labels,
                                    x_input_no_normal,
                                    labels_no_normal,
                                    trend_labels,
                                    timestamp,
                                    positive_batch,
                                    v_batch,
                                    config):

    # lz警告，这里要加一个判断，然后决定用来自那个model 文件下的指
    import model.MLP as net
    raw_metrics['ND'] = raw_metrics['ND'] + net.accuracy_ND(pre, labels.permute(1, 0, 2))
    raw_metrics['ND_each_prediction'] = raw_metrics['ND_each_prediction'] + net.accuracy_ND_each_predition(pre, labels.permute(1, 0, 2))

    raw_metrics['test_loss'] = raw_metrics['test_loss'] + [loss_fn(pred, labels, trend_labels) * pred.shape[0], pre.shape[0]]
    pre = torch.unsqueeze(v[:,:,0] -v[:,:,1], 1).repeat(1, pre.shape[1], 1) * pred + torch.unsqueeze(v[:,:,1], 1).repeat(1, pred.shape[1], 1)

    raw_metrics['ND_no_normal'] = raw_metrics['ND_no_normal'] + net.accuracy_ND(pred, labels_no_normal.permute(1, 0, 2))
    raw_metrics['RMSE'] = raw_metrics['RMSE'] + net.accuracy_RMSE(pred, labels_no_normal.permute(1, 0, 2))
    raw_metrics['num_each_prediction'] = raw_metrics['num_each_prediction'] + net.NUM_wrong_each_predition(pred,x_input_no_normal,labels_no_normal.permute( 1, 0, 2), timestamp, config, test=True)



def final_metrics_prediction_model(raw_metrics):
    summary_metric = {}
    summary_metric['ND'] = raw_metrics['ND'][0] / raw_metrics['ND'][1]
    summary_metric['test_loss'] = raw_metrics['test_loss'][0] / raw_metrics['test_loss'][1]
    return summary_metric


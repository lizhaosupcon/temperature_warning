import argparse
import logging
import os

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils

from evaluate import *
from dataloader import *
from experiments.configs.Base_Config import get_config


import matplotlib.pyplot as plt


# log写在开头 防止被随机import覆盖
logger = logging.getLogger('MODEL.Train')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train(model: nn.Module,
          optimizer: optim,
          loss_fn,
          train_loader: DataLoader,
          config,
          epoch: int) -> float:

    model.train()
    loss_epoch = np.zeros(len(train_loader))
    for i, (train_batch, idx, labels_batch, trend_label_batch, positive_batch) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        train_batch = train_batch.permute(1, 0, 2).to(torch.float32).to(config.TRAIN.DEVICE)  # not scaled
        labels_batch = labels_batch.permute(1, 0, 2).to(torch.float32).to(config.TRAIN.DEVICE)  # not scaled
        trend_label_batch = trend_label_batch.permute(1, 0, 2).to(torch.float32).to(config.TRAIN.DEVICE)
        positive_batch = positive_batch.permute(1, 0, 2).to(torch.float32).to(config.TRAIN.DEVICE)

        idx = idx.unsqueeze(0).to(config.TRAIN.DEVICE)

        mu = model(train_batch, idx)
        loss = loss_fn(mu,  labels_batch, trend_label_batch, positive_batch)

        loss.backward()
        optimizer.step()

        loss_epoch[i] = loss

    logger.info(f'train_loss: {loss_epoch.mean().item()}')

    return loss_epoch



def train_and_evaluate(model: nn.Module,
                       train_loader: DataLoader,
                       test_loader: DataLoader,
                       optimizer: optim, loss_fn,
                       config,
                       EPOCH_NUM ) -> None:

    # 模型的恢复与保存
    epoch_start = 0
    if EPOCH_NUM != -1:
        restore_path = os.path.join('experiments', config.TRAIN.MODEL_FILE_NAME, 'epoch_' + str(EPOCH_NUM) + '.pth.tar')
        logger.info('Restoring parameters from {}'.format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
        epoch_start = EPOCH_NUM


    logger.info('begin training and evaluation')

    best_test = float('inf')
    train_len = len(train_loader)
    epoch_summary = np.zeros(config.TRAIN.EPOCH)
    loss_summary = np.zeros((train_len * config.TRAIN.EPOCH))

    for epoch in range(epoch_start, config.TRAIN.EPOCH):
        if epoch == 100:
            print(111111111111111111)
        logger.info('Epoch {}/{}'.format(epoch + 1, config.TRAIN.EPOCH))
        loss_summary[epoch * train_len:(epoch + 1) * train_len] = train(model, optimizer, loss_fn, train_loader,
                                                                        config, epoch)

        # 模型的保存部分
        test_metrics= evaluate(model, loss_fn, test_loader, config)


        # 按照test的权重采样思路取nd
        epoch_summary[epoch] = test_metrics['ND']
        is_best = test_metrics['test_loss'] <= best_test and epoch >= 5
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              epoch=epoch,
                              is_best=is_best,
                              checkpoint=model_dir)

        if is_best:
            logger.info('- Found new best loss')
            best_test_ND = test_metrics['test_loss']
            best_json_path = os.path.join(model_dir, 'metrics_test_best_weights.json')
            utils.save_dict_to_json(test_metrics, best_json_path)


        logger.info('Current Best loss is: %.5f' % best_test_ND)

        utils.plot_all_epoch(epoch_summary[:epoch + 1], config.DATA.DATASET + '_ND', train_plot_dir)
        utils.plot_all_epoch(loss_summary[:(epoch + 1) * train_len], config.DATA.DATASET + '_loss', train_plot_dir)
        last_json_path = os.path.join(model_dir, 'metrics_test_last_weights.json')
        utils.save_dict_to_json(test_metrics, last_json_path)



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='base.yaml', required=False, metavar="FILE",
                        help='config file name', )
    parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')


    # 将参数读取出来，相当于解析。parse_known_args的好处是如果输入的格式不对，不报错，存在unparsed里面
    args, unparsed = parser.parse_known_args()
    logger.info('try to merge config from {}'.format(args.cfg))
    config = get_config(args)

    # 模型种类的区分
    if config.MODEL.MODEL_TYPE == 'classification':
        import model.MLP as net
    else:
        import model.LSTM as net

    # 设置文件夹地址，并检查文件地址是否可达
    model_dir = os.path.join('experiments', config.TRAIN.MODEL_FILE_NAME)
    data_dir = os.path.join(config.DATA.DATASET, config.DATA.SAVE_NAME)
    train_plot_dir = os.path.join('experiments', config.TRAIN.MODEL_FILE_NAME, config.TRAIN.PLOT_DIR)
    test_plot_dir = os.path.join('experiments', config.TRAIN.MODEL_FILE_NAME, config.TRAIN.PLOT_DIR)
    try:
        os.mkdir(model_dir)
        os.mkdir(train_plot_dir)
        os.mkdir(test_plot_dir)
    except FileExistsError:
        pass



    # log标记训练开始位置
    utils.set_logger(os.path.join(model_dir, 'train.log'))

    # gpu cpu环境检测
    cuda_exist = torch.cuda.is_available()
    if cuda_exist:
        if config.TRAIN.DEVICE == 'cpu':
            logger.info('Using cpu, but can set DEVICE:cuda in yaml file to use Cuda...')
            assert False
        else:
            logger.info('Using Cuda...')
            model = net.Net(config).cuda()
    else:
        if config.TRAIN.DEVICE == 'cpu':
            logger.info('Using cpu...')
        else:
            logger.info('Can not find Cuda, please set DEVICE:cpu in yaml file ...')
            assert False
        model = net.Net(config)


    # 读取数据，得到data loader函数
    logger.info('Loading the datasets...')
    train_set = TrainDataset(data_dir, config.DATA.SAVE_NAME, config.MODEL.MODEL_TYPE)
    test_set = TestDataset(data_dir, config.DATA.SAVE_NAME, config.MODEL.MODEL_TYPE)
    # sampler = WeightedSampler(data_dir, config.DATA.SAVE_NAME)  # Use weighted sampler instead of random sampler
    train_loader = DataLoader(train_set, batch_size=config.TRAIN.BATCH_SIZE, sampler=RandomSampler(train_set), num_workers=0)
    test_loader = DataLoader(test_set, batch_size=config.TEST.BATCH_SIZE, sampler=RandomSampler(test_set), num_workers=0)
    logger.info('Loading complete.')


    # 模型的优化器设定、损失函数设定
    # lz警告，学习率的warm up还未添加，此外还没有根据模型可以自动的选择损失函数，这里需要注意
    logger.info(f'Model: \n{str(model)}')
    optimizer = optim.Adam(model.parameters(), lr=config.MODEL.LEARNING_RATE)
    loss_fn = net.loss_fn


    # 模型开始训练
    logger.info('Starting training for {} epoch(s)'.format(config.TRAIN.EPOCH))
    train_and_evaluate(model,
                       train_loader,
                       test_loader,
                       optimizer,
                       loss_fn,
                       config,
                       config.TRAIN.EPOCH_NUM)


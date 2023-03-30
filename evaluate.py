import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm


import model.MLP as net
from dataloader import *
from experiments.configs.Base_Config import get_config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
logger = logging.getLogger('MODEL.Eval')




def update_metrics_LSTM(raw_metrics, loss_fn, pred, labels, x_input_no_normal,labels_no_normal, trend_labels, timestamp,v, predict_start, config, samples=None, relative=False):

    raw_metrics['ND'] = raw_metrics['ND'] + net.accuracy_ND(pred, labels.permute(1, 0, 2), relative=relative)
    raw_metrics['ND_each_prediction'] = raw_metrics['ND_each_prediction'] + net.accuracy_ND_each_predition(pred, labels.permute(1, 0, 2), relative=relative)
    raw_metrics['test_loss'] = raw_metrics['test_loss'] + [loss_fn(pred, labels, trend_labels) * pred.shape[0], pred.shape[0]]

    pred = torch.unsqueeze(v[:,:,0] -v[:,:,1], 1).repeat(1, pred.shape[1], 1) * pred + torch.unsqueeze(v[:,:,1], 1).repeat(1, pred.shape[1], 1)
    raw_metrics['ND_no_normal'] = raw_metrics['ND_no_normal'] + net.accuracy_ND(pred, labels_no_normal.permute(1, 0, 2), relative=relative)
    raw_metrics['RMSE'] = raw_metrics['RMSE'] + net.accuracy_RMSE(pred, labels_no_normal.permute(1, 0, 2), relative=relative)
    raw_metrics['num_each_prediction'] = raw_metrics['num_each_prediction'] + net.NUM_wrong_each_predition(pred,x_input_no_normal,labels_no_normal.permute( 1, 0, 2), timestamp ,config,test=False)

    return raw_metrics

def evaluate(model, loss_fn, test_loader, config):

    model.eval()
    with torch.no_grad():

        if config.MODEL.MODEL_TYPE == 'classification':
            from utils import init_metrics_classification_model as init_metrics
            from utils import update_metrics_classification_model as update_metrics
            from utils import final_metrics_classification_model as final_metrics
        else:
            from utils import init_metrics_prediction_model as init_metrics
            from utils import update_metrics_prediction_model as update_metrics
            from utils import final_metrics_prediction_model as final_metrics
        summary_metric = {}
        raw_metrics = init_metrics(config)

        for i, (test_batch, id_batch, v, labels, x_input_no_normal, labels_no_normal, trend_labels, timestamp, positive_batch) in enumerate(tqdm(test_loader)):
            test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(config.TRAIN.DEVICE)
            id_batch = id_batch.unsqueeze(0).to(config.TRAIN.DEVICE)
            v_batch = v.to(torch.float32).to(config.TRAIN.DEVICE)
            labels = labels.permute(1, 0, 2).to(torch.float32).to(config.TRAIN.DEVICE)
            labels_no_normal = labels_no_normal.permute(1, 0, 2).to(torch.float32).to(config.TRAIN.DEVICE)
            trend_labels = trend_labels.permute(1, 0, 2).to(torch.float32).to(config.TRAIN.DEVICE)
            positive_batch = positive_batch.permute(1, 0, 2).to(torch.float32).to(config.TRAIN.DEVICE)
            pred = model(test_batch, id_batch)


            raw_metrics = update_metrics(raw_metrics,
                                         loss_fn,
                                         pred,
                                         labels,
                                         x_input_no_normal,
                                         labels_no_normal,
                                         trend_labels,
                                         timestamp,
                                         positive_batch,
                                         v_batch[:, config.DATA.COV_DIM:],
                                         config)

            summary_metric = final_metrics(raw_metrics)
    return summary_metric




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='mse_trend_loss.yaml', required=False, metavar="FILE",
                        help='config file name', )
    parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')

    # 将参数读取出来，相当于解析。parse_known_args的好处是如果输入的格式不对，不报错，存在unparsed里面
    args, unparsed = parser.parse_known_args()
    logger.info('try to merge config from {}'.format(args.cfg))
    config = get_config(args)

    # 设置文件夹地址，并检查文件地址是否可达
    model_dir = os.path.join('experiments', config.TRAIN.MODEL_FILE_NAME)
    data_dir = os.path.join(config.DATA.DATASET, config.DATA.SAVE_NAME)
    plot_dir = os.path.join('experiments', config.TRAIN.MODEL_FILE_NAME, config.TRAIN.PLOT_DIR)


    try:
        os.mkdir(plot_dir)
    except FileExistsError:
        pass

    # log标记训练开始位置
    utils.set_logger(os.path.join(model_dir, 'eval.log'))

    # gpu cpu环境检测
    cuda_exist = torch.cuda.is_available()
    if cuda_exist:
        if config.TRAIN.DEVICE == 'cpu':
            logger.info('Using cpu, but can use DEVICE:cuda in yaml file to use Cuda...')
            assert False
        else:
            logger.info('Using Cuda...')

            model = net.Net(config).cuda()
    else:
        if config.TRAIN.DEVICE == 'cpu':
            logger.info('Using cpu...')
        else:
            logger.info('Can not find Cuda, please use DEVICE:cpu in yaml file ...')
            assert False
        model = net.Net(config)


    # 读取数据，得到data loader函数
    logger.info('Loading the datasets...')
    eval_set = EvalDataset(data_dir, config.DATA.SAVE_NAME, config.DATA.NUM_CLASS)
    eval_loader = DataLoader(eval_set, batch_size=config.TRAIN.TEST_BATCH_SIZE, sampler=RandomSampler(eval_set),
                             num_workers=4)
    logger.info('Loading complete.')
    logger.info(f'Model: \n{str(model)}')


    # 指定loss
    loss_type = config.TRAIN.LOSS_TYPE
    if loss_type == 1:
        loss_fn = net.loss_fn
    elif loss_type == 2:
        loss_fn = net.loss_fn_with_weight
    elif loss_type == 3:
        loss_fn = net.loss_fn_with_mse_trend_weight
    else:
        logger.info('loss未指定，程序中止')
        assert False


    # 模型加载
    model_checkpoint_name = config.TRAIN.RESTORE_FILE
    # model_checkpoint_name = 'epoch_79'
    utils.load_checkpoint(os.path.join(model_dir, model_checkpoint_name + '.pth.tar'), model)
    test_metrics, ND_each_prediction, num_each_prediction_1, num_each_prediction_2 = evaluate(model, loss_fn, eval_loader, config,  sample=config.TRAIN.SAMPLING)
    save_path = os.path.join(model_dir, 'metrics_eval_{}.json'.format(config.TRAIN.RESTORE_FILE))
    utils.save_dict_to_json(test_metrics, save_path)



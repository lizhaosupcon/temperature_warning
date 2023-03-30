import os
import yaml
from yacs.config import CfgNode as CN

# 所有参数的默认值。一般不予修改。如果需要修改。请在模型运行时 使用--cfg 指定yaml的参数文件地址

_C = CN()
_C.BASE = ['']


# _C.DATA 记录数据相关的参数
_C.DATA = CN()
# 数据文件夹
_C.DATA.DATASET = 'data'
# 数据集地址：可以在数据集文件夹下，按照需要建立多个训练文件夹
_C.DATA.SAVE_NAME = 'elect'
# 对数据集进行下采样
_C.DATA.DOWN_SAMPLE = True
# 数据集有几类：一床二床算两类
_C.DATA.NUM_CLASS = 3
# 数据自己的特征维度
_C.DATA.COV_DIM = 12
# 一个床输出12个温度，因此输出维度12
_C.DATA.OUTPUT_DIM = 12
# 数据需不需要归一化
_C.DATA.NORMALIZATION = True
_C.DATA.WINDOWS = 180
_C.DATA.PREDICT_START = 120
# 数据分割长度
_C.DATA.PREDICT_STEP = 60


# _C.MODEL 记录model的相关参数
_C.MODEL = CN()
# 模型训练的类型，有3个选项[classification, positive, negative]
_C.MODEL.TYPE = 'classification'
# 采用模型的名称
_C.MODEL.NAME = '中金'
# LSTM的相关参数
_C.MODEL.LSTM_LAYERS = 6
_C.MODEL.LSTM_HIDDEN_DIM = 50
# dropout 学习率 embedding维度
_C.MODEL.DROPOUT = 0.1
_C.MODEL.LEARNING_RATE = 1e-3
_C.MODEL.EMBEDDING_DIM = 20



# 训练时关于数据集合的一些参数
_C.TRAIN = CN()
_C.TRAIN.DEVICE = 'cpu'
_C.TRAIN.SAVE_BEST = False
# 模型的保存地址
_C.TRAIN.MODEL_FILE_NAME = 'base_model'
# 模型的重载名称
_C.TRAIN.RESTORE_FILE = 'best'
# 模型的重载名称，-1就是从头开始训练
_C.TRAIN.EPOCH_NUM = -1
# 模型过程中的画图保存地址
_C.TRAIN.PLOT_DIR = 'figures'


# 训练时关于数据集合的一些参数
_C.TEST = CN()
_C.TEST.DEVICE = 'cpu'
_C.TEST.PLOT_DIR = 'figures'




# 参数合并原子函数。
def _update_config_from_file(config, cfg_file):
    config.defrost()
    cfg_file = os.path.join('experiments', 'configs', cfg_file)
    if os.path.isfile(cfg_file):
        with open(cfg_file, 'r', encoding='utf-8') as f:

            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

        for cfg in yaml_cfg.setdefault('BASE', ['']):
            if cfg:
                _update_config_from_file(
                    config, os.path.join(os.path.dirname(cfg_file), cfg)
                )
        print('=> merge config from {}'.format(cfg_file))
        config.merge_from_file(cfg_file)
    else:
        print('=> No such yaml file at {}, we use base_config'.format(cfg_file))

    config.freeze()

# 更新参数，可以将if等逻辑写进去
def update_config(config, args):
    _update_config_from_file(config, args.cfg)


# 获取参数接口
def get_config(args):
    config = _C.clone()
    update_config(config, args)
    return config

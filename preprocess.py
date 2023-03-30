from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np
import random
from tqdm import tqdm
from tqdm import trange
from experiments.configs.Base_Config import get_config

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from math import sqrt
from pandas import read_csv, DataFrame
from scipy import stats


# 对类名进行编码
def gen_col2series(columns, inputs):
    columns = columns.values
    col2series_dict = {}
    series2id_dict = {}
    # 这个是列号 对应的床号
    series2bed_reactor_dic = {}

    j = 0

    for i, column in enumerate(columns):

        col2series_dict[i] = column

        num_weihao = len(inputs[0])
        for m, input_list in enumerate(inputs):
            if column in input_list:
                series2bed_reactor_dic[i] = m * num_weihao + input_list.index(column)


        if series2id_dict.get(column) is None:
            series2id_dict[column] = j
            j += 1

    save_path = os.path.join(config.DATA.DATASET, config.DATA.SAVE_NAME)
    try:
        os.mkdir(save_path)
    except FileExistsError:
        pass

    np.save(os.path.join(save_path, 'col2series_dict'), col2series_dict)
    np.save(os.path.join(save_path, 'series2id_dict'), series2id_dict)
    np.save(os.path.join(save_path, 'series2bed_reactor_dic'), series2bed_reactor_dic)

    return col2series_dict, series2id_dict, series2bed_reactor_dic


# 打乱顺 训练集：测试集: 验证集= 7：2：1
# 按照业内经验，一般的正负样本比例不会超过8：2

def split_train_test_data(x_input,
                          x_input_no_normal,
                          label,
                          label_no_normal,
                          trend_label,
                          v_input,
                          positive_sample,
                          negative_sample,
                          positive_input,
                          timestamp,
                          num_batch,
                          model_type,  # model_type in ['classification', 'positive', 'negative']
                          train_ratio=0.7,
                          test_ratio=0.2):

    # 首先是对正负样本进行采样
    assert model_type in ['classification', 'positive', 'negative'], \
        print('model_type must be in [\'classification', 'positive', 'negative\'] ')
    num_positive_sample = len(positive_sample)
    num_negative_sample = len(negative_sample)
    if config.DATA.DOWN_SAMPLE and model_type == 'classification':
        if num_positive_sample >= num_negative_sample * 2:
            if num_negative_sample == 0:
                num_negative_sample = 1
            new_positive_sample_list = np.random.choice(positive_sample, num_negative_sample * 2, replace= False)
            new_data_list = np.concatenate([new_positive_sample_list, negative_sample]).astype(np.int32)
            x_input = x_input[new_data_list]
            x_input_no_normal = x_input_no_normal[new_data_list]
            label = label[new_data_list]
            v_input = v_input[new_data_list]
            label_no_normal = label_no_normal[new_data_list]
            trend_label = trend_label[new_data_list]
            positive_input = positive_input[new_data_list]
            timestamp = timestamp[new_data_list]
            print(f'正样本个数为：{num_positive_sample}， 采样后为：{num_negative_sample * 2}')
        else:
            print('无需采样')



    # 数据集的切分
    x_len = x_input.shape[0]
    shuffle_idx = np.random.permutation(x_len)
    train_x_len = int(x_len * train_ratio)
    test_x_len = int(x_len * test_ratio)

    train_shuffle_idx = shuffle_idx[:train_x_len]
    test_shuffle_idx = shuffle_idx[train_x_len: train_x_len+test_x_len]
    eval_shuffle_idx = shuffle_idx[train_x_len+test_x_len:]

    train_x_input = x_input[train_shuffle_idx]
    train_label = label[train_shuffle_idx]
    train_v = v_input[train_shuffle_idx]
    train_label_no_normal = label_no_normal[train_shuffle_idx]
    train_x_input_no_normal = x_input_no_normal[train_shuffle_idx]
    train_trend_label = trend_label[train_shuffle_idx]
    train_positive_label = positive_input[train_shuffle_idx]
    train_timestamp = timestamp[train_shuffle_idx]

    test_x_input = x_input[test_shuffle_idx]
    test_label = label[test_shuffle_idx]
    test_v = v_input[test_shuffle_idx]
    test_label_no_normal = label_no_normal[test_shuffle_idx]
    test_x_input_no_normal = x_input_no_normal[test_shuffle_idx]
    test_trend_label = trend_label[test_shuffle_idx]
    test_positive_label = positive_input[test_shuffle_idx]
    test_timestamp = timestamp[test_shuffle_idx]

    eval_x_input = x_input[eval_shuffle_idx]
    eval_label = label[eval_shuffle_idx]
    eval_v = v_input[eval_shuffle_idx]
    eval_label_no_normal = label_no_normal[eval_shuffle_idx]
    eval_x_input_no_normal = x_input_no_normal[eval_shuffle_idx]
    eval_trend_label = trend_label[eval_shuffle_idx]
    eval_positive_label = positive_input[eval_shuffle_idx]
    eval_timestamp = timestamp[eval_shuffle_idx]

    # 数据集的保存
    save_path = os.path.join(config.DATA.DATASET, config.DATA.SAVE_NAME, model_type, num_batch)

    try:
        os.mkdir(save_path)
    except FileExistsError:
        pass

    prefix = os.path.join(save_path, 'train_')
    np.save(prefix + 'data_' + config.DATA.SAVE_NAME, train_x_input)
    np.save(prefix + 'v_' + config.DATA.SAVE_NAME, train_v)
    np.save(prefix + 'label_' + config.DATA.SAVE_NAME, train_label)
    np.save(prefix + 'label_no_normal_' + config.DATA.SAVE_NAME, train_label_no_normal)
    np.save(prefix + 'x_input_no_normal_' + config.DATA.SAVE_NAME, train_x_input_no_normal)
    np.save(prefix + 'trend_label_' + config.DATA.SAVE_NAME, train_trend_label)
    np.save(prefix + 'positive_label_' + config.DATA.SAVE_NAME, train_positive_label)
    np.save(prefix + 'timestamp_' + config.DATA.SAVE_NAME, train_timestamp)

    prefix = os.path.join(save_path, 'test_')
    np.save(prefix + 'data_' + config.DATA.SAVE_NAME, test_x_input)
    np.save(prefix + 'v_' + config.DATA.SAVE_NAME, test_v)
    np.save(prefix + 'label_' + config.DATA.SAVE_NAME, test_label)
    np.save(prefix + 'label_no_normal_' + config.DATA.SAVE_NAME, test_label_no_normal)
    np.save(prefix + 'x_input_no_normal_' + config.DATA.SAVE_NAME, test_x_input_no_normal)
    np.save(prefix + 'trend_label_' + config.DATA.SAVE_NAME, test_trend_label)
    np.save(prefix + 'positive_label_' + config.DATA.SAVE_NAME, test_positive_label)
    np.save(prefix + 'timestamp_' + config.DATA.SAVE_NAME, test_timestamp)


    prefix = os.path.join(save_path, 'eval_')
    np.save(prefix + 'data_' + config.DATA.SAVE_NAME, eval_x_input)
    np.save(prefix + 'v_' + config.DATA.SAVE_NAME, eval_v)
    np.save(prefix + 'label_' + config.DATA.SAVE_NAME, eval_label)
    np.save(prefix + 'label_no_normal_' + config.DATA.SAVE_NAME, eval_label_no_normal)
    np.save(prefix + 'x_input_no_normal_' + config.DATA.SAVE_NAME, eval_x_input_no_normal)
    np.save(prefix + 'trend_label_' + config.DATA.SAVE_NAME, eval_trend_label)
    np.save(prefix + 'positive_label_' + config.DATA.SAVE_NAME, eval_positive_label)
    np.save(prefix + 'timestamp_' + config.DATA.SAVE_NAME, eval_timestamp)



def prep_data(data, index, cfg, NUM_CLASS):
    # 一个有num class个矿床个数。因此windows 要设计好总的个数
    window_size = cfg.DATA.WINDOWS
    input_size = cfg.DATA.PREDICT_START
    total_time = len(data)
    windows_each_series = np.full(NUM_CLASS, (total_time - window_size) // cfg.DATA.PREDICT_STEP)
    windows_per_series = np.full(NUM_CLASS, (total_time - window_size)*cfg.DATA.OUTPUT_DIM // cfg.DATA.PREDICT_STEP)
    total_windows = np.sum(windows_per_series)
    positive_sample_list = []
    negative_sample_list = []

    # 初始化每一个windows x_input :输入特征， label：对应的预测的数据 v_input:对应归一化的存储的数据。
    # xi = 特征值+床号编码
    x_input = np.zeros((total_windows, input_size, cfg.DATA.COV_DIM + 1 + 1), dtype='float32')
    x_input_no_normal = np.zeros((total_windows, input_size, cfg.DATA.COV_DIM + 1 + 1), dtype='float32')
    label = np.zeros((total_windows, window_size-input_size, 1), dtype='float32')
    label_no_normal = np.zeros((total_windows, window_size-input_size, 1), dtype='float32')
    v_input = np.zeros((total_windows, cfg.DATA.COV_DIM + 1, 3), dtype='float32')
    positive_input = np.zeros(total_windows, dtype='float32')

    # 趋势部分的label
    trend_label = np.zeros((total_windows, window_size - input_size, 1), dtype='float32')

    # 时间戳
    timestamp = np.full(total_windows, 'FAR21:TI610417L 2020-00-00T00:00:00')

    # 开始预处理数据。
    count = 0
    for series in trange(NUM_CLASS):
        for i in range(windows_each_series[series]):
            # 每次取一段窗口数据。并进行处理
            window_start = cfg.DATA.PREDICT_STEP * i
            window_end = window_start + window_size
            x_input_end = window_start + input_size


            # 确定窗口位置
            label_start = NUM_CLASS * cfg.DATA.COV_DIM + series * cfg.DATA.OUTPUT_DIM
            label_end = NUM_CLASS * cfg.DATA.COV_DIM + (series + 1) * cfg.DATA.OUTPUT_DIM

            # 10s采样成1分钟，还有0值，代表至少五个点为空，此时该条数据就可以丢掉了。
            if (data[window_start:window_end, series * cfg.DATA.COV_DIM: (series + 1) * cfg.DATA.COV_DIM] == 0).sum() >= 1\
                    and ((data[window_start:window_end, label_start: label_end]) == 0).sum() >= 1:

                continue


            # 对窗口内出局进行处理
            for output_id in range(cfg.DATA.OUTPUT_DIM):


                x_input[count, :, 0:cfg.DATA.COV_DIM] = data[window_start:x_input_end, series * cfg.DATA.COV_DIM: (series + 1) * cfg.DATA.COV_DIM]
                x_input[count, 0: input_size, -2] = data[window_start + 1: x_input_end + 1, label_start + output_id]
                x_input[count, :, -1] = series2bed_reactor[label_start + output_id]
                x_input_no_normal[count, :, :] = x_input[count, :, :]

                # label的延后处理
                label[count, :-1, 0] = data[x_input_end + 1:window_end, label_start+output_id]
                label_no_normal[count, :-1, 0] = data[x_input_end + 1:window_end, label_start+output_id]
                if window_end < len(data):
                    label[count, -1, 0] = data[window_end, label_start+output_id]
                    label_no_normal[count, -1, 0] = data[window_end, label_start+output_id]
                else:
                    label[count, -1, 0] = data[-1, label_start+output_id]
                    label_no_normal[count, -1, 0] = data[-1, label_start+output_id]

                # 这个就是增速
                trend_label[count, 0] = label[count, 0] - x_input[count, -1, -2]
                trend_label[count, 1:] = label[count, 1:] - label[count, 0:-1]
                v_input[count, :, 2] = x_input[count, -1, :-1]

                # 时间戳
                timestamp[count] = col2series[label_start + output_id] + '_' + index[window_start]

                # 正负样本区分
                window_label_max = np.max(data[window_start+1:window_end + 1, label_start+output_id])
                window_label_min = np.min(data[window_start+1:window_end + 1, label_start+output_id])


                if (window_label_max - window_label_min) >= 3:
                    negative_sample_list.append(count)
                else:
                    positive_sample_list.append(count)
                    positive_input[count] = 1

                # 是否需要归一化.最大最小均值法
                if cfg.DATA.NORMALIZATION:
                    # input的归一化部分
                    x_input_max = np.max(x_input[count, 0:input_size, :-1], axis=0)
                    x_input_min = np.min(x_input[count, 0:input_size, :-1], axis=0)
                    not_equal_index = (x_input_min != x_input_max)
                    if np.any(not_equal_index == False):
                        for k, v in enumerate(not_equal_index):
                            if v:
                                x_input[count, :, k] = (x_input[count, :, k] - x_input_min[k]) / (x_input_max[k] - x_input_min[k])
                            else:
                                x_input[count, :, k] = x_input[count, :, k] - x_input_min[k]
                    else:
                        x_input[count, :, :-1] = (x_input[count, :, :-1] - x_input_min) / (x_input_max - x_input_min)
                    v_input[count, :, 0] = x_input_max
                    v_input[count, :, 1] = x_input_min



                    # label的归一化
                    label_max = np.max(x_input_no_normal[count, 0:input_size, -2])
                    label_min = np.min(x_input_no_normal[count, 0:input_size, -2])
                    not_equal_index = (label_max != label_min)
                    if np.any(not_equal_index == False):
                        for k, v in enumerate(not_equal_index):
                            if v:
                                label[count, :, k] = (label[count, :, k] - label_min[k]) / (label_max[k] - label_min[k])
                            else:
                                label[count, :, k] = label[count, :, k] - label_min[k]
                    else:
                        label[count, :, :] = (label[count, :, :] - label_min) / (label_max - label_min)
                    v_input[count, cfg.DATA.COV_DIM:, 0] = label_max
                    v_input[count, cfg.DATA.COV_DIM:, 1] = label_min

                    # 归一化之后的增速

                    trend_label[count, 0] = label[count, 0] - x_input[count, -1, -2]
                    trend_label[count, 1:] = label[count, 1:] - label[count, 0:-1]

                count += 1



    # 最大最小均值法 容易出现除以0。 单独考虑除以0的情况
    x_input[np.isnan(x_input)] = 0
    label[np.isnan(label)] = 0
    trend_label[np.isnan(trend_label)] = 0


    return x_input[count], x_input_no_normal[count], label[count], label_no_normal[count], trend_label[count], \
        v_input[count], positive_sample_list, negative_sample_list, positive_input[count], timestamp[count]




if __name__ == '__main__':

    import argparse

    # 首先获取所需要的参数
    # 创建ArgumentParser 对象。接受参数对象
    parser = argparse.ArgumentParser('pre data', add_help=False)
    parser.add_argument('--cfg', type=str, default='mse_weight_loss_10_test.yaml', required=False, metavar="FILE", help='path to config file', )
    args, unparsed = parser.parse_known_args()
    config = get_config(args)



    # 整理所需要的数据：获取特征、采样
    # lz警告，临时写法，后续记得修改
    input = [
        # 新增进料压力，进料温度，混合原料，循环氢,二段进料加热炉,出口压力
        ['FAR21:TIC610416A', 'FAR21:TI610416B', 'FAR21:TI610416C', 'FAR21:FI610406', 'FAR21:TIC610416ASP',
                             'FAR21:PI610403', 'FAR21:TI610405', 'FAR21:TI610406', 'FAR21:FIC610706', 'FAR21:FIC610701', 'FAR21:TI610801', 'FAR21:PI610404'],  # 一床入口
        ['FAR21:TIC610418A', 'FAR21:TI610418B', 'FAR21:TI610418C', 'FAR21:FI610407', 'FAR21:TIC610418ASP',
                             'FAR21:PI610403', 'FAR21:TI610405', 'FAR21:TI610406', 'FAR21:FIC610706', 'FAR21:FIC610701', 'FAR21:TI610801', 'FAR21:PI610404'],  # 二床入口
        ['FAR21:TIC610420A', 'FAR21:TI610420B', 'FAR21:TI610420C', 'FAR21:FI610408', 'FAR21:TIC610420ASP',
                             'FAR21:PI610403', 'FAR21:TI610405', 'FAR21:TI610406', 'FAR21:FIC610706', 'FAR21:FIC610701', 'FAR21:TI610801', 'FAR21:PI610404']  # 三床入口
    ]
    output = [
        ['FAR21:TI610417A', 'FAR21:TI610417B', 'FAR21:TI610417C', 'FAR21:TI610417D', 'FAR21:TI610417E',
         'FAR21:TI610417F', 'FAR21:TI610417G', 'FAR21:TI610417H', 'FAR21:TI610417I', 'FAR21:TI610417J',
         'FAR21:TI610417K', 'FAR21:TI610417L'],  # 一床出口

        ['FAR21:TI610419A', 'FAR21:TI610419B', 'FAR21:TI610419C', 'FAR21:TI610419D', 'FAR21:TI610419E',
         'FAR21:TI610419F', 'FAR21:TI610419G', 'FAR21:TI610419H', 'FAR21:TI610419I', 'FAR21:TI610419J',
         'FAR21:TI610419K', 'FAR21:TI610419L'],  # 二床出口

        ['FAR21:TI610421A', 'FAR21:TI610421B', 'FAR21:TI610421C', 'FAR21:TI610421D', 'FAR21:TI610421E',
         'FAR21:TI610421F', 'FAR21:TI610421G', 'FAR21:TI610421H', 'FAR21:TI610421I', 'FAR21:TI610421J',
         'FAR21:TI610421K', 'FAR21:TI610421L']  # 三床出口
    ]

    # 获得需要的数据
    input_len = len(input)
    output_len = len(output)
    input_str_list = [col for cols in input for col in cols]
    output_str_list = [col for cols in output for col in cols]
    Model_Type = ['classification', 'positive', 'negative']

    # 读取到所有的数据并进行拼接，
    # lz警告，先跑个100天的再说
    csv_dir = r'D:\zhongjinshihua\data\2019-2022-RawData'
    file_csv_list = []
    for m, (root, dirs, files) in enumerate(os.walk(csv_dir)):
        for n, file in enumerate(files):
            if os.path.splitext(file)[1] == '.csv' and '2022年07月30' >= os.path.splitext(file)[0] >= '2021年03月01':
                # os.path.splitext("abc.csv")[1]='.csv'
                file_csv_list.append(file)

    num_batch = 0
    batch_size = 250
    # 从这里开始分批处理。每次处理300个csv文件
    print('接下来分批处理处理数据')
    for tmp_num_batch, tmp_num_id in enumerate(tqdm(range(0, len(file_csv_list), batch_size))):
        file_csv_list_new = file_csv_list[tmp_num_batch*batch_size: (tmp_num_batch+1)*batch_size]
        data_frame = pd.read_csv(os.path.join(csv_dir, file_csv_list_new[0]), index_col=0, encoding='GBK', parse_dates=True)
        data_frame = data_frame[input_str_list + output_str_list]
        print(f'第{tmp_num_batch+1}批数据读取中')
        for i, csv in enumerate(tqdm(file_csv_list_new[1:])):
            csv_data = pd.read_csv(os.path.join(csv_dir, csv), index_col=0, encoding='GBK', parse_dates=True)
            data_frame = pd.concat([data_frame, csv_data[input_str_list + output_str_list]])
        print(f'第{tmp_num_batch + 1}批数据读取完毕')
        # 保存数据的列名,
        col2series, series2id, series2bed_reactor = gen_col2series(data_frame.columns, output)
        # 重新采,因为涉及采样，采样过程忽略nan但是不会忽略0值，因此将0值标记为nan，然后采样，再将nan值转成0值
        # 这样采就不会影响到数据了
        data_frame.replace(0, float('nan'), inplace=True)
        data_frame = data_frame.resample('1min', label='left', closed='right').mean()
        data_frame.fillna(0, inplace=True)
        data_frame_index = np.datetime_as_string(data_frame.index.values, unit='s')

        # 对数据进行处理，并进行切分
        print(f'第{tmp_num_batch + 1}批数据处理中')
        x_input, x_input_no_normal, label, label_no_normal, trend_label, v_input, positive_sample_list, \
            negative_sample_list, positive_input, timestamp = \
            prep_data(data_frame.values, data_frame_index, config, input_len)

        # 切分得到分类部分的数据
        for model_type in Model_Type:

            split_train_test_data(x_input,
                                  x_input_no_normal,
                                  label,
                                  label_no_normal,
                                  trend_label,
                                  v_input,
                                  positive_sample_list,
                                  negative_sample_list,
                                  positive_input,
                                  timestamp,
                                  str(num_batch),
                                  model_type,  # model_type in ['classification', 'positive', 'negative']
                                  train_ratio=0.7,
                                  test_ratio=0.2)

        num_batch = tmp_num_batch

        # 清空数据
        del x_input
        del x_input_no_normal
        del label
        del label_no_normal
        del trend_label
        del v_input
        del positive_sample_list
        del negative_sample_list
        del timestamp


    print(f'开始合并数据')
    classes_npy = ['data_', 'v_', 'label_', 'label_no_normal_', 'x_input_no_normal_', 'trend_label_', 'positive_label_','timestamp_']
    classes_data = ['train_', 'test_', 'eval_']
    for model_type in Model_Type:
        save_path = os.path.join(config.DATA.DATASET, config.DATA.SAVE_NAME, model_type)
        for class_data in tqdm(classes_data):
            for class_npy in classes_npy:
                arrays = []
                for num in range(num_batch):
                    prefix = os.path.join(save_path, str(num), class_data)
                    arrays.append(np.load(os.path.join(prefix + class_npy + config.DATA.SAVE_NAME + '.npy')))
                combined_array = np.concatenate(arrays, axis=0)
                np.save(os.path.join(save_path, class_data + class_npy + config.DATA.SAVE_NAME), combined_array)
    print(f'数据合并完毕')


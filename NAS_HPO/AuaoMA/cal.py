
import numpy as np
import os
from AuaoMA.f1 import *
from AuaoMA.utils import *

bf_search_min = 0
bf_search_max = 1000
bf_search_step_size = 1

def cal_f1(test_score, label):

    test_score = np.sum(test_score, axis=1)
    t, th, predict = bf_search(test_score, label,
                                start=bf_search_min,
                                end=bf_search_max,
                                step_num=int((bf_search_max-bf_search_min) // bf_search_step_size),
                                display_freq=1000,
                                calc_latency=True)
    res_point_list = {
        'machine_id': [],
        'tp': [],
        'fp': [],
        'fn': [],
        'p': [],
        'r': [],
        'f1': [],
        'threshold': []
    }

    label_item = label
    predict_item = predict.astype(int)

    res_point_list['tp'].append(np.sum(((label_item == 1) & (predict_item == 1)).astype(int)))
    res_point_list['fp'].append(np.sum(((label_item == 0) & (predict_item == 1)).astype(int)))
    res_point_list['fn'].append(np.sum(((label_item == 1) & (predict_item == 0)).astype(int)))

    p = round(res_point_list['tp'][-1] / (res_point_list['tp'][-1] + res_point_list['fp'][-1] + 1e-9), 4)
    r = round(res_point_list['tp'][-1] / (res_point_list['tp'][-1] + res_point_list['fn'][-1] + 1e-9), 4)
    f1 = round(2 * p * r / (p + r + 1e-9), 4)

    f1_formatted = f"{f1:.4f}"  
    return f1_formatted


def cal_AuaoMA(dataset2, test_score, test_score_replace_ano, cycle):
    th_high = 95
    th_low = 5
    vae_coeff = 1.5

    if dataset2 == 'ASD':
        cycle_metric = [10, 11, 12, 13, 14, 15, 16, 17, 18]
    elif dataset2 == 'SMD':
        cycle_metric = [0, 2, 6, 8, 12, 14, 18, 19, 20, 21, 23, 24, 27, 30, 31, 34, 35]

    data1 = DataProcess(test_score.transpose(), cycle, th_high, th_low)
    data2 = DataProcess(test_score_replace_ano.transpose(), cycle, th_high, th_low)

    for data in [data1, data2]:
        data.vae_arg(th_high, th_low, vae_coeff)

    sum1 = get_sum_2(data1.arg_data, cycle_metric)
    sum1 = data1.normalization(sum1, th_high, th_low)
    l1 = data1.get_sum_var(sum1)

    sum2 = get_sum_2(data2.arg_data, cycle_metric)
    sum2 = data2.normalization(sum2, th_high, th_low)
    l2 = data2.get_sum_var(sum2)

    return l1 -l2

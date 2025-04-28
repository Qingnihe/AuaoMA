import numpy as np
from matplotlib import pyplot as plt
import os

def get_sum(data, start=0, end=None):
    if end is not None:
        data = data[start:end]
    else:
        data = data[start:]
    return np.sum(data, 0)

def get_sum_2(data, index_list):
    result = []
    for index in index_list:
        result.append(data[index])
    return np.array(result)

class DataProcess:
    def __init__(self, data, cycle, high, low):
        self.data = data
        self.cycle = cycle
        self.metrics_num, self.time_len = self.data.shape
        self.pad = 0 if self.time_len % self.cycle == 0 else self.cycle - self.time_len % self.cycle
        self.data = self.normalization(self.data, high, low)
        self.mean_data_no_extrem = self.get_mean_data_wo_anomaly(self.data, high, low)
        self.arg_data = []

    @staticmethod
    def del_extreme_value(data, high, low):
        high = np.percentile(data, high)
        low = np.percentile(data, low)
        data = np.where(data > high, high, data)
        data = np.where(data < low, low, data)
        return data


    def normalization(self,data, high, low):
        one_flag = False
        if len(data.shape) == 1:
            data = np.reshape(data, (1, -1))
            one_flag = True
        mean_data_no_extrem = self.get_mean_data_wo_anomaly(data, high, low)

        min_ = np.min(mean_data_no_extrem, axis=1)
        min_ = np.tile(min_, (data.shape[1], 1)).T
        max_ = np.max(mean_data_no_extrem, axis=1)
        max_ = np.tile(max_, (data.shape[1], 1)).T
        data = (data - min_) / (max_ - min_) + 1
        if one_flag:
            data = np.squeeze(data)
        return data

    def get_mean_data_wo_anomaly(self, data, high, low):
        mean_data_no_extrem = []
        for i_data in data:
            new_i_data = np.pad(i_data, (0, self.pad), mode='constant',
                                constant_values=-100)

            new_i_data = np.reshape(new_i_data, (-1, self.cycle))
            new_i_data = np.transpose(new_i_data)
            mean_i_data = []
            for index_j, j in enumerate(new_i_data):
                j = j[j != -100]
                mean_i_data.append(np.mean(self.del_extreme_value(j, high, low)))
            mean_i_data = np.tile(mean_i_data, (self.time_len // self.cycle) + 1)[:self.time_len]
            mean_data_no_extrem.append(mean_i_data)
        return np.asarray(mean_data_no_extrem)

    def vae_arg(self, th_high, th_low, arg_coefficient):
        for d, m_d in zip(self.data, self.mean_data_no_extrem):
            i_arg_data = []
            diff = np.abs(d - m_d)
            diff_big = diff[diff > 0]
            if len(diff_big) == 0:
                th = 0
            else:
                diff_big = self.del_extreme_value(diff_big, th_high, th_low)
                diff_mean = np.mean(diff_big)
                th = diff_mean * 3
            for i in range(len(d)):
                coefficient = 1
                if d[i] - m_d[i] > th:
                    coefficient *= arg_coefficient
                i_arg_data.append(m_d[i] * coefficient)

            self.arg_data.append(i_arg_data)
        self.arg_data = np.asarray(self.arg_data)

    # def ae_arg(self, th_high, th_low, arg_coefficient):
    #     for d, m_d in zip(self.data, self.mean_data_no_extrem):
    #         i_arg_data = []
    #         diff = np.abs(d - m_d)
    #         diff = self.del_extreme_value(diff, th_high, th_low)
    #         diff_mean = np.mean(np.abs(diff))
    #         th = diff_mean * 8
    #         for i in range(len(d)):
    #             coefficient = 1
    #             if d[i] - m_d[i] > th:
    #                 if m_d[i] >= 0:
    #                     coefficient *= arg_coefficient
    #                 else:
    #                     coefficient /= arg_coefficient
    #             elif d[i] - m_d[i] < -th:
    #                 if m_d[i] >= 0:
    #                     coefficient /= arg_coefficient
    #                 else:
    #                     coefficient *= arg_coefficient
    #             i_arg_data.append(m_d[i] * coefficient)
    #         self.arg_data.append(i_arg_data)
    #     self.arg_data = np.asarray(self.arg_data)

    def get_m_var(self, data):
        variances = 0
        for i_data in data:
            new_i_data = np.pad(i_data, (0, self.pad), mode='constant',
                                constant_values=-100)
            new_i_data = np.reshape(new_i_data, (-1, self.cycle))
            new_i_data = np.transpose(new_i_data)
            for index_j, j in enumerate(new_i_data):
                j = j[j != -100]  
                variances += np.var(j)
        return variances

    def get_sum_var(self, data):
        variances = 0
        data = np.pad(data, (0, self.pad), mode='constant',
                            constant_values=-100)
        data = np.reshape(data, (-1, self.cycle))
        data = np.transpose(data)
        for index_j, j in enumerate(data):
            j = j[j != -100]  
            variances += np.var(j)
        return variances


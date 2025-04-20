import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MultivariateDataGenerator:
    def __init__(self, data):
        self.data = np.array(data)
        self.dim = len(self.data)
        self.STREAM_LENGTH = len(self.data[0])
        self.label = np.zeros(self.STREAM_LENGTH, dtype=int)
        self.data_origin = self.data.copy()

    def point_global_outliers(self, dim_nos, ratio, factor, radius):
        num_outliers = round(self.STREAM_LENGTH * ratio)
        
        position = np.random.randint(0, self.STREAM_LENGTH - radius, num_outliers)  
        for i in position:
            for dim_no in dim_nos:
                maximum, minimum = max(self.data[dim_no]), min(self.data[dim_no])

                for j in range(i, min(i + radius, self.STREAM_LENGTH)):
                    local_std = self.data_origin[dim_no][int(max(0, j - radius)):int(min(j + radius, self.STREAM_LENGTH))].std()
                    self.data[dim_no][j] = self.data_origin[dim_no][j] * factor * local_std
                    if 0 <= self.data[dim_no][j] < maximum:
                        self.data[dim_no][j] = maximum + np.random.uniform(0.2, 0.4)
                    self.label[j] = 1

    def point_contextual_outliers(self, dim_nos, ratio, factor, radius):
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio)) * self.STREAM_LENGTH).astype(int)

        for i in position:
            for dim_no in dim_nos:
                maximum, minimum = max(self.data[dim_no]), min(self.data[dim_no])
                local_std = self.data_origin[dim_no][int(max(0, i - radius)):int(min(i + radius, self.STREAM_LENGTH))].std()
                self.data[dim_no][i] = self.data_origin[dim_no][i] * factor * local_std
                if self.data[dim_no][i] > maximum: 
                    self.data[dim_no][i] = maximum * min(0.5, abs(np.random.normal(0, 1)))
                self.label[i] = 1

if __name__ == '__main__':
    base_path = 'ASD'
    for i in range(1, 13):
        data_path = os.path.join(base_path, str(i), f'test_replace.npy')
        data = np.load(data_path)
        data = data.T
        multivariate_data = MultivariateDataGenerator(data)
        global_outliers_dims = [1, 3, 5, 7, 9, 11, 15, 17]
        contextual_outliers_dims = [0, 2, 4, 12, 18]

        multivariate_data.point_global_outliers(dim_nos=global_outliers_dims, ratio=0.005, factor=3, radius=1)
        
        columns = [f'col_{i}' for i in range(multivariate_data.dim)]
        df = pd.DataFrame(multivariate_data.data.T, columns=columns)
        
        np.save(os.path.join(base_path, str(i), 'test.npy'), multivariate_data.data.T)
        np.save(os.path.join(base_path, str(i), 'label.npy'), multivariate_data.label)

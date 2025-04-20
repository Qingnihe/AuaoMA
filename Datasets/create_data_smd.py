import numpy as np
import os

base_path = 'server-machine-dataset/'
s_path = 'SMD'

def del_extreme_value(data, high, low):
    threshold_high = np.percentile(data, high)
    threshold_low = np.percentile(data, low)
    data = np.clip(data, threshold_low, threshold_high) 
    return data

for folder in range(1, 29): 
    folder_path1 = os.path.join(base_path, str(folder))
    folder_path2 = os.path.join(s_path, str(folder))
    train_file = os.path.join(folder_path1, f'train.npy')
    train_data = np.load(train_file)
    os.makedirs(folder_path2, exist_ok=True)  
    save_path1 = os.path.join(folder_path2, 'test_replace.npy')
    save_path2 = os.path.join(folder_path2, 'train.npy')
    train_data = del_extreme_value(train_data, high=95, low=5)
    period = 1440 
    num_periods = train_data.shape[0] // period  
    train_data_reshaped = train_data[:num_periods * period].reshape(num_periods, period, -1)
    mean_data = train_data_reshaped.mean(axis=0)
    num_simulated_periods = 6  
    simulated_data = np.tile(mean_data, (num_simulated_periods, 1))
    np.save(save_path2, train_data)  
    np.save(save_path1, simulated_data)  


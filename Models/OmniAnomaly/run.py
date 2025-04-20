import os
import numpy as np
import torch
from omnianomaly.nni_model import OmniAnomaly
from get_eval_result import *
import requests
import time, random
from data_config import *
import warnings
warnings.filterwarnings('ignore')

class Config:
    x_dims = ''
    z_dims = global_z_dim
    max_epochs = global_epochs
    batch_size = global_batch_size
    window_size = global_window_size

    exp_dir = exp_dir
    save_dir = exp_dir/ 'model'
    result_dir = exp_dir/'result'


def end_to_end(i, config, train_data_item, test_data_item):

    feature_dim = train_data_item.shape[1]
    model = OmniAnomaly(x_dims=feature_dim,
                z_dims=config.z_dims,
                max_epochs=config.max_epochs,
                batch_size=config.batch_size,
                window_size=config.window_size,
                learning_rate=global_learning_rate,
                model_dim=global_model_dim)

    (config.exp_dir).mkdir(parents=True, exist_ok=True)
    (config.save_dir).mkdir(parents=True, exist_ok=True)
    (config.result_dir).mkdir(parents=True, exist_ok=True)

    x_train_list = []
    train_id = i

    x_train, x_test = preprocess_meanstd(train_data_item, test_data_item)
    x_train_list.append(x_train)

    save_path = config.save_dir/ 'model.pkl'

    log_path = save_path.parent / 'time.txt'
    if log_path.exists():
        print(f"{log_path} exists.")
        return

    total_time = 0
    start_time = time.time()
    model.fit(x_train_list, save_path, valid_portion=0.05)
 
    model.restore(save_path)
    score, recon_mean, recon_std, z = model.predict(x_test, save_path)

    end_time = time.time()
    total_time = end_time - start_time
    log_path = save_path.parent / 'time.txt'
    with open(log_path, 'a') as f:
        f.write(str(total_time))
  
    if score is not None:
        np.save(config.result_dir/'test_score.npy', -score)
        np.save(config.result_dir/'recon_mean.npy', recon_mean)
        np.save(config.result_dir/'recon_std.npy', recon_std)

def torch_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def by_entity():
    for i,(tr,te,lab) in enumerate(zip(train_data,test_data,label)):
        torch_seed()
        if i+1 not in chosed_index:
            continue
        tr=np.array(tr)
        te=np.array(te)
        end_to_end(i, config, tr.T, te.T)

    get_exp_result()

if __name__ == '__main__':
    config = Config()
    print(config.exp_dir)
    by_entity()


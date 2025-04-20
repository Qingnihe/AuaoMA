import os
import numpy as np
import torch
from sdfvae.nni_model import SDFVAE
import time, random
from get_eval_result import *
from data_config import *
import requests


class Config:
    exp_dir = exp_dir
    save_dir = exp_dir/ 'model'
    result_dir = exp_dir/'result'

def end_to_end(i, config: Config, train_data_item, test_data_item):
    total_train_time = 0
    feature_dim = train_data_item.shape[1]
    model = SDFVAE(
        s_dim=global_s_dim, d_dim=global_d_dim, conv_dim=global_conv_dim, 
        hidden_dim=global_hidden_dim, T=global_T, w=global_window_size, 
        n=feature_dim, enc_dec='CNN', nonlinearity=None, loss_fn=global_loss_fn)

    x_train_list = []
    train_id = i
    x_train, x_test = preprocess(train_data_item, test_data_item)
    x_train_list.append(x_train)

    (config.exp_dir).mkdir(parents=True, exist_ok=True)
    (config.save_dir).mkdir(parents=True, exist_ok=True)
    (config.result_dir).mkdir(parents=True, exist_ok=True)

    save_path = config.save_dir/ 'model.pkl'
    log_path = save_path.parent / 'time.txt'
    if log_path.exists():
        print(f"{log_path} exists.")
        return

    total_time = 0
    start_time = time.time()

    model.fit(x_train_list, save_path, max_epoch=global_epochs, cluster_id=i)
    
    model.restore(save_path)
    score, recon_mean, recon_std = model.predict(x_test)

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

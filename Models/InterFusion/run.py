import os
import numpy as np
import torch
from interfusion.nni_model import InterFusion
from get_eval_result import *
import requests
import time, random
from data_config import *
import re
import warnings
warnings.filterwarnings('ignore')
class Config:
    z1_dims = global_z1_dim
    z2_dims = global_z2_dim
    train_max_epochs = global_train_epochs
    pretrain_max_epochs = global_pretrain_epochs
    batch_size = global_batch_size
    window_size = global_window_size
    exp_dir = exp_dir
    save_dir = exp_dir/ 'model'
    result_dir = exp_dir/'result'


def end_to_end(i, config, train_data_item, test_data_item):
    feature_dim = train_data_item.shape[1]
    pretrain_model = InterFusion(x_dims=feature_dim,
                z1_dims=config.z1_dims,
                z2_dims=config.z2_dims,
                max_epochs=config.train_max_epochs,
                pre_max_epochs=config.pretrain_max_epochs,
                batch_size=config.batch_size,
                window_size=config.window_size,
                learning_rate=pretrain_lr,
                output_padding_list=output_padding_list,
                model_dim=global_rnn_dims)
    x_train_list = []
    x_test_list = []
    train_id = i

    x_train, x_test = preprocess_meanstd(train_data_item, test_data_item)
    x_train_list.append(x_train)
    x_test_list.append(x_test)

    (config.exp_dir).mkdir(parents=True, exist_ok=True)
    (config.save_dir).mkdir(parents=True, exist_ok=True)
    (config.result_dir).mkdir(parents=True, exist_ok=True)

    pre_save_path = config.save_dir/ 'pre_model.pkl'
    save_path = config.save_dir/ 'model.pkl'

    total_time = 0
    start_time = time.time()

    pretrain_model.prefit(x_train_list, pre_save_path, valid_portion=0.01)
    model = InterFusion(x_dims=feature_dim,
        z1_dims=config.z1_dims,
        z2_dims=config.z2_dims,
        max_epochs=config.train_max_epochs,
        pre_max_epochs=config.pretrain_max_epochs,
        batch_size=config.batch_size,
        window_size=config.window_size,
        learning_rate=train_lr,
        output_padding_list=output_padding_list,
        model_dim=global_rnn_dims)
    model.restore(pre_save_path)
    model.fit(x_train_list, x_test_list, save_path, valid_portion=0.01)

    model.restore(save_path)
    score, recon_mean, recon_std, z = model.predict(x_test, save_path, if_pretrain=False)

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


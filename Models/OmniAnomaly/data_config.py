import pathlib
import json
import numpy as np
import os
import argparse
import torch
from sklearn.preprocessing import MinMaxScaler
from nni.utils import merge_parameter
import nni
import logging


def preprocess_meanstd(df_train, df_test):
    df_train = np.asarray(df_train, dtype=np.float32)

    if len(df_train.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df_train)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df_train = np.nan_to_num(df_train)
    
    k = 5
    e = 1e-3
    mean_array = np.mean(df_train, axis=0, keepdims=True)
    std_array = np.std(df_train, axis=0, keepdims=True)
    std_array[np.where(std_array==0)] = e
    df_train = np.where(df_train > mean_array + k * std_array, mean_array + k * std_array, df_train)
    df_train = np.where(df_train < mean_array - k * std_array, mean_array - k * std_array, df_train)
    
    train_mean_array = np.mean(df_train, axis=0, keepdims=True)
    train_std_array = np.std(df_train, axis=0, keepdims=True)
    train_std_array[np.where(train_std_array==0)] = e
    
    df_train_new = (df_train - train_mean_array) / train_std_array
    
    df_test = np.where(df_test > train_mean_array + k * train_std_array, train_mean_array + k * train_std_array, df_test)
    df_test = np.where(df_test < train_mean_array - k * train_std_array, train_mean_array - k * train_std_array, df_test)
    df_test_new = (df_test - train_mean_array) / train_std_array

    return df_train_new, df_test_new

def preprocess_minmax(df_train, df_test):
    scaler = MinMaxScaler().fit(df_train)
    train = scaler.transform(df_train)
    test = scaler.transform(df_test)
    test = np.clip(test, a_min=-3.0, a_max=3.0)
    return train, test

def load_data_from_json(s):
    json_file = json.load(open(s))
    data = json_file['data']
    label = json_file['label']
    return data,label

def global_ELBO_loss(self, x, z, z_flowed, q_zx, p_z, p_xz, flow_log_det: torch.Tensor, test_id=None):

    log_p_xz = torch.sum(p_xz.log_prob(x), dim=-1)

    log_q_zx = torch.sum(q_zx.log_prob(z), dim=-1) - flow_log_det.sum(dim=-1)
    log_p_z = p_z.log_prob(z_flowed)
    return -torch.mean(log_p_xz+log_p_z-log_q_zx), -torch.mean(log_p_xz)


def get_params():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset_type', type=str, default='application-server-dataset')
    parser.add_argument('--out_dir', type=str, default='hpo_result')
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--base_model_dir', type=str, default="test")

    parser.add_argument('--z_dim', type=int, default=3)
    parser.add_argument('--model_dim', type=int, default=500)
    parser.add_argument('--window_size', type=int, default=60)

    parser.add_argument('--epochs', type=int, default= 10) 
    parser.add_argument('--index_weight', type=int, default= 10) 
    parser.add_argument('--lr', type=float, default= 1e-3)
    parser.add_argument('--seed', type=int, default=2020)  
    parser.add_argument('--train_type', type=str, default='noshare')
    parser.add_argument('--valid_epoch', type=int, default=5) 

    parser.add_argument('--min_std', type=float, default=0.01) 
    parser.add_argument('--entity', type=int, default=1)

    return parser.parse_known_args()[0]

logger = logging.getLogger('omnianomaly_AutoML')

project_path = pathlib.Path(os.path.abspath(__file__)).parent

if 'args' not in globals():
    tuner_params= nni.get_next_parameter()
    logger.debug("tuner_params:", tuner_params)
    # args = vars(merge_parameter(get_params(), tuner_params))
    args = vars(get_params())
    logger.debug("updated args:", args)


out_dir =  "out/" + args['out_dir']
GPU_index = str(args['gpu_id'])
global_device = torch.device(f'cuda:{GPU_index}')

dataset_type = args['dataset_type']
global_epochs = args['epochs']
seed = args['seed']

global_z_dim= args['z_dim']
global_model_dim = args['model_dim']
global_batch_size= args['batch_size']
global_learning_rate = args['lr']

train_type = args['train_type']

if_freeze_seq = True 

global_min_std = args['min_std']

exp_key = ''
exp_key += f"modeldim{global_model_dim}"
exp_key += f"_lr{args['lr']}"
exp_key += f"_batchsize{args['batch_size']}"
exp_key += f"_epoch{args['epochs']}"
exp_key += f"_ws{args['window_size']}"


exp_dir = project_path / out_dir / dataset_type/ str(args['entity']) / exp_key
base_model_dir = args['base_model_dir']

learning_rate_decay_by_epoch = 10
learning_rate_decay_factor = 1
global_valid_epoch_freq = args['valid_epoch']

dataset_root = pathlib.Path(f"../../Datasets/{dataset_type}")

train_data_json = dataset_root / f"{dataset_type}-train.json"
test_data_json = dataset_root / f"{dataset_type}-test.json"

train_data,_ = load_data_from_json(train_data_json)
test_data,label = load_data_from_json(test_data_json)

min_std = global_min_std
global_window_size = args['window_size']

bf_search_min = 0
bf_search_max = 1000
bf_search_step_size = 1
chosed_index = [args['entity']]








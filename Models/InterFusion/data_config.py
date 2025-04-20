import math
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

    k = 20
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

def preprocess(df_train, df_test):
    return preprocess_meanstd(df_train, df_test)

def load_data_from_json(s):
    json_file = json.load(open(s))
    data = json_file['data']
    label = json_file['label']
    return data,label

def global_pretrain_ELBO_loss(self, x, z, q_zx, p_z, p_xz):
    index_loss_weight = [1 for _ in range(self.x_dims)]
    index_loss_weight_tensor = torch.tensor(index_loss_weight).to(device=global_device)
    log_p_xz = torch.sum(p_xz.log_prob(x).mul(index_loss_weight_tensor)/torch.sum(index_loss_weight_tensor), dim=(-2, -1)) # samplen, batch, time, xdim
    log_q_zx = torch.sum(q_zx.log_prob(z), dim=(-2, -1))  # samplen, batch, xdim, time
    log_p_z = torch.sum(p_z.log_prob(z), dim=(-2, -1))  # samplen, batch, xdim, time
    return -torch.mean(log_p_xz+log_p_z-log_q_zx)
    
def global_train_ELBO_loss(self, x: torch.Tensor, z1_sampled: torch.Tensor, z1_sampled_flowed: torch.Tensor, z1_flow_log_det: torch.Tensor, z2_sampled: torch.Tensor, 
    p_xz_dist, q_z1_dist, q_z2_dist, pz1_dist, pz2_dist):
    index_loss_weight = [1 for _ in range(self.x_dims)]
    index_loss_weight_tensor = torch.tensor(index_loss_weight).to(device=global_device)
    log_p_xz = torch.sum(p_xz_dist.log_prob(x),dim=(-2, -1))
    log_p_z1 = torch.sum(pz1_dist.log_prob(z1_sampled_flowed), dim=(-2, -1))
    log_p_z2 = torch.sum(pz2_dist.log_prob(z2_sampled), dim=(-2, -1))
    log_p_z = log_p_z1 + log_p_z2
    log_q_z1 = torch.sum(
        torch.sum(q_z1_dist.log_prob(z1_sampled), dim=(-1,)) - 
        torch.sum(z1_flow_log_det, dim=-1), 
        dim=(-1,))
    log_q_z2 = torch.sum(q_z2_dist.log_prob(z2_sampled), dim=(-2, -1))
    log_q_zx = log_q_z1 + log_q_z2

    return -torch.mean(log_p_xz+log_p_z-log_q_zx), -torch.mean(log_p_xz), -torch.mean(log_p_z-log_q_zx)

def get_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset_type', type=str, default='application-server-dataset')
    parser.add_argument('--out_dir', type=str, default='hpo_result')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--base_model_dir', type=str, default="test")

    parser.add_argument('--z1_dim', type=int, default=3)
    parser.add_argument('--model_dim', type=int, default=100)

    parser.add_argument('--epochs', type=int, default= 60)
    parser.add_argument('--pretrain_epochs', type=int, default= 60)
    parser.add_argument('--train_lr', type=float, default= 5e-4)
    parser.add_argument('--pretrain_lr', type=float, default= 8e-3)
    parser.add_argument('--seed', type=int, default=2020)  
    parser.add_argument('--train_type', type=str, default='noshare')
    parser.add_argument('--valid_epoch', type=int, default=5) 
    parser.add_argument('--entity', type=int, default=1) 
    parser.add_argument('--window_size', type=int, default=60) 

    return parser.parse_known_args()[0]

logger = logging.getLogger('interfusion_AutoML')

project_path = pathlib.Path(os.path.abspath(__file__)).parent

if 'args' not in globals():
    tuner_params= nni.get_next_parameter()
    logger.debug("tuner_params:", tuner_params)
    # args = vars(merge_parameter(get_params(), tuner_params))
    args = vars(get_params())
    logger.debug("updated args:", args)

single_score_th = 10000

out_dir = "out/" + args['out_dir']
GPU_index = str(args['gpu_id'])
global_device = torch.device(f'cuda:{GPU_index}')

dataset_type = args['dataset_type']
global_train_epochs = args['epochs']

global_pretrain_epochs = args['pretrain_epochs']
global_pretrain_epochs = args['epochs']
seed = args['seed']

global_z1_dim= args['z1_dim']
global_rnn_dims= args['model_dim']
global_dense_dims= args['model_dim']
global_batch_size= args['batch_size']
global_window_size = args['window_size']

pretrain_lr = args['pretrain_lr']
train_lr = args['train_lr']
clip_std_min = math.exp(-5)
clip_std_max = math.exp(2)
if_freeze_seq = True
train_type = args['train_type']

exp_key = ''
exp_key += f"z1dim{global_z1_dim}"
exp_key += f"_modeldim{global_dense_dims}"
exp_key += f'_prelr{pretrain_lr}'
exp_key += f"_lr{train_lr}"
exp_key += f"_batchsize{global_batch_size}"
exp_key += f"_epoch{global_train_epochs}"

exp_key += f"_ws{global_window_size}"
exp_dir = project_path / out_dir / dataset_type/ str(args['entity']) / exp_key
base_model_dir = args['base_model_dir']

learning_rate_decay_by_epoch = 10
learning_rate_decay_factor = 1
global_valid_epoch_freq = args['valid_epoch']

output_padding_list = [0, 1, 1]
global_z2_dim = 8
bf_search_min = 0
bf_search_max = 1000
bf_search_step_size = 1
noshare_save_dir = project_path / base_model_dir

dataset_root = pathlib.Path(f"../../Datasets/{dataset_type}")
train_data_json = dataset_root / f"{dataset_type}-train.json"
test_data_json = dataset_root / f"{dataset_type}-test.json"
train_data,_ = load_data_from_json(train_data_json)
test_data,label = load_data_from_json(test_data_json)
chosed_index = [args['entity']]






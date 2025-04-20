from cgi import print_environ
import pathlib
import json
import numpy as np
import os
import argparse
import torch
from nni.utils import merge_parameter
import nni
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_minmax(df_train, df_test):
    """
    normalize raw data
    """
    print('minmax', end=' ')
    df_train = np.asarray(df_train, dtype=np.float32)
    df_test = np.asarray(df_test, dtype=np.float32)
    if len(df_train.shape) == 1 or len(df_test.shape) == 1:
        raise ValueError('Data must be a 2-D array')
    if np.any(sum(np.isnan(df_train)) != 0):
        print('train data contains null values. Will be replaced with 0')
        df_train = np.nan_to_num()
    if np.any(sum(np.isnan(df_test)) != 0):
        print('test data contains null values. Will be replaced with 0')
        df_test = np.nan_to_num()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(df_train)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)
    return df_train, df_test

def preprocess_meanstd(df_train, df_test):
    # print('meanstd', end=' ')
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

def preprocess(df_train, df_test):
    return preprocess_meanstd(df_train, df_test)

def load_data_from_json(s):
    json_file = json.load(open(s))
    data = json_file['data']
    label = json_file['label']
    return data,label

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset_type', type=str, default='application-server-dataset')
    parser.add_argument('--out_dir', type=str, default='hpo_result')
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--base_model_dir', type=str, default='test')

    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--s_dim', type=int, default=8)
    parser.add_argument('--d_dim', type=int, default=10)
    parser.add_argument('--model_dim', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=2020)  
    parser.add_argument('--train_type', type=str, default='noshare')
    parser.add_argument('--valid_epoch', type=int, default=5) 

    parser.add_argument('--min_std', type=float, default=-3) 
    parser.add_argument('--global_window_size', type=int, default=60) 
    parser.add_argument('--entity', type=int, default=1)

    return parser.parse_known_args()[0]

def global_loss_fn(original_seq, recon_seq_mu, recon_seq_logsigma, s_mean,
                s_logvar, d_post_mean, d_post_logvar, d_prior_mean, d_prior_logvar,cluster_id):
        batch_size = original_seq.size(0)

        # print(f"original_seq.shape:{original_seq.shape} recon_seq_mu:{recon_seq_mu.shape}")
        loglikelihood = -0.5 * torch.sum(
            (torch.pow(((original_seq.float() - recon_seq_mu.float()) / torch.exp(recon_seq_logsigma.float())), 2) + 
            2 * recon_seq_logsigma.float()+ np.log(np.pi * 2))
            )
        # See https://arxiv.org/pdf/1606.05908.pdf, Page 9, Section 2.2, Equation (7) for details.
        kld_s = -0.5 * torch.sum(1 + s_logvar - torch.pow(s_mean, 2) - torch.exp(s_logvar))
        # See https://arxiv.org/pdf/1606.05908.pdf, Page 9, Section 2.2, Equation (6) for details.
        d_post_var = torch.exp(d_post_logvar)
        d_prior_var = torch.exp(d_prior_logvar)
        kld_d = 0.5 * torch.sum(d_prior_logvar - d_post_logvar
                                + ((d_post_var + torch.pow(d_post_mean - d_prior_mean, 2)) / d_prior_var)
                                - 1)
        # loss, llh, kld_s, kld_d
        return (-loglikelihood + kld_s + kld_d) / batch_size, -loglikelihood / batch_size, kld_s / batch_size, kld_d / batch_size

logger = logging.getLogger('sdfvae_AutoML')

# global_device = torch.device('cpu')
project_path = pathlib.Path(os.path.abspath(__file__)).parent


if 'args' not in globals():
    tuner_params= nni.get_next_parameter()
    logger.debug("tuner_params:", tuner_params)
    # args = vars(merge_parameter(get_params(), tuner_params))
    args = vars(get_params())
    logger.debug("updated args:", args)


min_std = args['min_std']

single_score_th = 10000
out_dir = "out/" + args['out_dir']

GPU_index = str(args['gpu_id'])
global_device = torch.device(f'cuda:{GPU_index}')
dataset_type = args['dataset_type']
global_epochs = args['epochs']
seed = args['seed']
global_valid_epoch = args['valid_epoch']

global_s_dim = args['s_dim']
global_d_dim = args['d_dim']
global_conv_dim = args['model_dim']
global_hidden_dim = args['model_dim']
global_T = args['T']

global_batch_size= args['batch_size']
global_learning_rate = args['lr']

if_freeze_seq = True
train_type = args['train_type']
global_window_size=args['global_window_size']

exp_key = ''
exp_key += f's{global_s_dim}'
exp_key += f'_d{global_d_dim}'
exp_key += f"_modeldim{args['model_dim']}"
exp_key += f"_lr{args['lr']}"
exp_key += f"_batchsize{args['batch_size']}"
exp_key += f"_epoch{args['epochs']}"
exp_key += f"_ws{global_window_size}"
exp_dir = project_path / out_dir / dataset_type/ str(args['entity']) / exp_key

min_log_sigma = min_std

base_model_dir = args['base_model_dir']

learning_rate_decay_by_step = 100
learning_rate_decay_factor = 1
global_valid_step_freq = None

dataset_root = pathlib.Path(f"../../Datasets/{dataset_type}")
train_data_json = dataset_root / f"{dataset_type}-train.json"
test_data_json = dataset_root / f"{dataset_type}-test.json"

g_min_lr = 1e-4
learning_rate_decay_by_step = 5
learning_rate_decay_factor = 1

global_window_size = 60

bf_search_min = 0
bf_search_max = 1000
bf_search_step_size = 5


train_data,_ = load_data_from_json(train_data_json)
test_data,label = load_data_from_json(test_data_json)

chosed_index = [args['entity']]




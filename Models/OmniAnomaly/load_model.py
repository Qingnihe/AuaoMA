import os
import re
import numpy as np
import torch
import warnings
from omnianomaly.nni_model import OmniAnomaly
from get_eval_result import *
import os
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Suppress warnings
warnings.filterwarnings('ignore')

# Function to extract model parameters from path using regex
def extract_model_params(path):
    params = {}
    params['model_dim'] = int(re.search(r'modeldim(\d+)', path).group(1)) if re.search(r'modeldim(\d+)', path) else None
    params['learning_rate'] = float(re.search(r'lr([0-9e\.-]+)', path).group(1)) if re.search(r'lr([0-9e\.-]+)', path) else None
    params['batch_size'] = int(re.search(r'batchsize(\d+)', path).group(1)) if re.search(r'batchsize(\d+)', path) else None
    params['epochs'] = int(re.search(r'epoch(\d+)', path).group(1)) if re.search(r'epoch(\d+)', path) else None
    params['window_size'] = int(re.search(r'ws(\d+)', path).group(1)) if re.search(r'ws(\d+)', path) else None
    return params

# Function to restore the model with extracted parameters
def load_model(model_path, params, dim):
    model = OmniAnomaly(
        x_dims=dim,
        z_dims=3,
        max_epochs=params['epochs'],
        batch_size=params['batch_size'],
        window_size=params['window_size'],
        learning_rate=params['learning_rate'],
        model_dim=params['model_dim']
    )
    model.restore(model_path)
    return model

# Function to preprocess data
def preprocess_data(train_data, test_data):
    return preprocess_meanstd(train_data, test_data)

# Function to make predictions and save results
def predict_and_save(model, x_test, result_path, suffix, model_path):
    score, mean, _, _ = model.predict(x_test, model_path)
    np.save(os.path.join(result_path, f'test_score{suffix}.npy'), -score)

# Function to load dataset files
def load_data(dataset_path, filenames):
    return {filename: np.load(os.path.join(dataset_path, filename)) for filename in filenames}

def load_create_data(setting, dataset1, dataset2, entity):
    folder = f'out/hpo_result/{dataset1}/{entity}/{setting}_ws60'
    params = extract_model_params(folder)
    
    model_path = os.path.join(folder, 'model', 'model.pkl')
    result_path = os.path.join(folder, 'result_create_data')
    os.makedirs(result_path, exist_ok=True)

    if dataset2 == "ASD":
        dim = 19
    else:
        dim = 38
    model = load_model(model_path, params, dim)

    dataset_path = f'../../Datasets/{dataset2}/{entity}'
    filenames = ['train.npy', 'test.npy', 'test_replace.npy', 'label.npy']
    data = load_data(dataset_path, filenames)

    x_train, x_test = preprocess_data(data['train.npy'], data['test.npy'])
    predict_and_save(model, x_test, result_path, '', model_path)

    x_train, x_test = preprocess_data(data['train.npy'], data['test_replace.npy'])
    predict_and_save(model, x_test, result_path,'_replace', model_path)

def parse_arguments():
    parser = argparse.ArgumentParser(description="specified parameters")
    parser.add_argument('--setting', type=str, required=True, help='Setting')
    parser.add_argument('--dataset1', type=str, required=True, help='The first dataset name')
    parser.add_argument('--dataset2', type=str, required=True, help='The second dataset name')
    parser.add_argument('--entity', type=int, required=True, help='Entity number')
    return parser.parse_args()

def main():
    args = parse_arguments()
    load_create_data(args.setting, args.dataset1, args.dataset2, args.entity)
    
if __name__ == "__main__":
    main()
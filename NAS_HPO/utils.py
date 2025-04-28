import os
import numpy as np
import os
from DbAN.cal import *

def run_model(model_name, params, dataset1, dataset2, entity):
    model_dirs = {
        "IF": "../Models/InterFusion",
        "Omni": "../Models/OmniAnomaly",
        "sdfvae": "../Models/SDFVAE"
    }

    prev_dir = os.getcwd()
    os.chdir(model_dirs[model_name])

    if model_name == "IF":
        command = (
            f"python run.py "
            f"--dataset_type='{dataset1}' --entity={entity} "
            f"--z1_dim={params['z1_dim']} "
            f"--model_dim={params['model_dim']} "
            f"--train_lr={params['train_lr']} "
            f"--pretrain_lr={params['pretrain_lr']} "
            f"--batch_size={params['batch_size']} "
            f"--epochs={params['epochs']}"
        )
    elif model_name == "Omni":
        command = (
            f"python run.py "
            f"--dataset_type='{dataset1}' --entity={entity} "
            f"--model_dim={params['model_dim']} "
            f"--lr={params['lr']} "
            f"--batch_size={params['batch_size']} "
            f"--epochs={params['epochs']}"
        )
    elif model_name == "sdfvae":
        command = (
            f"python run.py "
            f"--dataset_type='{dataset1}' --entity={entity} "
            f"--s_dim={params['s_dim']} "
            f"--d_dim={params['d_dim']} "
            f"--model_dim={params['model_dim']} "
            f"--lr={params['lr']} "
            f"--batch_size={params['batch_size']} "
            f"--epochs={params['epochs']}"
        )
    else:
        pass 

    print(f"Running command for {model_name}:")
    print(command)
    result = os.system(command)

    if result == 0:
        print(f"{model_name} training executed successfully")
    else:
        print(f"{model_name} training execution failed")

    model_settings = {
        "IF": lambda p: f"z1dim{p['z1_dim']}_modeldim{p['model_dim']}_prelr{p['pretrain_lr']}_lr{p['train_lr']}_batchsize{p['batch_size']}_epoch{p['epochs']}",
        "Omni": lambda p: f"modeldim{p['model_dim']}_lr{p['lr']}_batchsize{p['batch_size']}_epoch{p['epochs']}",
        "sdfvae": lambda p: f"s{p['s_dim']}_d{p['d_dim']}_modeldim{p['model_dim']}_lr{p['lr']}_batchsize{p['batch_size']}_epoch{p['epochs']}"
    }

    setting = model_settings[model_name](params)
    command = (
        f"python load_model.py "
        f"--setting={setting} "
        f"--dataset1={dataset1} "
        f"--dataset2={dataset2} "
        f"--entity={entity}"
    )

    print(f"Running command for {model_name}:")
    print(command)
    result = os.system(command)

    if result == 0:
        print(f"{model_name} load_model executed successfully")
    else:
        print(f"{model_name} load_model execution failed")

    os.chdir(prev_dir) 


def evaluate_model(model_name, params, dataset1, dataset2, entity, data_len, create_len, cycle):

    model_settings = {
        "IF": lambda p: f"z1dim{p['z1_dim']}_modeldim{p['model_dim']}_prelr{p['pretrain_lr']}_lr{p['train_lr']}_batchsize{p['batch_size']}_epoch{p['epochs']}",
        "Omni": lambda p: f"modeldim{p['model_dim']}_lr{p['lr']}_batchsize{p['batch_size']}_epoch{p['epochs']}",
        "sdfvae": lambda p: f"s{p['s_dim']}_d{p['d_dim']}_modeldim{p['model_dim']}_lr{p['lr']}_batchsize{p['batch_size']}_epoch{p['epochs']}"
    }


    setting = model_settings[model_name](params)

    print(f"Evaluating {model_name} with setting: {setting}")

    f1_test_score, label, test_score, test_score_replace_ano = read_model_results(model_name, setting, dataset1, dataset2, entity, data_len, create_len)

    f1 = cal_f1(f1_test_score, label)
    DbAN = float(cal_DbAN(dataset2, test_score, test_score_replace_ano, cycle))

    print(f"F1 Score: {float(f1):.4f}, DbAN: {float(DbAN):.4f}")

    return DbAN


def read_model_results(model_name, setting, dataset1, dataset2, entity, data_len, create_len):
    paths = {
        "IF": f'../Models/InterFusion/out/hpo_result/{dataset1}/{entity}',
        "Omni": f'../Models/OmniAnomaly/out/hpo_result/{dataset1}/{entity}',
        "sdfvae": f'../Models/SDFVAE/out/hpo_result/{dataset1}/{entity}',
    }
    path = paths[model_name]

    matching_folder = next((folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder)) and setting in folder), None)
    if not matching_folder:
        raise FileNotFoundError(f"Could not find folder with setting {setting} in {path}")

    result_path1 = os.path.join(path, matching_folder, 'result_create_data', 'test_score.npy')
    result_path2 = os.path.join(path, matching_folder, 'result_create_data', 'test_score_replace.npy')

    f1_result_path = os.path.join(path, matching_folder, 'result', 'test_score.npy')
    label_path = f'../Datasets/{dataset1}/{entity}/test_label.npy'

    test_score = np.load(result_path1)[-create_len:]
    test_score_replace_ano = np.load(result_path2)[-create_len:]
    
    f1_test_score = np.load(f1_result_path)[-data_len:]
    label = np.load(label_path)[-data_len:]

    return f1_test_score, label, test_score, test_score_replace_ano
from sdfvae.evaluate import bf_search
import numpy as np
import pandas as pd
from data_config import *
from tqdm import tqdm


def get_data(data_idx):
    test_score_name = f"test_score"
    test_score = np.load(exp_dir/f'result/{test_score_name}.npy')
    y_test = np.array(label[data_idx][-(len(test_score)):])
    return test_score, None, y_test

def get_bf_by_machine_threhold(calc_latency):
    res_prefix = 'bf' if calc_latency else 'pf'
    machine_best_df = pd.DataFrame()

    for machine_id in tqdm(chosed_index):
        machine_id-=1
        res_point_list = {
            'machine_id': [],
            'tp': [],
            'fp': [],
            'fn': [],
            'p': [],
            'r': [],
            'f1': [],
            'threshold': []
        }

        test_score, _, y_test = get_data(machine_id)

        test_score = np.sum(test_score, axis=-1)
        t, th, predict = bf_search(test_score, y_test,
                            start=bf_search_min,
                            end=bf_search_max,
                            step_num=int((bf_search_max-bf_search_min)//bf_search_step_size),
                            display_freq=1000,
                            calc_latency=calc_latency)
        label_item = y_test
        predict_item = predict.astype(int)
    
        tp_index = np.where((label_item == 1) & (predict_item == 1))
        fp_index = np.where((label_item == 0) & (predict_item == 1))
        fn_index = np.where((label_item == 1) & (predict_item == 0))
        
        res_point_list['machine_id'].append(machine_id)  
        res_point_list['tp'].append(np.sum(((label_item == 1) & (predict_item == 1)).astype(int))) 
        res_point_list['fp'].append(np.sum(((label_item == 0) & (predict_item == 1)).astype(int)))
        res_point_list['fn'].append(np.sum(((label_item == 1) & (predict_item == 0)).astype(int)))
        p = round(res_point_list['tp'][-1] / (res_point_list['tp'][-1]+res_point_list['fp'][-1]+1e-9), 4)
        r = round(res_point_list['tp'][-1] / (res_point_list['tp'][-1]+res_point_list['fn'][-1]+1e-9), 4)
        f1 = round(2*p*r / (p+r+1e-9), 4)
        res_point_list['p'].append(p)
        res_point_list['r'].append(r)
        res_point_list['f1'].append(f1)
        res_point_list['threshold'].append(th)  
        machine_best_df = machine_best_df.append(pd.DataFrame(res_point_list))

    (exp_dir/'evaluation_result').mkdir(exist_ok=True, parents=True)
    machine_best_df = machine_best_df.sort_values(by=['machine_id'])
    machine_best_df.to_csv(exp_dir/f'evaluation_result/{res_prefix}_best_f1.csv', index=False)

def read_bf_all_result(calc_latency):
    test_index = []
    res_prefix = 'bf' if calc_latency else 'pf'
    best_df = pd.read_csv(exp_dir/f'evaluation_result/{res_prefix}_best_f1.csv')
    tp = np.sum(best_df['tp'].values)
    fp = np.sum(best_df['fp'].values)
    fn = np.sum(best_df['fn'].values)
    p = tp / (tp+fp)
    r = tp / (tp+fn)
    f1 = 2*p*r / (p+r)
    res_file = open(exp_dir/f'evaluation_result/{res_prefix}_res.txt', mode='a')
    print(f"all result get_bf f1: {round(f1, 4)} p:{round(p, 4)} r:{round(r, 4)}")
    print(f"all result get_bf f1: {round(f1, 4)} p:{round(p, 4)} r:{round(r, 4)}", file=res_file)
    return round(f1, 4)

def get_exp_result():
    # print("get_exp_result:", exp_key)
    get_bf_by_machine_threhold(True)
    f1 = read_bf_all_result(True)
    return f1
# -*- coding: utf-8 -*-
import numpy as np

from sdfvae.spot import SPOT
from numpy.fft import fft, ifft

def cal_period(time_series: np.ndarray, sampling_rate=1):
    time_series_in_freq = fft(time_series)
    sample_num = len(time_series)
    sample_time = sample_num / sampling_rate

    time_series_in_freq[0] = 0
    main_freq = np.argmax(time_series_in_freq)
    p = round(sample_time / main_freq)

    if 3 * p > sample_num:
        p = 1
    return p


def mse_obj(x_truth: np.ndarray, x_estimated: np.ndarray):
    assert x_truth.shape == x_estimated.shape

    return np.average(np.square(x_truth - x_estimated))


def POF(x_t_back, x_t, x_t_forward):
    eps = 1

    def d(x1, x2):
        return np.sqrt(np.sum(np.square(x1 - x2)))
    
    d_xt_xt_b = d(x_t, x_t_back)
    d_xt_xt_f = d(x_t, x_t_forward)
    d_xt_b_xt_f = d(x_t_back, x_t_forward)

    y1 = (d_xt_xt_b + d_xt_xt_f) / (d_xt_xt_b + d_xt_b_xt_f + eps)
    y2 = (d_xt_xt_b + d_xt_xt_f) / (d_xt_xt_f + d_xt_b_xt_f + eps)

    pof = (y1 + y2) / 2

    if np.isnan(pof):
        pof = 2

    return pof
    

def nf_obj(x_estimated: np.ndarray):
    p = cal_period(x_estimated)

    pos_ls = []
    for pos in range(p, len(x_estimated)- 2*p + 1):
        x_t_back = x_estimated[pos-p : pos]
        x_t = x_estimated[pos : pos+p]
        x_t_forward = x_estimated[pos+p : pos+2*p]

        pos_ls.append(POF(x_t_back, x_t, x_t_forward))

    return np.average(pos_ls)


def mse_nf(x_truth: np.ndarray, x_estimated: np.ndarray):
    return mse_obj(x_truth, x_estimated) + nf_obj(x_estimated)

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t, predict
    else:
        # predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        predict = score > threshold
        t = list(calc_point2point(predict, label))
        t.append(0)

        return t, predict


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=False, calc_latency=False):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).


    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    p = None
    fp_fn_sum = 1e9
    for i in range(search_step):
        threshold += search_range / float(search_step)
        # target, predict = calc_seq(score, label, threshold, calc_latency=True)
        target, predict = calc_seq(score, label, threshold, calc_latency=calc_latency)
        if target[0] >= m[0]:
        # if target[-2]+target[-3]<fp_fn_sum:
            m_t = threshold
            m = target
            p = predict
            fp_fn_sum = target[-2]+target[-3]
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    # print(m, m_t)
    return m, m_t, p



def pot_eval(init_score, score, label, q=1e-3, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            For `OmniAnomaly`, it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            For `OmniAnomaly`, it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t

    Returns:
        dict: pot result dict
    """
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)  # data import
    s.initialize(level=level, min_extrema=True, verbose=False)  # initialization step
    ret = s.run(dynamic=False)  # run
    # print(len(ret['alarms']))
    # print(len(ret['thresholds']))
    pot_th = -np.mean(ret['thresholds'])
    print(f"pot_th:{pot_th}")
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    p_t = calc_point2point(pred, label)
    print('POT result: ', p_t, pot_th, p_latency)
    return {
        'pot-f1': p_t[0],
        'pot-precision': p_t[1],
        'pot-recall': p_t[2],
        'pot-TP': p_t[3],
        'pot-TN': p_t[4],
        'pot-FP': p_t[5],
        'pot-FN': p_t[6],
        'pot-threshold': pot_th,
        'pot-latency': p_latency
    }


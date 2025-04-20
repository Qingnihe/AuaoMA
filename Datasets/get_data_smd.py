import numpy as np
import os
import json

with open('server-machine-dataset/server-machine-dataset-train.json', 'r') as file:
    train_data = json.load(file)

with open('server-machine-dataset/server-machine-dataset-test.json', 'r') as file:
    test_data = json.load(file)

dataset = "server-machine-dataset"
if not os.path.exists(dataset):
    raise FileNotFoundError(f"The folder {dataset} does not exist.")

for i in range(28):
    train1 = train_data['data'][i]
    test1 = test_data['data'][i]
    test_label1 = test_data['label'][i]
    test_label1 = np.array(test_label1)

    train1 = np.array(train1).T
    test1 = np.array(test1).T

    sub_folder = os.path.join(dataset, str(i + 1))
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    np.save(os.path.join(sub_folder, f'train.npy'), train1)
    np.save(os.path.join(sub_folder, f'test.npy'), test1)
    np.save(os.path.join(sub_folder, f'test_label.npy'), test_label1)

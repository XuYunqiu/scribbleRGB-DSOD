import os
import cv2
import argparse
from sklearn.metrics import mean_absolute_error


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./dataset', help='path where to save testing data')
parser.add_argument('--save_dir', default='./results', help='path where to save predicted maps')
opt = parser.parse_args()

dataset_list = ['NJU2K_Test', 'STERE', 'DES', 'NLPR_Test', 'LFSD', 'SIP', 'SSD']
all_mae = []
for dataset in dataset_list:
    gt_dir = os.path.join(opt.data_dir, 'test_data', 'gt', dataset)
    pred_dir = os.path.join(opt.save_dir, dataset)
    mae = 0
    for file in os.listdir(pred_dir):
        pred = cv2.imread(os.path.join(pred_dir, file),0) / 255
        gt = cv2.imread(os.path.join(gt_dir, file),0) / 255
        mae += mean_absolute_error(gt, pred)
    mae = mae / len(os.listdir(pred_dir))
    all_mae.append(mae)

for dataset, mae in zip(dataset_list, all_mae):
    print('{}: {:0.4f}'.format(dataset, mae))

import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn.functional as F

from model.denet_models import DENet_VGG
from data import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--warmup_stage', action='store_true', help='warmup stage without consistency training')
parser.add_argument('--second_round', action='store_true', help='second-round model')
parser.add_argument('--data_dir', default='./dataset', help='path where to save testing data')
parser.add_argument('--model_path', default='./models/teacher_scribble_30.pth', help='path where to save trained model')
parser.add_argument('--save_dir', default='./results', help='path where to save predicted maps')
opt = parser.parse_args()

model = DENet_VGG(channel=32, warmup_stage=opt.warmup_stage or opt.second_round)
model.load_state_dict(torch.load(opt.model_path))

model.cuda()
model.eval()

test_datasets = ['DES', 'LFSD','NJU2K_Test','NLPR_Test','SIP','STERE', 'SSD']
for dataset in test_datasets:
    save_path = os.path.join(opt.save_dir, dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = os.path.join(opt.data_dir, 'test_data', 'img', dataset)
    depth_root = os.path.join(opt.data_dir, 'test_data', 'depth', dataset)
    test_loader = test_dataset(image_root, depth_root, opt.testsize)
    for i in tqdm(range(test_loader.size)):
        image, depth, HH, WW, name, image_edge, depth_edge = test_loader.load_data()
        _, _, res, _ = model(image.cuda(), depth.cuda(), image_edge.cuda(), depth_edge.cuda())
        res = F.interpolate(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = np.round(res * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_path, name), res)
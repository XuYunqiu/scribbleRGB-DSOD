import os
import argparse
from datetime import datetime
import numpy as np
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F

from model.denet_models import DENet_VGG
from data import get_loader
from utils import clip_gradient, adjust_lr, label_edge_prediction, visualize_prediction
from losses import smooth_loss, ssim_loss


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')

parser.add_argument('--sm_loss_weight', type=float, default=0.1, help='weight for smoothness loss')
parser.add_argument('--edge_loss_weight', type=float, default=1.0, help='weight for edge loss')
parser.add_argument('--con_loss_weight', type=float, default=0.01, help='consistency loss weight')
parser.add_argument('--l1_loss_weight', type=float, default=0.15, help='l1 loss weight')
parser.add_argument('--ssim_loss_weight', type=float, default=0.85, help='ssim loss weight')

parser.add_argument('--warmup_stage', action='store_true', help='warmup stage without consistency training')
parser.add_argument('--keep_teacher', action='store_true', help='keep teacher model during training')
parser.add_argument('--resize_max', type=int, default=480, help='max size when resize')
parser.add_argument('--resize_min', type=int, default=256, help='min size when resize')
parser.add_argument('--mom_coef', type=float, default=0.999, help='momentum coefficient for teacher model updating')

parser.add_argument('--data_dir', default='./dataset', help='path where to save training data')
parser.add_argument('--output_dir', default='./models', help='path where to save trained models')
parser.add_argument('--vis_dir', default='', help='path where to save visualizations, empty for no saving')
parser.add_argument('--warmup_model', default='./models/scribble_30.pth', help='path where to save warm-up model')
opt = parser.parse_args()

CE = torch.nn.BCELoss()
smooth_loss = smooth_loss.smoothness_loss(size_average=True)
SSIM = ssim_loss.SSIM()


@torch.no_grad()
def teacher_model_update(teacher_model, student_model, m=0.999):
    student_model_dict = student_model.state_dict()
    new_teacher_dict = OrderedDict()
    for key, value in teacher_model.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = student_model_dict[key] * (1 - m) + value * m
        else:
            raise Exception("{} is not found in student model".format(key))

    teacher_model.load_state_dict(new_teacher_dict)
    return teacher_model


def train(train_loader, model, teacher_model, optimizer, epoch):
    model.train()
    if not opt.warmup_stage:
        teacher_model.train()

    total_step = len(train_loader)
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        if opt.warmup_stage:
            images, depths, gts, masks, grays, image_edges, depth_edges = pack
        else:
            images, depths, gts, masks, grays, image_edges, depth_edges, \
            images_t, depths_t, image_edges_t, depth_edges_t, flip_flag = pack

            new_size = np.random.randint(opt.resize_min, opt.resize_max + 1)
            images = F.interpolate(images, size=new_size, mode='bilinear', align_corners=False)
            depths = F.interpolate(depths, size=new_size, mode='bilinear', align_corners=False)
            image_edges = F.interpolate(image_edges, size=new_size, mode='bilinear', align_corners=False)
            depth_edges = F.interpolate(depth_edges, size=new_size, mode='bilinear', align_corners=False)
            gts = F.interpolate(gts, size=new_size, mode='bilinear', align_corners=False)
            masks = F.interpolate(masks, size=new_size, mode='bilinear', align_corners=False)
            grays = F.interpolate(grays, size=new_size, mode='bilinear', align_corners=False)

        images = images.cuda()
        depths = depths.cuda()
        gts = gts.cuda()
        masks = masks.cuda()
        grays = grays.cuda()
        image_edges = image_edges.cuda()
        depth_edges = depth_edges.cuda()

        sal1, edge_map, sal2, edge_map_depth = model(images, depths, image_edges, depth_edges)

        img_size = images.size(2) * images.size(3) * images.size(0)
        ratio = img_size / torch.sum(masks)

        if not opt.warmup_stage:
            with torch.no_grad():
                sal1_teacher, edge_map_teacher, sal2_teacher, edge_map_depth_teacher = teacher_model(
                    images_t.cuda(), depths_t.cuda(), image_edges_t.cuda(), depth_edges_t.cuda()
                )
                torch.flip(sal2_teacher[torch.where(flip_flag == 1)], dims=[1])
                sal2_teacher_prob = torch.sigmoid(sal2_teacher)

        sal1_prob = torch.sigmoid(sal1)
        sal1_prob = sal1_prob * masks
        sal2_prob = torch.sigmoid(sal2)
        sal2_prob_all = sal2_prob.clone()
        sal2_prob = sal2_prob * masks

        smoothLoss_cur1 = opt.sm_loss_weight * smooth_loss(torch.sigmoid(sal1), grays)

        sal_loss1 = ratio * CE(sal1_prob, gts * masks) + smoothLoss_cur1
        smoothLoss_cur2 = opt.sm_loss_weight * smooth_loss(torch.sigmoid(sal2), grays)
        sal_loss2 = ratio * CE(sal2_prob, gts * masks) + smoothLoss_cur2
        edges_gt = torch.sigmoid(sal2).detach()
        edge_loss = opt.edge_loss_weight * CE(torch.sigmoid(edge_map), label_edge_prediction(edges_gt))
        edge_loss += opt.edge_loss_weight * CE(torch.sigmoid(edge_map_depth), label_edge_prediction(edges_gt))

        warm_loss = sal_loss1 + edge_loss + sal_loss2
        if opt.vis_dir:
            if not os.path.exists(opt.vis_dir):
                os.makedirs(opt.vis_dir)
            visualize_prediction(torch.sigmoid(sal1), 'sal1', opt.vis_dir)
            visualize_prediction(torch.sigmoid(sal2), 'sal2', opt.vis_dir)
            visualize_prediction(torch.sigmoid(edge_map), 'edge', opt.vis_dir)
            visualize_prediction(torch.sigmoid(edge_map_depth), 'depth_edge', opt.vis_dir)

        if opt.warmup_stage:
            loss = warm_loss
        else:
            sal2_prob_all = F.interpolate(
                sal2_prob_all, size=images_t.size(2), mode='bilinear', align_corners=False
            )
            l1_loss = torch.mean(torch.abs(sal2_prob_all - sal2_teacher_prob))
            ssim_loss = torch.mean(1 - SSIM(sal2_prob_all, sal2_teacher_prob))
            consist_loss = l1_loss * opt.l1_loss_weight + ssim_loss * opt.ssim_loss_weight
            loss = warm_loss + consist_loss * opt.con_loss_weight
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        if not opt.warmup_stage:
            teacher_model = teacher_model_update(teacher_model, model, opt.mom_coef)

        if i % 10 == 0 or i == total_step:
            if opt.warmup_stage:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], sal1_loss: {:0.4f}, edge_loss: {:0.4f}, sal2_loss: {:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, sal_loss1.data, edge_loss.data, sal_loss2.data))
            else:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], sal1_loss: {:0.4f}, edge_loss: {:0.4f}, sal2_loss: {:0.4f}, consist_loss: {:0.4f}'.format(
                        datetime.now(), epoch, opt.epoch, i, total_step, sal_loss1.data, edge_loss.data, sal_loss2.data, consist_loss.data))

    save_path = opt.output_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % 10 == 0:
        if opt.warmup_stage:
            torch.save(model.state_dict(), os.path.join(save_path, 'scribble_{}.pth'.format(epoch)))
        else:
            torch.save(teacher_model.state_dict(), os.path.join(save_path, 'teacher_scribble_{}.pth'.format(epoch)))


def main():
    print("Scribble it! (warm-up stage)") if opt.warmup_stage else print("Scribble it! (mutual learning stage)")
    print('Learning Rate: {}'.format(opt.lr))
    if opt.warmup_stage:
        model = DENet_VGG(channel=32, warmup_stage=True)
        teacher_model = None
    else:
        model = DENet_VGG(channel=32, warmup_stage=False)
        model.load_state_dict(torch.load(opt.warmup_model))
        teacher_model = DENet_VGG(channel=32)
        if opt.keep_teacher:
            teacher_model.load_state_dict(torch.load(opt.warmup_model))
            teacher_model.cuda()
            for teacher_param in teacher_model.parameters():
                teacher_param.detach_()

    model.cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    train_loader = get_loader(
        os.path.join(opt.data_dir, 'train_data', 'img'), os.path.join(opt.data_dir, 'train_data', 'depth'),
        os.path.join(opt.data_dir, 'train_data', 'gt'), os.path.join(opt.data_dir, 'train_data', 'mask'),
        os.path.join(opt.data_dir, 'train_data', 'gray'), batchsize=opt.batchsize, 
        trainsize=opt.trainsize, warmup_stage=opt.warmup_stage
    )

    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        if not opt.keep_teacher and teacher_model is not None:
            teacher_model.load_state_dict(torch.load(opt.warmup_model))
            teacher_model.cuda()
            for teacher_param in teacher_model.parameters():
                teacher_param.detach_()
        train(train_loader, model, teacher_model, optimizer, epoch)


if __name__ == '__main__':
    main()
